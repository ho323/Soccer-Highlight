"""
Highlight Generator Module for Soccer Videos
CALF inference 결과를 활용하여 축구 영상의 하이라이트를 자동 생성합니다.
"""

import os
import gc
import logging
import time
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Dict, Optional, Tuple
import tempfile

import numpy as np
import torch
import ffmpy

from dataset import SoccerNetClipsTesting
from model import ContextAwareModel
from preprocessing import batch2long, timestamps2long, NMS
from config.classes import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

# Fixing seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class HighlightGenerator:
    """축구 영상 하이라이트 생성기"""
    
    def __init__(
        self,
        video_path: str,
        model_name: str = "CALF",
        output_dir: str = "inference/outputs/highlights",
        num_features: int = 512,
        framerate: int = 2,
        chunk_size: int = 120,
        receptive_field: int = 40,
        dim_capsule: int = 16,
        gpu: int = -1
    ):
        self.video_path = video_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_features = num_features
        self.framerate = framerate
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.dim_capsule = dim_capsule
        
        # GPU 설정
        if gpu >= 0:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "clips"), exist_ok=True)
        
        self.model = None
        self.dataset = None
        self.detections = None
        
    def _load_model(self) -> None:
        """CALF 모델 로드"""
        logging.info("Loading CALF model...")
        
        # 데이터셋 초기화 (feature extraction 포함)
        self.dataset = SoccerNetClipsTesting(
            path=self.video_path,
            features="ResNET_PCA512.npy",
            framerate=self.framerate,
            chunk_size=self.chunk_size * self.framerate,
            receptive_field=self.receptive_field * self.framerate
        )
        
        # 모델 생성
        self.model = ContextAwareModel(
            weights=None,
            input_size=self.num_features,
            num_classes=self.dataset.num_classes,
            chunk_size=self.chunk_size * self.framerate,
            dim_capsule=self.dim_capsule,
            receptive_field=self.receptive_field * self.framerate,
            num_detections=self.dataset.num_detections,
            framerate=self.framerate
        ).cuda()
        
        # 체크포인트 로드 (스크립트 상위 디렉토리의 models 폴더)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(script_dir, "..", "models", self.model_name, "model.pth.tar")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        logging.info("Model loaded successfully")
        
    def extract_timestamps(
        self,
        confidence_threshold: float = 0.5,
        event_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        CALF 모델로 이벤트 timestamp 추출
        
        Args:
            confidence_threshold: 검출 신뢰도 임계값
            event_types: 포함할 이벤트 타입 리스트 (None이면 모든 이벤트)
            
        Returns:
            이벤트 정보 딕셔너리 리스트 [{timestamp, event_type, confidence}, ...]
        """
        if self.model is None:
            self._load_model()
            
        logging.info("Extracting timestamps...")
        
        # 데이터 로더 생성
        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        chunk_size = self.model.chunk_size
        receptive_field = self.model.receptive_field
        
        spotting_predictions = []
        
        # 추론 실행
        with torch.no_grad():
            for feat_half1, size in test_loader:
                feat_half1 = feat_half1.cuda().squeeze(0).unsqueeze(1)
                output_segmentation, output_spotting = self.model(feat_half1)
                
                timestamp_long = timestamps2long(
                    output_spotting.cpu().detach(),
                    size,
                    chunk_size,
                    receptive_field
                )
                spotting_predictions.append(timestamp_long)
        
        # NMS 적용
        detections_numpy = []
        for detection in spotting_predictions:
            detections_numpy.append(NMS(detection.numpy(), 20 * self.framerate))
        
        self.detections = detections_numpy[0]
        
        # 이벤트 타입 인덱스 필터링
        if event_types is not None:
            valid_indices = [EVENT_DICTIONARY_V2[e] for e in event_types if e in EVENT_DICTIONARY_V2]
        else:
            valid_indices = list(range(17))
        
        # timestamp 추출
        events = []
        frames, classes = np.where(self.detections >= confidence_threshold)
        
        for frame_idx, class_idx in zip(frames, classes):
            if class_idx not in valid_indices:
                continue
                
            confidence = self.detections[frame_idx, class_idx]
            timestamp_sec = frame_idx / self.framerate
            
            events.append({
                'timestamp': timestamp_sec,
                'frame': frame_idx,
                'event_type': INVERSE_EVENT_DICTIONARY_V2[class_idx],
                'event_idx': class_idx,
                'confidence': float(confidence)
            })
        
        # 시간순 정렬
        events.sort(key=lambda x: x['timestamp'])
        
        logging.info(f"Found {len(events)} events above threshold {confidence_threshold}")
        for event in events:
            logging.info(f"  {event['event_type']} at {event['timestamp']:.1f}s (conf: {event['confidence']:.3f})")
        
        return events
    
    def extract_clips(
        self,
        events: List[Dict],
        before_event: float = 5.0,
        after_event: float = 5.0,
        max_clips: Optional[int] = None
    ) -> List[str]:
        """
        각 이벤트 timestamp 주변 클립 추출
        
        Args:
            events: extract_timestamps()에서 반환된 이벤트 리스트
            before_event: 이벤트 전 시간(초)
            after_event: 이벤트 후 시간(초)
            max_clips: 최대 클립 수 (None이면 무제한)
            
        Returns:
            추출된 클립 파일 경로 리스트
        """
        logging.info(f"Extracting clips (before: {before_event}s, after: {after_event}s)...")
        
        clips_dir = os.path.join(self.output_dir, "clips")
        
        # 기존 클립 정리
        for f in os.listdir(clips_dir):
            if f.endswith('.mp4'):
                os.remove(os.path.join(clips_dir, f))
        
        if max_clips is not None:
            events = events[:max_clips]
        
        clip_paths = []
        
        for i, event in enumerate(events):
            start_time = max(0, event['timestamp'] - before_event)
            duration = before_event + after_event
            
            clip_filename = f"clip_{i:03d}_{event['event_type'].replace(' ', '_')}_{event['timestamp']:.1f}s.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)
            
            logging.info(f"  Extracting clip {i+1}/{len(events)}: {event['event_type']} at {event['timestamp']:.1f}s")
            
            try:
                ff = ffmpy.FFmpeg(
                    inputs={self.video_path: f"-ss {start_time}"},
                    outputs={clip_path: f"-t {duration} -c:v libx264 -c:a aac -y"}
                )
                ff.run()
                clip_paths.append(clip_path)
            except Exception as e:
                logging.error(f"  Failed to extract clip: {e}")
                continue
        
        logging.info(f"Extracted {len(clip_paths)} clips")
        return clip_paths
    
    def merge_clips(
        self,
        clip_paths: List[str],
        output_filename: str = "highlight.mp4"
    ) -> str:
        """
        클립들을 하나의 하이라이트 영상으로 병합
        
        Args:
            clip_paths: 클립 파일 경로 리스트
            output_filename: 출력 파일명
            
        Returns:
            최종 하이라이트 영상 경로
        """
        if not clip_paths:
            logging.warning("No clips to merge")
            return ""
            
        logging.info(f"Merging {len(clip_paths)} clips...")
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        # concat 파일 리스트 생성
        list_file = os.path.join(self.output_dir, "clips", "concat_list.txt")
        with open(list_file, 'w') as f:
            for clip_path in clip_paths:
                # 절대 경로로 변환
                abs_path = os.path.abspath(clip_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        try:
            ff = ffmpy.FFmpeg(
                inputs={list_file: "-f concat -safe 0"},
                outputs={output_path: "-c copy -y"}
            )
            ff.run()
            logging.info(f"Highlight saved to: {output_path}")
        except Exception as e:
            logging.error(f"Failed to merge clips: {e}")
            # re-encode 시도
            logging.info("Retrying with re-encoding...")
            try:
                ff = ffmpy.FFmpeg(
                    inputs={list_file: "-f concat -safe 0"},
                    outputs={output_path: "-c:v libx264 -c:a aac -y"}
                )
                ff.run()
                logging.info(f"Highlight saved to: {output_path}")
            except Exception as e2:
                logging.error(f"Failed to merge clips with re-encoding: {e2}")
                return ""
        
        return output_path
    
    def cleanup(self):
        """GPU 리소스 해제"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.dataset is not None:
            del self.dataset
            self.dataset = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("HighlightGenerator GPU resources released")

    def generate(
        self,
        confidence_threshold: float = 0.5,
        event_types: Optional[List[str]] = None,
        before_event: float = 5.0,
        after_event: float = 5.0,
        max_clips: Optional[int] = None,
        output_filename: str = "highlight.mp4"
    ) -> str:
        """
        전체 하이라이트 생성 파이프라인 실행
        
        Args:
            confidence_threshold: 검출 신뢰도 임계값
            event_types: 포함할 이벤트 타입 리스트
            before_event: 이벤트 전 시간(초)
            after_event: 이벤트 후 시간(초)
            max_clips: 최대 클립 수
            output_filename: 출력 파일명
            
        Returns:
            최종 하이라이트 영상 경로
        """
        logging.info("=" * 60)
        logging.info("Starting highlight generation pipeline")
        logging.info("=" * 60)
        
        start_time = time.time()
        
        # 1. Timestamp 추출
        events = self.extract_timestamps(confidence_threshold, event_types)
        
        if not events:
            logging.warning("No events found. Highlight generation aborted.")
            return ""
        
        # 2. 클립 추출
        clip_paths = self.extract_clips(events, before_event, after_event, max_clips)
        
        if not clip_paths:
            logging.warning("No clips extracted. Highlight generation aborted.")
            return ""
        
        # 3. 클립 병합
        output_path = self.merge_clips(clip_paths, output_filename)
        
        elapsed = time.time() - start_time
        logging.info("=" * 60)
        logging.info(f"Highlight generation completed in {elapsed:.1f}s")
        logging.info(f"Output: {output_path}")
        logging.info("=" * 60)
        
        return output_path


def get_available_events() -> List[str]:
    """사용 가능한 이벤트 타입 목록 반환"""
    return list(EVENT_DICTIONARY_V2.keys())


def main():
    """CLI 진입점"""
    parser = ArgumentParser(
        description='Soccer Highlight Generator using CALF model',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    # 필수 인자
    parser.add_argument(
        '--video_path',
        required=True,
        type=str,
        help='입력 영상 경로'
    )
    
    # 출력 설정
    parser.add_argument(
        '--output_path',
        required=False,
        type=str,
        default="highlight.mp4",
        help='출력 하이라이트 영상 파일명'
    )
    parser.add_argument(
        '--output_dir',
        required=False,
        type=str,
        default="inference/outputs/highlights",
        help='출력 디렉토리'
    )
    
    # 클립 설정
    parser.add_argument(
        '--clip_duration',
        required=False,
        type=float,
        default=10.0,
        help='총 클립 길이(초) - before_event + after_event 대신 사용시'
    )
    parser.add_argument(
        '--before_event',
        required=False,
        type=float,
        default=None,
        help='이벤트 전 시간(초)'
    )
    parser.add_argument(
        '--after_event',
        required=False,
        type=float,
        default=None,
        help='이벤트 후 시간(초)'
    )
    
    # 이벤트 필터링
    parser.add_argument(
        '--confidence_threshold',
        required=False,
        type=float,
        default=0.5,
        help='검출 신뢰도 임계값'
    )
    parser.add_argument(
        '--event_types',
        required=False,
        nargs='+',
        type=str,
        default=None,
        help=f'포함할 이벤트 타입 (가능한 값: {", ".join(get_available_events())})'
    )
    parser.add_argument(
        '--max_clips',
        required=False,
        type=int,
        default=None,
        help='최대 클립 수'
    )
    
    # 모델 설정
    parser.add_argument(
        '--model_name',
        required=False,
        type=str,
        default="CALF_benchmark",
        help='모델 이름 (models/ 디렉토리 내)'
    )
    parser.add_argument(
        '--num_features',
        required=False,
        type=int,
        default=512,
        help='입력 feature 차원'
    )
    parser.add_argument(
        '--framerate',
        required=False,
        type=int,
        default=2,
        help='Feature extraction framerate'
    )
    parser.add_argument(
        '--chunk_size',
        required=False,
        type=int,
        default=120,
        help='Chunk size (초)'
    )
    parser.add_argument(
        '--receptive_field',
        required=False,
        type=int,
        default=40,
        help='Receptive field (초)'
    )
    parser.add_argument(
        '--dim_capsule',
        required=False,
        type=int,
        default=16,
        help='Capsule dimension'
    )
    
    # 기타
    parser.add_argument(
        '--GPU',
        required=False,
        type=int,
        default=-1,
        help='GPU ID (-1 for default)'
    )
    parser.add_argument(
        '--loglevel',
        required=False,
        type=str,
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(
        args.output_dir,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S_highlight.log')
    )
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # before/after 설정
    if args.before_event is None and args.after_event is None:
        before_event = args.clip_duration / 2
        after_event = args.clip_duration / 2
    else:
        before_event = args.before_event if args.before_event is not None else 5.0
        after_event = args.after_event if args.after_event is not None else 5.0
    
    # 이벤트 타입 검증
    if args.event_types is not None:
        available = get_available_events()
        for et in args.event_types:
            if et not in available:
                logging.error(f"Unknown event type: {et}")
                logging.error(f"Available: {available}")
                return
    
    # 파라미터 로깅
    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(f"  {arg.rjust(20)} : {getattr(args, arg)}")
    
    # 하이라이트 생성
    generator = HighlightGenerator(
        video_path=args.video_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_features=args.num_features,
        framerate=args.framerate,
        chunk_size=args.chunk_size,
        receptive_field=args.receptive_field,
        dim_capsule=args.dim_capsule,
        gpu=args.GPU
    )
    
    output_path = generator.generate(
        confidence_threshold=args.confidence_threshold,
        event_types=args.event_types,
        before_event=before_event,
        after_event=after_event,
        max_clips=args.max_clips,
        output_filename=args.output_path
    )
    
    if output_path:
        logging.info(f"Highlight generated successfully: {output_path}")
    else:
        logging.error("Highlight generation failed")


if __name__ == '__main__':
    main()


import logging
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional

from gpu_memory_manager import gpu_phase
from highlight_generator import HighlightGenerator, get_available_events
from result_writer import InferenceResultWriter
from scene_describer import SceneDescriber
from scene_search import SceneSearchEngine
from tts_generator import TTSGenerator

logger = logging.getLogger(__name__)


class EnhancedHighlightGenerator:
    def __init__(
        self,
        video_path: str,
        model_name: str = "CALF_benchmark",
        output_dir: str = "inference/outputs/highlights",
        num_features: int = 512,
        framerate: int = 2,
        chunk_size: int = 120,
        receptive_field: int = 40,
        dim_capsule: int = 16,
        gpu: int = -1,
        vlm_model_id: Optional[str] = None,
        language: str = "auto",
        search_engine: str = "semantic",
        hybrid_alpha: float = 0.7,
        enable_tts: bool = False,
        tts_model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.language = language
        self.enable_tts = enable_tts
        self.search_engine_type = search_engine
        self.hybrid_alpha = hybrid_alpha

        self.highlight_gen = HighlightGenerator(
            video_path=video_path,
            model_name=model_name,
            output_dir=output_dir,
            num_features=num_features,
            framerate=framerate,
            chunk_size=chunk_size,
            receptive_field=receptive_field,
            dim_capsule=dim_capsule,
            gpu=gpu,
        )
        self.describer = SceneDescriber(
            output_dir=output_dir,
            model_id=vlm_model_id,
            language=language,
        )
        self.search_engine = SceneSearchEngine(
            output_dir=output_dir,
            engine_type=search_engine,
            hybrid_alpha=hybrid_alpha,
        )
        self.tts = TTSGenerator(output_dir=output_dir, model_id=tts_model_id)
        self.writer = InferenceResultWriter(output_dir=output_dir)

        self.events: List[Dict] = []
        self.clip_paths: List[str] = []
        self.descriptions: List[Dict] = []
        self.tts_entries: List[Dict] = []
        self.latest_result_path: str = ""

    def set_search_engine(self, engine_type: str, hybrid_alpha: Optional[float] = None):
        self.search_engine_type = engine_type
        if hybrid_alpha is not None:
            self.hybrid_alpha = hybrid_alpha
        self.search_engine.set_engine(engine_type, hybrid_alpha=self.hybrid_alpha)

    def set_vlm_model(self, model_id: str):
        self.describer.switch_model(model_id)

    def run_full_pipeline(
        self,
        confidence_threshold: float = 0.5,
        event_types: Optional[List[str]] = None,
        before_event: float = 5.0,
        after_event: float = 5.0,
        max_clips: Optional[int] = None,
        output_filename: str = "highlight.mp4",
        skip_description: bool = False,
        query_signal: str = "",
    ) -> Dict:
        logger.info("Starting enhanced pipeline")
        start = time.time()
        result = {
            "video_path": self.video_path,
            "events": [],
            "clip_paths": [],
            "descriptions": [],
            "tts": [],
            "highlight_path": "",
            "index_ready": False,
            "inference_result_path": "",
        }

        with gpu_phase("Phase 1-2: Feature Extraction + CALF Inference"):
            self.events = self.highlight_gen.extract_timestamps(
                confidence_threshold=confidence_threshold,
                event_types=event_types,
            )
            result["events"] = self.events
            if not self.events:
                logger.warning("No events detected. Stopping.")
                return result

        with gpu_phase("Phase 3: Clip Extraction"):
            self.highlight_gen.cleanup()
            self.clip_paths = self.highlight_gen.extract_clips(
                events=self.events,
                before_event=before_event,
                after_event=after_event,
                max_clips=max_clips,
            )
            result["clip_paths"] = self.clip_paths
            if not self.clip_paths:
                logger.warning("No clips extracted. Stopping.")
                return result

            result["highlight_path"] = self.highlight_gen.merge_clips(self.clip_paths, output_filename)

        if not skip_description:
            with gpu_phase("Phase 4: VLM Description"):
                self.descriptions = self.describer.describe_all_clips(
                    events=self.events,
                    clip_paths=self.clip_paths,
                    language_signal=query_signal,
                )
                self.describer.save_descriptions(self.descriptions)
                result["descriptions"] = self.descriptions
                self.describer.unload_model()

            with gpu_phase("Phase 5: Search Index Build"):
                self.search_engine.build_index(self.descriptions)
                self.search_engine.save_index()
                result["index_ready"] = True

            if self.enable_tts:
                with gpu_phase("Phase 6: Description TTS"):
                    self.tts_entries = self.tts.generate_for_descriptions(self.descriptions)
                    result["tts"] = self.tts_entries

        payload = self.writer.build_payload(
            video_path=self.video_path,
            pipeline_config={
                "confidence_threshold": confidence_threshold,
                "event_types": event_types or [],
                "before_event": before_event,
                "after_event": after_event,
                "max_clips": max_clips,
                "output_filename": output_filename,
                "skip_description": skip_description,
                "vlm_model_id": self.describer.active_model_id,
                "language": self.language,
                "search_engine": self.search_engine_type,
                "hybrid_alpha": self.hybrid_alpha,
                "enable_tts": self.enable_tts,
                "tts_model_id": self.tts.model_id,
            },
            events=self.events,
            clip_paths=self.clip_paths,
            descriptions=self.descriptions,
            tts_entries=self.tts_entries,
            search_index={
                "engine_type": self.search_engine_type,
                "index_files": os.path.join(self.output_dir, "descriptions"),
                "embedding_model": getattr(self.search_engine.engine, "embedding_model", None),
            },
        )
        self.latest_result_path = self.writer.save(payload)
        result["inference_result_path"] = self.latest_result_path

        logger.info("Pipeline completed in %.1fs", time.time() - start)
        logger.info("Result JSON: %s", self.latest_result_path)
        return result

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        search_engine: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> List[Dict]:
        if search_engine:
            self.set_search_engine(search_engine, hybrid_alpha=hybrid_alpha)
        if not self.search_engine.load_index():
            logger.error("No search index available. Run full pipeline first.")
            return []
        return self.search_engine.search(query=query, top_k=top_k, min_score=min_score)

    def generate_search_highlight(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        output_filename: str = "search_highlight.mp4",
        search_engine: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> str:
        results = self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            search_engine=search_engine,
            hybrid_alpha=hybrid_alpha,
        )
        if not results:
            return ""

        results.sort(key=lambda x: x.get("timestamp", 0))
        clip_paths = [r["clip_path"] for r in results if os.path.exists(r.get("clip_path", ""))]
        if not clip_paths:
            return ""
        return self.highlight_gen.merge_clips(clip_paths, output_filename)


def main():
    parser = ArgumentParser(
        description="Enhanced Soccer Highlight Generator with pluggable VLM/TTS/Search",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video_path", required=True, type=str, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="inference/outputs/highlights")
    parser.add_argument("--output_path", type=str, default="highlight.mp4")

    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--event_types", nargs="+", type=str, default=None)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--before_event", type=float, default=5.0)
    parser.add_argument("--after_event", type=float, default=5.0)

    parser.add_argument("--model_name", type=str, default="CALF_benchmark")
    parser.add_argument("--num_features", type=int, default=512)
    parser.add_argument("--framerate", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=120)
    parser.add_argument("--receptive_field", type=int, default=40)
    parser.add_argument("--dim_capsule", type=int, default=16)
    parser.add_argument("--GPU", type=int, default=-1)

    parser.add_argument("--skip_description", action="store_true")
    parser.add_argument("--search_query", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--generate_highlight", action="store_true")

    parser.add_argument("--vlm_model_id", type=str, default=None)
    parser.add_argument("--search_engine", type=str, default="semantic", choices=["semantic", "bm25", "hybrid"])
    parser.add_argument("--hybrid_alpha", type=float, default=0.7)
    parser.add_argument("--enable_tts", action="store_true")
    parser.add_argument("--tts_model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "ko", "en"])
    parser.add_argument("--loglevel", type=str, default="INFO")

    args = parser.parse_args()
    level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_enhanced.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    if args.event_types is not None:
        available = get_available_events()
        for event_type in args.event_types:
            if event_type not in available:
                raise ValueError(f"Unknown event type: {event_type}. Available: {available}")

    generator = EnhancedHighlightGenerator(
        video_path=args.video_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_features=args.num_features,
        framerate=args.framerate,
        chunk_size=args.chunk_size,
        receptive_field=args.receptive_field,
        dim_capsule=args.dim_capsule,
        gpu=args.GPU,
        vlm_model_id=args.vlm_model_id,
        language=args.language,
        search_engine=args.search_engine,
        hybrid_alpha=args.hybrid_alpha,
        enable_tts=args.enable_tts,
        tts_model_id=args.tts_model_id,
    )

    if args.search_query:
        results = generator.search(
            query=args.search_query,
            top_k=args.top_k,
            min_score=args.min_score,
            search_engine=args.search_engine,
            hybrid_alpha=args.hybrid_alpha,
        )
        if not results:
            print("No results found.")
            return

        print(f"\n{'=' * 70}")
        print(f"Search Results for: '{args.search_query}'")
        print(f"{'=' * 70}")
        for row in results:
            print(
                f"#{row['search_rank']} [{row['search_engine']}:{row['search_score']:.3f}] "
                f"{row['event_type']} @ {row['timestamp']:.1f}s"
            )
            print(f"  {row.get('description', '')[:160]}")

        if args.generate_highlight:
            path = generator.generate_search_highlight(
                query=args.search_query,
                top_k=args.top_k,
                min_score=args.min_score,
                output_filename="search_highlight.mp4",
                search_engine=args.search_engine,
                hybrid_alpha=args.hybrid_alpha,
            )
            if path:
                print(f"Search highlight saved: {path}")
        return

    result = generator.run_full_pipeline(
        confidence_threshold=args.confidence_threshold,
        event_types=args.event_types,
        before_event=args.before_event,
        after_event=args.after_event,
        max_clips=args.max_clips,
        output_filename=args.output_path,
        skip_description=args.skip_description,
        query_signal=args.search_query or "",
    )
    if result.get("highlight_path"):
        logger.info("Highlight generated: %s", result["highlight_path"])
    else:
        logger.error("Failed to generate highlight")


if __name__ == "__main__":
    main()

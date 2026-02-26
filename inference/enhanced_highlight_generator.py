import logging
import os
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional

import ffmpy

from gpu_memory_manager import gpu_phase
from highlight_generator import HighlightGenerator, get_available_events
from path_utils import default_highlight_output_dir
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
        output_dir: str = default_highlight_output_dir(),
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

    @staticmethod
    def _format_mmss(seconds: float) -> str:
        s = max(0, int(seconds))
        return f"{s // 60:02d}:{s % 60:02d}"

    def _fallback_descriptions_from_events(self, events: List[Dict], clip_paths: List[str]) -> List[Dict]:
        rows: List[Dict] = []
        count = min(len(events), len(clip_paths))
        for i in range(count):
            ev = events[i]
            ts = float(ev.get("timestamp", 0.0))
            et = ev.get("event_type", "Unknown")
            rows.append(
                {
                    "clip_index": i,
                    "clip_path": clip_paths[i],
                    "timestamp": ts,
                    "event_type": et,
                    "confidence": float(ev.get("confidence", 0.0)),
                    "description": f"{et} scene at {self._format_mmss(ts)}.",
                    "language": "en",
                    "model_id": "event_fallback",
                    "prompt_version": "fallback_v1",
                }
            )
        return rows

    @staticmethod
    def _filter_overlapping_events(events: List[Dict], before_event: float, after_event: float) -> List[Dict]:
        if not events:
            return []

        def interval(ev: Dict):
            ts = float(ev.get("timestamp", 0.0))
            return max(0.0, ts - before_event), ts + after_event

        # Keep only the highest-confidence event for overlaps of the same event_type.
        ranked = sorted(
            events,
            key=lambda e: (str(e.get("event_type", "")), -float(e.get("confidence", 0.0))),
        )
        selected: List[Dict] = []
        selected_by_type: Dict[str, List] = {}

        for ev in ranked:
            s, e = interval(ev)
            event_type = str(ev.get("event_type", ""))
            overlapped = False
            for ss, ee in selected_by_type.get(event_type, []):
                if not (e <= ss or s >= ee):
                    overlapped = True
                    break
            if overlapped:
                continue
            selected.append(ev)
            selected_by_type.setdefault(event_type, []).append((s, e))

        selected.sort(key=lambda e: float(e.get("timestamp", 0.0)))
        return selected

    @staticmethod
    def _select_non_overlapping_highlight_indices(events: List[Dict], before_event: float, after_event: float) -> List[int]:
        if not events:
            return []

        intervals = []
        for i, ev in enumerate(events):
            ts = float(ev.get("timestamp", 0.0))
            intervals.append(
                {
                    "idx": i,
                    "start": max(0.0, ts - before_event),
                    "end": ts + after_event,
                    "timestamp": ts,
                    "confidence": float(ev.get("confidence", 0.0)),
                }
            )

        intervals.sort(key=lambda x: x["start"])
        selected = []
        cursor = 0
        n = len(intervals)
        while cursor < n:
            cluster = [intervals[cursor]]
            cluster_end = intervals[cursor]["end"]
            j = cursor + 1
            while j < n and intervals[j]["start"] < cluster_end:
                cluster.append(intervals[j])
                cluster_end = max(cluster_end, intervals[j]["end"])
                j += 1

            best = max(cluster, key=lambda x: (x["confidence"], -x["timestamp"]))
            selected.append(best["idx"])
            cursor = j

        selected.sort()
        return selected

    def _mix_tts_into_clips(self, clip_paths: List[str], tts_entries: List[Dict]) -> List[str]:
        if not clip_paths or not tts_entries:
            return clip_paths

        mixed_paths: List[str] = []
        out_dir = os.path.join(self.output_dir, "clips_tts")
        os.makedirs(out_dir, exist_ok=True)

        for i, clip_path in enumerate(clip_paths):
            audio_path = ""
            if i < len(tts_entries):
                audio_path = tts_entries[i].get("audio_path", "")
            if not audio_path or not os.path.exists(audio_path):
                mixed_paths.append(clip_path)
                continue

            mixed_path = os.path.join(out_dir, f"clip_tts_{i:04d}.mp4")
            try:
                ff = ffmpy.FFmpeg(
                    inputs={clip_path: None, audio_path: None},
                    outputs={
                        mixed_path: (
                            '-filter_complex "[0:a]volume=0.22[a0];[1:a]volume=1.0[a1];'
                            '[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]" '
                            '-map 0:v -map "[aout]" -c:v copy -c:a aac -y'
                        )
                    },
                )
                ff.run()
                if os.path.exists(mixed_path):
                    mixed_paths.append(mixed_path)
                else:
                    mixed_paths.append(clip_path)
            except Exception:
                mixed_paths.append(clip_path)

        return mixed_paths

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
            "warnings": [],
        }

        with gpu_phase("Phase 1-2: Feature Extraction + CALF Inference"):
            self.events = self.highlight_gen.extract_timestamps(
                confidence_threshold=confidence_threshold,
                event_types=event_types,
            )
            original_count = len(self.events)
            self.events = self._filter_overlapping_events(
                self.events,
                before_event=before_event,
                after_event=after_event,
            )
            if len(self.events) < original_count:
                logger.info(
                    "Removed %s overlapping events by confidence-based filtering (%s -> %s).",
                    original_count - len(self.events),
                    original_count,
                    len(self.events),
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

            highlight_indices = self._select_non_overlapping_highlight_indices(
                self.events,
                before_event=before_event,
                after_event=after_event,
            )
            highlight_clip_paths = [self.clip_paths[i] for i in highlight_indices if i < len(self.clip_paths)]
            if not highlight_clip_paths:
                highlight_clip_paths = self.clip_paths
            result["highlight_path"] = self.highlight_gen.merge_clips(highlight_clip_paths, output_filename)

        if not skip_description:
            try:
                with gpu_phase("Phase 4: VLM Description"):
                    self.descriptions = self.describer.describe_all_clips(
                        events=self.events,
                        clip_paths=self.clip_paths,
                        language_signal=query_signal,
                    )
                    self.describer.save_descriptions(self.descriptions)
                    result["descriptions"] = self.descriptions
            except Exception as exc:
                msg = f"Description phase skipped due to error: {exc}"
                logger.warning(msg)
                result["warnings"].append(msg)
                self.descriptions = []
            finally:
                self.describer.unload_model()

            if not self.descriptions:
                self.descriptions = self._fallback_descriptions_from_events(self.events, self.clip_paths)
                result["descriptions"] = self.descriptions
                result["warnings"].append("Using fallback event-based descriptions for search/TTS.")

            with gpu_phase("Phase 5: Search Index Build"):
                self.search_engine.build_index(self.descriptions)
                self.search_engine.save_index()
                result["index_ready"] = True

            if self.enable_tts:
                with gpu_phase("Phase 6: Description TTS"):
                    self.tts_entries = self.tts.generate_for_descriptions(self.descriptions)
                    result["tts"] = self.tts_entries

                with gpu_phase("Phase 7: TTS Mixdown"):
                    mixed_clip_paths = self._mix_tts_into_clips(self.clip_paths, self.tts_entries)
                    highlight_indices = self._select_non_overlapping_highlight_indices(
                        self.events,
                        before_event=before_event,
                        after_event=after_event,
                    )
                    mixed_for_highlight = [mixed_clip_paths[i] for i in highlight_indices if i < len(mixed_clip_paths)]
                    if not mixed_for_highlight:
                        mixed_for_highlight = mixed_clip_paths
                    tts_highlight = self.highlight_gen.merge_clips(mixed_for_highlight, "highlight_tts.mp4")
                    if tts_highlight and os.path.exists(tts_highlight):
                        result["highlight_path"] = tts_highlight
                        result["warnings"].append("Highlight audio includes generated TTS narration mix.")
        else:
            self.descriptions = self._fallback_descriptions_from_events(self.events, self.clip_paths)
            result["descriptions"] = self.descriptions
            with gpu_phase("Phase 5: Search Index Build (Fallback)"):
                self.search_engine.build_index(self.descriptions)
                self.search_engine.save_index()
                result["index_ready"] = True

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
    parser.add_argument("--output_dir", type=str, default=default_highlight_output_dir())
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

import json
import os
from datetime import datetime
from typing import Dict, List


def _rel_or_abs(path: str, base: str) -> str:
    if not path:
        return path
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return path


class InferenceResultWriter:
    def __init__(self, output_dir: str = "inference/outputs/highlights"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_payload(
        self,
        video_path: str,
        pipeline_config: Dict,
        events: List[Dict],
        clip_paths: List[str],
        descriptions: List[Dict],
        tts_entries: List[Dict],
        search_index: Dict,
    ) -> Dict:
        run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        clips = []
        for i, path in enumerate(clip_paths):
            event = events[i] if i < len(events) else {}
            start = max(0.0, float(event.get("timestamp", 0.0)) - float(pipeline_config.get("before_event", 0.0)))
            end = max(start, float(event.get("timestamp", 0.0)) + float(pipeline_config.get("after_event", 0.0)))
            clips.append(
                {
                    "clip_id": f"clip_{i:03d}",
                    "clip_path": _rel_or_abs(path, self.output_dir),
                    "start_sec": start,
                    "end_sec": end,
                    "event_ref": i if i < len(events) else None,
                }
            )

        payload = {
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "video_path": video_path,
            "pipeline_config": pipeline_config,
            "events": events,
            "clips": clips,
            "descriptions": [
                {
                    "clip_id": f"clip_{int(d.get('clip_index', idx)):03d}",
                    "clip_index": d.get("clip_index", idx),
                    "language": d.get("language", "en"),
                    "text": d.get("description", ""),
                    "model_id": d.get("model_id", ""),
                    "prompt_version": d.get("prompt_version", "v1"),
                }
                for idx, d in enumerate(descriptions)
            ],
            "tts": [
                {
                    **entry,
                    "audio_path": _rel_or_abs(entry.get("audio_path", ""), self.output_dir)
                    if entry.get("audio_path")
                    else "",
                }
                for entry in tts_entries
            ],
            "search_index": search_index,
        }
        return payload

    def save(self, payload: Dict, filename: str = "inference_result.json") -> str:
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

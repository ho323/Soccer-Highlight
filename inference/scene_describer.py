import json
import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from language_router import resolve_language
from model_registry import ModelRegistry
from path_utils import default_highlight_output_dir, default_model_registry_path
from vlm_backends import BACKEND_BY_TYPE

logger = logging.getLogger(__name__)

EVENT_CONTEXT_EN = {
    "Penalty": "a penalty kick situation",
    "Kick-off": "a kick-off at the start or restart of play",
    "Goal": "a goal being scored",
    "Substitution": "a player substitution",
    "Offside": "an offside call",
    "Shots on target": "a shot on target (towards the goal)",
    "Shots off target": "a shot off target (missing the goal)",
    "Clearance": "a defensive clearance",
    "Ball out of play": "the ball going out of play",
    "Throw-in": "a throw-in",
    "Foul": "a foul being committed",
    "Indirect free-kick": "an indirect free-kick",
    "Direct free-kick": "a direct free-kick",
    "Corner": "a corner kick",
    "Yellow card": "a yellow card being shown",
    "Red card": "a red card being shown",
    "Yellow->red card": "a second yellow card leading to a red card",
}

EVENT_CONTEXT_KO = {
    "Penalty": "페널티킥 상황",
    "Kick-off": "킥오프 상황",
    "Goal": "골이 발생하는 장면",
    "Substitution": "선수 교체 장면",
    "Offside": "오프사이드 판정 장면",
    "Shots on target": "유효 슈팅 장면",
    "Shots off target": "무효 슈팅 장면",
    "Clearance": "수비 클리어링 장면",
    "Ball out of play": "볼이 아웃되는 장면",
    "Throw-in": "스로인 장면",
    "Foul": "파울 장면",
    "Indirect free-kick": "간접 프리킥 장면",
    "Direct free-kick": "직접 프리킥 장면",
    "Corner": "코너킥 장면",
    "Yellow card": "옐로카드 장면",
    "Red card": "레드카드 장면",
    "Yellow->red card": "경고 누적으로 퇴장되는 장면",
}


class SceneDescriber:
    def __init__(
        self,
        output_dir: str = default_highlight_output_dir(),
        registry_path: str = default_model_registry_path(),
        model_id: Optional[str] = None,
        language: str = "auto",
    ):
        self.output_dir = output_dir
        self.descriptions_dir = os.path.join(output_dir, "descriptions")
        os.makedirs(self.descriptions_dir, exist_ok=True)

        self.registry = ModelRegistry(registry_path=registry_path)
        self.model_config = self.registry.get_model(model_id=model_id)
        if self.model_config is None:
            raise RuntimeError("No VLM model is configured in registry.")

        backend_type = self.model_config.get("model_type", "qwen_vl")
        backend_cls = BACKEND_BY_TYPE.get(backend_type)
        if backend_cls is None:
            raise RuntimeError(f"Unsupported backend type: {backend_type}")

        self.backend = backend_cls(self.model_config)
        self.language = language

    @property
    def active_model_id(self) -> str:
        return self.model_config.get("id", "")

    def switch_model(self, model_id: str) -> None:
        if model_id == self.active_model_id:
            return

        self.unload_model()
        model_config = self.registry.get_model(model_id=model_id)
        if model_config is None:
            raise ValueError(f"Model id not found in registry: {model_id}")

        backend_type = model_config.get("model_type", "qwen_vl")
        backend_cls = BACKEND_BY_TYPE.get(backend_type)
        if backend_cls is None:
            raise RuntimeError(f"Unsupported backend type: {backend_type}")

        self.model_config = model_config
        self.backend = backend_cls(self.model_config)

    def load_model(self) -> None:
        self.backend.load_model()

    def unload_model(self) -> None:
        self.backend.unload_model()

    def _sample_frames(self, clip_path: str, num_frames: int = 4, max_size: int = 448) -> List[Image.Image]:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            logger.error("Cannot open video: %s", clip_path)
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames

    def _build_prompt(self, event_type: str, language: str) -> str:
        if language == "ko":
            context = EVENT_CONTEXT_KO.get(event_type, "축구 액션 장면")
            return (
                f"이 프레임들은 축구 경기 영상의 {context}입니다. "
                f"장면에서 일어나는 핵심 행동을 객관적으로 2~3문장으로 설명하세요. "
                f"핵심 이벤트({event_type}), 주요 선수 움직임, 경기 흐름을 포함하세요."
            )

        context = EVENT_CONTEXT_EN.get(event_type, "a soccer action")
        return (
            f"These frames are from a soccer match clip showing {context}. "
            f"Describe the scene objectively in 2-3 sentences. "
            f"Include the key event ({event_type}), player actions, and game context."
        )

    def describe_clip(
        self,
        clip_path: str,
        event_type: str = "Unknown",
        language_signal: str = "",
    ) -> Dict:
        frames = self._sample_frames(clip_path, num_frames=4)
        if not frames:
            return {"language": "en", "description": f"[Failed to extract frames from {os.path.basename(clip_path)}]"}

        lang = resolve_language(self.language, signal_text=language_signal)
        prompt_text = self._build_prompt(event_type, lang)
        description = self.backend.describe_frames(frames, prompt_text)
        return {"language": lang, "description": description}

    def describe_all_clips(
        self,
        events: List[Dict],
        clip_paths: List[str],
        language_signal: str = "",
    ) -> List[Dict]:
        if len(events) != len(clip_paths):
            logger.warning(
                "Event count (%s) != clip count (%s). Using min.",
                len(events),
                len(clip_paths),
            )

        self.load_model()
        out = []
        count = min(len(events), len(clip_paths))
        for i in range(count):
            event = events[i]
            clip_path = clip_paths[i]
            event_type = event.get("event_type", "Unknown")
            desc = self.describe_clip(
                clip_path=clip_path,
                event_type=event_type,
                language_signal=language_signal,
            )
            out.append(
                {
                    "clip_index": i,
                    "clip_path": clip_path,
                    "timestamp": event.get("timestamp", 0.0),
                    "event_type": event_type,
                    "confidence": event.get("confidence", 0.0),
                    "description": desc["description"],
                    "language": desc["language"],
                    "model_id": self.model_config.get("id", ""),
                    "prompt_version": "v2",
                }
            )

        logger.info("Generated descriptions for %s clips", len(out))
        return out

    def save_descriptions(self, descriptions: List[Dict], filename: str = "scene_descriptions.json") -> str:
        output_path = os.path.join(self.descriptions_dir, filename)
        save_data = []
        for desc in descriptions:
            entry = dict(desc)
            if os.path.isabs(entry.get("clip_path", "")):
                entry["clip_path"] = os.path.relpath(entry["clip_path"], self.output_dir)
            save_data.append(entry)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        logger.info("Descriptions saved to: %s", output_path)
        return output_path

    def load_descriptions(self, filename: str = "scene_descriptions.json") -> List[Dict]:
        path = os.path.join(self.descriptions_dir, filename)
        if not os.path.exists(path):
            logger.warning("Description file not found: %s", path)
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            clip_path = entry.get("clip_path", "")
            if clip_path and not os.path.isabs(clip_path):
                entry["clip_path"] = os.path.join(self.output_dir, clip_path)
        return data

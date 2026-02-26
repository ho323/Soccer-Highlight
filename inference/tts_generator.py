import json
import logging
import os
import wave
from typing import Dict, List

import numpy as np
from path_utils import default_highlight_output_dir

logger = logging.getLogger(__name__)


class TTSGenerator:
    def __init__(
        self,
        output_dir: str = default_highlight_output_dir(),
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ):
        self.output_dir = output_dir
        self.model_id = model_id
        self.audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

        self._pipeline = None
        self._load_error = None

    def _load_pipeline(self):
        if self._pipeline is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text-to-speech", model=self.model_id)
            logger.info("Loaded TTS model: %s", self.model_id)
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning("Failed to load TTS model '%s': %s", self.model_id, exc)

    def _write_wav(self, path: str, audio: np.ndarray, sample_rate: int) -> None:
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = audio.squeeze()
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    def _synthesize(self, text: str) -> Dict:
        self._load_pipeline()
        if self._pipeline is None:
            return {
                "status": "failed",
                "error": self._load_error or "TTS pipeline unavailable",
            }

        try:
            result = self._pipeline(text)
            audio = result["audio"]
            sample_rate = int(result.get("sampling_rate", 24000))
            return {"status": "ok", "audio": audio, "sample_rate": sample_rate}
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    def generate_for_descriptions(self, descriptions: List[Dict]) -> List[Dict]:
        manifest = []
        for desc in descriptions:
            clip_idx = int(desc.get("clip_index", 0))
            clip_id = f"clip_{clip_idx:03d}"
            text = desc.get("description", "")
            out_path = os.path.join(self.audio_dir, f"{clip_id}.wav")

            synth = self._synthesize(text)
            if synth["status"] != "ok":
                entry = {
                    "clip_id": clip_id,
                    "clip_index": clip_idx,
                    "audio_path": "",
                    "voice_model_id": self.model_id,
                    "sample_rate": None,
                    "duration_sec": None,
                    "status": "failed",
                    "error": synth.get("error", "unknown"),
                }
                manifest.append(entry)
                continue

            self._write_wav(out_path, synth["audio"], synth["sample_rate"])
            duration_sec = float(len(np.asarray(synth["audio"]).squeeze()) / synth["sample_rate"])
            entry = {
                "clip_id": clip_id,
                "clip_index": clip_idx,
                "audio_path": out_path,
                "voice_model_id": self.model_id,
                "sample_rate": synth["sample_rate"],
                "duration_sec": duration_sec,
                "status": "ok",
            }
            manifest.append(entry)

        manifest_path = os.path.join(self.audio_dir, "tts_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("TTS manifest saved: %s", manifest_path)

        return manifest

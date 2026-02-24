import json
import os
from typing import Dict, List, Optional


DEFAULT_MODELS = {
    "models": [
        {
            "id": "qwen2vl_soccer_merged_local",
            "label": "Qwen2VL Soccer Merged (Local)",
            "model_type": "qwen_vl",
            "hf_path_or_local_path": "C:/Models/qwen2vl_soccer_merged",
            "dtype": "float16",
            "device_map": "auto",
            "enabled": True,
        }
    ]
}


class ModelRegistry:
    def __init__(self, registry_path: str = "inference/config/models.json"):
        self.registry_path = registry_path
        self._data = None

    def _ensure_registry(self) -> None:
        if os.path.exists(self.registry_path):
            return
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_MODELS, f, ensure_ascii=False, indent=2)

    def load(self) -> Dict:
        self._ensure_registry()
        with open(self.registry_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        if "models" not in self._data:
            self._data["models"] = []
        return self._data

    def list_models(self, enabled_only: bool = True) -> List[Dict]:
        data = self.load()
        models = data.get("models", [])
        if not enabled_only:
            return models
        return [m for m in models if m.get("enabled", True)]

    def get_model(self, model_id: Optional[str] = None) -> Optional[Dict]:
        models = self.list_models(enabled_only=True)
        if not models:
            return None

        if model_id:
            for model in models:
                if model.get("id") == model_id:
                    return model

        return models[0]

    def get_model_choices(self) -> List[str]:
        return [m.get("label", m.get("id", "unknown")) for m in self.list_models(True)]

    def resolve_id_from_label(self, label: str) -> Optional[str]:
        for model in self.list_models(True):
            if model.get("label") == label:
                return model.get("id")
        return None

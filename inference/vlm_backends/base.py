from abc import ABC, abstractmethod
from typing import Dict, List

from PIL.Image import Image


class SceneDescriberBackend(ABC):
    def __init__(self, model_config: Dict):
        self.model_config = model_config

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def describe_frames(self, frames: List[Image], prompt_text: str) -> str:
        pass

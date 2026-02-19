import logging
from typing import List

import torch
from PIL.Image import Image

from gpu_memory_manager import cleanup_pytorch, log_gpu_memory
from vlm_backends.base import SceneDescriberBackend

logger = logging.getLogger(__name__)


class QwenVLBackend(SceneDescriberBackend):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        if self.model is not None:
            return

        model_id = self.model_config["hf_path_or_local_path"]
        dtype_name = self.model_config.get("dtype", "float16")
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        device_map = self.model_config.get("device_map", "auto")

        logger.info("Loading VLM model: %s", model_id)
        log_gpu_memory("VLM Load START")

        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=28 * 28 * 4,
            max_pixels=28 * 28 * 256,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )

        log_gpu_memory("VLM Load END")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        cleanup_pytorch()
        logger.info("VLM backend unloaded")

    def describe_frames(self, frames: List[Image], prompt_text: str) -> str:
        if self.model is None:
            self.load_model()

        image_content = [{"type": "image", "image": frame} for frame in frames]
        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt_text}],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=frames,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

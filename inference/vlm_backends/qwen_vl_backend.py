import logging
import json
import os
import tempfile
from glob import glob
from typing import List

import torch
from PIL.Image import Image

from gpu_memory_manager import cleanup_pytorch, log_gpu_memory
from vlm_backends.base import SceneDescriberBackend

logger = logging.getLogger(__name__)


class _DictConfig(dict):
    def to_dict(self):
        return dict(self)


class QwenVLBackend(SceneDescriberBackend):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = None
        self.processor = None

    @staticmethod
    def _sanitize_hf_config(config) -> None:
        from transformers import AutoConfig

        def _wrap_dict_configs(value):
            if isinstance(value, dict):
                wrapped = _DictConfig()
                for k, v in value.items():
                    wrapped[k] = _wrap_dict_configs(v)
                return wrapped
            if isinstance(value, list):
                return [_wrap_dict_configs(v) for v in value]
            return value

        for key, value in list(vars(config).items()):
            if not key.endswith("_config") or not isinstance(value, dict):
                continue

            model_type = value.get("model_type")
            if model_type:
                try:
                    kwargs = {k: v for k, v in value.items() if k != "model_type"}
                    setattr(config, key, AutoConfig.for_model(model_type, **kwargs))
                    continue
                except Exception as exc:
                    logger.warning("Failed to convert config.%s to AutoConfig: %s", key, exc)

            # Some exported local configs keep nested *_config as plain dict.
            # GenerationConfig expects objects exposing .to_dict(), so wrap recursively.
            setattr(config, key, _wrap_dict_configs(value))

    @staticmethod
    def _estimate_local_model_size_bytes(model_dir: str) -> int:
        total = 0
        for path in glob(os.path.join(model_dir, "*.safetensors")):
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
        return total

    def load_model(self) -> None:
        if self.model is not None:
            return

        model_id = self.model_config["hf_path_or_local_path"]
        is_local_model = os.path.isdir(model_id)
        if os.path.isdir(model_id):
            preprocessor_config = os.path.join(model_id, "preprocessor_config.json")
            processor_config = os.path.join(model_id, "processor_config.json")
            if not os.path.exists(preprocessor_config) and os.path.exists(processor_config):
                logger.warning(
                    "preprocessor_config.json not found. Creating from processor_config.json: %s",
                    model_id,
                )
                with open(processor_config, "r", encoding="utf-8") as f:
                    processor_data = json.load(f)
                image_processor_data = processor_data.get("image_processor", processor_data)
                with open(preprocessor_config, "w", encoding="utf-8") as f:
                    json.dump(image_processor_data, f, ensure_ascii=False, indent=2)

        dtype_name = self.model_config.get("dtype", "float16")
        dtype = torch.float16 if dtype_name == "float16" else torch.float32
        device_map = self.model_config.get("device_map", "auto")
        quantization = str(self.model_config.get("quantization", "")).strip().lower()
        use_8bit = quantization in ("8bit", "int8", "load_in_8bit")
        use_4bit = quantization in ("4bit", "int4", "load_in_4bit")
        allow_cpu_offload = bool(self.model_config.get("allow_cpu_offload", False))
        gpu_memory_ratio = float(self.model_config.get("gpu_memory_ratio", 0.75))
        cpu_memory_gb = int(self.model_config.get("cpu_memory_gb", 48))

        if is_local_model and torch.cuda.is_available() and str(device_map).lower() != "cpu":
            model_bytes = self._estimate_local_model_size_bytes(model_id)
            gpu_bytes = int(torch.cuda.get_device_properties(0).total_memory)
            scale = 1.0
            if use_8bit:
                scale = 0.55
            elif use_4bit:
                scale = 0.30
            effective_bytes = int(model_bytes * scale)
            # Large local checkpoints close to GPU VRAM size typically cause very long
            # offload stalls on Windows. Fail fast with an actionable message.
            if effective_bytes > int(gpu_bytes * 0.90) and not allow_cpu_offload:
                model_gb = model_bytes / (1024**3)
                effective_gb = effective_bytes / (1024**3)
                gpu_gb = gpu_bytes / (1024**3)
                raise RuntimeError(
                    f"Local VLM is too large for current GPU memory ({model_gb:.1f}GB raw, ~{effective_gb:.1f}GB effective vs {gpu_gb:.1f}GB VRAM). "
                    "Use stronger quantization/smaller model or enable Skip Description."
                )

        logger.info("Loading VLM model: %s", model_id)
        log_gpu_memory("VLM Load START")

        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=28 * 28 * 4,
                max_pixels=28 * 28 * 256,
                local_files_only=is_local_model,
            )
        except TypeError as e:
            # Some local Qwen2-VL exports include processor_config.json fields
            # that collide with AutoProcessor argument binding.
            if "multiple values for argument 'image_processor'" not in str(e):
                raise
            from transformers import AutoTokenizer
            from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLProcessor

            logger.warning(
                "AutoProcessor init failed due to processor config collision. Falling back to manual Qwen2VLProcessor assembly."
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=is_local_model)
            image_processor = Qwen2VLImageProcessor.from_pretrained(model_id, local_files_only=is_local_model)
            self.processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=tokenizer)

        hf_config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=is_local_model,
        )
        self._sanitize_hf_config(hf_config)

        quantization_kwargs = {}
        if use_8bit or use_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as exc:
                raise RuntimeError(
                    "8bit/4bit quantization requested but bitsandbytes is not available. "
                    "Install bitsandbytes in the active environment."
                ) from exc
            bnb_kwargs = {
                "load_in_8bit": use_8bit,
                "load_in_4bit": use_4bit,
            }
            if use_8bit:
                bnb_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
            offload_dir = os.path.join(tempfile.gettempdir(), "qwen2vl_int8_offload")
            os.makedirs(offload_dir, exist_ok=True)
            quantization_kwargs["offload_folder"] = offload_dir

        offload_kwargs = {}
        if allow_cpu_offload and str(device_map).lower() != "cpu":
            offload_dir = os.path.join(tempfile.gettempdir(), "qwen2vl_offload")
            os.makedirs(offload_dir, exist_ok=True)
            offload_kwargs["offload_folder"] = offload_dir
            offload_kwargs["offload_state_dict"] = True
            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_budget = max(4, int(total_gb * gpu_memory_ratio))
                offload_kwargs["max_memory"] = {0: f"{gpu_budget}GiB", "cpu": f"{cpu_memory_gb}GiB"}
            logger.warning(
                "CPU offload mode enabled (device_map=%s, max_memory=%s, offload_folder=%s)",
                device_map,
                offload_kwargs.get("max_memory"),
                offload_dir,
            )

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                config=hf_config,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=is_local_model,
                low_cpu_mem_usage=True,
                **quantization_kwargs,
                **offload_kwargs,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if use_8bit and ("normal_kernel_cpu" in msg or "Char" in msg):
                raise RuntimeError(
                    "8bit quantized loading failed on this environment (bitsandbytes/Windows CPU offload path). "
                    "Use a Linux CUDA environment or switch to non-quantized/smaller model."
                ) from exc
            raise

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

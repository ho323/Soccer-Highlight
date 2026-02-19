"""
GPU Memory Manager
TF2/PyTorch 간 GPU 메모리를 순차적으로 관리하여 16GB VRAM 제약 내에서 동작하도록 합니다.
"""

import gc
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def configure_tf2_memory_growth():
    """TF2가 전체 VRAM을 점유하지 않도록 memory_growth 설정"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            logger.info(f"TF2 memory_growth enabled for {len(gpus)} GPU(s)")
    except Exception as e:
        logger.warning(f"Failed to configure TF2 memory_growth: {e}")


def cleanup_pytorch():
    """PyTorch 모델 해제 및 GPU 캐시 정리"""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("PyTorch GPU cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch cleanup error: {e}")


def cleanup_tensorflow():
    """TensorFlow 세션 정리"""
    try:
        from tensorflow.keras import backend as K
        K.clear_session()
        logger.info("TensorFlow session cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"TensorFlow cleanup error: {e}")
    gc.collect()


def get_gpu_memory_info() -> dict:
    """현재 GPU VRAM 사용량 반환"""
    info = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 ** 2)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            info.update({
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_mb": total,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "free_mb": total - allocated,
            })
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU memory info error: {e}")
    return info


def log_gpu_memory(prefix: str = ""):
    """GPU 메모리 상태를 로그에 출력"""
    info = get_gpu_memory_info()
    if info.get("available"):
        tag = f"[{prefix}] " if prefix else ""
        logger.info(
            f"{tag}GPU Memory: {info['allocated_mb']:.0f}MB allocated / "
            f"{info['total_mb']:.0f}MB total "
            f"({info['free_mb']:.0f}MB free)"
        )
    else:
        logger.info(f"{'[' + prefix + '] ' if prefix else ''}GPU memory info not available")


@contextmanager
def gpu_phase(name: str):
    """
    Phase별 GPU 메모리 자동 관리 컨텍스트 매니저.

    Usage:
        with gpu_phase("Phase 2: CALF Inference"):
            # GPU 작업 수행
            ...
        # 자동으로 PyTorch + TF2 정리
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"Starting {name}")
    logger.info(f"{'=' * 60}")
    log_gpu_memory(f"{name} START")

    try:
        yield
    finally:
        logger.info(f"Finishing {name}, cleaning up GPU memory...")
        cleanup_pytorch()
        cleanup_tensorflow()
        gc.collect()
        log_gpu_memory(f"{name} END")
        logger.info(f"{'=' * 60}")

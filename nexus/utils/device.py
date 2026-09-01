"""GPU Acceleration & Hardware Device Manager for NEXUS.

Auto-detects and configures hardware accelerators (NVIDIA CUDA, Apple Metal MPS, AMD ROCm)
with PyTorch TensorFloat-32 (TF32), cuDNN benchmarking, and automatic memory caching.
"""

from typing import Any, Dict

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.utils.device")

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch: Any = None  # type: ignore[no-redef]


def get_best_device(enable_gpu: bool = True) -> str:
    """Select the optimal hardware device for PyTorch model training and inference.

    Args:
        enable_gpu: Whether GPU usage is enabled in config.

    Returns:
        str: Device identifier string ('cuda', 'mps', or 'cpu')
    """
    if not _HAS_TORCH or torch is None:
        logger.info("PyTorch not installed; using CPU")
        return "cpu"

    if not enable_gpu:
        logger.info("GPU disabled by configuration; using CPU")
        return "cpu"

    # 1. Check NVIDIA CUDA / AMD ROCm
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("NVIDIA CUDA GPU detected: %s (%.2f GB VRAM)", device_name, vram_gb)

        # Enable TensorFloat-32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return "cuda"

    # 2. Check Apple Silicon Metal (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Silicon GPU (Metal Performance Shaders - MPS) detected")
        return "mps"

    logger.info("No supported GPU detected; running on CPU")
    return "cpu"


def clear_gpu_cache() -> None:
    """Free cached GPU memory buffers."""
    if _HAS_TORCH and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (
        _HAS_TORCH
        and torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def get_device_info() -> Dict[str, Any]:
    """Retrieve detailed hardware accelerator metrics."""
    info: Dict[str, Any] = {
        "has_torch": _HAS_TORCH,
        "cuda_available": False,
        "mps_available": False,
        "device_name": "CPU",
        "vram_gb": 0.0,
    }

    if _HAS_TORCH and torch is not None:
        info["cuda_available"] = torch.cuda.is_available()
        if hasattr(torch.backends, "mps"):
            info["mps_available"] = torch.backends.mps.is_available()

        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        elif info["mps_available"]:
            info["device_name"] = "Apple Silicon MPS"

    return info


__all__ = ["get_best_device", "clear_gpu_cache", "get_device_info"]

"""Unit tests for GPU Acceleration & Hardware Device Manager."""

from nexus.utils.device import clear_gpu_cache, get_best_device, get_device_info


def test_get_best_device():
    device = get_best_device(enable_gpu=True)
    assert device in ("cpu", "cuda", "mps")

    # Disabled GPU should always yield cpu
    cpu_device = get_best_device(enable_gpu=False)
    assert cpu_device == "cpu"


def test_clear_gpu_cache():
    # Should execute safely on any platform without throwing exceptions
    clear_gpu_cache()


def test_get_device_info():
    info = get_device_info()
    assert isinstance(info, dict)
    assert "has_torch" in info
    assert "cuda_available" in info
    assert "mps_available" in info
    assert "device_name" in info
    assert "vram_gb" in info

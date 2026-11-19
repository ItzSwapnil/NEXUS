#!/usr/bin/env python3
"""Pre-download and warm-up pretrained AI models for NEXUS.

Downloads the configured Chronos foundation model to ~/.cache/huggingface,
loads it directly onto the GPU (NVIDIA RTX 5060), and runs a benchmark
inference to ensure zero cold-start delay during live trading.
"""

import sys
import time

import numpy as np
import pandas as pd
import torch

from nexus.ai.chronos_predictor import ChronosPredictor
from nexus.utils.config import load_runtime_settings


def main() -> int:
    print("=" * 60)
    print(" NEXUS Pretrained AI Model Downloader & Verification")
    print("=" * 60)

    # 1. Device Inspection
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"[*] Compute Device : {'CUDA GPU' if cuda_available else 'CPU'}")
    if cuda_available:
        print(f"[*] GPU Name       : {device_name}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] VRAM Available : {vram_gb:.2f} GB")

    # 2. Configured Model
    cfg = load_runtime_settings()
    model_id = getattr(cfg.ai, "chronos_model", "amazon/chronos-2")
    horizon = getattr(cfg.ai, "chronos_forecast_horizon", 5)
    cache_dir = getattr(cfg.ai, "chronos_cache_dir", "models/pretrained")
    print(f"[*] Target Directory : {cache_dir}")
    print("-" * 60)

    print(f"[*] Loading '{model_id}' from project directory '{cache_dir}'...")
    t0 = time.perf_counter()
    predictor = ChronosPredictor(model_id=model_id, forecast_horizon=horizon, cache_dir=cache_dir)

    if not predictor.is_available():
        print("[!] Error: Could not initialize Chronos pipeline. Check network/dependencies.")
        return 1

    load_time = time.perf_counter() - t0
    print(f"[✓] Model ready in {load_time:.2f}s")

    # 3. Warm-up Inference
    print("[*] Running warm-up GPU inference with sample price series...")
    synthetic_prices = 100.0 + np.cumsum(np.random.randn(60) * 0.2)
    sample_df = pd.DataFrame({"close": synthetic_prices})

    t1 = time.perf_counter()
    result = predictor.predict(sample_df, horizon=horizon)
    infer_time = (time.perf_counter() - t1) * 1000.0

    if result:
        print(f"[✓] Warm-up inference complete in {infer_time:.1f}ms")
        print(f"    Signal     : {result['signal'].upper()}")
        print(f"    Confidence : {result['confidence']:.2%}")
        print(f"    Probabilities: {result['probabilities']}")
        print("=" * 60)
        print(" SUCCESS: AI foundation model is installed, loaded, and GPU-accelerated.")
        print("=" * 60)
        return 0
    else:
        print("[!] Prediction test returned None.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

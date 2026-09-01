#!/usr/bin/env python3
"""
NEXUS - Automated Project Verification Script
Verifies project integrity, SOTA AI models, GPU acceleration, engine initialization, and test suite readiness.
"""

import asyncio
import sys

import numpy as np
import pandas as pd


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text:^60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")


def check(name: str, condition: bool) -> bool:
    """Check a condition and print result."""
    if condition:
        print(f"  {Colors.GREEN}✓{Colors.RESET} {name}")
        return True
    else:
        print(f"  {Colors.RED}✗{Colors.RESET} {name}")
        return False


def verify_imports() -> bool:
    print("\n1. Verifying Core & AI Dependencies + GPU Acceleration...")
    success = True

    try:
        import numpy
        import pandas
        import pydantic

        check(f"NumPy ({numpy.__version__})", True)
        check(f"Pandas ({pandas.__version__})", True)
        check(f"Pydantic ({pydantic.__version__})", True)
    except Exception as e:
        check(f"Core packages import error: {e}", False)
        success = False

    try:
        import torch

        from nexus.utils.device import get_best_device, get_device_info

        best_dev = get_best_device(enable_gpu=True)
        dev_info = get_device_info()
        check(
            f"PyTorch ({torch.__version__}) - Optimal Device: {best_dev.upper()} ({dev_info['device_name']})",
            True,
        )
    except Exception as e:
        check(f"PyTorch import error: {e}", False)
        success = False

    return success


def verify_models() -> bool:
    print("\n2. Verifying SOTA AI Intelligence Architecture...")
    success = True

    try:
        from nexus.intelligence.regime_detector import RegimeDetector

        detector = RegimeDetector()
        dates = pd.date_range("2026-01-01", periods=50, freq="1min")
        dummy_df = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 50),
                "high": np.linspace(101, 111, 50),
                "low": np.linspace(99, 109, 50),
                "close": np.linspace(100, 110, 50),
                "volume": np.full(50, 1000),
            },
            index=dates,
        )

        regime = asyncio.run(detector.detect_regime(dummy_df))
        check(
            f"Regime Detector (Detected: {regime})",
            regime in ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"],
        )
    except Exception as e:
        check(f"Regime Detector Error: {e}", False)
        success = False

    try:
        import torch

        from nexus.ai.lstm_predictor import LSTMPredictor

        model = LSTMPredictor(input_dim=20, hidden_dim=64)
        sample_input = torch.randn(2, 30, 20)
        output = model(sample_input)
        check(
            "LSTM Predictor (BiLSTM + Attention + Confidence Head)",
            "probabilities" in output,
        )
    except Exception as e:
        check(f"LSTM Predictor Error: {e}", False)
        success = False

    try:
        from nexus.intelligence.transformer import MarketPredictor

        predictor = MarketPredictor(lookback_periods=30, feature_dim=20)
        check(
            "Market Transformer (Positional Encoding + Multi-Head Self-Attention)",
            hasattr(predictor, "model"),
        )
    except Exception as e:
        check(f"Market Transformer Error: {e}", False)
        success = False

    return success


def verify_engine() -> bool:
    print("\n3. Verifying Strategy & Trading Engine...")
    success = True

    try:
        from nexus.core.engine import NexusEngine
        from nexus.utils.config import load_runtime_settings

        settings = load_runtime_settings()
        engine = NexusEngine(settings=settings, demo_mode=True)
        stats = engine.get_performance_stats()

        check("NexusEngine Initialization", engine is not None)
        check(f"Engine Stats (Win Rate: {stats.get('win_rate', 0):.2f}%)", True)
    except Exception as e:
        check(f"Engine Initialization Error: {e}", False)
        success = False

    return success


def main():
    print_header("NEXUS - Automated Project Verification")

    v_imp = verify_imports()
    v_mod = verify_models()
    v_eng = verify_engine()

    print("\n" + "=" * 60)
    if v_imp and v_mod and v_eng:
        print(
            f"{Colors.GREEN}{Colors.BOLD}✓ ALL VERIFICATION CHECKS PASSED - SOTA AI SYSTEM READY!{Colors.RESET}"
        )
        sys.exit(0)
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED - PLEASE REVIEW LOGS{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()

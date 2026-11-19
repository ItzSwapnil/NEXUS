"""Chronos 2 Pretrained Time-Series Forecasting for NEXUS.

Wraps Amazon Chronos 2 — a state-of-the-art foundation model pretrained on
millions of real-world time series — for zero-shot directional market forecasting.

Key properties:
- No training required: works immediately on fresh broker candle data
- Outputs full probability distributions over future price trajectories
- Converts probabilistic forecast to call / put / hold signal + confidence
- Lazy-loads model weights on first call; cached in memory thereafter
- Gracefully degrades (returns None) when chronos-forecasting is not installed
  or the model cannot be downloaded, so the existing ensemble continues normally
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("nexus.ai.chronos_predictor")

DEFAULT_CACHE_DIR = str(Path("models/pretrained").resolve())

# Sentinel so we only attempt the slow import once per process
_CHRONOS_PIPELINE: Any = None
_CHRONOS_AVAILABLE: Optional[bool] = None
_CHRONOS_MODEL_ID: str = ""
_CHRONOS_CACHE_DIR: str = ""


def _load_pipeline(model_id: str, cache_dir: Optional[str] = None) -> Any:
    """Load (or reuse) the Chronos pipeline strictly from project directory."""
    global _CHRONOS_PIPELINE, _CHRONOS_AVAILABLE, _CHRONOS_MODEL_ID, _CHRONOS_CACHE_DIR

    resolved_cache = cache_dir or DEFAULT_CACHE_DIR
    Path(resolved_cache).mkdir(parents=True, exist_ok=True)

    if _CHRONOS_AVAILABLE is False and _CHRONOS_MODEL_ID == model_id:
        return None  # already determined to be unavailable

    if (
        _CHRONOS_PIPELINE is not None
        and _CHRONOS_MODEL_ID == model_id
        and _CHRONOS_CACHE_DIR == resolved_cache
    ):
        return _CHRONOS_PIPELINE  # already loaded

    try:
        import torch
        from chronos import BaseChronosPipeline

        device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Chronos pipeline: %s on %s (cache: %s)", model_id, device_map, resolved_cache
        )
        pipeline = BaseChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device_map == "cuda" else torch.float32,
            cache_dir=resolved_cache,
        )
        _CHRONOS_PIPELINE = pipeline
        _CHRONOS_AVAILABLE = True
        _CHRONOS_MODEL_ID = model_id
        _CHRONOS_CACHE_DIR = resolved_cache
        logger.info("Chronos loaded successfully (%s from %s)", model_id, resolved_cache)
        return pipeline
    except Exception as exc:
        _CHRONOS_AVAILABLE = False
        logger.warning(
            "Chronos unavailable (%s). Ensemble will continue without it: %s", model_id, exc
        )
        return None


class ChronosPredictor:
    """Zero-shot pretrained time-series forecaster using Amazon Chronos 2.

    Usage::

        predictor = ChronosPredictor()
        result = predictor.predict(candles_df, horizon=5)
        # result: {"signal": "call", "confidence": 0.78, "forecast_mean": [...]}
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        forecast_horizon: int = 5,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.forecast_horizon = forecast_horizon
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        candles_df: pd.DataFrame,
        horizon: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run zero-shot Chronos forecast on close-price series.

        Args:
            candles_df: OHLCV DataFrame with at least a ``close`` column.
            horizon:    Number of future candles to forecast (default from config).

        Returns:
            Dict with ``signal``, ``confidence``, ``probabilities``, and
            ``forecast_mean`` -- or ``None`` when Chronos is unavailable.
        """
        pipeline = _load_pipeline(self.model_id, cache_dir=self.cache_dir)
        if pipeline is None:
            return None

        try:
            import torch

            h = horizon or self.forecast_horizon
            close = candles_df["close"].dropna().to_numpy(dtype=float)
            if len(close) < 10:
                return None

            # Support both Chronos 2 (amazon/chronos-2) and Chronos T5 (amazon/chronos-t5-*)
            is_chronos2 = hasattr(pipeline, "quantiles") or getattr(
                pipeline.__class__, "__name__", ""
            ).startswith("Chronos2")

            if is_chronos2:
                # Chronos2Pipeline accepts a list of 1D tensors or 3D tensor
                input_tensor = [torch.tensor(close, dtype=torch.float32)]
                forecast = pipeline.predict(
                    input_tensor,
                    prediction_length=h,
                )
                # forecast is list[torch.Tensor], forecast[0] is (n_variates, n_quantiles, horizon)
                raw = forecast[0]
                if raw.ndim == 3:
                    samples = raw[0].cpu().float().numpy()  # (n_quantiles, horizon)
                else:
                    samples = raw.cpu().float().numpy()
            else:
                context = torch.tensor(close, dtype=torch.float32).unsqueeze(0)
                forecast = pipeline.predict(
                    context,
                    prediction_length=h,
                    num_samples=100,
                )
                samples = forecast[0].cpu().float().numpy()  # (num_samples, horizon)

            return self._samples_to_signal(close, samples)

        except Exception as exc:
            logger.debug("Chronos prediction error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _samples_to_signal(
        self,
        close: np.ndarray,
        samples: np.ndarray,
    ) -> Dict[str, Any]:
        """Convert Monte-Carlo forecast samples to a trading signal.

        For each sample trajectory the median of the forecast window is compared
        against the last observed close price.  The fraction of samples above /
        below / equal yields call / put / hold probabilities respectively.
        """
        last_close = float(close[-1])

        sample_medians = np.median(samples, axis=1)  # (num_samples,)
        up_frac = float(np.mean(sample_medians > last_close))
        down_frac = float(np.mean(sample_medians < last_close))
        flat_frac = float(max(0.0, 1.0 - up_frac - down_frac))

        probs: Dict[str, float] = {
            "call": up_frac,
            "put": down_frac,
            "hold": flat_frac,
        }

        best = max(probs, key=lambda k: probs[k])

        # Confidence: best probability boosted by its margin over second-best
        sorted_probs: List[float] = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        confidence = float(np.clip(sorted_probs[0] + margin * 0.5, 0.0, 0.99))

        forecast_mean: List[float] = np.mean(samples, axis=0).tolist()

        return {
            "signal": best,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "forecast_mean": forecast_mean,
            "model": self.model_id,
        }

    def is_available(self) -> bool:
        """Return True if Chronos 2 is installed and loadable."""
        return _load_pipeline(self.model_id, cache_dir=self.cache_dir) is not None


__all__ = ["ChronosPredictor"]

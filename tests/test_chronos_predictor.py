"""Unit tests for ChronosPredictor and Chronos ensemble integration."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from nexus.ai.chronos_predictor import ChronosPredictor
from nexus.ai.ensemble_manager import AIEnsembleManager


def _make_candles(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": prices + np.random.randn(n) * 0.1,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": np.random.randint(100, 1000, n),
        }
    )


def test_chronos_samples_to_signal():
    predictor = ChronosPredictor(model_id="amazon/chronos-2", forecast_horizon=5)
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])

    bullish_samples = np.array(
        [
            [104.5, 105.0, 105.5, 106.0, 106.5],
            [104.2, 104.8, 105.2, 105.8, 106.1],
            [104.0, 104.3, 104.7, 105.0, 105.5],
        ]
    )
    result = predictor._samples_to_signal(close, bullish_samples)
    assert result["signal"] == "call"
    assert result["confidence"] > 0.6
    assert result["probabilities"]["call"] == 1.0
    assert len(result["forecast_mean"]) == 5


def test_chronos_predict_mock_chronos2():
    predictor = ChronosPredictor(model_id="amazon/chronos-2", forecast_horizon=5)
    df = _make_candles(30)

    mock_pipeline = MagicMock()
    mock_pipeline.quantiles = [0.1, 0.5, 0.9]
    mock_tensor = torch.tensor([[[105.0] * 5, [106.0] * 5, [107.0] * 5]], dtype=torch.float32)
    mock_pipeline.predict.return_value = [mock_tensor]

    with patch("nexus.ai.chronos_predictor._load_pipeline", return_value=mock_pipeline):
        res = predictor.predict(df)
        assert res is not None
        assert res["signal"] in {"call", "put", "hold"}
        assert "confidence" in res
        assert "forecast_mean" in res


def test_ensemble_with_chronos():
    manager = AIEnsembleManager()
    assert "chronos" in manager.weights
    assert manager.weights["chronos"] > 0.0

    preds = {
        "transformer": {"signal": "call", "confidence": 0.70, "probabilities": {"call": 0.7, "put": 0.3}},
        "lstm": {"signal": "call", "confidence": 0.65, "probabilities": {"call": 0.65, "put": 0.35}},
        "rl_agent": {"action": "call", "confidence": 0.60},
        "chronos": {"signal": "call", "confidence": 0.85, "probabilities": {"call": 0.85, "put": 0.15}},
    }

    result = manager.combine_predictions(
        transformer_pred=preds["transformer"],
        lstm_pred=preds["lstm"],
        rl_pred=preds["rl_agent"],
        chronos_pred=preds["chronos"],
    )

    assert result["signal"] == "call"
    assert result["confidence"] > 0.50
    assert "chronos" in result["weights"]

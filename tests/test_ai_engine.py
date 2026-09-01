"""Unit and integration tests for RealAITradingEngine."""

import numpy as np
import pandas as pd
import pytest

from nexus.ai.engine_ai import RealAITradingEngine
from nexus.core.engine import NexusEngine
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings


@pytest.fixture
def sample_candles_df() -> pd.DataFrame:
    n = 60
    base = 1.0850
    steps = np.random.randn(n) * 0.0005
    closes = base + np.cumsum(steps)
    highs = closes + np.random.rand(n) * 0.0003
    lows = closes - np.random.rand(n) * 0.0003
    opens = np.roll(closes, 1)
    opens[0] = base
    volumes = np.random.randint(100, 1000, size=n)

    return pd.DataFrame(
        {
            "time": range(n),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.mark.asyncio
async def test_real_ai_engine_analysis(sample_candles_df):
    ai = RealAITradingEngine()
    pred = await ai.analyze_market(sample_candles_df, asset="EURUSD", is_otc=False)

    assert "signal" in pred
    assert pred["signal"] in ("call", "put", "hold")
    assert 0.0 <= pred["confidence"] <= 1.0
    assert pred["recommended_expiration"] in (5, 15, 30, 60, 300)
    assert "breakdown" in pred
    assert "reasoning" in pred


def test_real_ai_engine_train_market(sample_candles_df):
    ai = RealAITradingEngine()
    result = ai.train_market("EURUSD", sample_candles_df)

    assert result["symbol"] == "EURUSD"
    assert "generation" in result
    assert "best_indicators" in result
    assert "accuracy" in result
    assert "blueprint" in result


@pytest.mark.asyncio
async def test_nexus_engine_ai_integration():
    settings = NexusSettings(
        quotex=QuotexSettings(email="test@example.com", password="secret_pass"),
        trading=TradingSettings(),
    )
    engine = NexusEngine(settings=settings, demo_mode=True, auto_login=False)

    prediction = await engine.get_ai_prediction("EURUSD", is_otc=False)
    assert isinstance(prediction, dict)
    assert "signal" in prediction
    assert "confidence" in prediction

    train_res = engine.train_market_ai("EURUSD")
    assert isinstance(train_res, dict)
    assert train_res.get("symbol") == "EURUSD"

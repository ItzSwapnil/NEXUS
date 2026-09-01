import pandas as pd
import pytest

from nexus.backtest.backtester import Backtester
from nexus.core.engine import NexusEngine
from nexus.strategies.meta_strategy import MetaStrategy
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings


# --- Stubs reused (lightweight) ---
class StubTransformer:
    async def predict(self, data, asset, timeframe, regime):
        return {
            "signal": "call",
            "confidence": 0.9,
            "reasoning": "stub_transformer",
            "features": {"trend_strength": 0.8, "volatility": 0.2, "momentum": 0.4},
        }


class StubRLAgent:
    async def predict(self, data, asset, timeframe, regime):
        return {
            "signal": "put",
            "confidence": 0.85,
            "reasoning": "stub_rl",
            "features": {"trend_strength": 0.75, "volatility": 0.25, "momentum": 0.45},
        }


class StubRegimeDetector:
    async def detect_regime(self, data):
        return "ranging"


@pytest.mark.asyncio
async def test_backtester_runs_and_collects_trades():
    # Build simple ascending OHLC data (placeholder)
    rows = 120
    df = pd.DataFrame(
        {
            "open": [float(i) for i in range(rows)],
            "close": [float(i) + 0.5 for i in range(rows)],
            "high": [float(i) + 1 for i in range(rows)],
            "low": [float(i) - 1 for i in range(rows)],
        }
    )

    # Settings + engine
    settings = NexusSettings(
        quotex=QuotexSettings(email="stub@example.com", password="pw"),
        trading=TradingSettings(base_trade_amount=5.0),
    )
    engine = NexusEngine(settings=settings, demo_mode=True)

    meta = MetaStrategy(
        transformer=StubTransformer(),
        rl_agent=StubRLAgent(),
        regime_detector=StubRegimeDetector(),
    )

    bt = Backtester(window=20, expiration=60)
    result = await bt.run(meta, engine, df, asset="EURUSD", timeframe=1)

    assert result.total_trades > 0, "Expected at least one trade to be generated"
    assert result.total_profit >= 0.0
    assert result.winning_trades + result.losing_trades == result.total_trades
    # Ensure meta information present
    assert result.meta.get("window") == 20
    assert result.meta.get("asset") == "EURUSD"
    # Trades detail sanity
    for t in result.trades[:5]:  # sample check
        assert t.amount >= 1.0
        assert t.expiration == 60

import pytest

from nexus.analysis.market_analyzer import MarketAnalyzer
from nexus.catalog.ingest import Market


class FakeAI:
    async def analyze_market(self, candles, asset, timeframe, is_otc):
        return {
            "signal": "call" if asset == "EURUSD" else "put",
            "confidence": 0.80,
            "regime": "BULL",
            "recommended_expiration": 60,
        }


@pytest.mark.asyncio
async def test_market_analyzer_uses_live_payout_and_estimates_scenarios():
    markets = [
        Market(
            "EURUSD",
            "Forex",
            82.0,
            payout_per_expiration={"60": 90.0},
            metadata={"payout_source": "live broker catalog"},
        ),
        Market("CLOSED", "Forex", 95.0, active=False),
    ]

    async def candles(_symbol):
        return [{"close": float(value)} for value in (1, 2, 3, 4, 5, 4)]

    result = await MarketAnalyzer(base_stake=1, cycle_seconds=60).analyze(
        markets,
        FakeAI(),
        candles_fetcher=candles,
        min_confidence=0.7,
        autonomy=0.5,
        use_price_gate=True,
    )

    assert len(result.opportunities) == 1
    opportunity = result.opportunities[0]
    assert opportunity.payout_1m == 90.0
    assert opportunity.payout_source == "live broker catalog"
    assert opportunity.recommended_entry_price is not None
    assert result.best_scenario is not None
    assert result.best_scenario.trades_10m >= 0
    assert result.best_scenario.eligible_markets == 1


@pytest.mark.asyncio
async def test_market_analyzer_does_not_claim_price_gate_without_candles():
    market = Market("EURUSD", "Forex", 90.0, payout_per_expiration={"60": 90.0})
    result = await MarketAnalyzer().analyze(
        [market], FakeAI(), candles_fetcher=lambda _symbol: _missing(), use_price_gate=True
    )
    assert result.opportunities[0].recommended_entry_price is None
    assert result.opportunities[0].price_gate == "No live candles"


async def _missing():
    return None

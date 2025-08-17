import pytest
from nexus.payouts.fetch import get_payout_for_market, is_payout_allowed, set_payout_override, is_override_enabled
from nexus.catalog.ingest import get_market_by_symbol
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings
from nexus.core.engine import NexusEngine


def test_payout_lookup_and_override():
    m = get_market_by_symbol("EURUSD")
    assert m is not None
    p = get_payout_for_market(m, "60")
    assert p >= 80.0
    low = get_market_by_symbol("USDJPY")
    assert low is not None
    low_payout = get_payout_for_market(low, "60")
    assert low_payout < 80.0
    assert is_payout_allowed(low_payout, 80.0) is False
    set_payout_override(True, user="test", reason="unit-test")
    assert is_override_enabled() is True
    assert is_payout_allowed(low_payout, 80.0) is True
    set_payout_override(False)

@pytest.mark.asyncio
async def test_engine_blocks_low_payout_without_override():
    settings = NexusSettings(
        quotex=QuotexSettings(email='a@b.com', password='pw'),
        trading=TradingSettings(payout_threshold=80.0)
    )
    engine = NexusEngine(settings=settings, demo_mode=False)
    result = await engine.execute_trade("USDJPY", "call", 10.0, "60")
    assert result.get("success") is False
    assert 'Payout' in result.get('error', '') or 'payout' in result.get('error', '')

@pytest.mark.asyncio
async def test_engine_allows_after_override():
    settings = NexusSettings(
        quotex=QuotexSettings(email='a@b.com', password='pw'),
        trading=TradingSettings(payout_threshold=80.0)
    )
    engine = NexusEngine(settings=settings, demo_mode=False)
    set_payout_override(True, user='test', reason='allow low payout')
    result = await engine.execute_trade("USDJPY", "call", 10.0, "60")
    assert result.get("success") is True
    set_payout_override(False)


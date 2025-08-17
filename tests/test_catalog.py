import pytest
from nexus.catalog.ingest import get_market_catalog, get_market_by_symbol

@pytest.mark.asyncio
async def test_catalog_contains_otc_and_standard_pairs():
    catalog = await get_market_catalog()
    symbols = {m.symbol for m in catalog}
    assert "EURUSD" in symbols
    assert any(m.otc for m in catalog), "Expected at least one OTC instrument"
    eurusd = get_market_by_symbol("EURUSD")
    assert eurusd is not None
    assert eurusd.payout_per_expiration is not None

@pytest.mark.asyncio
async def test_effective_payout_resolution():
    catalog = await get_market_catalog()
    eurusd = next(m for m in catalog if m.symbol == "EURUSD")
    base = eurusd.display_payout_percent
    exp60 = eurusd.effective_payout("60")
    assert exp60 == eurusd.payout_per_expiration["60"]
    assert base >= 80.0


import pytest

from nexus.catalog.ingest import (
    catalog_is_populated,
    fetch_live_catalog,
    get_market_by_symbol,
    get_market_catalog,
)


class LiveBrokerAdapterFixture:
    async def get_assets_with_payouts_async(self):
        return [
            {"symbol": "EURUSD", "type": "Forex", "payout": 88.0, "active": True},
            {"symbol": "EURUSD_otc", "type": "Forex", "payout": 92.0, "otc": True, "active": True},
            {"symbol": "USDJPY", "type": "Forex", "payout": 78.0, "active": True},
        ]


@pytest.mark.asyncio
async def test_catalog_live_extraction_and_resolution():
    adapter = LiveBrokerAdapterFixture()
    catalog = await fetch_live_catalog(adapter)

    assert catalog_is_populated() is True
    assert len(await get_market_catalog()) == len(catalog)
    symbols = {m.symbol for m in catalog}
    assert "EURUSD" in symbols
    assert any(m.otc for m in catalog), "Expected at least one OTC instrument"

    eurusd = get_market_by_symbol("EURUSD")
    assert eurusd is not None
    assert eurusd.payout_per_expiration is not None

    exp60 = eurusd.effective_payout("60")
    assert exp60 == eurusd.payout_per_expiration["60"]
    assert eurusd.display_payout_percent >= 80.0

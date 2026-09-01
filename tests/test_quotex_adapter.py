"""Unit tests for QuotexAdapter facade."""

import pytest

from nexus.adapters.quotex_adapter import QuotexAdapter


def test_quotex_adapter_init():
    adapter = QuotexAdapter(
        email="test@example.com",
        password="secret_pass",
        demo_mode=True,
    )
    assert adapter.email == "test@example.com"
    assert adapter.demo_mode is True
    assert adapter.authenticated is False


@pytest.mark.asyncio
async def test_quotex_adapter_methods():
    adapter = QuotexAdapter(
        email="test@example.com",
        password="secret_pass",
        demo_mode=True,
    )
    adapter.set_session(user_agent="Mozilla/5.0")
    await adapter.set_practice_mode(True)

    assets = await adapter.get_available_assets()
    assert isinstance(assets, list)

    assets_with_payouts = await adapter.get_assets_with_payouts_async()
    assert isinstance(assets_with_payouts, list)

    candles = await adapter.get_candles_async("EURUSD", timeframe_sec=60, limit=10)
    assert candles is None or isinstance(candles, list)

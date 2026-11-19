"""Unit tests for QuotexAdapter facade and session token management."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.adapters.quotex import QuotexAdapter as RealQuotexAdapter
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


@pytest.mark.asyncio
async def test_expired_session_token_fallback(tmp_path):
    """Verify that an expired cached session cleans up and falls back to fresh auth."""
    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps({"ssid": "expired_ssid_123", "cookies": "dummy=1"}), encoding="utf-8"
    )

    adapter = RealQuotexAdapter(
        email="user@test.com",
        password="valid_password",
        session_file=str(session_file),
    )

    mock_client1 = MagicMock()
    mock_client1.connect = AsyncMock(return_value=(False, "Invalid SSID / Session expired"))
    mock_client1.set_session = MagicMock()

    mock_client2 = MagicMock()
    mock_client2.connect = AsyncMock(return_value=(True, "ok"))
    mock_client2.check_connect = AsyncMock(return_value=True)

    with patch("nexus.adapters.quotex.Quotex", side_effect=[mock_client1, mock_client2]):
        success = await adapter.login()

    assert success is True
    assert adapter.authenticated is True
    assert not session_file.exists()


@pytest.mark.asyncio
async def test_expired_session_token_without_credentials(tmp_path):
    """Verify that an expired session without credentials fails gracefully without crashing."""
    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps({"ssid": "expired_ssid_123"}), encoding="utf-8")

    adapter = RealQuotexAdapter(
        email="",
        password="",
        session_file=str(session_file),
    )

    mock_client = MagicMock()
    mock_client.connect = AsyncMock(return_value=(False, "Unauthorized"))
    mock_client.set_session = MagicMock()

    with patch("nexus.adapters.quotex.Quotex", return_value=mock_client):
        success = await adapter.login()

    assert success is False
    assert adapter.authenticated is False
    assert not session_file.exists()


@pytest.mark.asyncio
async def test_ensure_authenticated_detects_disconnect_and_reauthenticates():
    """Verify that _ensure_authenticated detects a dead WebSocket connection and reauthenticates."""
    adapter = RealQuotexAdapter(email="user@test.com", password="pw")
    adapter.authenticated = True
    adapter.client = MagicMock()
    adapter.client.check_connect = AsyncMock(return_value=False)

    with patch.object(adapter, "login", new_callable=AsyncMock) as mock_login:
        mock_login.return_value = True
        result = await adapter._ensure_authenticated()
        assert result is True
        mock_login.assert_called_once()

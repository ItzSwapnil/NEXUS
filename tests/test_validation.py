"""Tests for validation utilities."""

import pytest

from nexus.utils.validation import (
    ValidationError,
    validate_asset_symbol,
    validate_confidence,
    validate_direction,
    validate_expiration,
    validate_payout,
    validate_timeframe,
    validate_trade_amount,
    validate_trade_params,
    validate_weights,
    validate_win_rate,
)


def test_validate_asset_symbol():
    """Test asset symbol validation."""
    assert validate_asset_symbol("eurusd") == "EURUSD"
    assert validate_asset_symbol("  btcusd  ") == "BTCUSD"
    assert validate_asset_symbol("S&P 500") == "S&P 500"

    with pytest.raises(ValidationError):
        validate_asset_symbol("")

    with pytest.raises(ValidationError):
        validate_asset_symbol("A")  # Too short

    with pytest.raises(ValidationError):
        validate_asset_symbol("INVALID@SYMBOL")  # Invalid chars


def test_validate_trade_amount():
    """Test trade amount validation."""
    assert validate_trade_amount(10.5) == 10.5
    assert validate_trade_amount(1.0) == 1.0

    with pytest.raises(ValidationError):
        validate_trade_amount(0.0)  # Zero

    with pytest.raises(ValidationError):
        validate_trade_amount(-5.0)  # Negative

    with pytest.raises(ValidationError):
        validate_trade_amount(0.5, min_amount=1.0)  # Below minimum

    with pytest.raises(ValidationError):
        validate_trade_amount(15000, max_amount=10000)  # Above maximum


def test_validate_direction():
    """Test direction validation."""
    assert validate_direction("call") == "call"
    assert validate_direction("CALL") == "call"
    assert validate_direction("buy") == "call"
    assert validate_direction("up") == "call"

    assert validate_direction("put") == "put"
    assert validate_direction("PUT") == "put"
    assert validate_direction("sell") == "put"
    assert validate_direction("down") == "put"

    with pytest.raises(ValidationError):
        validate_direction("")

    with pytest.raises(ValidationError):
        validate_direction("invalid")


def test_validate_expiration():
    """Test expiration validation."""
    assert validate_expiration(60) == 60
    assert validate_expiration("120") == 120

    with pytest.raises(ValidationError):
        validate_expiration(30, min_seconds=60)  # Below minimum

    with pytest.raises(ValidationError):
        validate_expiration(5000, max_seconds=3600)  # Above maximum

    with pytest.raises(ValidationError):
        validate_expiration("invalid")  # Invalid type


def test_validate_timeframe():
    """Test timeframe validation."""
    assert validate_timeframe(5) == 5
    assert validate_timeframe(60) == 60

    with pytest.raises(ValidationError):
        validate_timeframe(0)  # Zero

    with pytest.raises(ValidationError):
        validate_timeframe(-5)  # Negative


def test_validate_payout():
    """Test payout validation."""
    assert validate_payout(80.0) == 80.0
    assert validate_payout(95.5) == 95.5

    with pytest.raises(ValidationError):
        validate_payout(-10.0)  # Negative

    with pytest.raises(ValidationError):
        validate_payout(150.0, max_payout=100.0)  # Above maximum


def test_validate_win_rate():
    """Test win rate validation."""
    assert validate_win_rate(0.5) == 0.5
    assert validate_win_rate(0.0) == 0.0
    assert validate_win_rate(1.0) == 1.0

    with pytest.raises(ValidationError):
        validate_win_rate(-0.1)  # Below 0

    with pytest.raises(ValidationError):
        validate_win_rate(1.5)  # Above 1


def test_validate_confidence():
    """Test confidence validation."""
    assert validate_confidence(0.7) == 0.7
    assert validate_confidence(0.0) == 0.0
    assert validate_confidence(1.0) == 1.0

    with pytest.raises(ValidationError):
        validate_confidence(-0.1)

    with pytest.raises(ValidationError):
        validate_confidence(2.0)


def test_validate_weights():
    """Test weights validation and normalization."""
    weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}
    result = validate_weights(weights)
    assert abs(sum(result.values()) - 1.0) < 1e-9

    # Test normalization
    weights = {"model_a": 2.0, "model_b": 2.0}
    result = validate_weights(weights)
    assert result["model_a"] == 0.5
    assert result["model_b"] == 0.5

    with pytest.raises(ValidationError):
        validate_weights({})  # Empty

    with pytest.raises(ValidationError):
        validate_weights({"model_a": -1.0})  # Negative weight


def test_validate_trade_params():
    """Test complete trade parameter validation."""
    asset, direction, amount, expiration = validate_trade_params("eurusd", "call", 10.0, 60)

    assert asset == "EURUSD"
    assert direction == "call"
    assert amount == 10.0
    assert expiration == 60

    with pytest.raises(ValidationError):
        validate_trade_params("", "call", 10.0, 60)  # Invalid asset

    with pytest.raises(ValidationError):
        validate_trade_params("EURUSD", "invalid", 10.0, 60)  # Invalid direction

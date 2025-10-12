"""Tests for risk management utilities."""

import pytest
from nexus.utils.risk_manager import RiskManager, RiskLimits, PortfolioState


def test_risk_manager_initialization():
    """Test risk manager initialization."""
    rm = RiskManager()
    assert rm.state.current_equity == 10000.0
    assert rm.state.trades_today == 0
    assert rm.state.consecutive_losses == 0


def test_can_trade_daily_limit():
    """Test daily trade limit enforcement."""
    limits = RiskLimits(max_trades_per_day=5)
    rm = RiskManager(limits)

    # Should allow trading initially
    can_trade, reason = rm.can_trade()
    assert can_trade is True
    assert reason is None

    # Simulate reaching daily limit
    rm.state.trades_today = 5
    can_trade, reason = rm.can_trade()
    assert can_trade is False
    assert "Daily trade limit" in reason


def test_can_trade_daily_loss_limit():
    """Test daily loss limit enforcement."""
    limits = RiskLimits(max_daily_loss=100.0)
    rm = RiskManager(limits)

    # Simulate daily loss
    rm.state.daily_pnl = -100.0
    can_trade, reason = rm.can_trade()
    assert can_trade is False
    assert "Daily loss limit" in reason


def test_can_trade_drawdown_limit():
    """Test drawdown limit enforcement."""
    limits = RiskLimits(max_drawdown_percent=20.0)
    rm = RiskManager(limits)

    # Simulate drawdown
    rm.state.peak_equity = 10000.0
    rm.state.current_equity = 7900.0  # 21% drawdown

    can_trade, reason = rm.can_trade()
    assert can_trade is False
    assert "Drawdown limit" in reason


def test_can_trade_consecutive_losses():
    """Test consecutive loss limit enforcement."""
    limits = RiskLimits(max_consecutive_losses=3)
    rm = RiskManager(limits)

    # Simulate consecutive losses
    rm.state.consecutive_losses = 3
    can_trade, reason = rm.can_trade()
    assert can_trade is False
    assert "Consecutive loss limit" in reason


def test_calculate_position_size_basic():
    """Test basic position size calculation."""
    rm = RiskManager()

    # Base case with 0.5 confidence
    size = rm.calculate_position_size(10.0, confidence=0.5)
    assert 5.0 <= size <= 15.0  # Should scale with confidence

    # High confidence
    size_high = rm.calculate_position_size(10.0, confidence=0.9)
    size_low = rm.calculate_position_size(10.0, confidence=0.3)
    assert size_high > size_low  # Higher confidence = larger size


def test_calculate_position_size_kelly():
    """Test Kelly Criterion position sizing."""
    limits = RiskLimits(use_kelly_criterion=True, kelly_fraction=0.25)
    rm = RiskManager(limits)

    # Favorable Kelly scenario (60% win rate, 2:1 reward/risk)
    size = rm.calculate_position_size(
        base_amount=10.0,
        confidence=0.7,
        win_rate=0.6,
        avg_win=20.0,
        avg_loss=10.0
    )

    assert size > 0  # Should produce positive size
    assert size <= limits.max_position_size


def test_record_trade_updates_state():
    """Test that recording trades updates state correctly."""
    rm = RiskManager()
    initial_equity = rm.state.current_equity

    # Record winning trade
    rm.record_trade(profit=10.0, success=True)

    assert rm.state.current_equity == initial_equity + 10.0
    assert rm.state.daily_pnl == 10.0
    assert rm.state.trades_today == 1
    assert rm.state.consecutive_losses == 0

    # Record losing trade
    rm.record_trade(profit=-5.0, success=False)

    assert rm.state.current_equity == initial_equity + 5.0
    assert rm.state.daily_pnl == 5.0
    assert rm.state.trades_today == 2
    assert rm.state.consecutive_losses == 1


def test_record_trade_updates_peak_equity():
    """Test peak equity tracking."""
    rm = RiskManager()

    # Win increases peak
    rm.record_trade(profit=100.0, success=True)
    assert rm.state.peak_equity == 10100.0

    # Loss doesn't decrease peak
    rm.record_trade(profit=-50.0, success=False)
    assert rm.state.peak_equity == 10100.0


def test_get_risk_metrics():
    """Test risk metrics retrieval."""
    rm = RiskManager()
    rm.record_trade(profit=50.0, success=True)

    metrics = rm.get_risk_metrics()

    assert 'current_equity' in metrics
    assert 'peak_equity' in metrics
    assert 'daily_pnl' in metrics
    assert 'drawdown_percent' in metrics
    assert 'trades_today' in metrics
    assert 'can_trade' in metrics

    assert metrics['current_equity'] == 10050.0
    assert metrics['daily_pnl'] == 50.0
    assert metrics['trades_today'] == 1


def test_reset_clears_state():
    """Test that reset clears all state."""
    rm = RiskManager()

    # Create some state
    rm.record_trade(profit=50.0, success=True)
    rm.record_trade(profit=-30.0, success=False)

    # Reset
    rm.reset()

    assert rm.state.current_equity == 10000.0
    assert rm.state.daily_pnl == 0.0
    assert rm.state.trades_today == 0
    assert rm.state.consecutive_losses == 0
    assert len(rm.trade_history) == 0


def test_volatility_adjustment():
    """Test volatility-based position sizing adjustment."""
    limits = RiskLimits(volatility_adjustment=True)
    rm = RiskManager(limits)

    # Low volatility should allow larger positions
    size_low_vol = rm.calculate_position_size(10.0, volatility=0.005)

    # High volatility should reduce positions
    size_high_vol = rm.calculate_position_size(10.0, volatility=0.05)

    assert size_low_vol > size_high_vol
"""Tests for validation utilities."""

import pytest
from nexus.utils.validation import (
    ValidationError,
    validate_asset_symbol,
    validate_trade_amount,
    validate_direction,
    validate_expiration,
    validate_timeframe,
    validate_payout,
    validate_win_rate,
    validate_confidence,
    validate_weights,
    validate_trade_params,
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
    asset, direction, amount, expiration = validate_trade_params(
        "eurusd", "call", 10.0, 60
    )

    assert asset == "EURUSD"
    assert direction == "call"
    assert amount == 10.0
    assert expiration == 60

    with pytest.raises(ValidationError):
        validate_trade_params("", "call", 10.0, 60)  # Invalid asset

    with pytest.raises(ValidationError):
        validate_trade_params("EURUSD", "invalid", 10.0, 60)  # Invalid direction


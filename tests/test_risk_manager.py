"""Unit tests for Risk Management and Position Sizer."""

from nexus.risk.position_sizer import (
    DrawdownConfig,
    DrawdownProtection,
    KellyPositionSizer,
    PositionSizerConfig,
)


def test_kelly_position_sizer_calculates_valid_amount():
    config = PositionSizerConfig(
        fractional_kelly=0.25, min_trade_amount=1.0, max_account_percent=0.05
    )
    sizer = KellyPositionSizer(config)

    # 60% confidence, 85% payout, $10,000 balance
    amount = sizer.calculate_trade_amount(
        account_balance=10000.0, payout_rate=0.85, confidence=0.60
    )
    assert amount >= 1.0
    assert amount <= 500.0  # Max 5% of $10,000


def test_kelly_position_sizer_respects_minimum_amount():
    sizer = KellyPositionSizer()
    amount = sizer.calculate_trade_amount(account_balance=100.0, payout_rate=0.80, confidence=0.51)
    assert amount == 1.0


def test_drawdown_protection_allows_normal_trading():
    protection = DrawdownProtection()
    allowed, reason = protection.check_trade_allowed(current_balance=10000.0)
    assert allowed is True
    assert "allowed" in reason.lower()


def test_drawdown_protection_blocks_after_max_drawdown():
    config = DrawdownConfig(max_daily_drawdown_percent=0.05)
    protection = DrawdownProtection(config)
    protection.initialize_daily_balance(10000.0)

    # Account drops to $9,400 (6% drawdown > 5% max)
    allowed, reason = protection.check_trade_allowed(current_balance=9400.0)
    assert allowed is False
    assert "drawdown" in reason.lower()


def test_drawdown_protection_triggers_cooldown_on_loss_streak():
    config = DrawdownConfig(max_consecutive_losses=3, cooldown_minutes=15)
    protection = DrawdownProtection(config)
    protection.initialize_daily_balance(10000.0)

    protection.record_trade_outcome(is_win=False)
    protection.record_trade_outcome(is_win=False)
    protection.record_trade_outcome(is_win=False)

    allowed, reason = protection.check_trade_allowed(current_balance=9970.0)
    assert allowed is False
    assert "cooldown" in reason.lower()

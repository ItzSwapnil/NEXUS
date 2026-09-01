"""
Kelly Criterion Position Sizer and Drawdown Protection for NEXUS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.risk.position_sizer")


@dataclass
class PositionSizerConfig:
    """Configuration for Kelly Position Sizer."""

    fractional_kelly: float = 0.25  # Quarter-Kelly for conservative capital growth
    min_trade_amount: float = 1.0  # Absolute minimum trade amount
    max_account_percent: float = 0.05  # Max 5% of account per trade
    atr_volatility_scaling: bool = True  # Enable ATR volatility adjustment
    base_atr: float = 0.0010  # Reference baseline ATR


class KellyPositionSizer:
    """
    Dynamic position sizer using Fractional Kelly Criterion & Volatility Scaling.
    """

    def __init__(self, config: Optional[PositionSizerConfig] = None):
        self.config = config or PositionSizerConfig()

    def calculate_trade_amount(
        self,
        account_balance: float,
        payout_rate: float,
        confidence: float,
        current_atr: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal trade size.

        Args:
            account_balance: Current account balance in USD.
            payout_rate: Market payout rate (e.g., 0.85 for 85%).
            confidence: Model confidence score (e.g., 0.60 to 0.95).
            current_atr: Current ATR volatility indicator value.

        Returns:
            Optimal trade amount in USD.
        """
        if account_balance <= 0 or payout_rate <= 0:
            return self.config.min_trade_amount

        # Win probability (p) and loss probability (q)
        p = max(0.51, min(0.99, confidence))
        q = 1.0 - p
        b = max(0.1, payout_rate)

        # Full Kelly fraction: (p*b - q) / b
        kelly_fraction = (p * b - q) / b

        if kelly_fraction <= 0:
            return self.config.min_trade_amount

        # Apply fractional Kelly
        scaled_fraction = kelly_fraction * self.config.fractional_kelly

        # Calculate raw trade size
        raw_trade_amount = account_balance * scaled_fraction

        # Apply ATR volatility scaling if provided
        if self.config.atr_volatility_scaling and current_atr and current_atr > 0:
            volatility_factor = self.config.base_atr / current_atr
            # Clamp factor between 0.5x and 1.5x
            volatility_factor = max(0.5, min(1.5, volatility_factor))
            raw_trade_amount *= volatility_factor

        # Enforce min trade amount and max account percentage cap
        max_trade_amount = account_balance * self.config.max_account_percent
        final_amount = max(self.config.min_trade_amount, min(max_trade_amount, raw_trade_amount))

        logger.debug(
            f"Calculated trade amount: ${final_amount:.2f} (Kelly: {kelly_fraction:.4f}, Balance: ${account_balance:.2f})"
        )
        return round(final_amount, 2)


@dataclass
class DrawdownConfig:
    """Configuration for Drawdown Protection."""

    max_daily_drawdown_percent: float = 0.05  # 5% max daily drawdown limit
    max_consecutive_losses: int = 4  # Max losses before enforced pause
    cooldown_minutes: int = 30  # Cooldown pause duration in minutes


class DrawdownProtection:
    """
    Enforces real-time drawdown caps and loss streak cooldowns.
    """

    def __init__(self, config: Optional[DrawdownConfig] = None):
        self.config = config or DrawdownConfig()
        self.initial_daily_balance: Optional[float] = None
        self.current_consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

    def initialize_daily_balance(self, balance: float) -> None:
        """Set initial daily balance reference."""
        if self.initial_daily_balance is None or self._is_new_day():
            self.initial_daily_balance = balance

    def check_trade_allowed(self, current_balance: float) -> tuple[bool, str]:
        """
        Check if trading is permitted under risk controls.

        Args:
            current_balance: Current account balance.

        Returns:
            Tuple of (is_allowed: bool, reason: str).
        """
        # Check active cooldown timer
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            remaining = int((self.cooldown_until - datetime.now()).total_seconds() / 60)
            return False, f"Cooldown active due to loss streak ({remaining} mins remaining)"

        # Initialize daily reference balance if missing
        if self.initial_daily_balance is None:
            self.initial_daily_balance = current_balance

        # Calculate daily drawdown
        if self.initial_daily_balance > 0:
            drawdown = (self.initial_daily_balance - current_balance) / self.initial_daily_balance
            if drawdown >= self.config.max_daily_drawdown_percent:
                return (
                    False,
                    f"Max daily drawdown limit reached ({drawdown * 100:.1f}% >= {self.config.max_daily_drawdown_percent * 100:.1f}%)",
                )

        return True, "Trading allowed under risk limits"

    def record_trade_outcome(self, is_win: bool) -> None:
        """Update loss streak counter post-trade."""
        if is_win:
            self.current_consecutive_losses = 0
        else:
            self.current_consecutive_losses += 1
            if self.current_consecutive_losses >= self.config.max_consecutive_losses:
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.config.cooldown_minutes
                )
                logger.warning(
                    f"Loss streak limit reached ({self.current_consecutive_losses}). Cooldown active for {self.config.cooldown_minutes} mins."
                )

    def _is_new_day(self) -> bool:
        """Check if reset is needed for a new trading day."""
        return False

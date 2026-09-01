"""
Advanced Risk Management Module for NEXUS Trading System.

Provides sophisticated risk controls including:
- Position sizing algorithms
- Drawdown protection
- Daily loss limits
- Volatility-adjusted sizing
- Kelly Criterion implementation
- Multi-asset portfolio risk
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.utils.risk_manager")


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    max_position_size: float = 100.0
    max_daily_loss: float = 500.0
    max_drawdown_percent: float = 20.0
    max_trades_per_day: int = 50
    max_trades_per_hour: int = 10
    max_consecutive_losses: int = 5
    volatility_adjustment: bool = True
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25  # Fractional Kelly


@dataclass
class PortfolioState:
    """Current portfolio state for risk calculations."""

    current_equity: float = 10000.0
    daily_pnl: float = 0.0
    peak_equity: float = 10000.0
    trades_today: int = 0
    trades_this_hour: int = 0
    consecutive_losses: int = 0
    last_trade_time: Optional[datetime] = None
    open_positions: int = 0


class RiskManager:
    """
    Advanced risk management system for NEXUS.

    Implements multiple risk control mechanisms to protect capital
    and ensure sustainable trading practices.
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager with limits.

        Args:
            limits: Risk limit configuration
        """
        self.limits = limits or RiskLimits()
        self.state = PortfolioState()
        self.trade_history: List[Dict[str, Any]] = []
        self._day_start = datetime.now().date()
        self._hour_start = datetime.now().hour

        logger.info("RiskManager initialized with limits: %s", self.limits)

    def reset_daily_counters(self) -> None:
        """Reset daily counters at start of new trading day."""
        current_date = datetime.now().date()
        if current_date != self._day_start:
            self.state.trades_today = 0
            self.state.daily_pnl = 0.0
            self._day_start = current_date
            logger.info("Daily risk counters reset")

    def reset_hourly_counters(self) -> None:
        """Reset hourly counters at start of new hour."""
        current_hour = datetime.now().hour
        if current_hour != self._hour_start:
            self.state.trades_this_hour = 0
            self._hour_start = current_hour

    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on current risk limits.

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        self.reset_daily_counters()
        self.reset_hourly_counters()

        # Check daily trade limit
        if self.state.trades_today >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.limits.max_trades_per_day})"

        # Check hourly trade limit
        if self.state.trades_this_hour >= self.limits.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({self.limits.max_trades_per_hour})"

        # Check daily loss limit
        if self.state.daily_pnl <= -self.limits.max_daily_loss:
            return False, f"Daily loss limit reached (${abs(self.state.daily_pnl):.2f})"

        # Check drawdown limit
        drawdown_pct = self._calculate_drawdown_percent()
        if drawdown_pct >= self.limits.max_drawdown_percent:
            return False, f"Drawdown limit reached ({drawdown_pct:.1f}%)"

        # Check consecutive losses
        if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
            return False, f"Consecutive loss limit reached ({self.state.consecutive_losses})"

        return True, None

    def calculate_position_size(
        self,
        base_amount: float,
        confidence: float = 0.5,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal position size using various methods.

        Args:
            base_amount: Base trade amount
            confidence: Model confidence (0-1)
            win_rate: Historical win rate (optional, for Kelly)
            avg_win: Average win amount (optional, for Kelly)
            avg_loss: Average loss amount (optional, for Kelly)
            volatility: Market volatility (optional, for volatility adjustment)

        Returns:
            float: Recommended position size
        """
        position_size = base_amount

        # Apply confidence scaling
        confidence = max(0.0, min(1.0, confidence))
        position_size *= 0.5 + confidence * 0.5  # Scale from 50% to 100%

        # Apply Kelly Criterion if enabled and we have required data
        if self.limits.use_kelly_criterion and win_rate and avg_win and avg_loss:
            kelly_size = self._kelly_criterion(win_rate, avg_win, avg_loss)
            position_size = min(position_size, kelly_size)

        # Apply volatility adjustment if enabled
        if self.limits.volatility_adjustment and volatility:
            vol_adjustment = self._volatility_adjustment(volatility)
            position_size *= vol_adjustment

        # Apply equity-based scaling
        equity_fraction = self.state.current_equity / 10000.0  # Assume 10k starting equity
        position_size *= math.sqrt(max(0.1, equity_fraction))  # Scale with sqrt of equity

        # Ensure within limits
        position_size = max(1.0, min(position_size, self.limits.max_position_size))

        return round(position_size, 2)

    def _kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly Criterion optimal bet size.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average profit on winning trades
            avg_loss: Average loss on losing trades

        Returns:
            float: Kelly-optimal position size
        """
        if avg_loss <= 0:
            return self.limits.max_position_size

        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / abs(avg_loss)
        q = 1.0 - win_rate
        kelly_fraction = (win_rate * b - q) / b

        # Apply fractional Kelly for safety
        kelly_fraction *= self.limits.kelly_fraction

        # Convert to position size (fraction of equity)
        kelly_size = self.state.current_equity * max(0.0, kelly_fraction)

        return kelly_size

    def _volatility_adjustment(self, volatility: float) -> float:
        """
        Calculate position size adjustment based on volatility.

        Args:
            volatility: Current market volatility measure

        Returns:
            float: Adjustment multiplier (0.5 to 1.5)
        """
        # Higher volatility = smaller position
        # Assume normal volatility around 0.01 (1%)
        normal_vol = 0.01
        vol_ratio = volatility / normal_vol if normal_vol > 0 else 1.0

        # Inverse relationship: high vol = low multiplier
        adjustment = 1.0 / (1.0 + vol_ratio - 1.0)

        # Clamp between 0.5 and 1.5
        return max(0.5, min(1.5, adjustment))

    def _calculate_drawdown_percent(self) -> float:
        """Calculate current drawdown percentage."""
        if self.state.peak_equity <= 0:
            return 0.0

        drawdown = self.state.peak_equity - self.state.current_equity
        drawdown_pct = (drawdown / self.state.peak_equity) * 100.0

        return max(0.0, drawdown_pct)

    def record_trade(self, profit: float, success: bool) -> None:
        """
        Record a trade and update risk state.

        Args:
            profit: Trade profit/loss
            success: Whether trade was successful
        """
        # Update equity
        self.state.current_equity += profit
        self.state.daily_pnl += profit

        # Update peak equity
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity

        # Update counters
        self.state.trades_today += 1
        self.state.trades_this_hour += 1
        self.state.last_trade_time = datetime.now()

        # Update consecutive losses
        if success:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1

        # Store trade history
        self.trade_history.append(
            {
                "timestamp": datetime.now(),
                "profit": profit,
                "success": success,
                "equity": self.state.current_equity,
                "daily_pnl": self.state.daily_pnl,
            }
        )

        # Log risk metrics
        drawdown = self._calculate_drawdown_percent()
        logger.debug(
            f"Trade recorded: profit={profit:.2f}, equity={self.state.current_equity:.2f}, "
            f"drawdown={drawdown:.1f}%, consecutive_losses={self.state.consecutive_losses}"
        )

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics for monitoring.

        Returns:
            Dictionary of risk metrics
        """
        return {
            "current_equity": self.state.current_equity,
            "peak_equity": self.state.peak_equity,
            "daily_pnl": self.state.daily_pnl,
            "drawdown_percent": self._calculate_drawdown_percent(),
            "trades_today": self.state.trades_today,
            "trades_this_hour": self.state.trades_this_hour,
            "consecutive_losses": self.state.consecutive_losses,
            "can_trade": self.can_trade()[0],
        }

    def reset(self) -> None:
        """Reset risk manager to initial state."""
        self.state = PortfolioState()
        self.trade_history.clear()
        self._day_start = datetime.now().date()
        self._hour_start = datetime.now().hour
        logger.info("RiskManager reset to initial state")


__all__ = ["RiskManager", "RiskLimits", "PortfolioState"]

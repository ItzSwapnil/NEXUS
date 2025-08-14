"""Core trading engine implementation (lightweight stub for tests).

Provides registry management for strategies/models/risk modules, basic emotional
state updates, and simple risk management sizing. Designed to satisfy current
unit tests without external service dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nexus.utils.config import NexusSettings


@dataclass
class EngineState:
    initialized: bool = False
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0


class NexusEngine:
    """Lightweight engine facade.

    Args:
        settings: NexusSettings instance
        demo_mode: Whether to run in demo mode
        auto_login: Placeholder flag retained for compatibility
    """

    def __init__(self, settings: NexusSettings, demo_mode: bool = True, auto_login: bool = False) -> None:
        self.settings = settings
        self.demo_mode = demo_mode
        self.auto_login = auto_login

        # Registries
        self.strategy_registry: Dict[str, Any] = {}
        self.model_registry: Dict[str, Any] = {}
        self.risk_registry: Dict[str, Any] = {}

        # Optional higher-level strategy orchestrator
        self.meta_strategy: Optional[Any] = None

        # Emotional state (values kept within [0,1])
        self.emotion_state: Dict[str, float] = {
            "greed": 0.5,
            "fear": 0.5,
            "confidence": 0.5,
        }

        self._state = EngineState()

    # ---------------------------- Initialization ---------------------------- #
    async def initialize_components(self) -> None:
        """Async initialization hook (placeholder)."""
        self._state.initialized = True

    # -------------------------- Registry Management ------------------------ #
    def register_strategy(self, name: str, strategy: Any) -> None:
        self.strategy_registry[name] = strategy

    def unregister_strategy(self, name: str) -> None:
        self.strategy_registry.pop(name, None)

    # -------------------------- Emotional State Logic ---------------------- #
    def update_emotional_state(self, trade_result: Dict[str, Any]) -> None:
        """Update basic emotional state from a trade result.

        Expected trade_result keys:
            success: bool indicating winning trade
            profit: optional numeric profit (positive/negative)
        """
        success = bool(trade_result.get("success"))
        profit = float(trade_result.get("profit", 0.0))

        # Greed rises on wins, falls on losses
        delta_greed = 0.05 if success else -0.05
        # Fear rises on losses, decays on wins
        delta_fear = 0.05 if not success else -0.05

        # Profit amplifies adjustments slightly
        if profit != 0:
            scale = max(min(abs(profit) / 100.0, 0.1), 0.01)
            if profit > 0:
                delta_greed += scale
                delta_fear -= scale / 2
            else:
                delta_fear += scale
                delta_greed -= scale / 2

        self.emotion_state["greed"] = self._clamp(self.emotion_state["greed"] + delta_greed)
        self.emotion_state["fear"] = self._clamp(self.emotion_state["fear"] + delta_fear)
        # Confidence inversely tied to fear for this simple model
        self.emotion_state["confidence"] = self._clamp(1.0 - self.emotion_state["fear"] * 0.7)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return hi if value > hi else lo if value < lo else value

    # -------------------------- Risk Management ---------------------------- #
    def advanced_risk_management(self, context: Dict[str, Any], base_amount: float) -> float:
        """Compute a position size based on emotional state and simple caps.

        Reduces size if fear elevated; modestly increases if confidence high.
        Ensures minimum size of 1.0.
        """
        fear = self.emotion_state.get("fear", 0.5)
        confidence = self.emotion_state.get("confidence", 0.5)
        # Base modifier: confidence boosts up to +20%, fear reduces up to -40%
        size = base_amount * (1 + 0.2 * (confidence - 0.5) - 0.4 * (fear - 0.5))
        # Enforce min & not exceed a simple risk cap derived from settings
        max_risk_pct = getattr(self.settings.trading, "max_risk_per_trade_percent", 2.0) / 100.0
        equity_reference = 10000  # Placeholder equity baseline
        max_allowed = equity_reference * max_risk_pct
        if size > max_allowed:
            size = max_allowed
        if size < 1.0:
            size = 1.0
        return round(size, 2)

    # -------------------------- Performance Stats -------------------------- #
    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "total_trades": self._state.total_trades,
            "winning_trades": self._state.winning_trades,
            "losing_trades": self._state.losing_trades,
            "total_profit": round(self._state.total_profit, 2),
        }

    # -------------------------- Trade Logging (Optional) ------------------- #
    def record_trade(self, success: bool, profit: float) -> None:
        self._state.total_trades += 1
        if success:
            self._state.winning_trades += 1
        else:
            self._state.losing_trades += 1
        self._state.total_profit += profit
        self.update_emotional_state({"success": success, "profit": profit})


__all__ = ["NexusEngine"]


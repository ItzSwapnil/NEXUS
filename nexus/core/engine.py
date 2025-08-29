"""Lightweight NexusEngine used by tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import asyncio

from nexus.utils.config import NexusSettings
from nexus.utils.logger import get_nexus_logger
from nexus.payouts.fetch import get_payout_for_market, is_payout_allowed, is_override_enabled
from nexus.intelligence.exploration import ExplorationController
try:
    from nexus.adapters.quotex_adapter import QuotexAdapter as _QuotexAdapter  # type: ignore
except Exception:  # pragma: no cover
    _QuotexAdapter = None  # type: ignore

logger = get_nexus_logger("nexus.core.engine")


@dataclass
class EngineState:
    initialized: bool = False
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0


class NexusEngine:
    """Main engine facade for trading and tests."""

    def __init__(self, settings: NexusSettings, demo_mode: bool = True, auto_login: bool = False) -> None:
        self.settings = settings
        self.demo_mode = demo_mode
        self.auto_login = auto_login
        self.strategy_registry: Dict[str, Any] = {}
        self.model_registry: Dict[str, Any] = {}
        self.risk_registry: Dict[str, Any] = {}
        self._adapter_lock = asyncio.Lock()
        self.meta_strategy: Optional[Any] = None
        self.emotion_state: Dict[str, float] = {
            "greed": 0.5,
            "fear": 0.5,
            "confidence": 0.5,
        }
        self._state = EngineState()
        self.exploration_controller = ExplorationController(settings)
        self._quotex = None  # type: ignore

    async def initialize_components(self) -> None:
        """Initialize async components."""
        self._state.initialized = True
        if self.auto_login and not self.demo_mode:
            try:
                await self.login_broker()
            except Exception as e:  # pragma: no cover
                logger.warning(f"Auto-login failed: {e}")

    def register_strategy(self, name: str, strategy: Any) -> None:
        self.strategy_registry[name] = strategy

    def unregister_strategy(self, name: str) -> None:
        self.strategy_registry.pop(name, None)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    async def login_broker(self) -> bool:
        """Login to Quotex when not in demo mode."""
        if self.demo_mode:
            return True
        if _QuotexAdapter is None:
            raise RuntimeError("Quotex adapter unavailable. Ensure pyquotex is installed.")
        if self._quotex is None:
            qcfg = self.settings.quotex
            self._quotex = _QuotexAdapter(
                email=qcfg.email,
                password=qcfg.password,
                lang=qcfg.lang,
                demo_mode=bool(qcfg.demo_mode),
                use_real=not self.demo_mode,
            )
        try:
            await self._quotex.connect()  # type: ignore[attr-defined]
            return True
        except Exception as e:
            logger.error(f"Quotex login failed: {e}")
            raise

    def update_emotional_state(self, trade_result: Dict[str, Any]) -> None:
        """Adjust greed/fear/confidence based on trade result."""
        success = bool(trade_result.get("success"))
        profit = float(trade_result.get("profit", 0.0) or 0.0)
        delta_greed = 0.05 if success else -0.05
        delta_fear = -0.05 if success else 0.05
        if profit != 0:
            scale = min(abs(profit) / 100.0, 0.1)
            delta_conf = scale if profit > 0 else -scale
        else:
            delta_conf = 0.02 if success else -0.02
        self.emotion_state["greed"] = self._clamp(self.emotion_state["greed"] + delta_greed)
        self.emotion_state["fear"] = self._clamp(self.emotion_state["fear"] + delta_fear)
        self.emotion_state["confidence"] = self._clamp(self.emotion_state["confidence"] + delta_conf)

    def advanced_risk_management(self, context: Dict[str, Any], base_amount: float) -> float:
        """Return a size >= 1.0 using emotion-based scaling with equity cap."""
        greed = self.emotion_state.get("greed", 0.5)
        fear = self.emotion_state.get("fear", 0.5)
        confidence = self.emotion_state.get("confidence", 0.5)
        modifier = (greed - 0.5) * 0.4 - (fear - 0.5) * 0.4 + (confidence - 0.5) * 0.2
        size = base_amount * (1.0 + modifier)
        if size < 1.0:
            size = 1.0
        equity_reference = float(context.get("equity", base_amount * 100.0))
        max_risk_pct = float(self.settings.trading.max_risk_per_trade_percent)
        max_allowed = equity_reference * (max_risk_pct / 100.0)
        if size > max_allowed:
            size = max_allowed
        return round(size, 2)

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "total_trades": self._state.total_trades,
            "winning_trades": self._state.winning_trades,
            "losing_trades": self._state.losing_trades,
            "total_profit": round(self._state.total_profit, 2),
        }

    def record_trade(self, success: bool, profit: float) -> None:
        self._state.total_trades += 1
        if success:
            self._state.winning_trades += 1
        else:
            self._state.losing_trades += 1
        self._state.total_profit += profit
        self.update_emotional_state({"success": success, "profit": profit})

    async def execute_trade(self, asset: str, signal_type: str, amount: float, expiration: str) -> Dict[str, Any]:
        """Execute a trade (simulated in demo; guarded/real otherwise)."""
        direction = signal_type.lower()
        if direction not in {"call", "put"}:
            direction = "call"
        exp_key = str(expiration)
        try:
            duration = int(exp_key)
        except Exception:
            duration = int(getattr(self.settings.trading, "default_expiration", 60))

        if not self.demo_mode:
            payout = get_payout_for_market(asset, exp_key)
            threshold = float(self.settings.trading.payout_threshold)
            if not is_payout_allowed(payout, threshold):
                logger.warning(f"Blocked real trade for {asset} due to low payout ({payout} < {threshold})")
                return {"success": False, "error": "Payout below threshold", "asset": asset, "direction": direction}
            if is_override_enabled():
                profit = amount * 0.1
                self.record_trade(True, profit)
                return {
                    "success": True,
                    "profit": profit,
                    "asset": asset,
                    "direction": direction,
                    "expiration": exp_key,
                    "real_executed": False,
                    "note": "simulated due to payout override",
                }
            async with self._adapter_lock:
                try:
                    await self.login_broker()
                except Exception as e:
                    return {"success": False, "error": f"Login failed: {e}", "asset": asset, "direction": direction}
                try:
                    resp = await self._quotex.buy_simple(asset, float(amount), direction, int(duration))  # type: ignore[attr-defined]
                    placed = bool(resp) or resp is not None
                    result: Dict[str, Any] = {
                        "success": placed,
                        "asset": asset,
                        "direction": direction,
                        "expiration": exp_key,
                        "real_executed": True,
                    }
                    if resp is not None:
                        result["broker_response"] = resp
                    if placed:
                        self.record_trade(True, 0.0)
                    return result
                except Exception as e:
                    logger.error(f"Real trade placement failed: {e}")
                    return {"success": False, "error": f"Placement failed: {e}", "asset": asset, "direction": direction}

        profit = amount * 0.1
        self.record_trade(True, profit)
        return {
            "success": True,
            "profit": profit,
            "asset": asset,
            "direction": direction,
            "expiration": exp_key,
            "real_executed": False,
        }

    async def get_account_balance(self) -> float:
        """Return account balance for current mode."""
        if self.demo_mode:
            demo_balance = getattr(self.settings.trading, "demo_balance", 10000.0)
            return float(demo_balance)
        if _QuotexAdapter is None:
            raise RuntimeError("Quotex adapter unavailable. Ensure pyquotex is installed.")
        async with self._adapter_lock:
            try:
                await self.login_broker()
                balance = await self._quotex.get_balance()  # type: ignore[attr-defined]
                return float(balance)
            except Exception as e:
                logger.error(f"Failed to fetch account balance: {e}")
                raise RuntimeError(f"Could not fetch account balance: {e}")

__all__ = ["NexusEngine"]

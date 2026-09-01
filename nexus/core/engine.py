"""
NEXUS Core Trading Engine
Author: Swapnil De Sarkar
Created: 2025

Core NexusEngine implementation for autonomous trading.

Responsibilities:
- Maintain strategy/model/risk registries
- Track performance stats (trades, wins, losses, profit)
- Emotional state adjustments based on trade outcomes
- Payout threshold enforcement for non-demo trades with override support
- Advanced risk management with position sizing
- Execute trades with broker integration
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

from nexus.payouts.fetch import (
    get_payout_for_market,
    is_override_enabled,
    is_payout_allowed,
)
from nexus.utils.config import NexusSettings
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.core.engine")

# Seedable RNG for deterministic profit simulation if desired
_seed_env = os.getenv("NEXUS_ENGINE_RNG_SEED")
if _seed_env:
    try:
        random.seed(int(_seed_env))
    except ValueError:
        random.seed(_seed_env)


class TradeResult(TypedDict, total=False):
    success: bool
    profit: float
    asset: str
    direction: str
    expiration: str
    error: str
    real_executed: bool
    order_id: str


@dataclass
class EngineState:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0


class NexusEngine:
    def __init__(
        self,
        settings: NexusSettings,
        demo_mode: bool = True,
        auto_login: Optional[bool] = None,
    ) -> None:
        self.settings = settings
        self._demo_mode = bool(demo_mode)
        self.auto_login = bool(settings.auto_login) if auto_login is None else bool(auto_login)

        # Public registries
        self.strategy_registry: Dict[str, Any] = {}
        self.model_registry: Dict[str, Any] = {}
        self.risk_registry: Dict[str, Any] = {}

        # Optional future extension placeholders
        self.meta_strategy: Any = None

        # Emotional state (bounded in [0,1])
        self.emotion_state: Dict[str, float] = {
            "greed": 0.5,
            "fear": 0.5,
            "confidence": 0.5,
        }

        self._state = EngineState()
        # Persistence configuration
        self._persist_enabled = os.getenv("NEXUS_PERSIST_ENGINE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self._state_path = Path("models/engine_state.json")
        if self._persist_enabled:
            self._load_state()
        # Initialize proper exploration controller
        from nexus.intelligence.exploration import ExplorationController

        self.exploration_controller = ExplorationController(settings)
        self._initialized = False
        # Risk / drawdown tracking
        self.active_positions: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.virtual_demo_balance: float = 10000.0
        self.virtual_real_balance: float = 0.0
        self._peak_equity: float = 10_000.0
        self._max_drawdown_pct: float = 0.0
        self.circuit_breaker_active: bool = False
        self._broker: Any = None
        self.ai_engine: Any = None
        self._bg_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[Any] = None
        logger.debug(
            "NexusEngine initialized (demo_mode=%s, auto_login=%s)", self.demo_mode, self.auto_login
        )

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    @demo_mode.setter
    def demo_mode(self, value: bool) -> None:
        val = bool(value)
        self._demo_mode = val
        if self._broker is not None:
            self._broker.demo_mode = val
            try:
                if hasattr(self._broker, "set_practice_mode"):
                    self.run_async(self._broker.set_practice_mode(val))
            except Exception as err:
                logger.debug("Failed setting broker practice mode: %s", err)

    @property
    def virtual_balance(self) -> float:
        return self.virtual_demo_balance if self._demo_mode else self.virtual_real_balance

    @virtual_balance.setter
    def virtual_balance(self, val: float) -> None:
        if self._demo_mode:
            self.virtual_demo_balance = float(val)
        else:
            self.virtual_real_balance = float(val)

    async def get_ai_prediction(self, asset: str, is_otc: Optional[bool] = None) -> Dict[str, Any]:
        """Lazy load RealAITradingEngine and generate SOTA multi-model prediction."""
        # Live trading must never fall back to synthetic candles.  A missing
        # broker feed is a HOLD condition, not a reason to invent a signal.
        candles = None
        if self._broker is not None and getattr(self._broker, "authenticated", False):
            try:
                if hasattr(self._broker, "get_candles_async"):
                    candles = await self._broker.get_candles_async(asset, 60, 120)
            except Exception as err:
                logger.warning("Live candle fetch failed for %s: %s", asset, err)
        if candles is None or len(candles) < 20:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "recommended_expiration": 60 if not (is_otc or "otc" in asset.lower()) else 5,
                "reasoning": "No live broker candles available; trading blocked",
                "data_source": "unavailable",
            }
        if not hasattr(candles, "columns"):
            try:
                import pandas as pd

                candles = pd.DataFrame(candles)
            except Exception as err:
                logger.warning("Live candle data could not be normalized for %s: %s", asset, err)
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "recommended_expiration": 60,
                    "reasoning": "Live broker candles have an invalid format; trading blocked",
                    "data_source": "unavailable",
                }
        if self.ai_engine is None:
            try:
                from nexus.ai.engine_ai import RealAITradingEngine

                self.ai_engine = RealAITradingEngine()
            except Exception as err:
                logger.warning("Could not load RealAITradingEngine: %s", err)
                default_exp = (
                    60
                    if is_otc is False
                    else 5
                    if is_otc is True
                    else (5 if "otc" in asset.lower() else 60)
                )
                return {
                    "signal": "hold",
                    "confidence": 0.0,
                    "stake": 10.0,
                    "recommended_expiration": default_exp,
                    "reasoning": "AI engine unavailable; trading blocked",
                }
        try:
            result = await self.ai_engine.analyze_market(
                candles, asset=asset, timeframe=60, is_otc=is_otc
            )  # type: ignore[no-any-return]
            result["data_source"] = "live broker candles"
            return result
        except Exception as e:
            logger.warning("AI analysis failed for %s: %s", asset, e)
            default_exp = (
                60
                if is_otc is False
                else 5
                if is_otc is True
                else (5 if "otc" in asset.lower() else 60)
            )
            return {
                "signal": "hold",
                "confidence": 0.0,
                "stake": 10.0,
                "recommended_expiration": default_exp,
                "reasoning": f"Live AI analysis unavailable: {e}",
                "data_source": "unavailable",
            }

    def train_market_ai(self, asset: str) -> Dict[str, Any]:
        """Train AI models and dynamically select best indicators for a specific market."""
        if self.ai_engine is None:
            try:
                from nexus.ai.engine_ai import RealAITradingEngine

                self.ai_engine = RealAITradingEngine()
            except Exception as err:
                logger.warning("Could not load RealAITradingEngine for training: %s", err)
                return {"symbol": asset, "error": str(err)}
        return cast(Dict[str, Any], self.ai_engine.train_market(asset))

    def train_all_markets_ai(self, assets: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Train AI models and optimize indicators for all available markets."""
        if self.ai_engine is None:
            try:
                from nexus.ai.engine_ai import RealAITradingEngine

                self.ai_engine = RealAITradingEngine()
            except Exception as err:
                logger.warning("Could not load RealAITradingEngine for batch training: %s", err)
                return []
        if not assets:
            from nexus.adapters.quotex import COMMON_ASSETS

            assets = list(COMMON_ASSETS)
        return cast(List[Dict[str, Any]], self.ai_engine.train_all_markets(assets))

    def run_async(self, coro: Any) -> Any:
        """Run a coroutine safely on the engine's persistent background event loop."""
        if self._bg_loop is None or self._bg_loop.is_closed():
            import threading

            self._bg_loop = asyncio.new_event_loop()

            def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            self._bg_thread = threading.Thread(
                target=_loop_worker, args=(self._bg_loop,), daemon=True
            )
            self._bg_thread.start()

        future = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return future.result()

    async def initialize_components(
        self,
    ) -> None:  # pragma: no cover - placeholder for GUI compatibility
        """Async initialization hook (GUI expects this)."""
        self._initialized = True
        # Auto-login to broker if enabled
        if self.auto_login:
            try:
                await self.login_broker()
            except Exception as e:  # pragma: no cover
                logger.warning("Auto-login failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Broker integration
    # ------------------------------------------------------------------
    async def login_broker(self) -> bool:
        """Connect to Quotex using stored credentials; set practice/real mode."""
        try:
            from nexus.adapters.quotex_adapter import QuotexAdapter as BrokerAdapter
        except Exception as e:  # pragma: no cover
            logger.error("Broker adapter unavailable: %s", e)
            return False
        q = self.settings.quotex
        email = (
            getattr(q, "email", "") or os.getenv("QUOTEX_EMAIL") or os.getenv("QUOTEX__EMAIL") or ""
        )
        password = (
            getattr(q, "password", "")
            or os.getenv("QUOTEX_PASSWORD")
            or os.getenv("QUOTEX__PASSWORD")
            or ""
        )

        if not email or not password:
            logger.warning("Broker credentials missing; running in simulated mode")
            return False

        # Create adapter if needed
        if self._broker is None:
            self._broker = BrokerAdapter(
                email=email,
                password=password,
                lang=getattr(q, "lang", "en"),
                demo_mode=self.demo_mode,
            )
            # Forward optional session params
            ua = getattr(q, "user_agent", None)
            cookies = getattr(q, "cookies", None)
            ssid = getattr(q, "ssid", None)
            if ua:
                try:
                    self._broker.set_session(ua, cookies=cookies, ssid=ssid)
                except Exception:
                    pass
        ok = await self._broker.connect()
        if not ok:
            logger.error("Failed to connect to Quotex broker")
            return False
        # Ensure correct account mode
        try:
            await self._broker.set_practice_mode(bool(self.demo_mode))
        except Exception:
            pass
        logger.info("Connected to Quotex (mode=%s)", "DEMO" if self.demo_mode else "REAL")
        return True

    async def get_account_balance(self) -> float:
        try:
            if self._broker is not None:
                if hasattr(self._broker, "set_practice_mode"):
                    try:
                        await self._broker.set_practice_mode(self.demo_mode)
                    except Exception:
                        pass
                # Prefer the async broker call: the synchronous accessor reads
                # pyquotex's cached balance and can lag after settlements.
                if hasattr(self._broker, "get_balance_async"):
                    bal_obj = self._broker.get_balance_async()
                else:
                    bal_obj = self._broker.get_balance()
                bal: Any = await bal_obj if asyncio.iscoroutine(bal_obj) else bal_obj
                if isinstance(bal, (int, float)):
                    b_val = float(bal)
                    if b_val <= 0:
                        raise RuntimeError("Broker returned no usable account balance")
                    if self.demo_mode:
                        self.virtual_demo_balance = b_val
                        return float(self.virtual_demo_balance)
                    else:
                        self.virtual_real_balance = b_val
                        return float(self.virtual_real_balance)
            return float(self.virtual_demo_balance if self.demo_mode else self.virtual_real_balance)
        except Exception:
            if self._broker is not None:
                raise
            return float(self.virtual_demo_balance if self.demo_mode else self.virtual_real_balance)

    # ------------------------------------------------------------------
    # Persistence helpers (engine state + emotions)
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            if self._state_path.exists():
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                es = raw.get("engine_state", {})
                self._state.total_trades = int(es.get("total_trades", 0))
                self._state.winning_trades = int(es.get("winning_trades", 0))
                self._state.losing_trades = int(es.get("losing_trades", 0))
                self._state.total_profit = float(es.get("total_profit", 0.0))
                em = raw.get("emotions", {})
                for k in ("greed", "fear", "confidence"):
                    if k in em:
                        self.emotion_state[k] = float(em[k])
                # Restore drawdown stats if present
                dd = raw.get("drawdown", {})
                self._peak_equity = float(dd.get("peak_equity", self._peak_equity))
                self._max_drawdown_pct = float(dd.get("max_drawdown_pct", 0.0))
                self.circuit_breaker_active = bool(dd.get("circuit_breaker", False))
                logger.info("Loaded engine state from %s", self._state_path)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed loading engine state: {e}")

    def _save_state(self) -> None:
        if not self._persist_enabled:
            return
        try:
            self._state_path.parent.mkdir(exist_ok=True, parents=True)
            payload = {
                "engine_state": {
                    "total_trades": self._state.total_trades,
                    "winning_trades": self._state.winning_trades,
                    "losing_trades": self._state.losing_trades,
                    "total_profit": self._state.total_profit,
                },
                "emotions": self.emotion_state,
                "drawdown": {
                    "peak_equity": self._peak_equity,
                    "max_drawdown_pct": self._max_drawdown_pct,
                    "circuit_breaker": self.circuit_breaker_active,
                },
            }
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed saving engine state: {e}")

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def register_strategy(self, name: str, strategy: Any) -> None:
        self.strategy_registry[name] = strategy

    def unregister_strategy(self, name: str) -> None:  # noqa: D401
        self.strategy_registry.pop(name, None)

    # ------------------------------------------------------------------
    # Emotional state handling
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo if x < lo else hi if x > hi else x

    def update_emotional_state(self, trade_result: Dict[str, Any]) -> None:
        success = bool(trade_result.get("success"))
        profit = float(trade_result.get("profit", 0.0) or 0.0)

        # Base adjustments
        delta_greed = 0.05 if success else -0.05
        delta_fear = -0.05 if success else 0.05

        # Confidence: scale by profit magnitude (soft cap)
        if profit != 0:
            magnitude = min(1.0, abs(profit) / (self.settings.trading.base_trade_amount or 1.0))
            scale = 0.1 * magnitude
            delta_conf = scale if profit > 0 else -scale
        else:
            delta_conf = 0.02 if success else -0.02

        self.emotion_state["greed"] = self._clamp(self.emotion_state["greed"] + delta_greed)
        self.emotion_state["fear"] = self._clamp(self.emotion_state["fear"] + delta_fear)
        self.emotion_state["confidence"] = self._clamp(
            self.emotion_state["confidence"] + delta_conf
        )

    # ------------------------------------------------------------------
    # Risk / position sizing
    # ------------------------------------------------------------------
    def advanced_risk_management(self, context: Dict[str, Any], base_amount: float) -> float:
        """Return adjusted position size (>=1.0).
        Simple heuristic: modify by confidence and inverse fear.
        Circuit breaker: if active clamp to minimum safe size (1.0).
        """
        if self.circuit_breaker_active:
            return 1.0
        greed = self.emotion_state.get("greed", 0.5)
        fear = self.emotion_state.get("fear", 0.5)
        confidence = self.emotion_state.get("confidence", 0.5)
        modifier = 1.0 + (greed - 0.5) * 0.2 + (confidence - 0.5) * 0.3 - (fear - 0.5) * 0.2
        size = max(1.0, base_amount * max(0.2, modifier))
        return round(size, 2)

    # ------------------------------------------------------------------
    # Trade execution (broker-backed when available; simulation fallback)
    # ------------------------------------------------------------------
    async def execute_trade(
        self,
        asset: str,
        signal_type: str,
        amount: float,
        expiration: str | int,
    ) -> TradeResult:
        direction = (signal_type or "call").lower()
        if direction not in {"call", "put"}:
            direction = "call"
        exp_key = str(expiration)

        # Validate order inputs before touching the market catalog or broker.
        # This is especially important for the live path, where malformed
        # values should never be forwarded to an external order API.
        try:
            amount_value = float(amount)
            expiration_value = int(expiration)
        except (TypeError, ValueError, OverflowError):
            return {
                "success": False,
                "error": "Trade amount and expiration must be numeric",
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }
        if not math.isfinite(amount_value) or amount_value <= 0:
            return {
                "success": False,
                "error": "Trade amount must be a finite value greater than zero",
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }
        if expiration_value <= 0:
            return {
                "success": False,
                "error": "Expiration must be greater than zero seconds",
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }
        amount = amount_value

        if self.circuit_breaker_active:
            return {
                "success": False,
                "error": "Circuit breaker active",
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }

        from nexus.catalog.ingest import get_market_by_symbol

        m_info = get_market_by_symbol(asset)
        payout = get_payout_for_market(asset, exp_key) or (
            m_info.display_payout_percent if m_info else 0.0
        )
        is_active = m_info.active if m_info else (payout > 0.0)
        # The static catalog can lag Quotex or contain a different OTC
        # snapshot. For an authenticated broker, reconcile the symbol and
        # payout against the broker's current catalog before rejecting it.
        if self._broker is not None and getattr(self._broker, "authenticated", False):
            try:
                live_assets = await self._broker.get_assets_with_payouts_async()
                target = str(asset).strip().upper()
                for live in live_assets or []:
                    if not isinstance(live, dict) or str(live.get("symbol", "")).strip().upper() != target:
                        continue
                    live_payout = float(live.get("payout", 0.0) or 0.0)
                    payout = get_payout_for_market(asset, exp_key) or live_payout
                    is_active = bool(live.get("active", True))
                    break
            except Exception as catalog_err:
                logger.warning("Live broker catalog reconciliation failed for %s: %s", asset, catalog_err)
        if not is_active or payout <= 0.0:
            return {
                "success": False,
                "error": f"Market {asset} is currently OFFLINE / CLOSED.",
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }

        # Non‑demo: enforce payout threshold unless override active
        if not self.demo_mode:
            threshold = float(self.settings.trading.payout_threshold)
            if not is_override_enabled() and not is_payout_allowed(payout, threshold):
                result: TradeResult = {
                    "success": False,
                    "error": f"Payout below threshold ({payout} < {threshold})",
                    "asset": asset,
                    "direction": direction,
                    "expiration": exp_key,
                }
                return result

        # If broker is connected and not forced to simulate, place a real (demo/real) order
        force_sim = os.getenv("NEXUS_FORCE_SIM", "0").lower() in {"1", "true", "yes"}
        if self._broker is not None and not force_sim:
            if not self.demo_mode:
                current_bal = await self.get_account_balance()
                if current_bal < float(amount):
                    return {
                        "success": False,
                        "error": f"Insufficient REAL balance (${current_bal:.2f}). Switch to DEMO mode or deposit funds to trade REAL.",
                        "asset": asset,
                        "direction": direction,
                        "expiration": exp_key,
                        "real_executed": True,
                    }
            try:
                await self._broker.set_practice_mode(bool(self.demo_mode))
                order = await self._broker.buy_simple_async(
                    asset=asset,
                    direction=direction,
                    amount=float(amount),
                    expiration=int(exp_key),
                )

                if isinstance(order, dict) and order.get("success"):
                    def _extract_order_id(value: Any) -> Any:
                        for _ in range(4):
                            if isinstance(value, dict):
                                value = value.get("id") or value.get("order_id") or value.get("order")
                            elif isinstance(value, str) and value.lstrip().startswith("{"):
                                try:
                                    value = ast.literal_eval(value)
                                except (SyntaxError, ValueError):
                                    break
                            else:
                                break
                        return value

                    # Quotex has returned: ID string, {id: ID}, and nested
                    # {id: {id: ID}} shapes across its client methods.
                    broker_order_id = _extract_order_id(
                        order.get("order_id") or order.get("order")
                    )
                    return {
                        "success": True,
                        "order_id": str(broker_order_id) if broker_order_id else "",
                        "asset": asset,
                        "direction": direction,
                        "expiration": exp_key,
                        "real_executed": True,
                    }
                else:
                    err_msg = (
                        order.get("error", "Broker did not confirm order")
                        if isinstance(order, dict)
                        else "Broker order placement failed"
                    )
                    logger.warning("Broker trade rejected: %s", err_msg)
                    return {
                        "success": False,
                        "error": err_msg,
                        "asset": asset,
                        "direction": direction,
                        "expiration": exp_key,
                        "real_executed": True,
                    }
            except Exception as e:
                logger.error("Broker trade placement error: %s", e)
                return {
                    "success": False,
                    "error": str(e),
                    "asset": asset,
                    "direction": direction,
                    "expiration": exp_key,
                    "real_executed": True,
                }

        enable_stochastic = os.getenv("NEXUS_ENABLE_STOCHASTIC", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        # Base amount for profit/loss calculations
        base = float(amount)
        if enable_stochastic:
            # Configurable success probability
            try:
                p_win = float(os.getenv("NEXUS_P_WIN", "0.6"))
            except ValueError:
                p_win = 0.6
            p_win = min(0.99, max(0.01, p_win))
            win = random.random() < p_win

            # Profit / loss multiplier ranges
            def _parse_range(
                env_key: str, default_low: float, default_high: float
            ) -> tuple[float, float]:
                raw = os.getenv(env_key)
                if not raw:
                    return (default_low, default_high)
                try:
                    lo_s, hi_s = raw.split(",", 1)
                    lo_v = float(lo_s.strip())
                    hi_v = float(hi_s.strip())
                    if hi_v < lo_v:
                        lo_v, hi_v = hi_v, lo_v
                    return (lo_v, hi_v)
                except Exception:
                    return (default_low, default_high)

            win_lo, win_hi = _parse_range("NEXUS_PROFIT_MULT_RANGE", 0.05, 0.15)
            loss_lo, loss_hi = _parse_range("NEXUS_LOSS_MULT_RANGE", 0.05, 0.15)
            if win:
                mult = random.uniform(win_lo, win_hi)
                profit = base * mult
            else:
                mult = random.uniform(loss_lo, loss_hi)
                profit = -base * mult
            self.record_trade(win, profit)
            return {
                "success": bool(win),
                "profit": round(profit, 4),
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }
        else:
            # Legacy deterministic-positive profit path (retains original test expectations)
            profit_multiplier = 0.1
            try:
                profit_multiplier = random.uniform(0.05, 0.15)
            except Exception:  # pragma: no cover
                profit_multiplier = 0.1
            profit = base * profit_multiplier
            self.record_trade(True, profit)
            return {
                "success": True,
                "profit": round(profit, 4),
                "asset": asset,
                "direction": direction,
                "expiration": exp_key,
                "real_executed": False,
            }

    # ------------------------------------------------------------------
    # Accounting helpers
    # ------------------------------------------------------------------
    def record_trade(self, success: bool, profit: float) -> None:
        self._state.total_trades += 1
        if success:
            self._state.winning_trades += 1
        else:
            self._state.losing_trades += 1
        self._state.total_profit += profit
        self.update_emotional_state({"success": success, "profit": profit})
        # Update drawdown metrics & circuit breaker
        current_equity = 10_000.0 + self._state.total_profit
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = self._peak_equity - current_equity
        dd_pct = (drawdown / self._peak_equity) * 100 if self._peak_equity > 0 else 0.0
        if dd_pct > self._max_drawdown_pct:
            self._max_drawdown_pct = dd_pct
        threshold_env = os.getenv("NEXUS_MAX_DRAWDOWN_PCT")
        if threshold_env is not None:
            try:
                thresh = float(threshold_env)
                if self._max_drawdown_pct >= thresh and not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    logger.warning(
                        "Circuit breaker activated: drawdown %.2f%% >= threshold %.2f%%",
                        self._max_drawdown_pct,
                        thresh,
                    )
            except ValueError:
                pass
        # Persist after each trade (lightweight JSON)
        self._save_state()

    def get_performance_stats(self) -> Dict[str, Any]:
        return {
            "total_trades": self._state.total_trades,
            "winning_trades": self._state.winning_trades,
            "losing_trades": self._state.losing_trades,
            "total_profit": round(self._state.total_profit, 2),
            "max_drawdown_pct": round(self._max_drawdown_pct, 2),
            "circuit_breaker": self.circuit_breaker_active,
        }

    def get_risk_state(self) -> Dict[str, Any]:
        return {
            "peak_equity": self._peak_equity,
            "max_drawdown_pct": self._max_drawdown_pct,
            "circuit_breaker": self.circuit_breaker_active,
        }

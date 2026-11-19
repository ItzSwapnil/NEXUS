"""FastAPI application for the NEXUS browser dashboard.

The web layer is deliberately thin: the NexusEngine remains the single
owner of broker state, AI state, and the durable trade ledger.  The browser
only receives broker-confirmed data and never calculates trade outcomes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from nexus.core.engine import NexusEngine
from nexus.data.trade_history import TradeHistory
from nexus.features import get_feature_provider_catalog
from nexus.intelligence.continual import ContinualLearningService
from nexus.utils.config import NexusSettings
from nexus.utils.device import get_device_info
from nexus.web.security import has_valid_service_token

logger = logging.getLogger("nexus.web")

WEB_ROOT = Path(__file__).with_name("static")
RESEARCH_REPORT = Path(__file__).resolve().parents[2] / "reports" / "robust_market_research.json"
RESEARCH_PROGRESS = (
    Path(__file__).resolve().parents[2] / "reports" / "robust_market_research.progress.json"
)


def _is_valid_market_asset(asset: str) -> bool:
    """Reject UI placeholders before they reach broker APIs."""
    value = str(asset or "").strip()
    if not value or value.lower() in {
        "undefined",
        "null",
        "unknown",
        "loading",
        "loading markets…",
        "—",
        "–",
        "...",
    }:
        return False
    return not any(marker in value for marker in ("—", "–", " "))


def read_research_integrity() -> dict[str, Any]:
    """Return compact, report-backed health metrics for the research console."""
    if not RESEARCH_REPORT.exists():
        return {"available": False, "detail": "Research report not found"}
    try:
        report = json.loads(RESEARCH_REPORT.read_text(encoding="utf-8"))
        records = [
            trade
            for market in report.get("markets", [])
            if market.get("status") == "complete"
            for trade in market.get("trade_records", [])
        ]
        confidences = [float(trade.get("confidence", 0.0) or 0.0) for trade in records]
        confidence_quality = report.get("confidence_quality", {})
        complete_records = [
            market for market in report.get("markets", []) if market.get("status") == "complete"
        ]
        error_records = [
            market for market in report.get("markets", []) if market.get("status") != "complete"
        ]
        sequence_model_rows = sum(
            int(market.get("sequence_model_rows", 0) or 0) for market in complete_records
        )
        walk_forward_splits = sum(len(market.get("folds", [])) for market in complete_records)
        promoted_records = [
            market.get("asset")
            for market in complete_records
            if market.get("promotion_status") == "promote"
        ]
        return {
            "available": True,
            "schema": report.get("report_schema", "unknown"),
            "markets": len(report.get("markets", [])),
            "completed_markets": report.get("completed_markets", len(complete_records)),
            "error_markets": report.get(
                "error_markets", [market.get("asset") for market in error_records]
            ),
            "oos_trades": report.get(
                "total_oos_trades",
                sum(int(market.get("trades", 0) or 0) for market in complete_records),
            ),
            "promoted_markets": report.get("promoted_markets", promoted_records),
            "orders_placed": report.get("orders_placed", 0),
            "max_confidence": max(confidences, default=0.0),
            "exact_100_confidence": sum(value >= 1.0 for value in confidences),
            "high_confidence_oos_trades": sum(value >= 0.70 for value in confidences),
            "confidence_evidence_requirement": "20 held-out trades at every 50/60/70/80/90/99% threshold",
            "under_50_confidence": sum(value < 0.5 for value in confidences),
            "monotonic_confidence_markets": confidence_quality.get("monotonic_markets", 0),
            "evaluated_confidence_markets": confidence_quality.get(
                "evaluated_markets", report.get("completed_markets", 0)
            ),
            "sequence_model_rows": sequence_model_rows,
            "walk_forward_splits": walk_forward_splits,
            "report_updated": time.strftime(
                "%Y-%m-%d %H:%M:%S UTC", time.gmtime(RESEARCH_REPORT.stat().st_mtime)
            ),
        }
    except (OSError, ValueError, TypeError) as exc:
        return {"available": False, "detail": f"Invalid research report: {exc}"}


def read_research_progress() -> dict[str, Any]:
    """Return live progress written by an external or dashboard trainer."""
    if not RESEARCH_PROGRESS.exists():
        return {"available": False, "status": "idle", "detail": "No live research progress file"}
    try:
        progress = json.loads(RESEARCH_PROGRESS.read_text(encoding="utf-8"))
        progress["available"] = True
        progress["updated_at_text"] = time.strftime(
            "%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(progress.get("updated_at", 0)))
        )
        return dict(progress)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "available": False,
            "status": "error",
            "detail": f"Invalid research progress: {exc}",
        }


def _json_safe(value: Any) -> Any:
    """Convert model outputs containing NumPy values into JSON data."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class TradeRequest(BaseModel):
    asset: str = Field(min_length=1, max_length=40)
    direction: str = Field(pattern="^(call|put)$")
    amount: float = Field(gt=0, le=1000)
    expiration: int = Field(gt=0, le=3600)


class WorkspacePracticeRequest(BaseModel):
    assets: list[str] = Field(min_length=1, max_length=12)
    confirmation: str = ""


class ModeRequest(BaseModel):
    mode: str = Field(pattern="^(PRACTICE|REAL)$")
    confirmation: str = ""


class ControlRequest(BaseModel):
    pause: bool | None = None
    reset_circuit_breaker: bool = False
    auto_trade_enabled: bool | None = None
    ai_select_timeframe: bool | None = None
    min_confidence: float | None = Field(default=None, ge=0.5, le=0.99)
    payout_threshold: float | None = Field(default=None, ge=0, le=100)
    max_open_trades: int | None = Field(default=None, ge=1, le=20)
    base_trade_amount: float | None = Field(default=None, gt=0, le=1000)
    auto_trade_interval_seconds: int | None = Field(default=None, ge=5, le=300)
    auto_trade_assets: list[str] | None = Field(default=None, max_length=50)


class ClearHistoryRequest(BaseModel):
    confirmation: str
    include_unresolved: bool = False


class BacktestRequest(BaseModel):
    asset: str | None = None
    assets: list[str] = Field(default_factory=list, max_length=100)
    market_scope: Literal["single", "multiple", "all"] = "single"
    learning_hours: int = Field(default=10, ge=1, le=24)
    evaluation_hours: int = Field(default=2, ge=1, le=12)
    candle_timeframe: int = Field(default=60, ge=1, le=3600)
    ai_select_timeframe: bool = True
    trade_expiration: int = Field(default=60, ge=1, le=3600)
    stake_mode: Literal["ai", "fixed"] = "ai"
    stake: float | None = Field(default=None, gt=0, le=1000)
    min_confidence: float = 0.60
    min_payout: float = Field(default=0.0, ge=0, le=100)

    @field_validator("asset", mode="before")
    @classmethod
    def _validate_asset(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("min_confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, v: Any) -> float:
        if v is None:
            return 0.60
        try:
            val = float(v)
        except (ValueError, TypeError):
            return 0.60
        if 0.0 < val < 0.05:
            val *= 100.0
        if val > 1.0:
            val /= 100.0
        return max(0.50, min(0.99, val))


class ResearchRequest(BaseModel):
    market_scope: Literal["all", "selected"] = "all"
    assets: list[str] = Field(default_factory=list, max_length=100)
    min_payout: float = Field(default=80.0, ge=0.0, le=100.0)
    search_depth: Literal["rapid", "deep", "exhaustive"] = "deep"
    indicator_scope: Literal["core", "full"] = "core"
    folds: int = Field(default=3, ge=2, le=6)


class WebState:
    def __init__(
        self, settings: NexusSettings, demo_mode: bool = True, engine: NexusEngine | None = None
    ) -> None:
        self.engine = engine or NexusEngine(settings, demo_mode=demo_mode, auto_login=False)
        self.history = TradeHistory()
        self.clients: set[WebSocket] = set()
        self.trading_paused = True
        self._broadcast_task: asyncio.Task[None] | None = None
        self._continual_task: asyncio.Task[None] | None = None
        self._auto_task: asyncio.Task[None] | None = None
        self._login_task: asyncio.Task[None] | None = None
        self.login_status: dict[str, Any] = {"status": "idle"}
        self.continual_learning = ContinualLearningService()
        self.provider_catalog = get_feature_provider_catalog()
        self.backtest_status: dict[str, Any] = {"status": "idle"}
        self.backtest_markets: list[dict[str, Any]] = []
        self._backtest_task: asyncio.Task[None] | None = None
        self.backtest_started_at: float | None = None
        self.research_status: dict[str, Any] = {"status": "idle", "orders_placed": 0}
        self._research_task: asyncio.Task[None] | None = None
        self._research_process: asyncio.subprocess.Process | None = None
        self.login_refresh_runner: Callable[[], Coroutine[Any, Any, None]] | None = None
        self._auto_trade_cursor: int = 0
        self.auto_trade_last_fired: float = 0.0
        self.auto_trade_count: int = 0
        self.auto_trade_assets: list[str] = []
        # pyquotex uses one stateful WebSocket client. Serialize broker I/O,
        # while allowing CPU-heavy AI work and browser requests to continue.
        self.broker_io_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.engine.initialize_components()
        # Web startup must never silently fall back to simulation.  A failed
        # broker login is reported by /api/status and disables trading.
        await self.engine.login_broker()
        self._broadcast_task = asyncio.create_task(self.broadcast_loop())
        self._continual_task = asyncio.create_task(self.continuous_learning_loop())
        self._auto_task = asyncio.create_task(self.auto_trading_loop())
        if self.login_refresh_runner is not None:
            self._login_task = asyncio.create_task(self.login_refresh_runner())

    async def stop(self) -> None:
        if self._login_task and not self._login_task.done():
            self._login_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._login_task
        if self._broadcast_task:
            self._broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast_task
        if self._continual_task and not self._continual_task.done():
            self._continual_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._continual_task
        if self._auto_task and not self._auto_task.done():
            self._auto_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._auto_task
        if self._backtest_task and not self._backtest_task.done():
            self._backtest_task.cancel()
        if self._research_task and not self._research_task.done():
            self._research_task.cancel()
        if self._research_process and self._research_process.returncode is None:
            self._research_process.terminate()

    async def snapshot(self) -> dict[str, Any]:
        broker = getattr(self.engine, "_broker", None)
        authenticated = bool(broker and getattr(broker, "authenticated", False))
        if authenticated:
            try:
                async with self.broker_io_lock:
                    balance = await asyncio.wait_for(self.engine.get_account_balance(), timeout=5)
            except (asyncio.TimeoutError, OSError):
                balance = None
        else:
            balance = None
        stats = self.engine.get_performance_stats()
        return {
            "mode": "PRACTICE" if self.engine.demo_mode else "REAL",
            "authenticated": authenticated,
            "balance": balance,
            "active_trades": len(getattr(self.engine, "active_positions", [])),
            "performance": stats,
            "feature_providers": self.provider_catalog,
            "compute": get_device_info(),
            "backtest": self.backtest_status,
            "backtest_markets": self.backtest_markets,
            "backtest_elapsed_seconds": round(time.time() - self.backtest_started_at, 1)
            if self.backtest_started_at
            else 0.0,
            "research_training": self.research_status,
            "research_progress": read_research_progress(),
            "research_report": read_research_integrity(),
            "continual_learning": self.continual_learning.snapshot(),
            "controls": {
                "trading_paused": self.trading_paused,
                "auto_trade_enabled": bool(self.engine.settings.trading.auto_trade_enabled),
                "ai_select_timeframe": bool(self.engine.settings.trading.ai_select_timeframe),
                "min_confidence": self.engine.settings.trading.min_confidence,
                "payout_threshold": self.engine.settings.trading.payout_threshold,
                "max_open_trades": self.engine.settings.trading.max_open_trades,
                "base_trade_amount": self.engine.settings.trading.base_trade_amount,
                "circuit_breaker": self.engine.circuit_breaker_active,
                "auto_trade_interval_seconds": self.engine.settings.trading.auto_trade_interval_seconds,
                "auto_trade_count": self.auto_trade_count,
                "auto_trade_last_fired": self.auto_trade_last_fired,
                "auto_trade_assets": list(self.auto_trade_assets),
            },
            "trades": self.history.get_trade_history(limit=50),
        }

    async def broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(3)
            if not self.clients:
                continue
            try:
                payload = json.dumps(await self.snapshot(), default=str)
            except Exception:
                # A stalled or disconnected broker must not kill the live UI
                # broadcaster or make the control plane appear frozen.
                continue
            stale: list[WebSocket] = []
            for client in self.clients:
                try:
                    await client.send_text(payload)
                except Exception:
                    stale.append(client)
            self.clients.difference_update(stale)

    async def continuous_learning_loop(self) -> None:
        """Observe rotating active markets every second; never places orders."""
        assets: list[str] = []
        cursor = 0
        last_catalog = 0.0
        self.continual_learning.running = True
        try:
            while True:
                await asyncio.sleep(1)
                broker = getattr(self.engine, "_broker", None)
                if not broker or not getattr(broker, "authenticated", False):
                    self.continual_learning.last_error = "broker not authenticated"
                    continue
                now = time.time()
                if not assets or now - last_catalog >= 30:
                    try:
                        async with self.broker_io_lock:
                            catalog = await asyncio.wait_for(
                                broker.get_assets_with_payouts_async(), timeout=8
                            )
                        assets = sorted(
                            {
                                str(item.get("symbol"))
                                for item in catalog or []
                                if isinstance(item, dict)
                                and item.get("active")
                                and item.get("symbol")
                            }
                        )
                        cursor = cursor % max(1, len(assets))
                        last_catalog = now
                    except Exception as exc:
                        self.continual_learning.last_error = str(exc)[:240]
                        continue
                if not assets:
                    continue
                asset = assets[cursor % len(assets)]
                cursor += 1

                async def fetch(asset_name: str = asset, broker_ref: Any = broker) -> pd.DataFrame:
                    async with self.broker_io_lock:
                        raw = await asyncio.wait_for(
                            broker_ref.get_candles_async(asset_name, 60, 180), timeout=8
                        )
                    frame = pd.DataFrame(raw).dropna().reset_index(drop=True)
                    if "time" in frame.columns:
                        frame = frame.sort_values("time").reset_index(drop=True)
                    return frame

                await self.continual_learning.tick(asset, fetch)
        except asyncio.CancelledError:
            raise
        finally:
            self.continual_learning.running = False

    async def _reconcile_open_positions(self) -> None:
        """Check status of open positions whose expiration time has passed."""
        if not hasattr(self.engine, "active_positions") or not self.engine.active_positions:
            return
        broker = getattr(self.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            return

        now = time.time()
        to_remove = []
        for pos in list(self.engine.active_positions):
            if now >= pos.get("expire_at", 0) + 2:
                broker_id = pos.get("broker_order_id")
                local_id = pos.get("local_id") or pos.get("id") or ""
                if broker_id and hasattr(broker, "get_trade_outcome"):
                    try:
                        async with self.broker_io_lock:
                            real_result = broker.get_trade_outcome(str(broker_id))
                        if real_result and real_result.get("status") == "SETTLED":
                            outcome = str(real_result.get("outcome", "")).upper()
                            profit = float(real_result.get("profit", 0.0))
                            if local_id:
                                self.history.record_settlement(local_id, outcome, profit)
                            if self.engine.demo_mode and hasattr(self.engine, "virtual_balance"):
                                gain = (
                                    (pos.get("amount", 0.0) + profit)
                                    if profit > 0
                                    else (pos.get("amount", 0.0) if profit == 0 else 0.0)
                                )
                                self.engine.virtual_balance += gain
                            ai_engine = getattr(self.engine, "ai_engine", None)
                            if ai_engine and pos.get("analysis"):
                                try:
                                    await ai_engine.learn_and_evolve(
                                        asset=pos["asset"],
                                        signal_type=str(pos["direction"]).lower(),
                                        success=outcome == "WIN",
                                        profit=profit,
                                        analysis=pos["analysis"],
                                    )
                                except Exception as err:
                                    logger.debug("AI feedback update note: %s", err)
                            to_remove.append(pos)
                        elif now >= pos.get("expire_at", 0) + 35:
                            if local_id:
                                self.history.record_settlement(
                                    local_id, "UNVERIFIED", 0.0, status="UNVERIFIED"
                                )
                            to_remove.append(pos)
                    except Exception as exc:
                        logger.debug("Reconcile error for order %s: %s", broker_id, exc)
                elif not broker_id:
                    to_remove.append(pos)

        for pos in to_remove:
            if pos in self.engine.active_positions:
                self.engine.active_positions.remove(pos)

    async def auto_trading_loop(self) -> None:
        """Continuously scan target markets and place PRACTICE trades when the AI
        finds an eligible signal. Prioritizes user's assigned markets (single or grid)
        so 1 market can easily generate 10+ trades an hour."""
        catalog_cache: list[str] = []
        last_catalog_refresh: float = 0.0
        CATALOG_TTL = 60.0

        try:
            while True:
                interval = int(
                    getattr(self.engine.settings.trading, "auto_trade_interval_seconds", 15) or 15
                )
                await asyncio.sleep(max(5, interval))

                if not getattr(self.engine.settings.trading, "auto_trade_enabled", False):
                    await self._reconcile_open_positions()
                    continue
                if self.trading_paused:
                    await self._reconcile_open_positions()
                    continue
                if getattr(self.engine, "circuit_breaker_active", False):
                    await self._reconcile_open_positions()
                    continue

                broker = getattr(self.engine, "_broker", None)
                if not broker or not getattr(broker, "authenticated", False):
                    continue

                controls = self.engine.settings.trading
                now = time.time()

                # 1. Check if user configured explicit target markets (e.g. grid or single market)
                target_assets = [
                    a for a in self.auto_trade_assets if _is_valid_market_asset(a)
                ]

                # 2. Fall back to broker's active markets meeting payout threshold
                if not target_assets:
                    if not catalog_cache or now - last_catalog_refresh >= CATALOG_TTL:
                        try:
                            async with self.broker_io_lock:
                                raw = await asyncio.wait_for(
                                    broker.get_assets_with_payouts_async(), timeout=10
                                )
                            catalog_cache = [
                                str(item["symbol"])
                                for item in (raw or [])
                                if isinstance(item, dict)
                                and item.get("active")
                                and item.get("symbol")
                                and float(item.get("payout") or 0) >= controls.payout_threshold
                            ]
                            if catalog_cache:
                                self._auto_trade_cursor = self._auto_trade_cursor % len(catalog_cache)
                                last_catalog_refresh = now
                        except Exception as exc:
                            logger.debug("Auto-trade catalog refresh failed: %s", exc)
                            continue
                    target_assets = catalog_cache

                if not target_assets:
                    continue

                # Always reconcile expired positions
                await self._reconcile_open_positions()

                open_count = len(getattr(self.engine, "active_positions", []))
                if open_count >= controls.max_open_trades:
                    continue

                # For single market or small grid, evaluate up to available trade capacity
                batch_limit = min(
                    max(1, len(target_assets)), controls.max_open_trades - open_count
                )
                for _ in range(batch_limit):
                    if len(getattr(self.engine, "active_positions", [])) >= controls.max_open_trades:
                        break
                    asset = target_assets[self._auto_trade_cursor % len(target_assets)]
                    self._auto_trade_cursor = (self._auto_trade_cursor + 1) % len(target_assets)

                    try:
                        async with self.broker_io_lock:
                            prediction = await asyncio.wait_for(
                                self.engine.get_ai_prediction(
                                    asset, is_otc="otc" in asset.lower()
                                ),
                                timeout=15,
                            )
                        signal = str(prediction.get("signal", "hold")).lower()
                        confidence = float(prediction.get("confidence", 0.0) or 0.0)

                        if signal not in {"call", "put", "long", "short"}:
                            continue
                        if confidence < controls.min_confidence:
                            continue

                        direction = "call" if signal in {"call", "long"} else "put"
                        # OTC supports 5s-60s rapid trading; non-OTC defaults to 60s
                        default_exp = 30 if "otc" in asset.lower() else 60
                        expiration = int(
                            prediction.get("recommended_expiration")
                            or controls.default_expiration
                            or default_exp
                        )
                        amount = float(controls.base_trade_amount or 10.0)

                        async with self.broker_io_lock:
                            result = await asyncio.wait_for(
                                self.engine.execute_trade(
                                    asset,
                                    direction,
                                    amount,
                                    expiration,
                                ),
                                timeout=30,
                            )
                        if result.get("success"):
                            order_id = str(result.get("order_id") or "")
                            self.auto_trade_count += 1
                            self.auto_trade_last_fired = time.time()
                            local_id = f"auto_{int(time.time()*1000)}_{self.auto_trade_count}"

                            trade_rec = {
                                "local_id": local_id,
                                "broker_order_id": order_id,
                                "asset": asset,
                                "direction": direction,
                                "amount": amount,
                                "status": "PLACED",
                                "expiration": expiration,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            self.history.record_placement(trade_rec)

                            pos_entry = {
                                "id": local_id,
                                "local_id": local_id,
                                "broker_order_id": order_id,
                                "asset": asset,
                                "direction": direction,
                                "amount": amount,
                                "expiration": expiration,
                                "placed_at": time.time(),
                                "expire_at": time.time() + expiration,
                                "analysis": prediction,
                            }
                            if not hasattr(self.engine, "active_positions"):
                                self.engine.active_positions = []
                            self.engine.active_positions.append(pos_entry)

                            logger.info(
                                "Auto-trade #%s: %s %s confidence=%.1f%% expiry=%ss order_id=%s",
                                self.auto_trade_count,
                                asset,
                                direction,
                                confidence * 100,
                                expiration,
                                order_id,
                            )
                    except asyncio.TimeoutError:
                        pass
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug("Auto-trade error for %s: %s", asset, exc)

                await self._reconcile_open_positions()

        except asyncio.CancelledError:
            raise


def create_app(
    settings: NexusSettings, demo_mode: bool = True, engine: NexusEngine | None = None
) -> FastAPI:
    state = WebState(settings, demo_mode=demo_mode, engine=engine)

    async def run_browser_login() -> None:
        """Run the interactive session refresher without blocking Uvicorn."""
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "scripts" / "login_browser.py"
        state.login_status = {"status": "starting", "detail": "Launching browser login…"}
        try:
            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "python",
                str(script),
                cwd=project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output: list[str] = []
            assert process.stdout is not None
            async for raw_line in process.stdout:
                line = raw_line.decode(errors="replace").strip()
                if line:
                    output.append(line)
                    state.login_status = {"status": "running", "detail": line[-240:]}
            return_code = await process.wait()
            if return_code != 0 or not any("Updated .env file" in line for line in output):
                state.login_status = {
                    "status": "error",
                    "detail": output[-1] if output else "Browser login failed",
                }
                return
            from dotenv import load_dotenv

            load_dotenv(project_root / ".env", override=True)
            async with state.broker_io_lock:
                state.engine._broker = None
                connected = await state.engine.login_broker()
            state.login_status = {
                "status": "complete" if connected else "error",
                "detail": "Fresh Quotex session connected"
                if connected
                else "Session captured but broker reconnect failed",
                "authenticated": connected,
            }
        except asyncio.CancelledError:
            state.login_status = {"status": "cancelled"}
        except Exception as exc:
            state.login_status = {"status": "error", "detail": str(exc)}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await state.start()
        try:
            yield
        finally:
            await state.stop()

    state.login_refresh_runner = run_browser_login
    app = FastAPI(title="NEXUS Trading Terminal", version=settings.version, lifespan=lifespan)
    app.state.nexus = state
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    service_token = settings.api_token.strip()
    if settings.api_token_required and not service_token:
        raise ValueError("NEXUS_API_REQUIRE_TOKEN=true requires NEXUS_API_TOKEN")

    def request_has_valid_service_token(headers: Any) -> bool:
        return has_valid_service_token(headers, service_token)

    @app.middleware("http")
    async def disable_dashboard_asset_cache(request: Any, call_next: Any) -> Any:
        """Keep the local control plane from running a stale JS/CSS bundle."""
        if (
            settings.api_token_required
            and request.url.path.startswith("/api/")
            and not request_has_valid_service_token(request.headers)
        ):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API token"})
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await state.snapshot()

    @app.get("/api/research/report")
    async def research_report(
        asset: str | None = None,
        min_confidence: float = 0.0,
        min_payout: float = 0.0,
        sort: str = "timestamp",
        descending: bool = True,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Return a bounded, filterable view of the research-only report."""
        if not RESEARCH_REPORT.exists():
            return {
                "report_schema": "unavailable",
                "catalog_assets": 0,
                "completed_markets": 0,
                "error_markets": [],
                "orders_placed": 0,
                "payout_filter_available": False,
                "payout_note": "No research report has been generated yet.",
                "filters": {
                    "asset": asset,
                    "min_confidence": min_confidence,
                    "min_payout": min_payout,
                    "sort": sort,
                    "descending": descending,
                },
                "matched_trades": 0,
                "returned_trades": 0,
                "aggregate": {"net_pnl": 0.0, "amount_allocated": 0.0, "win_rate": None},
                "markets": [],
                "trades": [],
            }
        try:
            report = json.loads(RESEARCH_REPORT.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Unable to read research report: {exc}"
            ) from exc
        markets = report.get("markets", [])
        selected_markets = [m for m in markets if not asset or str(m.get("asset")) == asset]
        payout_available = any(
            "payout_percent" in trade or "payout_rate" in trade
            for market in selected_markets
            for trade in market.get("trade_records", [])
        )
        trades: list[dict[str, Any]] = []
        for market in selected_markets:
            for trade in market.get("trade_records", []):
                confidence = float(trade.get("confidence", 0.0) or 0.0)
                payout = trade.get("payout_percent", trade.get("payout_rate"))
                if confidence < min_confidence:
                    continue
                if min_payout > 0 and (payout is None or float(payout) < min_payout):
                    continue
                item = dict(trade)
                item.setdefault("asset", market.get("asset"))
                item["payout_available"] = payout is not None
                trades.append(item)
        allowed_sort = {
            "timestamp",
            "confidence",
            "amount_allocated",
            "gross_pnl",
            "net_pnl",
            "account_balance_after",
            "regime",
        }
        sort_key = sort if sort in allowed_sort else "timestamp"

        def chronological_key(item: dict[str, Any]) -> tuple[int, Any]:
            value = item.get("timestamp")
            if value is not None:
                try:
                    return (0, float(value))
                except (TypeError, ValueError):
                    pass
            return (1, str(value or ""))

        def display_key(item: dict[str, Any]) -> tuple[int, Any]:
            return (
                chronological_key(item)
                if sort_key == "timestamp"
                else (item.get(sort_key) is None, item.get(sort_key, 0))
            )

        trades.sort(key=display_key, reverse=descending)
        # Re-run money management on the filtered sample. This prevents a
        # 70%+ view from inheriting balance/amounts from excluded trades.
        filtered_path: list[dict[str, Any]] = []
        filtered_capital = 10_000.0
        for original in sorted(trades, key=chronological_key):
            item = dict(original)
            risk_fraction = max(0.0, float(item.get("risk_fraction", 0.001) or 0.001))
            amount_before = filtered_capital * risk_fraction
            amount_after = float(max(1, round(amount_before)))
            original_amount = max(1.0, float(item.get("amount_allocated", 1.0) or 1.0))
            gross_rate = float(item.get("gross_pnl", 0.0) or 0.0) / original_amount
            net_rate = float(item.get("net_pnl", 0.0) or 0.0) / original_amount
            item["filtered_trade_amount_before"] = amount_before
            item["filtered_trade_amount_after"] = amount_after
            item["original_amount_allocated"] = original_amount
            item["account_balance_before"] = filtered_capital
            item["amount_allocated"] = amount_after
            item["gross_pnl"] = amount_after * gross_rate
            item["net_pnl"] = amount_after * net_rate
            item["fees"] = float(item.get("fees", 0.0) or 0.0) / original_amount * amount_after
            item["spread"] = float(item.get("spread", 0.0) or 0.0) / original_amount * amount_after
            item["slippage"] = (
                float(item.get("slippage", 0.0) or 0.0) / original_amount * amount_after
            )
            payout = item.get("payout_percent")
            item["payout_amount"] = (
                amount_after * float(payout) / 100.0
                if payout is not None and item.get("win")
                else 0.0
            )
            item["payment_after_payout"] = (
                amount_after * (1.0 + float(payout) / 100.0)
                if payout is not None and item.get("win")
                else 0.0
            )
            filtered_capital += float(item["net_pnl"])
            item["account_balance_after"] = filtered_capital
            filtered_path.append(item)
        # Reapply the requested display order after chronological money replay.
        trades = sorted(filtered_path, key=display_key, reverse=descending)
        bounded_limit = max(1, min(int(limit), 5000))
        matched_net = sum(float(item.get("net_pnl", 0.0) or 0.0) for item in filtered_path)
        matched_allocated = sum(
            float(item.get("amount_allocated", 0.0) or 0.0) for item in filtered_path
        )
        matched_wins = sum(bool(item.get("win")) for item in filtered_path)
        return {
            "report_schema": report.get("report_schema", "unknown"),
            "catalog_assets": report.get("assets", len(markets)),
            "completed_markets": report.get("completed_markets"),
            "error_markets": report.get("error_markets", []),
            "orders_placed": report.get("orders_placed", 0),
            "payout_filter_available": payout_available,
            "payout_note": "Historical payout was not captured in this report; payout filtering is unavailable."
            if not payout_available
            else "Historical payout field is available.",
            "filters": {
                "asset": asset,
                "min_confidence": min_confidence,
                "min_payout": min_payout,
                "sort": sort_key,
                "descending": descending,
            },
            "matched_trades": len(filtered_path),
            "returned_trades": min(len(filtered_path), bounded_limit),
            "aggregate": {
                "net_pnl": matched_net,
                "gross_pnl": sum(
                    float(item.get("gross_pnl", 0.0) or 0.0) for item in filtered_path
                ),
                "amount_allocated": matched_allocated,
                "wins": matched_wins,
                "losses": len(filtered_path) - matched_wins,
                "win_rate": matched_wins / len(filtered_path) if filtered_path else None,
                "roi": matched_net / matched_allocated if matched_allocated else 0.0,
            },
            "trades": trades[:bounded_limit],
            "markets": [
                {
                    "asset": m.get("asset"),
                    "status": m.get("status"),
                    "trades": m.get("trades", 0),
                    "win_rate": m.get("win_rate"),
                    "net_pnl": m.get("net_pnl"),
                    "promotion_status": m.get("promotion_status"),
                    "confidence_buckets": m.get("confidence_buckets", []),
                }
                for m in selected_markets
            ],
        }

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "nexus-web"}

    @app.get("/api/providers")
    async def providers() -> dict[str, Any]:
        """Expose the indicator providers used by the live analysis engine."""
        return state.provider_catalog

    @app.get("/api/backtest/status")
    async def backtest_status() -> dict[str, Any]:
        return state.backtest_status

    @app.get("/api/research/status")
    async def research_status() -> dict[str, Any]:
        return {"training": state.research_status, "report": read_research_integrity()}

    async def run_robust_research(request: ResearchRequest | None = None) -> None:
        """Run the full research trainer from the dashboard; never submits orders."""
        req = request or ResearchRequest()
        project_root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-u", str(project_root / "scripts" / "run_robust_research.py")]
        environment = os.environ.copy()
        environment.update({
            "NEXUS_FORCE_RETRAIN": "1",
            "NEXUS_HISTORY_TIMEOUT": "30",
            "NEXUS_MIN_PAYOUT": str(req.min_payout),
            "NEXUS_SEARCH_DEPTH": req.search_depth,
            "NEXUS_INDICATOR_SCOPE": req.indicator_scope,
            "NEXUS_WALK_FORWARD_FOLDS": str(req.folds),
        })
        if req.market_scope == "selected" and req.assets:
            environment["NEXUS_RESEARCH_ASSETS"] = ",".join(req.assets)
        state.research_status = {
            "status": "starting",
            "orders_placed": 0,
            "progress": 0.0,
            "strategy_discovery": {
                "active_trial": None,
                "trial_history": [],
                "champion": None,
                "feature_importance": [],
                "market_results": [],
            },
        }
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=project_root,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            state._research_process = process
            assert process.stdout is not None
            discovery = state.research_status.setdefault("strategy_discovery", {
                "active_trial": None,
                "trial_history": [],
                "champion": None,
                "feature_importance": [],
                "market_results": [],
            })
            async for raw_line in process.stdout:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue
                # Parse structured telemetry lines
                if line.startswith("[STRATEGY_TRIAL] "):
                    try:
                        payload = json.loads(line[17:])
                        discovery["active_trial"] = payload
                        history = discovery.setdefault("trial_history", [])
                        history.append(payload)
                        if len(history) > 30:
                            discovery["trial_history"] = history[-30:]
                        # Track best champion across all validated trials
                        if payload.get("stage") == "validated":
                            win_rate = float(payload.get("win_rate", 0))
                            current_champion = discovery.get("champion")
                            if not current_champion or win_rate > float(current_champion.get("win_rate", 0)):
                                discovery["champion"] = payload
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                    continue
                if line.startswith("[FEATURE_RANK] "):
                    try:
                        payload = json.loads(line[15:])
                        discovery["feature_importance"] = payload.get("top_features", [])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                    continue
                if line.startswith("[MARKET_SUMMARY] "):
                    try:
                        payload = json.loads(line[17:])
                        results_list = discovery.setdefault("market_results", [])
                        results_list.append(payload)
                        if len(results_list) > 50:
                            discovery["market_results"] = results_list[-50:]
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                    continue
                # Legacy progress line parsing
                match = re.search(r"\[(\d+)/(\d+)\]\s+([^:]+):\s*(.*)", line)
                if match:
                    index, total, asset, detail = match.groups()
                    state.research_status.update({
                        "status": "running",
                        "market_index": int(index),
                        "markets_total": int(total),
                        "progress": round(int(index) / max(1, int(total)), 4),
                        "asset": asset.strip(),
                        "detail": detail[:240],
                        "orders_placed": 0,
                    })
                else:
                    state.research_status.update({
                        "status": "running",
                        "detail": line[-240:],
                        "orders_placed": 0,
                    })
            return_code = await process.wait()
            if return_code != 0:
                state.research_status = {
                    "status": "error",
                    "return_code": return_code,
                    "orders_placed": 0,
                }
                return
            enrich = await asyncio.create_subprocess_exec(
                sys.executable,
                str(project_root / "scripts" / "enrich_research_report.py"),
                cwd=project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await enrich.communicate()
            state.research_status = {
                "status": "complete",
                "progress": 1.0,
                "orders_placed": 0,
                "report": read_research_integrity(),
            }
        except asyncio.CancelledError:
            state.research_status = {"status": "cancelled", "orders_placed": 0}
            raise
        except Exception as exc:
            state.research_status = {
                "status": "error",
                "detail": str(exc)[:240],
                "orders_placed": 0,
            }
        finally:
            if state._research_process and state._research_process.returncode is None:
                state._research_process.terminate()
                with contextlib.suppress(ProcessLookupError):
                    await state._research_process.wait()
            state._research_process = None

    @app.post("/api/research/retrain")
    async def retrain_research(req: ResearchRequest = ResearchRequest()) -> dict[str, Any]:
        if state._research_task and not state._research_task.done():
            raise HTTPException(
                status_code=409, detail="Robust research training is already running"
            )
        if state._backtest_task and not state._backtest_task.done():
            raise HTTPException(
                status_code=409, detail="Stop the local backtest before robust retraining"
            )
        state._research_task = asyncio.create_task(run_robust_research(req))
        return {
            "status": "started",
            "orders_placed": 0,
            "detail": f"Robust research retraining started ({req.search_depth} search, {req.indicator_scope} scope, {req.folds} folds)",
        }

    @app.post("/api/research/stop")
    async def stop_research() -> dict[str, Any]:
        if state._research_task and not state._research_task.done():
            if state._research_process and state._research_process.returncode is None:
                state._research_process.terminate()
            state._research_task.cancel()
            return {"status": "stopping", "orders_placed": 0}
        return {"status": state.research_status.get("status", "idle"), "orders_placed": 0}

    @app.post("/api/login/refresh")
    async def refresh_login() -> dict[str, Any]:
        if state._login_task and not state._login_task.done():
            raise HTTPException(status_code=409, detail="Browser login refresh is already running")
        if state._backtest_task and not state._backtest_task.done():
            raise HTTPException(
                status_code=409, detail="Stop the backtest before refreshing broker login"
            )
        state._login_task = asyncio.create_task(run_browser_login())
        return {"status": "started", "detail": "Browser login launched"}

    @app.post("/api/shutdown")
    async def shutdown() -> dict[str, str]:
        """Request a graceful shutdown of the process hosting the dashboard."""
        pid = int(os.getenv("NEXUS_SERVER_PID", str(os.getpid())))

        async def stop_process() -> None:
            await asyncio.sleep(0.15)
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGINT)

        asyncio.create_task(stop_process())
        return {"status": "shutting_down", "detail": "NEXUS server shutdown requested"}

    async def run_single_local_backtest(
        request: BacktestRequest, asset: str, market_progress: list[dict[str, Any]]
    ) -> None:
        """Train and replay real broker candles locally; never place orders."""
        from nexus.ai.engine_ai import RealAITradingEngine

        def update_market_progress(**updates: Any) -> None:
            for item in market_progress:
                if item.get("asset") == asset:
                    item.update(updates)
                    break

        state.backtest_status = {"status": "fetching", "asset": asset, "orders_placed": 0}
        try:
            broker = getattr(state.engine, "_broker", None)
            if not broker or not getattr(broker, "authenticated", False):
                raise RuntimeError("Quotex broker is not authenticated")
            window_candles = int(
                (request.learning_hours + request.evaluation_hours)
                * 3600
                / int(request.candle_timeframe)
            )
            # Quotex can return a few fewer rows than requested around gaps
            # or the current candle. Fetch a small margin, then trim after
            # validation so exact 4h + 1h windows do not fail at 298/300.
            fetch_candles = min(max(window_candles + 20, 150), 1000)
            # Keep the requested dataset exactly 10h + 2h. Expiry outcomes
            # must also remain inside that 12h window, so the final candles
            # are intentionally not scored if their expiry falls later.
            async with state.broker_io_lock:
                raw = await asyncio.wait_for(
                    broker.get_candles_async(asset, int(request.candle_timeframe), fetch_candles),
                    timeout=20,
                )
            if not raw:
                raise RuntimeError("No live broker candles available")
            frame = pd.DataFrame(raw)
            required = ["open", "high", "low", "close", "volume"]
            frame = frame[required + (["time"] if "time" in frame.columns else [])].copy()
            frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=required).reset_index(drop=True)
            # Broker APIs are not consistent about newest-first versus
            # oldest-first responses. The replay must always be chronological.
            if "time" in frame.columns:
                frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
                frame = frame.sort_values("time").reset_index(drop=True)
            learning_rows = int(request.learning_hours * 3600 / int(request.candle_timeframe))
            evaluation_rows = int(request.evaluation_hours * 3600 / int(request.candle_timeframe))
            learn_end = learning_rows
            eval_end = learning_rows + evaluation_rows
            if learn_end < 30 or len(frame) < eval_end:
                raise RuntimeError(
                    f"Insufficient candles for this interval: need {learning_rows + evaluation_rows} total "
                    f"({learning_rows} learning + {evaluation_rows} evaluation), received {len(frame)}. "
                    "Choose a shorter candle interval or a longer learning window."
                )
            frame = frame.iloc[:eval_end].copy()
            ai = RealAITradingEngine()

            async def run_ai(coro_factory: Any) -> Any:
                """Run CPU-heavy async model work away from the web event loop."""
                return await asyncio.to_thread(lambda: asyncio.run(coro_factory()))

            state.backtest_status = {
                "status": "learning",
                "asset": asset,
                "progress": 0,
                "orders_placed": 0,
            }

            def training_progress(stage: str, progress: float, detail: str) -> None:
                update_market_progress(
                    status="training", stage=stage, progress=round(progress, 4), detail=detail
                )
                state.backtest_status = {
                    "status": "learning",
                    "asset": asset,
                    "stage": stage,
                    "progress": round(progress, 4),
                    "detail": detail,
                    "learning_candles": learn_end,
                    "evaluation_candles": evaluation_rows,
                    "orders_placed": 0,
                }

            # Feature generation/model fitting can be expensive. Keep it off
            # the event loop so WebSocket snapshots, charts, and controls
            # remain responsive while this market is training.
            training = await asyncio.to_thread(
                ai.train_market,
                asset,
                frame.iloc[:learn_end].copy(),
                training_progress,
            )
            payout = 80.0
            async with state.broker_io_lock:
                payout_assets = await asyncio.wait_for(
                    broker.get_assets_with_payouts_async(), timeout=10
                )
            for item in payout_assets:
                if str(item.get("symbol", "")).upper() == asset.upper():
                    payout = float(item.get("payout") or payout)
                    break
            async with state.broker_io_lock:
                balance = await asyncio.wait_for(state.engine.get_account_balance(), timeout=5)
            results = []
            expirations: list[int] = []
            amounts: list[float] = []
            learning_updates = 0
            candidate_signals = 0
            filtered_low_confidence = 0
            max_observed_confidence = 0.0
            confidence_samples: list[tuple[float, bool]] = []
            for index, row_index in enumerate(range(learn_end, eval_end)):
                analysis = await run_ai(
                    lambda row_index=row_index: ai.analyze_market(
                        frame.iloc[: row_index + 1].tail(180),
                        asset,
                        int(request.candle_timeframe),
                        "otc" in asset.lower(),
                    )
                )
                expiration = int(
                    analysis.get("recommended_expiration", request.trade_expiration)
                    if request.ai_select_timeframe
                    else request.trade_expiration
                )
                expiration = max(int(request.candle_timeframe), expiration)
                steps = max(1, round(expiration / int(request.candle_timeframe)))
                exit_index = row_index + steps
                if exit_index >= eval_end:
                    break
                signal = str(analysis.get("signal", "hold")).lower()
                confidence = float(analysis.get("confidence", 0.0))
                if confidence > max_observed_confidence:
                    max_observed_confidence = confidence
                if signal in {"call", "put"}:
                    candidate_signals += 1

                # Enforce multi-model consensus & volatile regime protection (matching live risk gate)
                breakdown = analysis.get("breakdown", {})
                votes = []
                for key in ("transformer", "lstm", "rl_agent"):
                    val = breakdown.get(key, {})
                    d = str(val.get("signal") or val.get("action") or "hold").lower()
                    if d in {"call", "put"}:
                        votes.append(d)
                regime = str(analysis.get("regime", "")).upper()

                if signal in {"call", "put"}:
                    if (
                        confidence < request.min_confidence
                        or votes.count(signal) < 2
                        or regime == "VOLATILE"
                    ):
                        filtered_low_confidence += 1
                        signal = "hold"
                amount = float(request.stake or state.engine.settings.trading.base_trade_amount)
                if request.stake_mode == "ai" or request.stake is None:
                    raw_amount = ai.position_sizer.calculate_trade_amount(
                        balance,
                        payout / 100.0,
                        confidence,
                        float(analysis.get("features", {}).get("atr", 0.0) or 0.0),
                    )
                    # Protect against catastrophic single-test drawdowns on large demo balances
                    amount = min(50.0, max(1.0, round(raw_amount, 2)))
                row = frame.iloc[row_index]
                entry = float(row["close"])
                exit_price = float(frame.iloc[exit_index]["close"])
                decided = signal in {"call", "put"}
                win = decided and (
                    (signal == "call" and exit_price > entry)
                    or (signal == "put" and exit_price < entry)
                )
                trade_profit = amount * payout / 100 if win else (-amount if decided else 0.0)
                results.append(
                    {
                        "signal": signal,
                        "amount": amount,
                        "expiration": expiration,
                        "win": bool(win),
                        "profit": trade_profit,
                    }
                )
                if decided:
                    confidence_samples.append((confidence, bool(win)))
                    await run_ai(
                        lambda signal=signal, win=win, trade_profit=trade_profit, analysis=analysis: (
                            ai.learn_and_evolve(
                                asset, signal, bool(win), float(trade_profit), analysis, False
                            )
                        )
                    )
                    learning_updates += 1
                    expirations.append(expiration)
                    amounts.append(amount)
                state.backtest_status = {
                    "status": "evaluating",
                    "asset": asset,
                    "progress": round((index + 1) / max(1, eval_end - learn_end), 4),
                    "decisions": index + 1,
                    "orders_placed": 0,
                }
                update_market_progress(
                    status="evaluating",
                    progress=round((index + 1) / max(1, eval_end - learn_end), 4),
                    decisions=index + 1,
                )
            trades = [item for item in results if item.get("signal") in {"call", "put"}]
            wins = sum(1 for item in trades if item.get("win"))
            profit = 0.0
            for item in results:
                raw_profit = item.get("profit", 0.0)
                if isinstance(raw_profit, (int, float, str)):
                    with contextlib.suppress(ValueError, TypeError):
                        profit += float(raw_profit)
            confidence_buckets = []
            for lower, upper in (
                (0.50, 0.60),
                (0.60, 0.70),
                (0.70, 0.80),
                (0.80, 0.90),
                (0.90, 1.01),
            ):
                bucket = [
                    (score, won) for score, won in confidence_samples if lower <= score < upper
                ]
                bucket_wins = sum(won for _, won in bucket)
                confidence_buckets.append(
                    {
                        "range": f"{int(lower * 100)}-{min(100, int(upper * 100))}%",
                        "trades": len(bucket),
                        "wins": bucket_wins,
                        "win_rate": round(bucket_wins / len(bucket), 4) if bucket else None,
                    }
                )
            # Feed only the held-out evaluation outcomes back into this
            # market's confidence calibration. The learning window is never
            # used as evidence for this mapping.
            ai.get_dynamic_asset_params(asset)["ensemble_confidence_bands"] = [
                {
                    "lower": lower,
                    "upper": upper,
                    "trades": sum(1 for score, _ in confidence_samples if lower <= score < upper),
                    "wins": sum(
                        int(won) for score, won in confidence_samples if lower <= score < upper
                    ),
                }
                for lower, upper in (
                    (0.50, 0.60),
                    (0.60, 0.70),
                    (0.70, 0.80),
                    (0.80, 0.90),
                    (0.90, 1.01),
                )
            ]
            ai._save_asset_stats()
            state.backtest_status = {
                "status": "complete",
                "asset": asset,
                "learning_hours": request.learning_hours,
                "evaluation_hours": request.evaluation_hours,
                "candles": len(frame),
                "decisions": len(results),
                "candidate_signals": candidate_signals,
                "filtered_low_confidence": filtered_low_confidence,
                "min_confidence": request.min_confidence,
                "trades_scored": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "win_rate": round(wins / len(trades), 4) if trades else 0.0,
                "net_profit": round(profit, 4),
                "payout_percent": payout,
                "break_even_win_rate": round(1 / (1 + payout / 100), 4) if payout > 0 else None,
                "timeframes_used": sorted(set(expirations)),
                "amount_min": round(min(amounts), 2) if amounts else 0.0,
                "amount_max": round(max(amounts), 2) if amounts else 0.0,
                "learning_updates": learning_updates,
                "confidence_buckets": confidence_buckets,
                "confidence_note": "Market-specific confidence calibrated from held-out replay outcomes",
                "training": {
                    k: training.get(k)
                    for k in (
                        "market_model_status",
                        "validation_accuracy",
                        "validation_baseline",
                        "indicator_count",
                        "research",
                    )
                },
                "orders_placed": 0,
                "data_source": "Quotex live candles; local replay only",
                "phase": "10h training + 2h local evaluation/adaptation",
            }
            update_market_progress(
                status="complete",
                progress=1.0,
                decisions=len(results),
                trades_scored=len(trades),
                candidate_signals=candidate_signals,
                max_confidence=round(max_observed_confidence, 4),
                wins=wins,
                losses=len(trades) - wins,
                net_profit=round(profit, 4),
            )
        except asyncio.CancelledError:
            update_market_progress(status="cancelled")
            state.backtest_status = {"status": "cancelled", "orders_placed": 0}
        except Exception as exc:
            update_market_progress(status="error", detail=str(exc))
            state.backtest_status = {"status": "error", "error": str(exc), "orders_placed": 0}

    async def run_local_backtest(request: BacktestRequest, assets: list[str]) -> None:
        """Run isolated 10h/2h replays for each selected market."""
        summaries: list[dict[str, Any]] = []
        try:
            state.backtest_started_at = time.time()
            state.backtest_markets = [
                {"asset": asset, "status": "queued", "progress": 0.0, "orders_placed": 0}
                for asset in assets
            ]
            concurrency = max(1, min(int(os.getenv("NEXUS_BACKTEST_CONCURRENCY", "2")), 4))
            semaphore = asyncio.Semaphore(concurrency)

            async def run_market(index: int, asset: str) -> None:
                async with semaphore:
                    state.backtest_status = {
                        "status": "queued",
                        "asset": asset,
                        "market_index": index + 1,
                        "markets_total": len(assets),
                        "parallel_markets": concurrency,
                        "orders_placed": 0,
                    }
                    await run_single_local_backtest(request, asset, state.backtest_markets)

            await asyncio.gather(*(run_market(index, asset) for index, asset in enumerate(assets)))
            summaries = [
                dict(item)
                for item in state.backtest_markets
                if item.get("status") in {"complete", "error", "cancelled"}
            ]
            if len(assets) > 1:
                scored = sum(int(item.get("trades_scored", 0)) for item in summaries)
                wins = sum(int(item.get("wins", 0)) for item in summaries)
                net_profit = sum(float(item.get("net_profit", 0.0)) for item in summaries)
                failed = sum(1 for item in summaries if item.get("status") == "error")
                total_decisions = sum(int(item.get("decisions", 0)) for item in summaries)
                total_candidates = sum(int(item.get("candidate_signals", 0)) for item in summaries)
                max_conf = max(
                    (float(item.get("max_confidence", 0.0)) for item in summaries), default=0.0
                )
                state.backtest_status = {
                    "status": "complete_with_errors" if failed else "complete",
                    "markets": summaries,
                    "markets_total": len(assets),
                    "markets_failed": failed,
                    "parallel_markets": concurrency,
                    "learning_hours": request.learning_hours,
                    "evaluation_hours": request.evaluation_hours,
                    "decisions": total_decisions,
                    "candidate_signals": total_candidates,
                    "max_confidence": round(max_conf, 4),
                    "min_confidence": request.min_confidence,
                    "trades_scored": scored,
                    "wins": wins,
                    "losses": scored - wins,
                    "win_rate": round(wins / scored, 4) if scored else 0.0,
                    "net_profit": round(net_profit, 4),
                    "orders_placed": 0,
                    "phase": f"isolated {request.learning_hours}h training + {request.evaluation_hours}h local evaluation/adaptation per market",
                }
        except asyncio.CancelledError:
            state.backtest_status = {"status": "cancelled", "orders_placed": 0}
            state.backtest_started_at = None

    @app.post("/api/backtest/start")
    async def start_backtest(request: BacktestRequest) -> dict[str, Any]:
        if state._backtest_task and not state._backtest_task.done():
            raise HTTPException(status_code=409, detail="A backtest is already running")
        if not getattr(state.engine, "demo_mode", True):
            raise HTTPException(status_code=403, detail="Backtesting requires PRACTICE mode")
        min_payout = float(request.min_payout or 0.0)
        logger.info(
            f"Backtest requested: scope={request.market_scope}, min_payout={min_payout}%, "
            f"raw_assets_count={len(request.assets)}"
        )
        if request.market_scope == "all":
            broker = getattr(state.engine, "_broker", None)
            if broker:
                async with state.broker_io_lock:
                    available = await asyncio.wait_for(
                        broker.get_assets_with_payouts_async(), timeout=10
                    )
            else:
                available = []
            assets = [
                str(item.get("symbol", ""))
                for item in available
                if item.get("symbol") and float(item.get("payout") or 0) >= min_payout
            ]
            logger.info(
                f"Scope 'all': found {len(assets)}/{len(available)} markets meeting >={min_payout}% payout: {assets}"
            )
        elif request.market_scope == "multiple":
            assets = list(dict.fromkeys(request.assets))
        else:
            assets = [request.asset] if request.asset else []
        if min_payout > 0 and assets and request.market_scope != "all":
            broker = getattr(state.engine, "_broker", None)
            if broker:
                async with state.broker_io_lock:
                    available = await asyncio.wait_for(
                        broker.get_assets_with_payouts_async(), timeout=10
                    )
                payouts = {
                    str(item.get("symbol", "")).upper(): float(item.get("payout") or 0)
                    for item in available
                }
                assets = [asset for asset in assets if payouts.get(asset.upper(), 0) >= min_payout]
                logger.info(
                    f"Scope '{request.market_scope}': filtered to {len(assets)} assets meeting >={min_payout}% payout: {assets}"
                )
        if not assets:
            detail = (
                f"No available markets meet the {request.min_payout:.1f}% minimum payout"
                if request.min_payout > 0
                else "Select at least one market"
            )
            raise HTTPException(status_code=400, detail=detail)
        request.assets = assets
        request.asset = assets[0]
        state._backtest_task = asyncio.create_task(run_local_backtest(request, assets))
        return {"status": "started", "orders_placed": 0}

    @app.post("/api/backtest/stop")
    async def stop_backtest() -> dict[str, Any]:
        if state._backtest_task and not state._backtest_task.done():
            state._backtest_task.cancel()
            return {"status": "stopping", "orders_placed": 0}
        if state.backtest_status.get("status") in {"error", "cancelled"}:
            state.backtest_status = {"status": "idle", "orders_placed": 0}
            state.backtest_markets = []
            state.backtest_started_at = None
        return {"status": "idle", "orders_placed": 0}

    @app.get("/api/markets")
    async def markets() -> list[dict[str, Any]]:
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            return await asyncio.wait_for(broker.get_assets_with_payouts_async(), timeout=10)

    @app.get("/api/markets/{asset}/candles")
    async def candles(asset: str, timeframe: int = 60, limit: int = 120) -> list[dict[str, float]]:
        if not _is_valid_market_asset(asset):
            raise HTTPException(status_code=400, detail="Invalid market identifier")
        if timeframe not in {5, 15, 30, 60, 300, 900, 1800, 3600}:
            raise HTTPException(status_code=400, detail="Unsupported timeframe")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            try:
                data = await asyncio.wait_for(
                    broker.get_candles_async(asset, timeframe, max(20, min(limit, 300))),
                    timeout=20,
                )
            except TimeoutError as exc:
                raise HTTPException(
                    status_code=504,
                    detail=f"Timed out waiting for live candles for {asset}",
                ) from exc
        if not data:
            raise HTTPException(status_code=404, detail="No live candles available for this market")
        return list(data)

    @app.get("/api/markets/{asset}/analysis")
    async def analysis(asset: str) -> dict[str, Any]:
        if not _is_valid_market_asset(asset):
            raise HTTPException(status_code=400, detail="Invalid market identifier")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            result = await asyncio.wait_for(
                state.engine.get_ai_prediction(asset, is_otc="otc" in asset.lower()), timeout=20
            )
        return dict(_json_safe(result))

    @app.get("/api/markets/{asset}/eligibility")
    async def market_eligibility(asset: str) -> dict[str, Any]:
        """Explain whether a market is presently eligible for PRACTICE action.

        This is diagnostic only: it never submits an order and makes every
        gate visible to the workspace instead of leaving an apparent idle.
        """
        if not _is_valid_market_asset(asset):
            raise HTTPException(status_code=400, detail="Invalid market identifier")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            prediction = await asyncio.wait_for(
                state.engine.get_ai_prediction(asset, is_otc="otc" in asset.lower()), timeout=20
            )
            catalog = await asyncio.wait_for(broker.get_assets_with_payouts_async(), timeout=10)
        listing = next(
            (item for item in catalog or [] if isinstance(item, dict) and str(item.get("symbol")) == asset),
            {},
        )
        payout = float(listing.get("payout") or 0.0)
        controls = state.engine.settings.trading
        signal = str(prediction.get("signal", "hold")).lower()
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        reasons: list[str] = []
        if signal not in {"call", "put", "long", "short"}:
            reasons.append("AI has no directional signal")
        if confidence < controls.min_confidence:
            reasons.append(f"confidence {confidence:.1%} is below {controls.min_confidence:.0%}")
        if payout < controls.payout_threshold:
            reasons.append(f"payout {payout:.0f}% is below {controls.payout_threshold:.0f}%")
        if state.trading_paused:
            reasons.append("PRACTICE execution is paused")
        if state.engine.circuit_breaker_active:
            reasons.append("circuit breaker is active")
        if len(getattr(state.engine, "active_positions", [])) >= controls.max_open_trades:
            reasons.append("maximum open-trade limit reached")
        return {
            "asset": asset,
            "eligible": not reasons,
            "reasons": reasons,
            "signal": signal,
            "confidence": confidence,
            "payout": payout,
            "recommended_expiration": prediction.get("recommended_expiration"),
            "practice_only": True,
        }

    @app.get("/api/trades")
    async def trades() -> list[dict[str, Any]]:
        return state.history.get_trade_history(limit=200)

    @app.post("/api/workspace/practice-scan")
    async def workspace_practice_scan(request: WorkspacePracticeRequest) -> dict[str, Any]:
        """Submit only broker-confirmed PRACTICE candidates in the workspace."""
        if request.confirmation != "RUN PRACTICE WORKSPACE":
            raise HTTPException(status_code=400, detail="Explicit PRACTICE workspace confirmation is required")
        if not state.engine.demo_mode:
            raise HTTPException(status_code=403, detail="Workspace execution is PRACTICE-only")
        if state.trading_paused:
            raise HTTPException(status_code=423, detail="PRACTICE execution is paused")
        available = max(0, state.engine.settings.trading.max_open_trades - len(getattr(state.engine, "active_positions", [])))
        results: list[dict[str, Any]] = []
        for asset in dict.fromkeys(request.assets):
            if available <= 0:
                break
            eligibility = await market_eligibility(asset)
            if not eligibility["eligible"]:
                results.append({"asset": asset, "status": "skipped", "reasons": eligibility["reasons"]})
                continue
            direction = "call" if eligibility["signal"] in {"call", "long"} else "put"
            async with state.broker_io_lock:
                result = await asyncio.wait_for(
                    state.engine.execute_trade(asset, direction, state.engine.settings.trading.base_trade_amount, eligibility["recommended_expiration"] or 60),
                    timeout=30,
                )
            order_id = str(result.get("order_id") or "")
            if result.get("success") and order_id:
                available -= 1
                local_id = f"work_{int(time.time()*1000)}_{len(results)}"
                exp_sec = int(eligibility.get("recommended_expiration") or 60)
                amt = float(state.engine.settings.trading.base_trade_amount or 10.0)
                trade_rec = {
                    "local_id": local_id,
                    "broker_order_id": order_id,
                    "asset": asset,
                    "direction": direction,
                    "amount": amt,
                    "status": "PLACED",
                    "expiration": exp_sec,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                state.history.record_placement(trade_rec)
                pos_entry = {
                    "id": local_id,
                    "local_id": local_id,
                    "broker_order_id": order_id,
                    "asset": asset,
                    "direction": direction,
                    "amount": amt,
                    "expiration": exp_sec,
                    "placed_at": time.time(),
                    "expire_at": time.time() + exp_sec,
                }
                if not hasattr(state.engine, "active_positions"):
                    state.engine.active_positions = []
                state.engine.active_positions.append(pos_entry)
                results.append({"asset": asset, "status": "submitted", "broker_order_id": order_id, "direction": direction})
            else:
                results.append({"asset": asset, "status": "not_confirmed", "reason": result.get("error", "Broker did not return an order ID")})
        return {"mode": "PRACTICE", "results": results, "confirmed_orders": sum(item["status"] == "submitted" for item in results)}

    @app.post("/api/account/mode")
    async def account_mode(request: ModeRequest) -> dict[str, Any]:
        if request.mode == "REAL":
            if os.getenv("NEXUS_ALLOW_REAL_WEB_TRADING", "0").lower() not in {"1", "true", "yes"}:
                raise HTTPException(
                    status_code=403,
                    detail="REAL web trading is disabled. Set NEXUS_ALLOW_REAL_WEB_TRADING=true first.",
                )
            if request.confirmation != "I UNDERSTAND REAL TRADING":
                raise HTTPException(
                    status_code=400, detail="Explicit REAL trading confirmation is required"
                )
        elif request.confirmation not in {"", "SWITCH TO PRACTICE"}:
            raise HTTPException(status_code=400, detail="Invalid PRACTICE confirmation")

        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        practice = request.mode == "PRACTICE"
        async with state.broker_io_lock:
            await asyncio.wait_for(broker.set_practice_mode(practice), timeout=10)
        # Avoid the synchronous property setter from inside the web event
        # loop; the broker mode is already changed asynchronously above.
        state.engine._demo_mode = practice
        broker.demo_mode = practice
        state.trading_paused = True
        return {"mode": request.mode, "trading_paused": True}

    @app.post("/api/control")
    async def controls(request: ControlRequest) -> dict[str, Any]:
        if request.pause is not None:
            state.trading_paused = request.pause
        if request.reset_circuit_breaker:
            state.engine.circuit_breaker_active = False
        trading = state.engine.settings.trading
        for name in (
            "auto_trade_enabled",
            "ai_select_timeframe",
            "min_confidence",
            "payout_threshold",
            "max_open_trades",
            "base_trade_amount",
            "auto_trade_interval_seconds",
        ):
            value = getattr(request, name)
            if value is not None:
                setattr(trading, name, value)
        if request.auto_trade_assets is not None:
            state.auto_trade_assets = [
                a for a in request.auto_trade_assets if _is_valid_market_asset(a)
            ]
        snapshot = await state.snapshot()
        return dict(snapshot.get("controls", {}))

    @app.post("/api/history/clear")
    async def clear_history(request: ClearHistoryRequest) -> dict[str, Any]:
        if request.confirmation != "CLEAR HISTORY":
            raise HTTPException(status_code=400, detail="Type CLEAR HISTORY to confirm")
        deleted = state.history.clear_history(request.include_unresolved)
        return {"deleted": deleted, "unresolved_retained": not request.include_unresolved}

    @app.post("/api/trades", status_code=202)
    async def place_trade(request: TradeRequest) -> dict[str, Any]:
        # The browser can request a practice order, but cannot enable REAL
        # mode or bypass the engine's broker/payout/risk validation.
        if state.trading_paused:
            raise HTTPException(status_code=423, detail="Trading is paused")
        if state.engine.circuit_breaker_active:
            raise HTTPException(status_code=423, detail="Circuit breaker is active")
        if not state.engine.demo_mode:
            # REAL mode is intentionally not reachable from the web API yet;
            # mode switching remains separately gated above.
            raise HTTPException(status_code=403, detail="Web trading is practice-only")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            result = await asyncio.wait_for(
                state.engine.execute_trade(
                    request.asset, request.direction, request.amount, request.expiration
                ),
                timeout=30,
            )
        if not result.get("success"):
            raise HTTPException(
                status_code=502, detail=result.get("error", "Broker rejected order")
            )
        order_id = str(result.get("order_id") or "")
        if order_id:
            local_id = f"manual_{int(time.time()*1000)}"
            exp_sec = int(request.expiration or 60)
            trade_rec = {
                "local_id": local_id,
                "broker_order_id": order_id,
                "asset": request.asset,
                "direction": request.direction,
                "amount": float(request.amount),
                "status": "PLACED",
                "expiration": exp_sec,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            state.history.record_placement(trade_rec)
            pos_entry = {
                "id": local_id,
                "local_id": local_id,
                "broker_order_id": order_id,
                "asset": request.asset,
                "direction": request.direction,
                "amount": float(request.amount),
                "expiration": exp_sec,
                "placed_at": time.time(),
                "expire_at": time.time() + exp_sec,
            }
            if not hasattr(state.engine, "active_positions"):
                state.engine.active_positions = []
            state.engine.active_positions.append(pos_entry)
        return dict(result)

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        if settings.api_token_required and not request_has_valid_service_token(websocket.headers):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        state.clients.add(websocket)
        try:
            await websocket.send_json(await state.snapshot())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(websocket)

    return app


if __name__ == "__main__":
    import os

    import uvicorn

    from nexus.utils.config import load_runtime_settings

    settings = load_runtime_settings()
    web_app = create_app(settings)
    host = os.getenv("NEXUS_HOST", "0.0.0.0")
    port = int(os.getenv("NEXUS_PORT", "8000"))
    print(f"Launching NEXUS web dashboard on http://{host}:{port}")
    print(f"Accessible in browser via: http://localhost:{port} or http://127.0.0.1:{port}")
    uvicorn.run(web_app, host=host, port=port)

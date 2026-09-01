"""FastAPI application for the NEXUS browser dashboard.

The web layer is deliberately thin: the NexusEngine remains the single
owner of broker state, AI state, and the durable trade ledger.  The browser
only receives broker-confirmed data and never calculates trade outcomes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus.core.engine import NexusEngine
from nexus.data.trade_history import TradeHistory
from nexus.features import get_feature_provider_catalog
from nexus.utils.config import NexusSettings

WEB_ROOT = Path(__file__).with_name("static")


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


class ClearHistoryRequest(BaseModel):
    confirmation: str
    include_unresolved: bool = False


class WebState:
    def __init__(
        self, settings: NexusSettings, demo_mode: bool = True, engine: NexusEngine | None = None
    ) -> None:
        self.engine = engine or NexusEngine(settings, demo_mode=demo_mode, auto_login=False)
        self.history = TradeHistory()
        self.clients: set[WebSocket] = set()
        self.trading_paused = True
        self._broadcast_task: asyncio.Task[None] | None = None
        self.provider_catalog = get_feature_provider_catalog()

    async def start(self) -> None:
        await self.engine.initialize_components()
        # Web startup must never silently fall back to simulation.  A failed
        # broker login is reported by /api/status and disables trading.
        await self.engine.login_broker()
        self._broadcast_task = asyncio.create_task(self.broadcast_loop())

    async def stop(self) -> None:
        if self._broadcast_task:
            self._broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast_task

    async def snapshot(self) -> dict[str, Any]:
        broker = getattr(self.engine, "_broker", None)
        authenticated = bool(broker and getattr(broker, "authenticated", False))
        balance = await self.engine.get_account_balance() if authenticated else None
        stats = self.engine.get_performance_stats()
        return {
            "mode": "PRACTICE" if self.engine.demo_mode else "REAL",
            "authenticated": authenticated,
            "balance": balance,
            "active_trades": len(getattr(self.engine, "active_positions", [])),
            "performance": stats,
            "feature_providers": self.provider_catalog,
            "controls": {
                "trading_paused": self.trading_paused,
                "auto_trade_enabled": bool(self.engine.settings.trading.auto_trade_enabled),
                "ai_select_timeframe": bool(self.engine.settings.trading.ai_select_timeframe),
                "min_confidence": self.engine.settings.trading.min_confidence,
                "payout_threshold": self.engine.settings.trading.payout_threshold,
                "max_open_trades": self.engine.settings.trading.max_open_trades,
                "base_trade_amount": self.engine.settings.trading.base_trade_amount,
                "circuit_breaker": self.engine.circuit_breaker_active,
            },
            "trades": self.history.get_trade_history(limit=50),
        }

    async def broadcast_loop(self) -> None:
        while True:
            await asyncio.sleep(3)
            if not self.clients:
                continue
            payload = json.dumps(await self.snapshot(), default=str)
            stale: list[WebSocket] = []
            for client in self.clients:
                try:
                    await client.send_text(payload)
                except Exception:
                    stale.append(client)
            self.clients.difference_update(stale)


def create_app(
    settings: NexusSettings, demo_mode: bool = True, engine: NexusEngine | None = None
) -> FastAPI:
    state = WebState(settings, demo_mode=demo_mode, engine=engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await state.start()
        try:
            yield
        finally:
            await state.stop()

    app = FastAPI(title="NEXUS Trading Terminal", version=settings.version, lifespan=lifespan)
    app.state.nexus = state
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await state.snapshot()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "nexus-web"}

    @app.get("/api/providers")
    async def providers() -> dict[str, Any]:
        """Expose the indicator providers used by the live analysis engine."""
        return state.provider_catalog

    @app.get("/api/markets")
    async def markets() -> list[dict[str, Any]]:
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        return await broker.get_assets_with_payouts_async()

    @app.get("/api/markets/{asset}/candles")
    async def candles(asset: str, timeframe: int = 60, limit: int = 120) -> list[dict[str, float]]:
        if timeframe not in {5, 15, 30, 60, 300, 900, 1800, 3600}:
            raise HTTPException(status_code=400, detail="Unsupported timeframe")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        data = await broker.get_candles_async(asset, timeframe, max(20, min(limit, 300)))
        if not data:
            raise HTTPException(status_code=404, detail="No live candles available for this market")
        return data

    @app.get("/api/markets/{asset}/analysis")
    async def analysis(asset: str) -> dict[str, Any]:
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        result = await state.engine.get_ai_prediction(asset, is_otc="otc" in asset.lower())
        return _json_safe(result)

    @app.get("/api/trades")
    async def trades() -> list[dict[str, Any]]:
        return state.history.get_trade_history(limit=200)

    @app.post("/api/account/mode")
    async def account_mode(request: ModeRequest) -> dict[str, Any]:
        if request.mode == "REAL":
            if os.getenv("NEXUS_ALLOW_REAL_WEB_TRADING", "0").lower() not in {"1", "true", "yes"}:
                raise HTTPException(
                    status_code=403,
                    detail="REAL web trading is disabled. Set NEXUS_ALLOW_REAL_WEB_TRADING=true first.",
                )
            if request.confirmation != "I UNDERSTAND REAL TRADING":
                raise HTTPException(status_code=400, detail="Explicit REAL trading confirmation is required")
        elif request.confirmation not in {"", "SWITCH TO PRACTICE"}:
            raise HTTPException(status_code=400, detail="Invalid PRACTICE confirmation")

        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        practice = request.mode == "PRACTICE"
        await broker.set_practice_mode(practice)
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
        ):
            value = getattr(request, name)
            if value is not None:
                setattr(trading, name, value)
        return (await state.snapshot())["controls"]

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
        result = await state.engine.execute_trade(
            request.asset, request.direction, request.amount, request.expiration
        )
        if not result.get("success"):
            raise HTTPException(status_code=502, detail=result.get("error", "Broker rejected order"))
        return dict(result)

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
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

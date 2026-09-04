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
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nexus.core.engine import NexusEngine
from nexus.data.trade_history import TradeHistory
from nexus.features import get_feature_provider_catalog
from nexus.utils.config import NexusSettings
from nexus.utils.device import get_device_info

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


class BacktestRequest(BaseModel):
    asset: str | None = Field(default=None, min_length=1, max_length=40)
    assets: list[str] = Field(default_factory=list, max_length=100)
    market_scope: Literal["single", "multiple", "all"] = "single"
    learning_hours: int = Field(default=10, ge=2, le=12)
    evaluation_hours: int = Field(default=2, ge=1, le=4)
    candle_timeframe: Literal[60, 300, 900] = 60
    ai_select_timeframe: bool = True
    trade_expiration: Literal[60, 300, 900] = 60
    stake_mode: Literal["ai", "fixed"] = "ai"
    stake: float | None = Field(default=None, gt=0, le=1000)
    min_confidence: float = Field(default=0.70, ge=0.5, le=0.99)
    min_payout: float = Field(default=0.0, ge=0, le=100)


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
        self.backtest_status: dict[str, Any] = {"status": "idle"}
        self.backtest_markets: list[dict[str, Any]] = []
        self._backtest_task: asyncio.Task[None] | None = None
        self.backtest_started_at: float | None = None
        self.login_refresh_runner: Callable[[], Awaitable[None]] | None = None
        # pyquotex uses one stateful WebSocket client. Serialize broker I/O,
        # while allowing CPU-heavy AI work and browser requests to continue.
        self.broker_io_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.engine.initialize_components()
        # Web startup must never silently fall back to simulation.  A failed
        # broker login is reported by /api/status and disables trading.
        await self.engine.login_broker()
        self._broadcast_task = asyncio.create_task(self.broadcast_loop())
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
        if self._backtest_task and not self._backtest_task.done():
            self._backtest_task.cancel()

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
            "backtest_elapsed_seconds": round(time.time() - self.backtest_started_at, 1) if self.backtest_started_at else 0.0,
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
                "uv", "run", "python", str(script), cwd=project_root,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
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
                state.login_status = {"status": "error", "detail": output[-1] if output else "Browser login failed"}
                return
            from dotenv import load_dotenv

            load_dotenv(project_root / ".env", override=True)
            async with state.broker_io_lock:
                state.engine._broker = None
                connected = await state.engine.login_broker()
            state.login_status = {
                "status": "complete" if connected else "error",
                "detail": "Fresh Quotex session connected" if connected else "Session captured but broker reconnect failed",
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

    @app.get("/api/backtest/status")
    async def backtest_status() -> dict[str, Any]:
        return state.backtest_status

    @app.post("/api/login/refresh")
    async def refresh_login() -> dict[str, Any]:
        if state._login_task and not state._login_task.done():
            raise HTTPException(status_code=409, detail="Browser login refresh is already running")
        if state._backtest_task and not state._backtest_task.done():
            raise HTTPException(status_code=409, detail="Stop the backtest before refreshing broker login")
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

            state.backtest_status = {"status": "learning", "asset": asset, "progress": 0, "orders_placed": 0}
            def training_progress(stage: str, progress: float, detail: str) -> None:
                update_market_progress(status="training", stage=stage, progress=round(progress, 4), detail=detail)
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
            confidence_samples: list[tuple[float, bool]] = []
            for index, row_index in enumerate(range(learn_end, eval_end)):
                analysis = await run_ai(
                    lambda row_index=row_index: ai.analyze_market(
                        frame.iloc[:row_index + 1].tail(180),
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
                steps = max(1, round(expiration / int(request.candle_timeframe)))
                exit_index = row_index + steps
                if exit_index >= eval_end:
                    break
                signal = str(analysis.get("signal", "hold")).lower()
                confidence = float(analysis.get("confidence", 0.0))
                if signal in {"call", "put"}:
                    candidate_signals += 1
                if signal in {"call", "put"} and confidence < request.min_confidence:
                    filtered_low_confidence += 1
                    signal = "hold"
                amount = float(request.stake or state.engine.settings.trading.base_trade_amount)
                if request.stake_mode == "ai" or request.stake is None:
                    amount = ai.position_sizer.calculate_trade_amount(balance, payout / 100.0, confidence, float(analysis.get("features", {}).get("atr", 0.0) or 0.0))
                row = frame.iloc[row_index]
                entry = float(row["close"])
                exit_price = float(frame.iloc[exit_index]["close"])
                decided = signal in {"call", "put"}
                win = decided and ((signal == "call" and exit_price > entry) or (signal == "put" and exit_price < entry))
                trade_profit = amount * payout / 100 if win else (-amount if decided else 0.0)
                results.append({"signal": signal, "amount": amount, "expiration": expiration, "win": bool(win), "profit": trade_profit})
                if decided:
                    confidence_samples.append((confidence, bool(win)))
                    await run_ai(
                        lambda signal=signal, win=win, trade_profit=trade_profit, analysis=analysis: ai.learn_and_evolve(
                            asset, signal, bool(win), float(trade_profit), analysis, False
                        )
                    )
                    learning_updates += 1
                    expirations.append(expiration)
                    amounts.append(amount)
                state.backtest_status = {"status": "evaluating", "asset": asset, "progress": round((index + 1) / max(1, eval_end - learn_end), 4), "decisions": index + 1, "orders_placed": 0}
                update_market_progress(status="evaluating", progress=round((index + 1) / max(1, eval_end - learn_end), 4), decisions=index + 1)
            trades = [item for item in results if item["signal"] in {"call", "put"}]
            wins = sum(item["win"] for item in trades)
            profit = sum(item["profit"] for item in results)
            confidence_buckets = []
            for lower, upper in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)):
                bucket = [(score, won) for score, won in confidence_samples if lower <= score < upper]
                bucket_wins = sum(won for _, won in bucket)
                confidence_buckets.append({
                    "range": f"{int(lower * 100)}-{min(100, int(upper * 100))}%",
                    "trades": len(bucket),
                    "wins": bucket_wins,
                    "win_rate": round(bucket_wins / len(bucket), 4) if bucket else None,
                })
            # Feed only the held-out evaluation outcomes back into this
            # market's confidence calibration. The learning window is never
            # used as evidence for this mapping.
            ai.get_dynamic_asset_params(asset)["ensemble_confidence_bands"] = [
                {
                    "lower": lower,
                    "upper": upper,
                    "trades": sum(
                        1 for score, _ in confidence_samples if lower <= score < upper
                    ),
                    "wins": sum(
                        int(won)
                        for score, won in confidence_samples
                        if lower <= score < upper
                    ),
                }
                for lower, upper in ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01))
            ]
            ai._save_asset_stats()
            state.backtest_status = {"status": "complete", "asset": asset, "learning_hours": request.learning_hours, "evaluation_hours": request.evaluation_hours, "candles": len(frame), "decisions": len(results), "candidate_signals": candidate_signals, "filtered_low_confidence": filtered_low_confidence, "min_confidence": request.min_confidence, "trades_scored": len(trades), "wins": wins, "losses": len(trades) - wins, "win_rate": round(wins / len(trades), 4) if trades else 0.0, "net_profit": round(profit, 4), "payout_percent": payout, "break_even_win_rate": round(1 / (1 + payout / 100), 4) if payout > 0 else None, "timeframes_used": sorted(set(expirations)), "amount_min": round(min(amounts), 2) if amounts else 0.0, "amount_max": round(max(amounts), 2) if amounts else 0.0, "learning_updates": learning_updates, "confidence_buckets": confidence_buckets, "confidence_note": "Market-specific confidence calibrated from held-out replay outcomes", "training": {k: training.get(k) for k in ("market_model_status", "validation_accuracy", "validation_baseline", "indicator_count", "research")}, "orders_placed": 0, "data_source": "Quotex live candles; local replay only", "phase": "10h training + 2h local evaluation/adaptation"}
            update_market_progress(status="complete", progress=1.0, decisions=len(results), trades_scored=len(trades), wins=wins, losses=len(trades) - wins, net_profit=round(profit, 4))
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
                state.backtest_status = {
                    "status": "complete_with_errors" if failed else "complete",
                    "markets": summaries,
                    "markets_total": len(assets),
                    "markets_failed": failed,
                    "parallel_markets": concurrency,
                    "trades_scored": scored,
                    "wins": wins,
                    "losses": scored - wins,
                    "win_rate": round(wins / scored, 4) if scored else 0.0,
                    "net_profit": round(net_profit, 4),
                    "orders_placed": 0,
                    "phase": "isolated 10h training + 2h local evaluation/adaptation per market",
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
                if item.get("symbol") and float(item.get("payout") or 0) >= request.min_payout
            ]
        elif request.market_scope == "multiple":
            assets = list(dict.fromkeys(request.assets))
        else:
            assets = [request.asset] if request.asset else []
        if request.min_payout > 0 and assets and request.market_scope != "all":
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
                assets = [asset for asset in assets if payouts.get(asset.upper(), 0) >= request.min_payout]
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
        if timeframe not in {5, 15, 30, 60, 300, 900, 1800, 3600}:
            raise HTTPException(status_code=400, detail="Unsupported timeframe")
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            data = await asyncio.wait_for(
                broker.get_candles_async(asset, timeframe, max(20, min(limit, 300))), timeout=20
            )
        if not data:
            raise HTTPException(status_code=404, detail="No live candles available for this market")
        return data

    @app.get("/api/markets/{asset}/analysis")
    async def analysis(asset: str) -> dict[str, Any]:
        broker = getattr(state.engine, "_broker", None)
        if not broker or not getattr(broker, "authenticated", False):
            raise HTTPException(status_code=503, detail="Quotex broker is not authenticated")
        async with state.broker_io_lock:
            result = await asyncio.wait_for(
                state.engine.get_ai_prediction(asset, is_otc="otc" in asset.lower()), timeout=20
            )
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
        async with state.broker_io_lock:
            result = await asyncio.wait_for(
                state.engine.execute_trade(
                    request.asset, request.direction, request.amount, request.expiration
                ),
                timeout=30,
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

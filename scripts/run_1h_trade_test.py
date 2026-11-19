#!/usr/bin/env python3
"""NEXUS 1-Hour Autonomous AI Trading Test.

Workflow:
  1. Connect to Quotex demo account.
  2. Discover ALL active markets with payout >= 80%.
  3. Train the AI ensemble (RealAITradingEngine) on each discovered market.
  4. Enter a 60-minute trading loop:
       - Pick the highest-payout open market.
       - Fetch live candles from broker.
       - Run AI ensemble analysis -> signal + confidence.
       - Execute demo trade if confidence >= threshold.
       - Wait for expiration; log outcome.
  5. Print full performance summary.

Usage:
    cd /home/swapnil/NEXUS
    uv run python scripts/run_1h_trade_test.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── project path ──────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402, I001
load_dotenv(project_root / ".env", override=True)

from nexus.ai.engine_ai import RealAITradingEngine  # noqa: E402
from nexus.core.engine import NexusEngine  # noqa: E402
from nexus.utils.config import load_runtime_settings  # noqa: E402
from nexus.utils.logger import get_nexus_logger  # noqa: E402

logger = get_nexus_logger("scripts.run_1h_trade_test")
console = Console()

# ── configuration ─────────────────────────────────────────────────────────────
MIN_PAYOUT_PCT       = 80.0
MIN_CONFIDENCE       = 0.60
MIN_EDGE_BUFFER      = 0.03
MAX_RISK_FRACTION    = 0.001  # 0.10% of current demo balance per settled trade.
MIN_TRADE_AMOUNT     = 1.0
MAX_TRADE_AMOUNT     = 20.0
MAX_TRADES_PER_ASSET = 3
EXPLORATION_MAX_RISK_FRACTION = 0.00025  # 0.025%; unpromoted PRACTICE only.
EXPLORATION_MAX_TRADE_AMOUNT = 5.0
EXPLORATION_MAX_CONSECUTIVE_LOSSES = 2
EXPLORATION_SESSION_LOSS_FRACTION = 0.0002  # 0.02% of starting demo balance.
TEST_DURATION_S      = 3600
MAX_CANDLE_LIMIT     = 150
REPORT_PATH          = project_root / "reports" / "robust_market_research.json"


def fmt_dir(d: str) -> str:
    if d == "call":
        return "[bold green]CALL[/bold green]"
    return "[bold red]PUT[/bold red]"


def break_even_win_rate(payout_percent: float) -> float:
    """Return the win rate required to break even for a binary option payout."""
    return 1.0 / (1.0 + max(0.0, payout_percent) / 100.0)


def expected_value(confidence: float, payout_percent: float) -> float:
    """Expected P&L for one currency unit; confidence must be calibrated."""
    payout = max(0.0, payout_percent) / 100.0
    return confidence * payout - (1.0 - confidence)


def is_broker_confirmed_demo_order(result: Dict[str, Any]) -> bool:
    """Accept only a real Quotex PRACTICE placement, never a local fallback."""
    order_id = str(result.get("order_id", "")).strip()
    return bool(
        result.get("success")
        and result.get("real_executed") is True
        and order_id
        and not order_id.upper().startswith("SIM-")
    )


def position_size(
    balance: float, confidence: float, payout_percent: float,
    max_risk_fraction: float = MAX_RISK_FRACTION, max_trade_amount: float = MAX_TRADE_AMOUNT,
    ai_requested_risk_fraction: Optional[float] = None,
) -> float:
    """Authorize an AI-requested stake fraction inside deterministic limits."""
    payout = max(0.01, payout_percent / 100.0)
    kelly = max(0.0, (confidence * (1.0 + payout) - 1.0) / payout)
    # Half-Kelly is the maximum confidence scale.  This produces a stake that
    # changes with both account balance and calibrated edge, while the mode's
    # deterministic risk budget remains an upper bound.
    confidence_scale = min(1.0, kelly / 0.50)
    fallback_risk_fraction = max_risk_fraction * confidence_scale
    requested = ai_requested_risk_fraction if ai_requested_risk_fraction is not None else fallback_risk_fraction
    risk_fraction = min(max_risk_fraction, max(0.0, requested))
    if risk_fraction <= 0.0:
        return 0.0
    return round(min(max_trade_amount, max(MIN_TRADE_AMOUNT, balance * risk_fraction)), 2)


def normalized_expiration(recommended_expiration: Any, is_otc: bool) -> int:
    """Keep the model's one-second resolution within the broker market rules."""
    try:
        expiration = int(round(float(recommended_expiration)))
    except (TypeError, ValueError):
        expiration = 5 if is_otc else 60
    minimum = 5 if is_otc else 60
    return max(minimum, min(900, expiration))


def ai_allocation_decision(
    analysis: Dict[str, Any], balance: float, payout_percent: float, exploration: bool = False,
    is_otc: bool = True,
) -> Dict[str, Any]:
    """Convert the AI forecast into a per-trade expiry and risk-capped stake.

    The AI's calibrated probability and recommended horizon determine the
    allocation.  Deterministic limits can only reduce/reject exposure; they
    never invent a trade or increase the AI-derived amount.
    """
    confidence = float(analysis.get("confidence", 0.0) or 0.0)
    expiration = normalized_expiration(analysis.get("recommended_expiration"), is_otc)
    payout = max(0.0, float(payout_percent))
    max_risk = EXPLORATION_MAX_RISK_FRACTION if exploration else MAX_RISK_FRACTION
    max_amount = EXPLORATION_MAX_TRADE_AMOUNT if exploration else MAX_TRADE_AMOUNT
    break_even = break_even_win_rate(payout)
    payout_ratio = max(0.01, payout / 100.0)
    kelly = max(0.0, (confidence * (1.0 + payout_ratio) - 1.0) / payout_ratio)
    exposure_scale = min(1.0, kelly / 0.50)
    fallback_risk_fraction = max_risk * exposure_scale
    try:
        requested_risk_fraction = float(analysis.get("recommended_risk_fraction"))
    except (TypeError, ValueError):
        requested_risk_fraction = fallback_risk_fraction
    requested_risk_fraction = max(0.0, requested_risk_fraction)
    requested_amount = balance * requested_risk_fraction
    amount = position_size(
        balance, confidence, payout, max_risk, max_amount, requested_risk_fraction,
    )
    return {
        "model": "RealAITradingEngine",
        "confidence_type": str(analysis.get("confidence_type", "calibrated probability")),
        "calibrated_confidence": confidence,
        "raw_confidence": float(analysis.get("raw_confidence", confidence) or confidence),
        "ai_recommended_expiration_seconds": expiration,
        "market_type": "otc" if is_otc else "non_otc",
        "ai_requested_risk_fraction": requested_risk_fraction,
        "ai_requested_amount": requested_amount,
        "risk_authorized_amount": amount,
        "expected_value_per_dollar": expected_value(confidence, payout),
        "break_even_win_rate": break_even,
        "ai_exposure_scale": exposure_scale,
        "risk_cap_fraction": max_risk,
        "risk_cap_amount": max_amount,
        "risk_limited": requested_risk_fraction > max_risk or requested_amount > max_amount,
        "allocation_mode": "practice_exploration" if exploration else "promoted",
    }


def promotion_is_eligible(evidence: Dict[str, Any], payout_percent: float) -> bool:
    """Evaluate held-out evidence using the *current* broker payout."""
    return (
        evidence.get("status") == "complete"
        and evidence.get("promotion_status") == "promote"
        and int(evidence.get("trades", 0)) >= 30
        and float(evidence.get("win_rate", 0.0)) > break_even_win_rate(payout_percent)
        and float(evidence.get("net_pnl", 0.0)) > 0.0
        and float(evidence.get("calibration_error", 1.0)) <= 0.15
    )


def load_walk_forward_gates() -> Dict[str, Tuple[bool, str, Dict[str, Any]]]:
    """Load the sole promotion source for demo orders, keyed by normalized asset."""
    if not REPORT_PATH.exists():
        return {}
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        records = report.get("markets", [])
    except (OSError, ValueError, TypeError):
        return {}
    gates: Dict[str, Tuple[bool, str, Dict[str, Any]]] = {}
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        asset = str(record.get("asset", "")).strip().upper()
        if not asset:
            continue
        payout = float(record.get("payout", 0.0) or 0.0)
        evidence = {
            "status": str(record.get("status", "unknown")),
            "trades": int(record.get("trades", 0) or 0),
            "win_rate": float(record.get("win_rate", 0.0) or 0.0),
            "net_pnl": float(record.get("net_pnl", 0.0) or 0.0),
            "calibration_error": float(record.get("calibration_error", 1.0) or 1.0),
            "promotion_status": str(record.get("promotion_status", "unknown")),
        }
        # Payout is live and can change after research.  A nonzero report
        # payout is useful only for a quick display; selection rechecks it
        # against the fresh broker catalog below.
        passed = promotion_is_eligible(evidence, payout) if payout > 0.0 else False
        reason = "walk-forward promotion criteria met" if passed else "no promoted, positive, calibrated OOS evidence"
        gates[asset] = (passed, reason, evidence)
    return gates


async def wait_for_verified_settlement(
    broker: Any, order_id: str, expiration: int, deadline: float
) -> Optional[Dict[str, Any]]:
    """Only accept a broker-reported closed outcome; never infer one from balance."""
    until = min(deadline, time.monotonic() + expiration + 45)
    while time.monotonic() < until:
        outcome = broker.get_trade_outcome(order_id)
        if isinstance(outcome, dict) and outcome.get("status") == "SETTLED":
            return outcome
        await asyncio.sleep(min(2.0, max(0.1, until - time.monotonic())))
    return None


def write_session_report(payload: Dict[str, Any]) -> Path:
    """Atomically retain an audit record without exposing broker credentials."""
    directory = project_root / "reports" / "demo_trade_sessions"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / f"session-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    temporary = report.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(report)
    return report


def write_session_table(payload: Dict[str, Any], report_path: Path) -> Path:
    """Write a human-readable broker-demo ledger beside the JSON audit."""
    table_path = report_path.with_suffix(".md")
    rows = [
        "# Quotex PRACTICE Trade Report",
        "",
        f"- Mode: `{payload['mode']}`",
        f"- Balance: `${payload['initial_balance']:.2f}` -> `${payload['final_balance']:.2f}`",
        f"- Net P&L: `${payload['net_pnl']:+.2f}`",
        "",
        "| # | Time | Market | Side | AI requested | Authorized stake | AI expiry | AI stake scale | Payout | Confidence | Status | Outcome | P&L | Broker order ID |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for index, trade in enumerate(payload.get("trades", []), 1):
        pnl = f"${float(trade.get('profit', 0.0) or 0.0):+.2f}" if trade.get("settlement") == "SETTLED" else "-"
        rows.append(
            "| {index} | {time} | {asset} | {direction} | ${requested_amount:.2f} | ${amount:.2f} | {expiration}s | {stake_scale:.2f}x | {payout:.0f}% | {confidence:.1%} | {status} | {outcome} | {pnl} | {order_id} |".format(
                index=index, time=trade.get("time", "-"), asset=trade.get("asset", "-"),
                direction=str(trade.get("direction", "-")).upper(), amount=float(trade.get("amount", 0.0) or 0.0),
                expiration=int(trade.get("expiration", 0) or 0), payout=float(trade.get("payout", 0.0) or 0.0),
                confidence=float(trade.get("confidence", 0.0) or 0.0),
                stake_scale=float(trade.get("ai_allocation", {}).get("ai_exposure_scale", 1.0) or 1.0),
                requested_amount=float(trade.get("ai_allocation", {}).get("ai_requested_amount", trade.get("amount", 0.0)) or 0.0),
                status=trade.get("settlement", "REJECTED" if not trade.get("success") else "UNVERIFIED"),
                outcome=trade.get("outcome", "-"), pnl=pnl, order_id=trade.get("order_id", "-"),
            )
        )
    if not payload.get("trades"):
        rows.append("| - | - | - | - | - | - | - | - | - | - | NO ORDERS | - | - | - |")
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return table_path


def build_summary_table(trades: List[Dict[str, Any]]) -> Table:
    tbl = Table(title="1-Hour Test — Trade Log", box=box.MINIMAL_DOUBLE_HEAD)
    tbl.add_column("#",       style="dim",    justify="right")
    tbl.add_column("Time",    style="cyan")
    tbl.add_column("Asset",   style="cyan")
    tbl.add_column("Dir",     style="white")
    tbl.add_column("Conf%",   justify="right")
    tbl.add_column("Payout%", justify="right")
    tbl.add_column("Amount",  justify="right")
    tbl.add_column("Status",  style="white")
    tbl.add_column("Outcome", style="white")
    tbl.add_column("P&L", justify="right")
    tbl.add_column("Order ID", style="dim")
    for i, t in enumerate(trades, 1):
        if t.get("settlement") == "SETTLED":
            status = "[green]SETTLED[/green]"
        elif t.get("success"):
            status = "[yellow]UNVERIFIED[/yellow]"
        else:
            status = f"[red]FAIL: {str(t.get('error', ''))[:30]}[/red]"
        tbl.add_row(
            str(i),
            t.get("time", ""),
            t.get("asset", ""),
            fmt_dir(t.get("direction", "call")),
            f"{t.get('confidence', 0)*100:.0f}%",
            f"{t.get('payout', 0):.0f}%",
            f"${float(t.get('amount', 0.0) or 0.0):.2f}",
            status,
            str(t.get("outcome", "-")),
            f"${float(t.get('profit', 0.0) or 0.0):+.2f}" if t.get("settlement") == "SETTLED" else "-",
            str(t.get("order_id", ""))[:16],
        )
    return tbl


async def run_1h_test(duration_s: int = TEST_DURATION_S, practice_exploration: bool = False) -> None:
    if os.getenv("NEXUS_FORCE_SIM", "0").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Demo-trade runner refuses NEXUS_FORCE_SIM; real Quotex PRACTICE execution is required")
    start_ts = time.monotonic()
    deadline = start_ts + duration_s
    trades: List[Dict[str, Any]] = []

    console.print(Panel.fit(
        "[bold cyan]NEXUS AI -- Evidence-Gated Demo Trading Test[/bold cyan]\n"
        f"[white]Min payout: {MIN_PAYOUT_PCT:.0f}%  |  "
        f"Min calibrated confidence: {MIN_CONFIDENCE*100:.0f}%  |  "
        f"Duration: {duration_s // 60} min[/white]",
        border_style="cyan",
    ))

    # ── Step 1: Connect ───────────────────────────────────────────────────────
    console.rule("[bold yellow]Step 1 . Connecting to Quotex Demo[/bold yellow]")
    settings = load_runtime_settings()
    engine   = NexusEngine(settings=settings, demo_mode=True, auto_login=False)

    connected = False
    for attempt in range(1, 6):
        console.print(f"Connecting to Quotex demo account (attempt {attempt}/5)...")
        connected = await engine.login_broker()
        if connected:
            break
        console.print("[yellow]Connection attempt failed. Retrying in 5 seconds...[/yellow]")
        await asyncio.sleep(5)

    if not connected:
        console.print(
            "[bold red]Could not connect to Quotex demo account after 5 attempts.\n"
            "Check QUOTEX_EMAIL / QUOTEX_PASSWORD in .env.[/bold red]"
        )
        sys.exit(1)

    initial_balance = await engine.get_account_balance()
    console.print(
        f"[green]Connected!   Demo balance: [bold]${initial_balance:,.2f}[/bold][/green]"
    )

    broker = engine._broker
    assert broker is not None

    # ── Step 2: Discover markets ──────────────────────────────────────────────
    console.rule("[bold yellow]Step 2 . Discovering high-payout markets[/bold yellow]")
    all_payouts: List[Dict[str, Any]] = await broker.get_assets_with_payouts_async() or []

    eligible: List[Dict[str, Any]] = []
    for a in all_payouts:
        sym    = str(a.get("symbol", "")).strip()
        payout = float(a.get("payout", 0.0) or 0.0)
        active = bool(a.get("active", True))
        if sym and active and payout >= MIN_PAYOUT_PCT:
            eligible.append({"symbol": sym, "payout": payout})

    if not eligible:
        console.print(f"[yellow]No markets >= {MIN_PAYOUT_PCT}% payout; using all.[/yellow]")
        eligible = [
            {"symbol": str(a.get("symbol", "")), "payout": float(a.get("payout", 0) or 0)}
            for a in all_payouts if str(a.get("symbol", "")).strip()
        ]

    eligible.sort(key=lambda x: x["payout"], reverse=True)

    tbl_mkt = Table(title=f"Markets (payout >= {MIN_PAYOUT_PCT}%)", box=box.SIMPLE)
    tbl_mkt.add_column("Symbol", style="cyan")
    tbl_mkt.add_column("Payout", justify="right")
    for m in eligible[:25]:
        tbl_mkt.add_row(m["symbol"], f"{m['payout']:.0f}%")
    console.print(tbl_mkt)
    console.print(f"[green]Found [bold]{len(eligible)}[/bold] eligible markets.[/green]")

    # ── Step 3: Load promoted models ──────────────────────────────────────────
    console.rule("[bold yellow]Step 3 . Validating walk-forward promotion[/bold yellow]")
    ai_engine = RealAITradingEngine()
    engine.ai_engine = ai_engine
    gates = load_walk_forward_gates()
    researched_eligible_assets = sorted(
        market["symbol"].upper() for market in eligible if market["symbol"].upper() in gates
    )
    promoted_assets = {
        market["symbol"].upper()
        for market in eligible
        if market["symbol"].upper() in gates
        and promotion_is_eligible(gates[market["symbol"].upper()][2], market["payout"])
    }
    if promoted_assets:
        console.print(f"[green]{len(promoted_assets)} promoted markets in current research report.[/green]")
    elif practice_exploration:
        console.print(
            "[yellow]No promoted market: running explicitly requested PRACTICE exploration. "
            "Orders remain unpromoted learning observations.[/yellow]"
        )
    elif not researched_eligible_assets:
        console.print(
            "[yellow]No currently eligible broker market has a research record. "
            "Ending without a pointless scan loop; run payout-filtered robust research first.[/yellow]"
        )
    else:
        rejected = ", ".join(researched_eligible_assets[:5])
        console.print(
            f"[yellow]Research records exist but none is promotable ({rejected}). "
            "Ending without sending orders.[/yellow]"
        )
    promotion_preflight = {
        "research_report": str(REPORT_PATH.relative_to(project_root)),
        "eligible_broker_markets": len(eligible),
        "eligible_markets_with_research": researched_eligible_assets,
        "promoted_markets": sorted(promoted_assets),
        "decision": "promoted_trade" if promoted_assets else "practice_exploration" if practice_exploration else "abstain",
    }

    # ── Step 4: Trading loop ──────────────────────────────────────────────────
    console.rule("[bold yellow]Step 4 . Evidence-Gated Demo Trading Loop[/bold yellow]")
    cycle = 0
    exploration_mode = practice_exploration and not promoted_assets

    while (promoted_assets or practice_exploration) and time.monotonic() < deadline:
        settled = [trade for trade in trades if trade.get("settlement") == "SETTLED"]
        consecutive_losses = 0
        for trade in reversed(settled):
            if trade.get("outcome") != "LOSS":
                break
            consecutive_losses += 1
        realized_pnl = sum(float(trade.get("profit", 0.0) or 0.0) for trade in settled)
        loss_limit = initial_balance * EXPLORATION_SESSION_LOSS_FRACTION
        if exploration_mode and (
            consecutive_losses >= EXPLORATION_MAX_CONSECUTIVE_LOSSES
            or realized_pnl <= -loss_limit
        ):
            console.print(
                "[yellow]Exploration circuit breaker: stopping after adverse settled evidence "
                f"(P&L ${realized_pnl:+.2f}, consecutive losses {consecutive_losses}).[/yellow]"
            )
            break
        cycle += 1
        elapsed   = time.monotonic() - start_ts
        remaining = max(0.0, duration_s - elapsed)
        console.print(
            f"\n[bold cyan]Cycle {cycle}  |  "
            f"elapsed {int(elapsed//60)}m{int(elapsed%60):02d}s  |  "
            f"remaining {int(remaining//60)}m{int(remaining%60):02d}s[/bold cyan]"
        )

        # Check connection health and reconnect if dropped
        if hasattr(broker, "check_connect"):
            try:
                is_conn = await broker.check_connect()
                if not is_conn:
                    console.print("  [yellow]Broker connection inactive. Reconnecting...[/yellow]")
                    await engine.login_broker()
                    broker = engine._broker
            except Exception:
                pass

        candidates: List[Dict[str, Any]] = []
        best_conf = 0.0

        for target in eligible:
            sym = target["symbol"]
            payout = target["payout"]
            gate = gates.get(sym.upper())
            is_promoted = gate is not None and promotion_is_eligible(gate[2], payout)
            if not is_promoted and not practice_exploration:
                continue

            # Fetch candles for this market
            try:
                raw = await broker.get_candles_async(sym, timeframe_sec=60, limit=MAX_CANDLE_LIMIT)
            except Exception:
                continue

            if not raw or len(raw) < 20:
                continue

            cdf = pd.DataFrame(raw)
            for col in ("open", "high", "low", "close"):
                cdf[col] = cdf[col].astype(float)
            if "volume" not in cdf.columns:
                cdf["volume"] = 0.0

            try:
                analysis = await engine.analyze_market_ai(
                    cdf, asset=sym, timeframe=60,
                    is_otc="otc" in sym.lower(),
                )
            except Exception:
                continue

            signal     = analysis.get("signal", "hold").lower()
            confidence = float(analysis.get("confidence", 0.0))
            regime     = analysis.get("regime", "Unknown")

            sig_color = "green" if signal == "call" else "red" if signal == "put" else "yellow"
            console.print(
                f"  [{sig_color}]{sym}[/{sig_color}] ({payout:.0f}% payout) -> "
                f"[{sig_color}]{signal.upper()}[/{sig_color}] "
                f"conf={confidence*100:.1f}% ({regime})"
            )

            allocation = ai_allocation_decision(
                analysis,
                balance=1.0,
                payout_percent=payout,
                exploration=exploration_mode,
                is_otc="otc" in sym.lower(),
            )
            expiration = int(allocation["ai_recommended_expiration_seconds"])
            min_confidence = max(MIN_CONFIDENCE, break_even_win_rate(payout) + MIN_EDGE_BUFFER)
            ev = expected_value(confidence, payout)
            best_conf = max(best_conf, confidence)
            if (
                signal in ("call", "put")
                and confidence >= min_confidence
                and ev > 0.0
                and regime.upper() not in {"SIDEWAYS", "UNKNOWN"}
                and sum(t.get("asset") == sym and t.get("success") for t in trades)
                < (1 if exploration_mode else MAX_TRADES_PER_ASSET)
            ):
                candidates.append({
                    "target": target, "signal": signal, "confidence": confidence,
                    "analysis": analysis, "allocation": allocation,
                    "expiration": expiration, "expected_value": ev,
                })

        if not candidates:
            console.print(f"  [yellow]HOLD: no promoted positive-edge opportunity (best confidence: {best_conf*100:.1f}%).[/yellow]")
            await asyncio.sleep(min(30, max(1.0, deadline - time.monotonic())))
            continue

        selected = max(candidates, key=lambda candidate: candidate["expected_value"])
        target, signal = selected["target"], selected["signal"]
        sym, payout = target["symbol"], target["payout"]
        confidence, expiration = selected["confidence"], selected["expiration"]
        balance = await engine.get_account_balance()
        allocation = ai_allocation_decision(
            selected["analysis"],
            balance,
            payout,
            exploration=exploration_mode,
            is_otc="otc" in sym.lower(),
        )
        amount = float(allocation["risk_authorized_amount"])
        if amount <= 0:
            console.print("  [yellow]HOLD: no positive risk-adjusted stake.[/yellow]")
            continue
        ts_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        console.print(
            f"\n  [bold green]>>> DEMO ORDER:[/bold green] {sym} {fmt_dir(signal)} "
            f"${amount:.2f} | {expiration}s | payout {payout:.0f}% | calibrated confidence {confidence*100:.1f}% | EV ${amount * selected['expected_value']:.2f}"
        )
        trade_res = await engine.execute_trade(sym, signal, amount, expiration)
        trade = {
            "time": ts_str, "asset": sym, "direction": signal, "confidence": confidence,
            "payout": payout, "amount": amount, "expiration": expiration,
            "expected_value": selected["expected_value"], "policy_mode": "exploration" if exploration_mode else "promoted",
            "success": bool(trade_res.get("success")),
            "ai_allocation": allocation,
            "real_executed": bool(trade_res.get("real_executed", False)),
            "order_id": str(trade_res.get("order_id", "")), "error": str(trade_res.get("error", "")),
        }
        trades.append(trade)
        if not is_broker_confirmed_demo_order(trade_res):
            detail = trade["error"] or "broker did not return a real PRACTICE order ID"
            trade.update({"success": False, "error": detail})
            console.print(f"  [bold red]✗ Order rejected: {detail}[/bold red]")
            continue
        console.print(f"  [green]✓ Broker accepted order {trade['order_id']}[/green]")
        settlement = await wait_for_verified_settlement(broker, trade["order_id"], expiration, deadline)
        if settlement is None:
            trade.update({"settlement": "UNVERIFIED", "outcome": "UNVERIFIED"})
            console.print("  [yellow]Outcome not broker-verified before deadline; no learning update applied.[/yellow]")
            continue
        profit = float(settlement.get("profit", 0.0) or 0.0)
        outcome = str(settlement.get("outcome", "UNVERIFIED")).upper()
        success = outcome == "WIN"
        trade.update({"settlement": "SETTLED", "outcome": outcome, "profit": profit})
        await ai_engine.learn_and_evolve(sym, signal, success, profit, selected["analysis"])
        console.print(f"  [green]Settled {outcome}: ${profit:+.2f}; recorded learning feedback.[/green]")

    # ── Step 5: Final summary ─────────────────────────────────────────────────
    console.rule("[bold green]Test Complete -- Summary[/bold green]")
    final_balance = await engine.get_account_balance()
    placed   = [t for t in trades if t["success"]]
    rejected = [t for t in trades if not t["success"]]
    settled = [t for t in placed if t.get("settlement") == "SETTLED"]
    unverified = [t for t in placed if t.get("settlement") != "SETTLED"]
    pnl      = final_balance - initial_balance

    console.print(build_summary_table(trades))

    summary = Table(title="Performance Summary", box=box.HEAVY_EDGE)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value",  style="bold white")
    summary.add_row("Total Cycles",         str(cycle))
    summary.add_row("Trades Placed",        str(len(placed)))
    summary.add_row("Trades Rejected",      str(len(rejected)))
    summary.add_row("Broker-Verified Settlements", str(len(settled)))
    summary.add_row("Unverified Orders", str(len(unverified)))
    summary.add_row("Initial Demo Balance", f"${initial_balance:,.2f}")
    summary.add_row("Final Demo Balance",   f"${final_balance:,.2f}")
    pnl_color = "green" if pnl >= 0 else "red"
    summary.add_row(
        "Net P&L",
        f"[{pnl_color}]{'+' if pnl>=0 else ''}{pnl:,.2f}[/{pnl_color}]",
    )
    console.print(summary)

    pnl_str = f"${pnl:+,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    console.print(Panel.fit(
        f"[bold green]NEXUS Demo Test Complete[/bold green]\n"
        f"Trades: {len(placed)} placed  |  "
        f"Balance: ${initial_balance:,.2f} -> ${final_balance:,.2f}  |  "
        f"P&L: {pnl_str}",
        border_style="green",
    ))
    session_payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_s,
        "mode": "broker_demo_practice_exploration" if practice_exploration and not promoted_assets else "broker_demo_evidence_gated",
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "net_pnl": pnl,
        "orders_placed": len(placed),
        "orders_settled": len(settled),
        "orders_unverified": len(unverified),
        "orders_rejected": len(rejected),
        "promotion_preflight": promotion_preflight,
        "trades": trades,
    }
    report_path = write_session_report(session_payload)
    table_path = write_session_table(session_payload, report_path)
    console.print(f"[dim]Audit report: {report_path.relative_to(project_root)}[/dim]")
    console.print(f"[dim]Trade table: {table_path.relative_to(project_root)}[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evidence-gated NEXUS demo trading.")
    parser.add_argument("--duration-minutes", type=int, default=60, help="Test duration (default: 60).")
    parser.add_argument("--practice-exploration", action="store_true", help="Use real Quotex PRACTICE observations when no market is promoted; never treats them as promotion evidence.")
    args = parser.parse_args()
    if args.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    try:
        asyncio.run(run_1h_test(args.duration_minutes * 60, args.practice_exploration))
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted.[/yellow]")

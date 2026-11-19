"""NEXUS EUR/JPY AI Training and Live Demo Trade Test.

Orchestrates:
1. Live Quotex broker connection in Demo mode and initial balance verification.
2. AI model training specifically for EUR/JPY:
   - Dynamic indicator screening and feature ranking.
   - Historical feature engineering via broad technical analysis provider suite.
   - HistGradientBoostingClassifier model training and walk-forward validation.
   - Deep learning sequence training for the Attention-LSTM predictor.
3. Live market candle retrieval from Quotex for EUR/JPY.
4. Full AI ensemble market analysis and directional signal generation (Market Regime,
   Transformer, Attention-LSTM, Market Model, and MetaStrategy confluence).
5. Live demo order execution on Quotex.
6. Settlement tracking and post-trade demo balance verification.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env", override=True)

from nexus.ai.engine_ai import RealAITradingEngine  # noqa: E402
from nexus.ai.lstm_predictor import MarketPredictor  # noqa: E402
from nexus.core.engine import NexusEngine  # noqa: E402
from nexus.data.provider import FeatureEngineer  # noqa: E402
from nexus.utils.config import load_runtime_settings  # noqa: E402
from nexus.utils.logger import get_nexus_logger  # noqa: E402

logger = get_nexus_logger("scripts.test_eurjpy_ai_trade")
console = Console()


def load_walk_forward_gate(asset: str, payout_percent: float) -> tuple[bool, str, dict[str, object]]:
    """Require current out-of-sample evidence before a PRACTICE order.

    A live directional score is an observation, not proof that a strategy is
    viable.  The robust-research report is produced from chronological,
    purged walk-forward folds and is intentionally the only promotion source
    for this direct broker test.
    """
    report_path = project_root / "reports" / "robust_market_research.json"
    if not report_path.exists():
        return False, "no current walk-forward research report", {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        records = report.get("markets", [])
        record = next(
            (item for item in records if str(item.get("asset", "")).upper() == asset.upper()),
            None,
        )
        if not isinstance(record, dict) or record.get("status") != "complete":
            return False, "no completed walk-forward result for this asset", {}
        payout_break_even = 1.0 / (1.0 + payout_percent / 100.0)
        win_rate = float(record.get("win_rate", 0.0) or 0.0)
        net_pnl = float(record.get("net_pnl", 0.0) or 0.0)
        trades = int(record.get("trades", 0) or 0)
        calibrated = float(record.get("calibration_error", 1.0) or 1.0)
        promoted = record.get("promotion_status") == "promote"
        passed = promoted and trades >= 30 and win_rate > payout_break_even and net_pnl > 0 and calibrated <= 0.15
        evidence = {
            "trades": trades,
            "win_rate": win_rate,
            "break_even": payout_break_even,
            "net_pnl": net_pnl,
            "calibration_error": calibrated,
            "promotion_status": record.get("promotion_status", "unknown"),
        }
        if passed:
            return True, "walk-forward promotion criteria met", evidence
        return False, "walk-forward promotion criteria not met", evidence
    except (OSError, ValueError, TypeError, KeyError) as error:
        return False, f"could not read walk-forward report: {error}", {}


async def run_eurjpy_test() -> None:
    """Execute complete EUR/JPY AI training and live demo trade test."""
    console.print(
        Panel.fit(
            "[bold cyan]NEXUS AI Trader - EUR/JPY AI Training & Live Demo Trade Test[/bold cyan]\n"
            "[white]Verifying real AI learning, signal generation, and broker execution[/white]",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------
    # 1. Initialize Engine and Connect to Quotex Live Demo
    # ------------------------------------------------------------------
    console.print("\n[bold yellow]Step 1: Connecting to Quotex Demo Account...[/bold yellow]")
    settings = load_runtime_settings()
    engine = NexusEngine(settings=settings, demo_mode=True)

    connected = await engine.login_broker()
    if not connected:
        console.print("[bold red]✗ Failed to connect to Quotex broker demo account.[/bold red]")
        sys.exit(1)

    initial_balance = await engine.get_account_balance()
    console.print("[bold green]✓ Connected to Quotex Demo Account![/bold green]")
    console.print(f"[bold white]  Initial Demo Balance:[/bold white] [bold green]${initial_balance:,.2f}[/bold green]")

    # Verify EUR/JPY asset availability and payout
    target_asset = "EURJPY"
    broker = engine._broker
    if broker is None:
        console.print("[bold red]✗ Broker adapter unavailable after login.[/bold red]")
        sys.exit(1)

    assets_with_payouts = await broker.get_assets_with_payouts_async()
    eurjpy_info = None
    for a in assets_with_payouts:
        sym = str(a.get("symbol", "")).strip().upper()
        if sym in ("EURJPY", "EURJPY_OTC", "OTC_EURJPY"):
            if a.get("active", False) and float(a.get("payout", 0.0)) > 0:
                eurjpy_info = a
                target_asset = a.get("symbol")
                break
    if not eurjpy_info:
        # Check standard EURJPY
        for a in assets_with_payouts:
            if "EUR" in str(a.get("symbol", "")).upper() and "JPY" in str(a.get("symbol", "")).upper():
                eurjpy_info = a
                target_asset = a.get("symbol")
                break

    payout = float(eurjpy_info.get("payout", 85.0)) if eurjpy_info else 85.0
    console.print(f"[bold white]  Target Symbol:[/bold white] [cyan]{target_asset}[/cyan] (Payout: [green]{payout:.0f}%[/green])")

    # A broker test may connect and train, but it must never bypass the
    # out-of-sample gate simply because the latest ensemble emits CALL/PUT.
    gate_passed, gate_reason, evidence = load_walk_forward_gate(target_asset, payout)
    gate_table = Table(title="Walk-Forward Promotion Gate", box=box.SIMPLE_HEAVY)
    gate_table.add_column("Metric", style="cyan")
    gate_table.add_column("Current EURJPY evidence", style="white")
    if evidence:
        gate_table.add_row("OOS decisions", str(evidence["trades"]))
        gate_table.add_row("OOS win rate", f"{float(evidence['win_rate']) * 100:.2f}%")
        gate_table.add_row("Payout break-even", f"{float(evidence['break_even']) * 100:.2f}%")
        gate_table.add_row("Modeled net P&L", f"${float(evidence['net_pnl']):.2f}")
        gate_table.add_row("Calibration error", f"{float(evidence['calibration_error']):.3f}")
        gate_table.add_row("Promotion", str(evidence["promotion_status"]).upper())
    gate_table.add_row("Decision", Text("ALLOW PRACTICE TEST" if gate_passed else "ABSTAIN", style="bold green" if gate_passed else "bold yellow"))
    console.print(gate_table)
    if not gate_passed:
        console.print(f"[bold yellow]No order sent: {gate_reason}. Run fresh walk-forward research after new data arrives.[/bold yellow]")
        return

    # ------------------------------------------------------------------
    # 2. Fetch Live Market Candles from Broker
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Step 2: Fetching live {target_asset} candles from Quotex...[/bold yellow]")
    raw_candles = await broker.get_candles_async(target_asset, timeframe_sec=60, limit=120)
    if not raw_candles or len(raw_candles) < 30:
        console.print(f"[bold red]✗ Insufficient live candles retrieved for {target_asset}[/bold red]")
        sys.exit(1)

    candles_df = pd.DataFrame(raw_candles)
    for col in ("open", "high", "low", "close"):
        candles_df[col] = candles_df[col].astype(float)
    if "volume" not in candles_df.columns:
        candles_df["volume"] = 0.0

    last_candle = candles_df.iloc[-1]
    console.print(f"[bold green]✓ Received {len(candles_df)} live 1-minute candles[/bold green]")
    console.print(
        f"  [white]Latest Close:[/white] [cyan]{last_candle['close']:.3f}[/cyan] | "
        f"[white]High:[/white] [cyan]{last_candle['high']:.3f}[/cyan] | "
        f"[white]Low:[/white] [cyan]{last_candle['low']:.3f}[/cyan]"
    )

    # ------------------------------------------------------------------
    # 3. Train AI Models for EUR/JPY
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Step 3: Training AI models on {target_asset}...[/bold yellow]")
    ai_engine = RealAITradingEngine()
    engine.ai_engine = ai_engine

    training_progress = []

    def progress_callback(stage: str, pct: float, detail: str) -> None:
        training_progress.append((stage, pct, detail))
        console.print(f"  [dim]• [{pct*100:3.0f}%] {stage.title()}: {detail}[/dim]")

    # A. Train Market Classification & Dynamic Feature Selection Model
    train_summary = ai_engine.train_market(
        target_asset,
        candles_df=candles_df,
        progress_callback=progress_callback,
    )

    console.print(f"\n[bold green]✓ Market AI model trained successfully for {target_asset}![/bold green]")
    table_train = Table(title=f"AI Training Results - {target_asset}", box=box.SIMPLE_HEAVY)
    table_train.add_column("Metric / Property", style="cyan")
    table_train.add_column("Trained Value / Specification", style="white")

    table_train.add_row("Trained Model File", f"models/markets/{target_asset}.joblib")
    table_train.add_row("Optimal RSI Period", str(train_summary.get("params", {}).get("rsi_period", 14)))
    table_train.add_row("Optimal EMA Fast/Slow", f"{train_summary.get('params', {}).get('ema_fast', 9)} / {train_summary.get('params', {}).get('ema_slow', 21)}")
    table_train.add_row("Optimal Stoch K/D", f"{train_summary.get('params', {}).get('stoch_k', 14)} / {train_summary.get('params', {}).get('stoch_d', 3)}")
    table_train.add_row("In-Sample Directional Hit Rate", f"{train_summary.get('accuracy', 0.0):.1f}%")
    table_train.add_row("Validation Accuracy", f"{train_summary.get('validation_accuracy', 'N/A')}%")
    table_train.add_row("Validation Baseline", f"{train_summary.get('validation_baseline', 'N/A')}%")
    table_train.add_row("Selected Features", ", ".join(train_summary.get("features", [])[:6]) + "...")
    table_train.add_row("Top Indicators", ", ".join(train_summary.get("best_indicators", [])[:2]))
    console.print(table_train)

    # B. Train Deep Learning Neural Sequence Predictor
    console.print("\n[bold yellow]Training Deep Learning Attention-LSTM Predictor...[/bold yellow]")
    engineer = FeatureEngineer()
    df_feat = engineer.add_features(candles_df.copy())
    df_feat = df_feat.dropna()

    feat_cols = [c for c in df_feat.columns if c not in ("open", "high", "low", "close", "volume", "time", "timestamp")][:20]
    seq_len = 15
    if len(df_feat) > seq_len + 5:
        X_seqs = []
        y_classes = []
        y_values = []
        for i in range(len(df_feat) - seq_len - 1):
            seq = df_feat[feat_cols].iloc[i : i + seq_len].values
            curr_c = df_feat["close"].iloc[i + seq_len - 1]
            nxt_c = df_feat["close"].iloc[i + seq_len]
            ret = (nxt_c - curr_c) / max(curr_c, 1e-6)
            c_label = 0 if ret > 0.0001 else (1 if ret < -0.0001 else 2)
            X_seqs.append(seq)
            y_classes.append(c_label)
            y_values.append(ret * 100.0)

        import torch
        device = ai_engine.transformer_predictor.device
        lstm_predictor = MarketPredictor(input_dim=len(feat_cols), hidden_dim=64, device=device)
        trainer = lstm_predictor.trainer
        t_x = torch.FloatTensor(np.array(X_seqs)).to(device)
        t_yc = torch.LongTensor(np.array(y_classes)).to(device)
        t_yv = torch.FloatTensor(np.array(y_values)).to(device)

        # Run 5 fast adaptation epochs
        for _ep in range(5):
            trainer.train_batch(t_x, t_yc, t_yv)
        console.print(f"[bold green]✓ Attention-LSTM adapted on {len(X_seqs)} sequences of {target_asset}![/bold green]")

    # ------------------------------------------------------------------
    # 4. Perform Real AI Market Analysis & Signal Generation
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Step 4: Running Real AI Market Analysis on live {target_asset}...[/bold yellow]")
    analysis = await engine.analyze_market_ai(
        candles_df, asset=target_asset, timeframe=60, is_otc=target_asset.endswith("_otc")
    )

    raw_signal = analysis.get("signal", "hold").lower()
    confidence = float(analysis.get("confidence", 0.0))
    reasoning = analysis.get("reasoning", "AI ensemble analysis")
    regime = analysis.get("regime", "Trend")
    indicators = analysis.get("indicators", [])

    table_ai = Table(title=f"AI Ensemble Analysis - {target_asset}", box=box.ROUNDED)
    table_ai.add_column("Signal Component", style="cyan")
    table_ai.add_column("AI Prediction / State", style="bold white")

    signal_style = "bold green" if raw_signal == "call" else "bold red" if raw_signal == "put" else "bold yellow"
    table_ai.add_row("Ensemble Signal", Text(raw_signal.upper(), style=signal_style))
    table_ai.add_row("Ensemble Confidence", f"{confidence * 100:.1f}%")
    table_ai.add_row("Detected Market Regime", Text(str(regime)))
    table_ai.add_row("AI Reasoning", Text(str(reasoning)))
    if indicators:
        table_ai.add_row("Active Technical Indicators", Text("\n".join(str(ind) for ind in indicators[:3])))
    console.print(table_ai)

    # A broker-execution test must not turn an abstention into an arbitrary
    # order.  If there is no actionable AI direction, the result is a valid
    # no-trade outcome rather than fabricated evidence.
    if raw_signal not in ("call", "put"):
        console.print("[yellow]No directional AI signal; EURJPY test abstained without placing an order.[/yellow]")
        return
    trade_direction = raw_signal

    trade_amount = 1.0
    trade_expiration = 60

    # ------------------------------------------------------------------
    # 5. Execute Trade on Real Quotex Demo Account
    # ------------------------------------------------------------------
    console.print(
        f"\n[bold yellow]Step 5: Executing Real Demo Trade on Quotex...[/bold yellow]\n"
        f"  [white]Asset:[/white] [cyan]{target_asset}[/cyan] | "
        f"[white]Direction:[/white] [{'green' if trade_direction == 'call' else 'red'}]{trade_direction.upper()}[/{'green' if trade_direction == 'call' else 'red'}] | "
        f"[white]Amount:[/white] [green]${trade_amount:.2f}[/green] | "
        f"[white]Duration:[/white] [cyan]{trade_expiration}s[/cyan]"
    )

    trade_result = await engine.execute_trade(
        asset=target_asset,
        signal_type=trade_direction,
        amount=trade_amount,
        expiration=trade_expiration,
    )

    table_exec = Table(title="Live Trade Execution Result", box=box.HEAVY_EDGE)
    table_exec.add_column("Field", style="cyan")
    table_exec.add_column("Value", style="bold white")

    is_success = trade_result.get("success", False)
    table_exec.add_row("Status", Text("ACCEPTED BY BROKER", style="bold green") if is_success else Text(f"REJECTED: {trade_result.get('error')}", style="bold red"))
    table_exec.add_row("Broker Order ID", Text(str(trade_result.get("order_id", "N/A"))))
    table_exec.add_row("Asset", Text(str(trade_result.get("asset", target_asset))))
    table_exec.add_row("Direction", Text(str(trade_result.get("direction", trade_direction)).upper()))
    table_exec.add_row("Amount", f"${trade_amount:.2f}")
    table_exec.add_row("Expiration", f"{trade_expiration}s")
    table_exec.add_row("Real Broker Executed", str(trade_result.get("real_executed", False)))
    console.print(table_exec)

    order_id = str(trade_result.get("order_id") or "")
    if not is_success or not order_id:
        console.print(f"[bold red]Trade placement did not complete: {trade_result.get('error')}[/bold red]")
        if not order_id:
            console.print("[bold red]Broker did not return an order ID; this is not a confirmed order.[/bold red]")
        sys.exit(1)

    console.print(f"\n[bold green]✓ Order successfully placed on Quotex Demo! Order ID: {order_id}[/bold green]")

    # ------------------------------------------------------------------
    # 6. Monitor Trade and Verify Account Balance
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Step 6: Monitoring trade on Quotex demo account ({trade_expiration}s)...[/bold yellow]")

    # Check balance right after trade placement
    await asyncio.sleep(3.0)
    current_balance = await engine.get_account_balance()
    console.print(f"  [white]Account Balance right after order placement:[/white] [green]${current_balance:,.2f}[/green]")

    # Wait for trade settlement window
    wait_time = trade_expiration + 5
    console.print(f"  [dim]Waiting {wait_time}s to inspect settlement state...[/dim]")
    await asyncio.sleep(wait_time)

    outcome = None
    if order_id:
        outcome = broker.get_trade_outcome(order_id)
        if outcome:
            console.print(f"  [bold white]Settlement Info:[/bold white] {outcome}")

    if not outcome or outcome.get("status") != "SETTLED" or outcome.get("outcome") == "UNVERIFIED":
        console.print("[bold red]Broker did not provide a verified settlement; test cannot be marked successful.[/bold red]")
        sys.exit(1)

    # The WebSocket client can retain the pre-settlement account-balance event.
    # Reconnect before the final check so a cached value cannot misreport the
    # balance delta for a broker-confirmed order.
    await broker.close()
    if not await engine.login_broker():
        console.print("[bold red]Unable to refresh the broker balance after settlement.[/bold red]")
        sys.exit(1)
    final_balance = await engine.get_account_balance()
    expected_profit = float(outcome.get("profit", 0.0) or 0.0)
    actual_profit = final_balance - initial_balance
    if abs(actual_profit - expected_profit) > 0.02:
        console.print(
            "[bold red]Settlement and refreshed balance disagree; test cannot be marked successful. "
            f"Expected change ${expected_profit:.2f}, observed ${actual_profit:.2f}.[/bold red]"
        )
        sys.exit(1)
    console.print(f"  [bold white]Final Quotex Demo Balance:[/bold white] [bold green]${final_balance:,.2f}[/bold green]")

    console.print(
        Panel.fit(
            f"[bold green]✓ EUR/JPY AI TRAINING & LIVE DEMO TRADE TEST COMPLETED SUCCESSFULLY![/bold green]\n"
            f"[white]Asset:[/white] {target_asset}  |  [white]Order ID:[/white] {order_id}\n"
            f"[white]Initial Balance:[/white] ${initial_balance:,.2f}  ➔  [white]Final Balance:[/white] ${final_balance:,.2f}\n"
            f"[white]Model Path:[/white] models/markets/{target_asset}.joblib",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(run_eurjpy_test())

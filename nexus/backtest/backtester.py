"""Lightweight Backtester Scaffold.

Purpose:
    Provide an asynchronous backtest harness that iterates over market data
    and invokes a meta strategy + engine to simulate trade execution.

Scope:
    - Sliding evaluation over provided DataFrame rows
    - Uses meta_strategy.generate_signal() (async)
    - Executes trades through NexusEngine (simulation mode)
    - Collects simple performance metrics

Not in scope (future roadmap):
    - Transaction costs, slippage modeling
    - Walk-forward retraining windows
    - Parallel asset evaluation
    - Equity curve drawdown analytics

Usage Example:
    backtester = Backtester(window=20)
    result = await backtester.run(meta_strategy, engine, df, asset="EURUSD", timeframe=1)

The design keeps dependencies minimal so tests remain fast.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from nexus.core.engine import NexusEngine
from nexus.payouts.fetch import get_payout_for_market
from nexus.strategies.meta_strategy import MetaStrategy, SignalType


@dataclass
class ExecutedTrade:
    index: int
    signal: SignalType
    amount: float
    profit: float
    asset: str
    expiration: int
    exploratory: bool = False


@dataclass
class BacktestResult:
    total_trades: int
    total_profit: float
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_profit: float
    max_drawdown: float
    profit_factor: float
    equity_curve: List[float]
    exploratory_trades: int
    trades: List[ExecutedTrade] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Backtester:
    def __init__(self, window: int = 50, expiration: int = 60):
        self.window = max(1, int(window))
        self.expiration = int(expiration)
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    async def run(
        self,
        meta_strategy: MetaStrategy,
        engine: NexusEngine,
        data: pd.DataFrame,
        asset: str,
        timeframe: int,
        *,
        mode: str = "sim",  # "sim" (default) retains current behavior; "market" uses OHLC outcomes
    ) -> BacktestResult:
        if data.empty:
            return BacktestResult(
                total_trades=0,
                total_profit=0.0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                average_profit=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                equity_curve=[],
                exploratory_trades=0,
                trades=[],
                meta={"note": "empty dataset"},
            )

        trades: List[ExecutedTrade] = []
        wins = 0
        losses = 0
        total_profit = 0.0
        equity_curve: List[float] = [0.0]
        peak_equity = 0.0
        max_drawdown = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        exploratory_count = 0

        # Precompute expiry steps w.r.t timeframe (in minutes)
        steps = 1
        try:
            tf_sec = max(1, int(timeframe) * 60)
            steps = max(1, int(round(self.expiration / tf_sec)))
        except Exception:
            steps = 1

        # Iterate sequentially; for each index >= window-1 evaluate full history slice
        for i in range(len(data)):
            if i + 1 < self.window:
                continue
            window_df = data.iloc[i + 1 - self.window : i + 1]

            # Ask strategy for signal
            try:
                sig_tuple = await meta_strategy.generate_signal(
                    window_df, asset=asset, timeframe=timeframe
                )
            except Exception:
                # Strategy failure -> skip silently (could log in future)
                continue

            if sig_tuple is None:
                continue

            sig_type, position_fraction = sig_tuple
            # Convert position fraction to notional using base trade amount
            base_amt = engine.settings.trading.base_trade_amount
            amount = max(1.0, base_amt * max(0.01, float(position_fraction)))

            if mode == "market":
                # Determine outcome using OHLC close after expiry steps
                entry_close = (
                    float(window_df.iloc[-1]["close"])
                    if "close" in window_df.columns
                    else float(window_df.iloc[-1])
                )
                exit_idx = min(i + steps, len(data) - 1)
                exit_close = (
                    float(data.iloc[exit_idx]["close"])
                    if "close" in data.columns
                    else float(data.iloc[exit_idx])
                )
                is_win = (sig_type.value == "call" and exit_close > entry_close) or (
                    sig_type.value == "put" and exit_close < entry_close
                )
                payout = get_payout_for_market(asset, str(self.expiration)) or 80.0
                profit = amount * (float(payout) / 100.0) if is_win else -amount
                success = bool(is_win)
                # Update engine accounting as if trade executed (without placing orders)
                try:
                    engine.record_trade(success, float(profit))
                except Exception:
                    pass
            else:
                # Execute trade (simulation via engine)
                trade_result = await engine.execute_trade(
                    asset, sig_type.value, amount, self.expiration
                )
                profit = float(trade_result.get("profit", 0.0) or 0.0)
                success = bool(trade_result.get("success")) and profit >= 0

            # Track gross profit/loss for profit factor
            if float(profit) >= 0:
                gross_profit += float(profit)
            else:
                gross_loss += abs(float(profit))
            if success:
                wins += 1
            else:
                losses += 1
            total_profit += float(profit)
            equity_curve.append(equity_curve[-1] + float(profit))
            # Drawdown calculation
            if equity_curve[-1] > peak_equity:
                peak_equity = equity_curve[-1]
            dd = peak_equity - equity_curve[-1]
            if dd > max_drawdown:
                max_drawdown = dd
            exploratory_flag = getattr(meta_strategy, "last_exploratory", False)
            if exploratory_flag:
                exploratory_count += 1
            trades.append(
                ExecutedTrade(
                    index=i,
                    signal=sig_type,
                    amount=amount,
                    profit=float(profit),
                    asset=asset,
                    expiration=self.expiration,
                    exploratory=exploratory_flag,
                )
            )

        total = len(trades)
        win_rate = (wins / total) if total else 0.0
        avg_profit = (total_profit / total) if total else 0.0
        profit_factor = (
            (gross_profit / gross_loss)
            if gross_loss > 0
            else (gross_profit if gross_profit > 0 else 0.0)
        )

        bt_result = BacktestResult(
            total_trades=total,
            total_profit=round(total_profit, 4),
            winning_trades=wins,
            losing_trades=losses,
            win_rate=round(win_rate, 4),
            average_profit=round(avg_profit, 4),
            max_drawdown=round(max_drawdown, 4),
            profit_factor=round(profit_factor, 4),
            equity_curve=[round(x, 4) for x in equity_curve[1:]],
            exploratory_trades=exploratory_count,
            trades=trades,
            meta={
                "window": self.window,
                "asset": asset,
                "timeframe": timeframe,
                "mode": mode,
            },
        )

        # Persist JSON report (compact)
        try:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            report_path = self.reports_dir / f"backtest_{asset}_{stamp}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "summary": {
                            "total_trades": bt_result.total_trades,
                            "total_profit": bt_result.total_profit,
                            "win_rate": bt_result.win_rate,
                            "max_drawdown": bt_result.max_drawdown,
                            "profit_factor": bt_result.profit_factor,
                            "exploratory_trades": bt_result.exploratory_trades,
                        },
                        "meta": bt_result.meta,
                    },
                    f,
                    indent=2,
                )
            bt_result.meta["report_path"] = str(report_path)
        except Exception:
            pass

        return bt_result


__all__ = [
    "Backtester",
    "BacktestResult",
    "ExecutedTrade",
]

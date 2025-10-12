"""Backtest report aggregation utilities.

Scans the reports/ directory for backtest_*.json summary files produced by Backtester
and computes a simple leaderboard ranked by composite fitness.

Composite fitness (temporary heuristic):
    score = w_profit * norm_profit + w_pf * norm_profit_factor + w_win * win_rate - w_dd * norm_drawdown
Where normalization heuristics:
    norm_profit = clip(total_profit / profit_ref, 0, 1) with profit_ref= base_trade_amount * 50 (approx)
    norm_profit_factor = clip(profit_factor / 5, 0, 1)
    norm_drawdown = clip(max_drawdown / 50, 0, 1)

Intended as an interim layer before full evolutionary pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json

@dataclass
class ReportEntry:
    path: Path
    asset: str
    total_trades: int
    total_profit: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    exploratory_trades: int
    score: float

DEFAULT_WEIGHTS = {
    "w_profit": 0.4,
    "w_pf": 0.25,
    "w_win": 0.25,
    "w_dd": 0.15,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def compute_score(summary: Dict[str, Any], base_trade_amount: float, weights: Dict[str, float] | None = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    total_profit = float(summary.get("total_profit", 0.0))
    win_rate = float(summary.get("win_rate", 0.0))
    max_dd = float(summary.get("max_drawdown", 0.0))
    pf = float(summary.get("profit_factor", 0.0))

    profit_ref = max(1.0, base_trade_amount * 50.0)
    norm_profit = _clip(total_profit / profit_ref, 0.0, 1.0)
    norm_pf = _clip(pf / 5.0, 0.0, 1.0)
    norm_dd = _clip(max_dd / 50.0, 0.0, 1.0)

    score = (
        w["w_profit"] * norm_profit +
        w["w_pf"] * norm_pf +
        w["w_win"] * win_rate -
        w["w_dd"] * norm_dd
    )
    return round(score, 6)


def aggregate_reports(reports_dir: str | Path = "reports", base_trade_amount: float = 5.0) -> List[ReportEntry]:
    p = Path(reports_dir)
    if not p.exists():
        return []
    entries: List[ReportEntry] = []
    for file in sorted(p.glob("backtest_*.json")):
        try:
            raw = json.loads(file.read_text(encoding="utf-8"))
            summary = raw.get("summary", {})
            meta = raw.get("meta", {})
            asset = meta.get("asset", "unknown")
            score = compute_score(summary, base_trade_amount)
            entries.append(ReportEntry(
                path=file,
                asset=asset,
                total_trades=int(summary.get("total_trades", 0)),
                total_profit=float(summary.get("total_profit", 0.0)),
                win_rate=float(summary.get("win_rate", 0.0)),
                max_drawdown=float(summary.get("max_drawdown", 0.0)),
                profit_factor=float(summary.get("profit_factor", 0.0)),
                exploratory_trades=int(summary.get("exploratory_trades", 0)),
                score=score,
            ))
        except Exception:
            continue
    # Sort by score descending then profit
    entries.sort(key=lambda e: (e.score, e.total_profit), reverse=True)
    return entries

__all__ = ["ReportEntry", "aggregate_reports", "compute_score"]


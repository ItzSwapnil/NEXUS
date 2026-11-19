"""Normalize an existing research report for auditable payout and stake views.

Payouts are a broker snapshot at normalization time; they are not backfilled
as historical facts.  Replay returns are preserved while stake sizing is
recomputed using the whole-dollar, minimum-$1 policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

REPORT = Path("reports/robust_market_research.json")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    try:
        with urlopen("http://127.0.0.1:8000/api/markets", timeout=5) as response:
            catalog = json.loads(response.read().decode("utf-8"))
    except Exception:
        catalog = []
    payouts = {str(item.get("symbol", "")).upper(): float(item.get("payout") or 0.0) for item in catalog if isinstance(item, dict)}
    for market in report.get("markets", []):
        if market.get("status") != "complete":
            continue
        payout = payouts.get(str(market.get("asset", "")).upper(), 0.0)
        market["payout_percent"] = payout
        market["payout_source"] = "current broker snapshot at report normalization"
        capital = 10_000.0
        for trade in market.get("trade_records", []):
            old_amount = max(1.0, float(trade.get("amount_allocated", 1.0) or 1.0))
            gross_rate = float(trade.get("gross_pnl", 0.0) or 0.0) / old_amount
            net_rate = float(trade.get("net_pnl", 0.0) or 0.0) / old_amount
            risk = float(trade.get("risk_fraction", 0.001) or 0.001)
            before = capital * risk
            amount = float(max(1, round(before)))
            trade["trade_amount_before"] = before
            trade["trade_amount_after"] = amount
            trade["amount_allocated_original"] = old_amount
            trade["account_balance_before"] = capital
            trade["amount_allocated"] = amount
            trade["gross_pnl"] = amount * gross_rate
            trade["net_pnl"] = amount * net_rate
            trade["fees"] = float(trade.get("fees", 0.0) or 0.0) / old_amount * amount
            trade["spread"] = float(trade.get("spread", 0.0) or 0.0) / old_amount * amount
            trade["slippage"] = float(trade.get("slippage", 0.0) or 0.0) / old_amount * amount
            trade["position_size"] = amount
            trade["payout_percent"] = payout
            trade["payout_amount"] = amount * payout / 100.0 if trade.get("win") else 0.0
            trade["payment_after_payout"] = amount * (1.0 + payout / 100.0) if trade.get("win") else 0.0
            capital += float(trade["net_pnl"])
            trade["account_balance_after"] = capital
        market["final_virtual_capital"] = capital
        market["net_pnl"] = capital - 10_000.0
        market["roi"] = capital / 10_000.0 - 1.0
        market["trade_amount_policy"] = "whole-dollar amounts, minimum $1"
    report["payout_note"] = "Payout fields are current broker snapshots captured during normalization, not historical payout guarantees."
    report["trade_amount_policy"] = "whole-dollar amounts, minimum $1"
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "payout_markets": sum(bool(payouts.get(str(m.get('asset', '')).upper(), 0.0)) for m in report.get('markets', [])), "policy": report["trade_amount_policy"]}, indent=2))


if __name__ == "__main__":
    main()

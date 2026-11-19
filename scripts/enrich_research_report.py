"""Add specification-required confidence buckets to an existing research report."""

from __future__ import annotations

import json
from pathlib import Path

from nexus.features.feature_engine import get_feature_provider_catalog


def buckets(records: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        chosen = [r for r in records if low <= float(r["confidence"]) <= high] if high == 1 else [r for r in records if low <= float(r["confidence"]) < high]
        wins = sum(bool(r["win"]) for r in chosen)
        allocated = sum(float(r["amount_allocated"]) for r in chosen)
        net = sum(float(r["net_pnl"]) for r in chosen)
        gross = sum(float(r["gross_pnl"]) for r in chosen)
        result.append({"range": f"{low:.0%}-{high:.0%}", "trades": len(chosen), "wins": wins, "losses": len(chosen) - wins, "win_rate": wins / len(chosen) if chosen else None, "average_confidence": sum(float(r["confidence"]) for r in chosen) / len(chosen) if chosen else None, "average_amount": allocated / len(chosen) if chosen else 0.0, "total_amount_allocated": allocated, "gross_pnl": gross, "net_pnl": net, "roi": net / allocated if allocated else 0.0, "expectancy": net / len(chosen) if chosen else 0.0})
    return result


def threshold_quality(records: list[dict[str, object]], minimum_trades: int = 20) -> dict[str, object]:
    """Measure confidence ordering only where every threshold has evidence.

    Empty or tiny high-confidence buckets must not make the confidence claim
    appear monotonic.  They are explicitly reported as insufficient evidence.
    """
    thresholds = (0.50, 0.60, 0.70, 0.80, 0.90, 0.99)
    values = []
    for threshold in thresholds:
        chosen = [r for r in records if float(r.get("confidence", 0.0)) >= threshold]
        values.append({"threshold": threshold, "trades": len(chosen), "win_rate": sum(bool(r.get("win")) for r in chosen) / len(chosen) if chosen else None})
    insufficient = [item["threshold"] for item in values if int(item["trades"]) < minimum_trades]
    observed = [float(item["win_rate"]) for item in values if int(item["trades"]) >= minimum_trades]
    monotonic = (
        not insufficient
        and len(observed) == len(thresholds)
        and all(right + 1e-12 >= left for left, right in zip(observed, observed[1:], strict=False))
    )
    return {
        "monotonic_oos": monotonic,
        "minimum_trades_required_per_threshold": minimum_trades,
        "insufficient_thresholds": insufficient,
        "thresholds": values,
    }


def main() -> None:
    path = Path("reports/robust_market_research.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    complete = [market for market in report.get("markets", []) if market.get("status") == "complete"]
    for market in complete:
        market["confidence_buckets"] = buckets(market.get("trade_records", []))
        market["confidence_quality"] = threshold_quality(market.get("trade_records", []))
        if not market["confidence_quality"]["monotonic_oos"]:
            reasons = list(market.get("promotion_rejection_reasons", []))
            if "held-out confidence thresholds are not monotonic" not in reasons:
                reasons.append("held-out confidence thresholds are not monotonic")
            market["promotion_rejection_reasons"] = reasons
            market["promotion_status"] = "reject"
    report["report_schema"] = "robust_research_v2"
    report["feature_provider_catalog"] = get_feature_provider_catalog()
    report["feature_selection"] = "causal sequence representation over up to 256 namespaced TA features; normalization and model fitting are fold-local"
    report["completed_markets"] = len(complete)
    report["error_markets"] = [market["asset"] for market in report.get("markets", []) if market.get("status") == "error"]
    report["total_oos_trades"] = sum(int(market.get("trades", 0)) for market in complete)
    report["total_net_pnl_independent_market_replays"] = sum(float(market.get("net_pnl", 0.0)) for market in complete)
    report["promoted_markets"] = [market["asset"] for market in complete if market.get("promotion_status") == "promote"]
    report["confidence_quality"] = {
        "monotonic_markets": sum(bool(market.get("confidence_quality", {}).get("monotonic_oos")) for market in complete),
        "evaluated_markets": len(complete),
        "definition": "higher confidence must have non-decreasing held-out win rate at 50/60/70/80/90/99 percent thresholds",
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "completed_markets": len(complete), "total_oos_trades": report["total_oos_trades"], "promoted": len(report["promoted_markets"])}, indent=2))


if __name__ == "__main__":
    main()

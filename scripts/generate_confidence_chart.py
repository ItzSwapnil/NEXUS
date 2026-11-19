"""Generate a dependency-free confidence versus win-rate SVG from a replay report."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    report_path = Path("reports/broker_active_market_backtest.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    trades = [t for market in report["markets"] for t in market.get("trade_records", [])]
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    values = []
    for threshold in thresholds:
        selected = [t for t in trades if float(t["confidence"]) >= threshold]
        values.append((100 * sum(bool(t["win"]) for t in selected) / len(selected) if selected else 0.0, len(selected)))
    width, height = 1000, 520
    left, bottom, chart_w, chart_h = 80, 410, 860, 330
    bar_w = chart_w / len(values) * 0.72
    bars = []
    labels = []
    for i, ((rate, count), threshold) in enumerate(zip(values, thresholds, strict=True)):
        x = left + i * chart_w / len(values) + (chart_w / len(values) - bar_w) / 2
        y = bottom - rate / 100 * chart_h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bottom-y:.1f}" fill="#3478c9"/><text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" font-size="13">{rate:.1f}% (n={count})</text>')
        labels.append(f'<text x="{x+bar_w/2:.1f}" y="440" text-anchor="middle" font-size="13">{threshold:.0%}+</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="500" y="35" text-anchor="middle" font-size="22" font-family="sans-serif">Local replay win rate by model confidence</text><line x1="{left}" y1="{bottom}" x2="940" y2="{bottom}" stroke="#333"/><line x1="{left}" y1="80" x2="{left}" y2="{bottom}" stroke="#333"/><line x1="{left}" y1="245" x2="940" y2="245" stroke="#cc4444" stroke-dasharray="6 5"/><text x="945" y="249" font-size="12" fill="#cc4444">50%</text>{''.join(bars)}{''.join(labels)}<text x="500" y="495" text-anchor="middle" font-size="15">Confidence bucket</text><text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle" font-size="15">Win rate (%)</text></svg>'''
    chart = Path("reports/broker_active_confidence_win_rate.svg")
    chart.write_text(svg, encoding="utf-8")
    report["chart"] = str(chart)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"chart": str(chart), "trades": len(trades), "buckets": values}, indent=2))


if __name__ == "__main__":
    main()

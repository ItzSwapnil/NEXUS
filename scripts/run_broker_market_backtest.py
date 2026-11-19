"""No-order backtest over the broker's refreshed active market catalog."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nexus.core.engine import NexusEngine
from nexus.intelligence.adaptive import (
    AdaptiveBacktestRunner,
    AdaptiveMarketIntelligence,
    AdaptiveSequenceEncoder,
    AdaptiveTrainer,
)
from nexus.utils.config import load_runtime_settings


async def main() -> None:
    engine = NexusEngine(load_runtime_settings(), demo_mode=True, auto_login=False)
    if not await engine.login_broker():
        raise RuntimeError("refreshed broker login failed")
    catalog = await engine._broker.get_assets_with_payouts_async()
    assets = sorted({str(x.get("symbol") or x.get("name")) for x in catalog or [] if isinstance(x, dict) and x.get("active", True)})
    provider = engine._broker
    reports: list[dict[str, object]] = []
    for number, asset in enumerate(assets, 1):
        try:
            raw = await provider.get_candles_async(asset, 60, 720)
            base_frame = pd.DataFrame(raw).dropna().reset_index(drop=True)
            # Let the local policy choose between chart resolutions using
            # training data only; the holdout remains untouched.
            candidates = {1: base_frame}
            if "time" in base_frame:
                indexed = base_frame.assign(_dt=pd.to_datetime(base_frame["time"], unit="s", errors="coerce")).dropna(subset=["_dt"]).set_index("_dt")
                candidates[5] = indexed[["open", "high", "low", "close", "volume"]].resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index(drop=True)
            timeframe, frame = 1, base_frame
            best_score = -float("inf")
            for candidate_tf, candidate in candidates.items():
                if len(candidate) < 180:
                    continue
                n = int(len(candidate) * 0.8)
                ret = candidate["close"].pct_change().to_numpy()[1:n]
                momentum = candidate["close"].pct_change(8).to_numpy()[1:n]
                score = float(np.nanmean(np.sign(momentum) * ret - 3.0 / 10_000.0))
                if score > best_score:
                    timeframe, frame, best_score = candidate_tf, candidate, score
            if len(frame) < 180:
                reports.append({"asset": asset, "status": "insufficient_data", "candles": len(frame)})
                continue
            train_rows = int(len(frame) * 0.8)
            np.random.seed(7)
            torch.manual_seed(7)
            ai = AdaptiveMarketIntelligence(":memory:")
            bundle = ai.representation.features(frame)
            ai.representation.fit(type(bundle)(bundle.values[:train_rows], bundle.names, bundle.schema_hash, train_rows))
            values = ai.representation.transform(bundle)
            close = frame["close"].to_numpy(float)
            returns = np.diff(close) / close[:-1]
            ref = float(np.std(returns[: train_rows - 1])) or 1e-6
            sequences, targets, regimes, vols = [], [], [], []
            for end in range(64, train_rows - 1):
                sequences.append(values[end - 64 : end])
                targets.append(float(close[end + 1] / close[end] - 1.0))
                local = returns[max(0, end - 16) : end]
                regimes.append(3 if np.std(local) > 2 * ref else 0 if np.mean(local) > 0 else 1 if np.mean(local) < 0 else 2)
                vols.append(float(np.std(local)) if len(local) else ref)
            x = torch.tensor(np.asarray(sequences), dtype=torch.float32)
            model = AdaptiveSequenceEncoder(x.shape[-1], 64)
            trainer = AdaptiveTrainer(model)
            trainer.train_batch(x, torch.tensor(targets), torch.tensor(regimes), torch.tensor(vols))
            ai.representation.encoder = model
            result = AdaptiveBacktestRunner(ai, cost_bps=2.0, slippage_bps=1.0).run(frame, asset, start_at=train_rows)
            reports.append({"asset": asset, "status": "complete", "chart_timeframe_minutes": timeframe, "trade_time_seconds": int(timeframe * 60), "candles": len(frame), "train_rows": train_rows, "test_rows": len(frame) - train_rows, **result})
            print(f"[{number}/{len(assets)}] {asset}: trades={result['trades']} net={float(result['net_return']):+.4%} dd={float(result['max_drawdown']):.4%}", flush=True)
        except Exception as exc:
            reports.append({"asset": asset, "status": "error", "error": str(exc)[:240]})
            print(f"[{number}/{len(assets)}] {asset}: ERROR {exc}", flush=True)
    output = Path("reports/broker_active_market_backtest.json")
    output.write_text(json.dumps({"mode": "local_replay", "catalog_source": "refreshed_broker", "active_markets": len(assets), "timeframe_selection": "training-only policy comparison: 1m vs 5m", "training_hours_target": 10, "evaluation_hours_target": 2, "total_window_hours": 12, "orders_placed": 0, "markets": reports}, indent=2), encoding="utf-8")
    complete = [x for x in reports if x.get("status") == "complete"]
    all_trades = [trade for item in complete for trade in item.get("trade_records", [])]
    chart = Path("reports/broker_active_confidence_win_rate.png")
    try:
        import matplotlib.pyplot as plt
        bins = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
        labels = [f"{bins[i]:.0%}-{bins[i + 1]:.0%}" for i in range(len(bins) - 1)]
        rates = [100.0 * np.mean([bool(t["win"]) for t in all_trades if bins[i] <= float(t["confidence"]) < bins[i + 1]]) if any(bins[i] <= float(t["confidence"]) < bins[i + 1] for t in all_trades) else np.nan for i in range(len(bins) - 1)]
        counts = [sum(bins[i] <= float(t["confidence"]) < bins[i + 1] for t in all_trades) for i in range(len(bins) - 1)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, rates, color="#3478c9")
        ax.axhline(50, color="#cc4444", linestyle="--", label="50% reference")
        for i, (rate, count) in enumerate(zip(rates, counts, strict=True)):
            if np.isfinite(rate):
                ax.text(i, rate + 2, f"{rate:.1f}%\n(n={count})", ha="center", fontsize=8)
        ax.set(title="Local replay win rate by model confidence", xlabel="Confidence bucket", ylabel="Win rate (%)")
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart, dpi=140)
        plt.close(fig)
    except Exception as exc:
        chart = Path(f"chart unavailable: {exc}")
    print(json.dumps({"report": str(output), "chart": str(chart), "active_markets": len(assets), "completed": len(complete), "total_trades": sum(int(x.get("trades", 0)) for x in complete), "winning_trades": sum(int(x.get("winning_trades", 0)) for x in complete), "losing_trades": sum(int(x.get("losing_trades", 0)) for x in complete), "win_rate": sum(int(x.get("winning_trades", 0)) for x in complete) / sum(int(x.get("trades", 0)) for x in complete) if sum(int(x.get("trades", 0)) for x in complete) else 0.0, "mean_net_return": float(np.mean([float(x["net_return"]) for x in complete])) if complete else 0.0, "orders_placed": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

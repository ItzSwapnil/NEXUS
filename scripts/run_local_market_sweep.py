"""Train and evaluate the adaptive core across every configured market locally."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import torch

from nexus.adapters.quotex import COMMON_ASSETS
from nexus.data.provider import DataProvider
from nexus.intelligence.adaptive import (
    AdaptiveBacktestRunner,
    AdaptiveMarketIntelligence,
    AdaptiveSequenceEncoder,
    AdaptiveTrainer,
    MarketRepresentation,
    RepresentationConfig,
)


async def main() -> None:
    assets = COMMON_ASSETS
    rows, train_rows = 360, 288  # 24h and 18h at five-minute candles
    provider = DataProvider()
    reports: list[dict[str, object]] = []
    for number, asset in enumerate(assets, 1):
        frame = await provider.get_ohlcv(asset, 5, rows, source="synthetic")
        frame = frame.dropna().reset_index(drop=True)
        np.random.seed(7)
        torch.manual_seed(7)
        representation = MarketRepresentation(RepresentationConfig(seed=7))
        bundle = representation.features(frame)
        representation.fit(type(bundle)(bundle.values[:train_rows], bundle.names, bundle.schema_hash, train_rows))
        values = representation.transform(bundle)
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
        y = torch.tensor(targets, dtype=torch.float32)
        r = torch.tensor(regimes, dtype=torch.long)
        v = torch.tensor(vols, dtype=torch.float32)
        losses = []
        for _ in range(2):
            batch_losses = [trainer.train_batch(x[i : i + 32], y[i : i + 32], r[i : i + 32], v[i : i + 32]) for i in range(0, len(x), 32)]
            losses.append(float(np.mean(batch_losses)))
        ai = AdaptiveMarketIntelligence(":memory:")
        ai.representation = representation
        ai.representation.encoder = model
        result = AdaptiveBacktestRunner(ai, cost_bps=2.0, slippage_bps=1.0).run(frame, asset, start_at=train_rows)
        reports.append({"asset": asset, "train_rows": train_rows, "test_rows": len(frame) - train_rows, "loss_start": losses[0], "loss_end": losses[-1], **result})
        print(f"[{number}/{len(assets)}] {asset}: trades={result['trades']} net={float(result['net_return']):+.4%} dd={float(result['max_drawdown']):.4%}", flush=True)
    output = Path("reports/local_market_sweep.json")
    output.write_text(json.dumps({"mode": "local", "assets": len(reports), "training_hours": 18, "evaluation_hours": 6, "total_window_hours": 24, "timeframe_minutes": 5, "orders_placed": 0, "markets": reports}, indent=2), encoding="utf-8")
    net = float(np.mean([float(item["net_return"]) for item in reports])) if reports else 0.0
    print(json.dumps({"report": str(output), "markets": len(reports), "mean_net_return": net, "orders_placed": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

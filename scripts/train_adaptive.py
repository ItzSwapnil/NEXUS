"""Reproducible adaptive encoder training on local/synthetic OHLCV data.

Use a real CSV through ``--csv`` when available.  The default is synthetic and
is explicitly labelled as such in the artifact metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nexus.intelligence.adaptive import (
    AdaptiveSequenceEncoder,
    AdaptiveTrainer,
    MarketRepresentation,
    RepresentationConfig,
)


async def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    if args.broker:
        from nexus.core.engine import NexusEngine
        from nexus.utils.config import load_runtime_settings

        engine = NexusEngine(load_runtime_settings(), demo_mode=True, auto_login=False)
        if not await engine.login_broker():
            raise RuntimeError("broker login failed; refusing to substitute synthetic candles")
        candles = await engine._broker.get_candles_async(args.asset, args.timeframe * 60, args.rows)
        if not candles:
            raise RuntimeError("broker returned no candles")
        return pd.DataFrame(candles)
    # Reuse the provider's deterministic synthetic generator without broker access.
    from nexus.data.provider import DataProvider

    return await DataProvider().get_ohlcv(args.asset, args.timeframe, args.rows, source="synthetic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument("--timeframe", type=int, default=5)
    parser.add_argument("--rows", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--csv")
    parser.add_argument("--broker", action="store_true", help="fetch recent candles from configured broker")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    frame = asyncio.run(load_data(args)).dropna().reset_index(drop=True)
    representation = MarketRepresentation(RepresentationConfig(seed=args.seed))
    bundle = representation.features(frame)
    split = max(128, int(len(bundle.values) * 0.8))
    representation.fit(type(bundle)(bundle.values[:split], bundle.names, bundle.schema_hash, split))
    values = representation.transform(bundle)
    lookback = representation.config.lookback
    sequences, returns, regimes, volatility = [], [], [], []
    close = frame["close"].to_numpy(float)
    ret = np.diff(close) / close[:-1]
    vol_ref = float(np.std(ret[: max(1, split - 1)])) or 1e-6
    for end in range(lookback, split - 1):
        sequences.append(values[end - lookback : end])
        future = float(close[end + 1] / close[end] - 1.0)
        returns.append(future)
        window = ret[max(0, end - 16) : end]
        regimes.append(3 if np.std(window) > 2 * vol_ref else 0 if np.mean(window) > 0 else 1 if np.mean(window) < 0 else 2)
        volatility.append(float(np.std(window)) if len(window) else vol_ref)
    x = torch.tensor(np.asarray(sequences), dtype=torch.float32)
    y = torch.tensor(returns, dtype=torch.float32)
    regime_y = torch.tensor(regimes, dtype=torch.long)
    vol_y = torch.tensor(volatility, dtype=torch.float32)
    requested_device = os.getenv("NEXUS_SEQUENCE_DEVICE", "auto").lower()
    device = torch.device("cuda" if requested_device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    model = AdaptiveSequenceEncoder(x.shape[-1], representation.config.embedding_dim).to(device)
    x, y, regime_y, vol_y = x.to(device), y.to(device), regime_y.to(device), vol_y.to(device)
    trainer = AdaptiveTrainer(model)
    losses = []
    for _ in range(args.epochs):
        epoch_losses = []
        for start in range(0, len(x), args.batch_size):
            end = start + args.batch_size
            epoch_losses.append(trainer.train_batch(x[start:end], y[start:end], regime_y[start:end], vol_y[start:end]))
        losses.append(float(np.mean(epoch_losses)))
    output = Path("models/adaptive")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{args.asset.upper()}_{args.timeframe}m.pt"
    torch.save({"model": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, checkpoint)
    metadata = {"asset": args.asset, "timeframe": args.timeframe, "rows": len(frame), "source": "broker" if args.broker else "csv" if args.csv else "synthetic", "feature_schema": bundle.schema_hash, "input_dim": int(x.shape[-1]), "lookback": lookback, "seed": args.seed, "epochs": args.epochs, "losses": losses, "train_rows": split, "device": str(device), "deterministic_backend": True, "data_resolution_policy": "CSV broker candles; no synthetic fallback when --csv is supplied"}
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), **metadata}, indent=2))


if __name__ == "__main__":
    main()

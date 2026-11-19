"""Fresh bounded research run with one broker session and eight local lanes.

Broker I/O stays serialized because the Quotex session is stateful.  Once a
fresh market batch is collected, evaluation/training runs in local worker
threads.  Each batch is processed once and the program exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from nexus.core.engine import NexusEngine
from nexus.features.feature_engine import get_feature_provider_catalog
from nexus.utils.config import load_runtime_settings

# The scripts directory is intentionally not a package; add the repository
# root so this runner can reuse the tested research primitives when invoked by
# file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_robust_research import (
    BrokerAuthorizationError,
    MarketDataUnavailableError,
    _chronological,
    _fetch_best_supported_resolution,
    _fetch_historical_range,
    _observed_timeframe_seconds,
    _refresh_login_browser,
    evaluate,
)


def _minimum_payout() -> float:
    parser = argparse.ArgumentParser(description="Run research-only market replay")
    parser.add_argument(
        "--min-payout",
        type=float,
        default=None,
        help="Only include active broker markets with payout >= this percent",
    )
    args = parser.parse_args()
    value = args.min_payout if args.min_payout is not None else float(os.getenv("NEXUS_MIN_PAYOUT", "0"))
    if not 0.0 <= value <= 100.0:
        raise ValueError("minimum payout must be between 0 and 100 percent")
    return value


async def main() -> None:
    min_payout = _minimum_payout()
    engine = NexusEngine(load_runtime_settings(), demo_mode=True, auto_login=False)
    if not await engine.login_broker():
        if not await _refresh_login_browser(Path(__file__).resolve().parents[1]) or not await engine.login_broker():
            raise BrokerAuthorizationError("broker login failed after login refresh")
    catalog = await engine._broker.get_assets_with_payouts_async()
    eligible = [
        item for item in catalog or []
        if isinstance(item, dict)
        and item.get("active", True)
        and float(item.get("payout") or 0.0) >= min_payout
    ]
    assets = sorted({str(item.get("symbol") or item.get("name")) for item in eligible})
    excluded = sorted({str(item.get("symbol") or item.get("name")) for item in catalog or [] if isinstance(item, dict) and item.get("active", True) and float(item.get("payout") or 0.0) < min_payout})
    batch_size = max(1, min(8, int(os.getenv("NEXUS_BATCH_SIZE", "8"))))
    workers = max(1, min(8, int(os.getenv("NEXUS_LOCAL_WORKERS", "8"))))
    history_days = max(7, int(os.getenv("NEXUS_HISTORY_DAYS", "7")))
    history_seconds = history_days * 86400
    history_limit = min(3200, max(3000, int(os.getenv("NEXUS_HISTORY_LIMIT", "3200"))))
    timeout = float(os.getenv("NEXUS_HISTORY_TIMEOUT", "30"))
    report = Path("reports/robust_market_research.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    payout_map = {str(item.get("symbol", "")).upper(): float(item.get("payout") or 0.0) for item in catalog if isinstance(item, dict)}

    def checkpoint() -> None:
        report.write_text(json.dumps({
            "mode": "research_only_batched_local_replay",
            "catalog_source": "refreshed_broker",
            "assets": len(assets),
            "run_assets": len(assets),
            "minimum_payout_percent": min_payout,
            "excluded_low_payout_markets": len(excluded),
            "completed_or_recorded": len(results),
            "batch_size": batch_size,
            "local_workers": workers,
            "orders_placed": 0,
            "feature_provider_catalog": get_feature_provider_catalog(),
            "timing_policy": "measured broker resolution only; no fabricated sub-minute data",
            "markets": results,
        }, indent=2), encoding="utf-8")

    async def collect(asset: str) -> tuple[str, pd.DataFrame, str]:
        preferred = int(os.getenv("NEXUS_OTC_TIMEFRAME_SECONDS", "60")) if "otc" in asset.lower() else 60
        try:
            raw, source_seconds = await _fetch_historical_range(engine, asset, preferred, history_seconds, timeout)
            source = "refreshed_broker_historical_range"
        except (MarketDataUnavailableError, asyncio.TimeoutError):
            raw, source_seconds = await _fetch_best_supported_resolution(engine, asset, preferred, history_limit, timeout)
            source = "refreshed_broker_legacy_limit"
        frame = _chronological(pd.DataFrame(raw).dropna().reset_index(drop=True))
        observed = _observed_timeframe_seconds(frame, preferred)
        if len(frame) < 3000:
            raise MarketDataUnavailableError(f"only {len(frame)} candles for {asset}")
        frame.attrs["asset"] = asset
        frame.attrs["timeframe_seconds"] = observed
        frame.attrs["payout_percent"] = payout_map.get(asset.upper(), 0.0)
        return asset, frame, f"{source}; observed_resolution={observed}; source_resolution={source_seconds}"

    for offset in range(0, len(assets), batch_size):
        batch = assets[offset:offset + batch_size]
        frames: list[tuple[str, pd.DataFrame, str]] = []
        for asset in batch:
            try:
                frames.append(await collect(asset))
                print(f"[batch {offset // batch_size + 1}] collected {asset}", flush=True)
            except Exception as exc:
                error_type = "authentication" if isinstance(exc, BrokerAuthorizationError) else "market_data_unavailable" if isinstance(exc, MarketDataUnavailableError) else "unexpected"
                results.append({"asset": asset, "status": "error", "error_type": error_type, "candles": 0, "orders_placed": 0, "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
                checkpoint()
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nexus-learn") as pool:
            futures = [loop.run_in_executor(pool, evaluate, frame) for _, frame, _ in frames]
            evaluated = await asyncio.gather(*futures, return_exceptions=True)
        for (asset, frame, source), result in zip(frames, evaluated, strict=True):
            if isinstance(result, Exception):
                results.append({"asset": asset, "status": "error", "error_type": "unexpected", "candles": len(frame), "orders_placed": 0, "error": f"{type(result).__name__}: {str(result)[:220]}"})
            else:
                results.append({"asset": asset, "status": "complete", "candles": len(frame), "history_source": source, "orders_placed": 0, **result})
            print(f"[batch {offset // batch_size + 1}] evaluated {asset}", flush=True)
        checkpoint()
    close = getattr(getattr(engine, "_broker", None), "close", None)
    if close is not None:
        await close()
    print(json.dumps({"report": str(report), "assets": len(assets), "minimum_payout_percent": min_payout, "excluded_low_payout_markets": len(excluded), "completed": sum(x.get("status") == "complete" for x in results), "orders_placed": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

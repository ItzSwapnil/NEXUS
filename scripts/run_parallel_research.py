"""Run disjoint research shards concurrently and merge their checkpoints.

This is a bounded batch runner: each invocation processes the catalog once,
each shard owns a disjoint slice, and the process exits after all shards finish.
It never enables execution or submits broker orders.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def _run_shard(index: int, total: int, root: Path, base_env: dict[str, str]) -> tuple[int, int, str]:
    env = {**base_env, "NEXUS_RESEARCH_SHARDS": str(total), "NEXUS_RESEARCH_SHARD_INDEX": str(index), "NEXUS_RESEARCH_REPORT": str(root / "reports" / f"robust_market_research_shard_{index}.json")}
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(root / "scripts" / "run_robust_research.py"),
        cwd=root, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    async for line in process.stdout:
        print(f"[shard {index + 1}/{total}] {line.decode(errors='replace').rstrip()}", flush=True)
    return index, await process.wait(), env["NEXUS_RESEARCH_REPORT"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    workers = max(1, min(8, args.workers))
    root = Path(__file__).resolve().parents[1]
    base_env = os.environ.copy()
    base_env["NEXUS_RESEARCH_ONLY_ERRORS"] = "0"
    base_env["NEXUS_FORCE_RETRAIN"] = "1"
    # One account/session refresh for the whole batch.  Individual workers
    # must not launch competing browser logins against the same WebSocket.
    if base_env.get("NEXUS_AUTO_BROWSER_REFRESH", "1").lower() not in {"0", "false", "no"}:
        login = await asyncio.create_subprocess_exec(
            "uv", "run", "python", str(root / "scripts" / "login_browser.py"),
            cwd=root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await login.communicate()
        print(output.decode(errors="replace"), flush=True)
        if login.returncode != 0:
            raise RuntimeError("broker login refresh failed before batch")
    base_env["NEXUS_AUTO_BROWSER_REFRESH"] = "0"
    tasks = [_run_shard(index, workers, root, base_env) for index in range(workers)]
    results = await asyncio.gather(*tasks)
    reports: list[dict[str, object]] = []
    for index, code, report_path in results:
        if code != 0:
            raise RuntimeError(f"research shard {index} exited with code {code}")
        reports.append(json.loads(Path(report_path).read_text(encoding="utf-8")))
    merged_markets = [market for report in reports for market in report.get("markets", []) if isinstance(market, dict)]
    merged_markets.sort(key=lambda market: str(market.get("asset", "")))
    output = root / "reports" / "robust_market_research.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        **reports[0],
        "assets": max(int(report.get("assets", 0)) for report in reports),
        "run_assets": len(merged_markets),
        "completed_or_recorded": len(merged_markets),
        "parallel_workers": workers,
        "orders_placed": 0,
        "markets": merged_markets,
    }
    output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "workers": workers, "markets": len(merged_markets), "orders_placed": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

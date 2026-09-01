from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, List

from nexus.adapters.quotex_adapter import QuotexAdapter
from nexus.catalog.ingest import fetch_live_catalog, get_market_by_symbol, get_market_catalog
from nexus.payouts.fetch import get_payout_for_market


async def _fetch_assets(
    email: str | None, password: str | None, demo: bool, otc_only: bool
) -> dict[str, Any]:
    qa = QuotexAdapter(email=email or "", password=password or "", demo_mode=bool(demo))
    connected = await qa.connect()

    # First try enriched assets with payouts from adapter
    enriched: List[dict[str, Any]] = []
    try:
        if hasattr(qa, "get_assets_with_payouts_async"):
            enriched = await qa.get_assets_with_payouts_async()  # type: ignore[attr-defined]
    except Exception:
        enriched = []

    # Determine base symbol list
    if enriched:
        base_symbols = [str(x.get("symbol")) for x in enriched if str(x.get("symbol"))]
    else:
        try:
            base_symbols = await qa.get_available_assets()
        except Exception:
            base_symbols = []

    # OTC inference from symbol patterns
    otc_sym = [s for s in base_symbols if "OTC" in str(s).upper()]

    # Refresh catalog (ingests enriched payouts when available)
    try:
        await fetch_live_catalog(qa)
        catalog = await get_market_catalog()
        otc_from_catalog = [m.symbol for m in catalog if getattr(m, "otc", False)]
    except Exception:
        otc_from_catalog = []

    # Combine unique OTC
    all_otc: list[str] = []
    seen: set[str] = set()
    for s in otc_sym + otc_from_catalog:
        u = str(s).strip()
        if u and u not in seen:
            seen.add(u)
            all_otc.append(u)

    # Prepare output selection
    out_assets = all_otc if otc_only else base_symbols

    # Build payouts for visible assets using enriched first, then catalog
    assets_with_payouts: list[dict[str, Any]] = []
    # Map from enriched for quick lookup
    enriched_map = {
        str(x.get("symbol")): float(x.get("payout", 0.0) or 0.0)
        for x in enriched
        if x.get("symbol")
    }

    for sym in out_assets:
        payout = float(enriched_map.get(sym, 0.0))
        if payout <= 0.0:
            # Try catalog resolution
            m = None
            try:
                m = get_market_by_symbol(sym)
            except Exception:
                m = None
            if m is None and "OTC" in str(sym).upper():
                base = (
                    str(sym)
                    .replace("-OTC", "")
                    .replace("OTC_", "")
                    .replace("OTC-", "")
                    .replace("_OTC", "")
                    .strip()
                )
                for variant in (f"OTC_{base}", f"{base}-OTC", base):
                    m = get_market_by_symbol(variant)
                    if m:
                        break
            if m is not None:
                try:
                    payout = float(get_payout_for_market(m, expiration="60"))
                except Exception:
                    payout = float(getattr(m, "display_payout_percent", 0.0) or 0.0)
        assets_with_payouts.append({"symbol": sym, "payout_percent": round(payout, 2)})

    return {
        "connected": bool(connected),
        "count": len(out_assets),
        "assets": out_assets,  # preserved for backward compatibility
        "assets_with_payouts": assets_with_payouts,
        "demo": bool(demo),
        "email_set": bool(email),
        "otc_only": bool(otc_only),
        "otc_detected": all_otc,
    }


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="List available Quotex assets (live) with payout %")
    p.add_argument("--email", help="Quotex email (or use QUOTEX__EMAIL)")
    p.add_argument("--password", help="Quotex password (or use QUOTEX__PASSWORD)")
    p.add_argument("--demo", action="store_true", help="Use practice (demo) account")
    p.add_argument("--json", action="store_true", help="Print JSON output")
    p.add_argument("--otc-only", action="store_true", help="Show only OTC instruments")
    args = p.parse_args()

    email = args.email or os.getenv("QUOTEX__EMAIL") or os.getenv("QUOTEX_EMAIL")
    password = args.password or os.getenv("QUOTEX__PASSWORD") or os.getenv("QUOTEX_PASSWORD")

    result = asyncio.run(_fetch_assets(email, password, bool(args.demo), bool(args.otc_only)))

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        if not result["connected"]:
            print("Not connected. Check credentials or network. Showing assets if available...")
        title = "OTC Assets" if args.otc_only else "Assets"
        print(f"{title} ({result['count']}):")
        for item in result["assets_with_payouts"]:
            sym = item["symbol"]
            pct = item.get("payout_percent", 0.0)
            print(f"{sym} - {pct:.0f}%")


if __name__ == "__main__":  # pragma: no cover
    main()

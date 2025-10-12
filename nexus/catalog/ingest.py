"""Market catalog ingestion (stable placeholder + optional live fetch).

Purpose:
- Provide deterministic placeholder markets for tests
- Expose get_market_catalog() and get_market_by_symbol()
- Optionally fetch live markets via adapter when explicitly called
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import asyncio


@dataclass
class Market:
    symbol: str
    asset_type: str
    display_payout_percent: float
    active: bool = True
    otc: bool = False
    payout_per_expiration: Dict[str, float] | None = None
    metadata: Dict[str, float | int | str] = field(default_factory=dict)

    def effective_payout(self, expiration: Optional[str] = None) -> float:
        if self.payout_per_expiration and expiration and expiration in self.payout_per_expiration:
            return float(self.payout_per_expiration[expiration])
        return float(self.display_payout_percent)


# Deterministic placeholder catalog (kept static for tests)
_PLACEHOLDER: List[Market] = [
    Market("EURUSD", "Forex", 87.0, payout_per_expiration={"60": 87.0, "120": 86.0}),
    Market("GBPUSD", "Forex", 82.0, payout_per_expiration={"60": 82.0, "300": 84.0}),
    Market("USDJPY", "Forex", 78.0, payout_per_expiration={"60": 78.0}),  # below threshold example
    Market("OTC_EURUSD", "OTC", 90.0, otc=True, payout_per_expiration={"60": 90.0}),
]
_symbol_index: Dict[str, Market] = {m.symbol: m for m in _PLACEHOLDER}
_catalog_lock = asyncio.Lock()


def get_market_by_symbol(symbol: str) -> Optional[Market]:
    return _symbol_index.get(symbol)


async def get_market_catalog(force_refresh: bool = False) -> List[Market]:  # force_refresh kept for API compatibility
    async with _catalog_lock:
        # Return a deep-ish copy so callers cannot mutate global state
        return [
            Market(
                symbol=m.symbol,
                asset_type=m.asset_type,
                display_payout_percent=m.display_payout_percent,
                active=m.active,
                otc=m.otc,
                payout_per_expiration=dict(m.payout_per_expiration) if m.payout_per_expiration else None,
                metadata=dict(m.metadata),
            )
            for m in _PLACEHOLDER
        ]


async def refresh_catalog() -> None:  # retained for backward compatibility (no-op)
    return None


def _normalize_payout_value(val: Any) -> float:
    try:
        p = float(val)
        # normalize fractional to percent
        if 0.0 < p <= 1.0:
            p *= 100.0
        # guard negative or nan
        if p < 0 or p != p:  # noqa: PLW0127 (nan check)
            return 0.0
        return p
    except Exception:
        return 0.0


def _extract_payout_from_record(rec: dict[str, Any]) -> float:
    # Accept several shapes
    if not isinstance(rec, dict):
        return 0.0
    # direct fields
    for key in ("payout", "payout_percent", "profit", "rate", "yield", "payment", "turbo_payment"):
        if key in rec and isinstance(rec[key], (int, float)):
            return _normalize_payout_value(rec[key])
    # nested dicts under profit/payout
    for key in ("profit", "payout", "rate", "yield"):
        v = rec.get(key)
        if isinstance(v, dict):
            for tkey in ("1M", "1", 1, "60", 60, "M1", "m1", "MIN1", "1min", "01", "0:01"):
                if tkey in v and isinstance(v[tkey], (int, float)):
                    return _normalize_payout_value(v[tkey])
    return 0.0


async def fetch_live_catalog(adapter: Any) -> List[Market]:  # pragma: no cover (network interaction)
    """Replace placeholder with live markets from adapter if possible.

    Adapter may expose get_assets(), get_available_assets(), or get_available_assets_async().
    Prefer enriched methods that include payout when available.
    """
    # 1) Prefer enriched API exposing payout
    enriched = None
    for name in ("get_assets_with_payouts_async", "get_assets_with_payouts"):
        if hasattr(adapter, name):
            try:
                fn = getattr(adapter, name)
                enriched = fn()
                if asyncio.iscoroutine(enriched):
                    enriched = await enriched
                if enriched:
                    break
            except Exception:
                enriched = None
    try:
        markets: List[Market] = []
        if isinstance(enriched, list) and enriched:
            for e in enriched:
                try:
                    if not isinstance(e, dict):
                        # support tuple/list formats
                        symbol = str(e[0]).strip()
                        payout = _normalize_payout_value(e[1]) if len(e) > 1 else 0.0
                        otc_flag = "OTC" in symbol.upper()
                        a_type = "Unknown"
                    else:
                        symbol = str(e.get("symbol") or e.get("name") or "").strip()
                        payout = _extract_payout_from_record(e)
                        otc_flag = bool(e.get("otc") or (symbol and "OTC" in symbol.upper()))
                        a_type = str(e.get("type") or e.get("asset_type") or "Unknown")
                    if not symbol:
                        continue
                    markets.append(
                        Market(
                            symbol=symbol,
                            asset_type=a_type,
                            display_payout_percent=payout,
                            otc=otc_flag,
                            payout_per_expiration={"60": payout} if payout > 0 else None,
                        )
                    )
                except Exception:
                    continue
        # 2) Fallback to generic asset getters
        if not markets:
            method_names = ["get_assets", "get_available_assets", "get_available_assets_async"]
            getter = next((getattr(adapter, n) for n in method_names if hasattr(adapter, n)), None)
            if getter is None:
                return await get_market_catalog()
            raw = getter()
            if asyncio.iscoroutine(raw):
                raw = await raw
            iterable = raw.values() if isinstance(raw, dict) else raw
            for entry in iterable or []:
                if isinstance(entry, dict):
                    symbol = entry.get("symbol") or entry.get("name") or ""
                    payout = _extract_payout_from_record(entry)
                    asset_type = entry.get("type") or entry.get("asset_type") or "Unknown"
                    otc_flag = bool(entry.get("otc") or (symbol and "OTC" in str(symbol).upper()))
                else:
                    symbol = str(entry).strip()
                    payout = 0.0
                    asset_type = "Unknown"
                    otc_flag = "OTC" in symbol.upper()
                if not symbol:
                    continue
                markets.append(
                    Market(
                        symbol=str(symbol),
                        asset_type=str(asset_type),
                        display_payout_percent=float(payout),
                        otc=bool(otc_flag),
                        payout_per_expiration={"60": float(payout)} if payout > 0 else None,
                    )
                )
        # Only replace the catalog if at least one positive payout present
        if markets and any(m.display_payout_percent > 0 for m in markets):
            async with _catalog_lock:
                global _PLACEHOLDER, _symbol_index
                _PLACEHOLDER = markets
                _symbol_index = {m.symbol: m for m in _PLACEHOLDER}
        return await get_market_catalog()
    except Exception:
        return await get_market_catalog()


__all__ = [
    "Market",
    "get_market_catalog",
    "get_market_by_symbol",
    "refresh_catalog",
    "fetch_live_catalog",
]
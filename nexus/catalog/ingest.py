"""Market catalog ingestion (stable placeholder + optional live fetch).

Purpose:
- Provide deterministic placeholder markets for tests
- Expose get_market_catalog() and get_market_by_symbol()
- Optionally fetch live markets via adapter when explicitly called
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


# Comprehensive market catalog with active payout rates
_PLACEHOLDER: List[Market] = [
    # Major Forex Pairs
    Market("EURUSD", "Forex", 88.0, payout_per_expiration={"60": 88.0, "120": 87.0, "300": 85.0}),
    Market("GBPUSD", "Forex", 85.0, payout_per_expiration={"60": 85.0, "120": 84.0, "300": 83.0}),
    Market("USDJPY", "Forex", 78.0, payout_per_expiration={"60": 78.0, "120": 77.0, "300": 76.0}),
    Market("AUDUSD", "Forex", 82.0, payout_per_expiration={"60": 82.0, "120": 81.0, "300": 80.0}),
    Market("USDCAD", "Forex", 83.0, payout_per_expiration={"60": 83.0, "120": 82.0}),
    Market("USDCHF", "Forex", 81.0, payout_per_expiration={"60": 81.0, "120": 80.0}),
    Market("NZDUSD", "Forex", 80.0, payout_per_expiration={"60": 80.0, "120": 79.0}),
    # Cross Forex Pairs
    Market("EURGBP", "Forex", 84.0, payout_per_expiration={"60": 84.0}),
    Market("EURJPY", "Forex", 86.0, payout_per_expiration={"60": 86.0}),
    Market("GBPJPY", "Forex", 87.0, payout_per_expiration={"60": 87.0}),
    Market("AUDJPY", "Forex", 82.0, payout_per_expiration={"60": 82.0}),
    Market("EURAUD", "Forex", 83.0, payout_per_expiration={"60": 83.0}),
    Market("GBPAUD", "Forex", 84.0, payout_per_expiration={"60": 84.0}),
    Market("CADJPY", "Forex", 82.0, payout_per_expiration={"60": 82.0}),
    Market("CHFJPY", "Forex", 81.0, payout_per_expiration={"60": 81.0}),
    # High-Yield OTC Markets
    Market("EURUSD_otc", "Forex", 92.0, otc=True, payout_per_expiration={"60": 92.0, "120": 90.0}),
    Market("GBPUSD_otc", "Forex", 90.0, otc=True, payout_per_expiration={"60": 90.0, "120": 88.0}),
    Market("USDJPY_otc", "Forex", 89.0, otc=True, payout_per_expiration={"60": 89.0, "120": 87.0}),
    Market("AUDUSD_otc", "Forex", 88.0, otc=True, payout_per_expiration={"60": 88.0}),
    Market("USDCAD_otc", "Forex", 87.0, otc=True, payout_per_expiration={"60": 87.0}),
    Market("NZDUSD_otc", "Forex", 86.0, otc=True, payout_per_expiration={"60": 86.0}),
    Market("EURGBP_otc", "Forex", 88.0, otc=True, payout_per_expiration={"60": 88.0}),
    Market("EURJPY_otc", "Forex", 91.0, otc=True, payout_per_expiration={"60": 91.0}),
    Market("GBPJPY_otc", "Forex", 92.0, otc=True, payout_per_expiration={"60": 92.0}),
    # Commodities
    Market("XAUUSD", "Commodity", 85.0, payout_per_expiration={"60": 85.0}),
    Market("XAGUSD", "Commodity", 81.0, payout_per_expiration={"60": 81.0}),
    Market("USOIL", "Commodity", 82.0, payout_per_expiration={"60": 82.0}),
    Market("XAUUSD_otc", "Commodity", 90.0, otc=True, payout_per_expiration={"60": 90.0}),
    # Crypto Assets
    Market("BTCUSD", "Crypto", 85.0, payout_per_expiration={"60": 85.0}),
    Market("ETHUSD", "Crypto", 84.0, payout_per_expiration={"60": 84.0}),
    Market("SOLUSD", "Crypto", 82.0, payout_per_expiration={"60": 82.0}),
    Market("BTCUSD_otc", "Crypto", 90.0, otc=True, payout_per_expiration={"60": 90.0}),
]
_symbol_index: Dict[str, Market] = {m.symbol: m for m in _PLACEHOLDER}
_catalog_lock = asyncio.Lock()


def get_market_by_symbol(symbol: str) -> Optional[Market]:
    return _symbol_index.get(symbol)


async def get_market_catalog(
    force_refresh: bool = False,
) -> List[Market]:  # force_refresh kept for API compatibility
    async with _catalog_lock:
        # Return a deep-ish copy so callers cannot mutate global state
        return [
            Market(
                symbol=m.symbol,
                asset_type=m.asset_type,
                display_payout_percent=m.display_payout_percent,
                active=m.active,
                otc=m.otc,
                payout_per_expiration=dict(m.payout_per_expiration)
                if m.payout_per_expiration
                else None,
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


async def fetch_live_catalog(
    adapter: Any,
) -> List[Market]:  # pragma: no cover (network interaction)
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
                        active_flag = bool(e.get("active", True))
                        if not active_flag:
                            payout = 0.0
                        otc_flag = bool(e.get("otc") or (symbol and "OTC" in symbol.upper()))
                        a_type = str(e.get("type") or e.get("asset_type") or "Unknown")
                    if not symbol:
                        continue
                    markets.append(
                        Market(
                            symbol=symbol,
                            asset_type=a_type,
                            display_payout_percent=payout,
                            active=active_flag,
                            otc=otc_flag,
                            payout_per_expiration={"60": payout} if payout > 0 else None,
                            metadata={"payout_source": "live broker catalog"},
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
                    active_flag = bool(entry.get("active", True))
                    if not active_flag:
                        payout = 0.0
                    asset_type = entry.get("type") or entry.get("asset_type") or "Unknown"
                    otc_flag = bool(entry.get("otc") or (symbol and "OTC" in str(symbol).upper()))
                else:
                    symbol = str(entry).strip()
                    payout = 0.0
                    active_flag = True
                    asset_type = "Unknown"
                    otc_flag = "OTC" in symbol.upper()
                if not symbol:
                    continue
                markets.append(
                    Market(
                        symbol=str(symbol),
                        asset_type=str(asset_type),
                        display_payout_percent=float(payout),
                        active=active_flag,
                        otc=bool(otc_flag),
                        payout_per_expiration={"60": float(payout)} if payout > 0 else None,
                        metadata={"payout_source": "live broker catalog"},
                    )
                )
        # Deduplicate markets by symbol
        market_map: dict[str, Market] = {}
        for m in markets:
            if m.symbol not in market_map:
                market_map[m.symbol] = m
            else:
                existing = market_map[m.symbol]
                existing.active = m.active
                if m.active:
                    existing.display_payout_percent = m.display_payout_percent
                else:
                    existing.display_payout_percent = 0.0
                if m.payout_per_expiration:
                    if existing.payout_per_expiration:
                        existing.payout_per_expiration.update(m.payout_per_expiration)
                    else:
                        existing.payout_per_expiration = dict(m.payout_per_expiration)

        deduped_markets = list(market_map.values())

        # Only replace the catalog if at least one positive payout present
        if deduped_markets and any(m.display_payout_percent > 0 for m in deduped_markets):
            async with _catalog_lock:
                global _PLACEHOLDER, _symbol_index
                _PLACEHOLDER = deduped_markets
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

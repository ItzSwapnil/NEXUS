"""Market catalog ingestion (spec-aligned with optional live fetch).

Responsibilities (Spec §2):
  * Provide an in-memory catalog of markets with payout percentages
  * Support OTC flagging and per-expiration payout mapping
  * Optional live ingestion via Quotex adapter (fallback to placeholder)
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

# Placeholder deterministic catalog used by tests
_PLACEHOLDER: List[Market] = [
    Market("EURUSD", "Forex", 87.0, payout_per_expiration={"60": 87.0, "120": 86.0}),
    Market("GBPUSD", "Forex", 82.0, payout_per_expiration={"60": 82.0, "300": 84.0}),
    Market("USDJPY", "Forex", 78.0, payout_per_expiration={"60": 78.0}),  # Below threshold example
    Market("OTC_EURUSD", "OTC", 90.0, otc=True, payout_per_expiration={"60": 90.0}),
]
_symbol_index = {m.symbol: m for m in _PLACEHOLDER}
_catalog_lock = asyncio.Lock()

async def get_market_catalog(force_refresh: bool = False) -> List[Market]:
    async with _catalog_lock:
        return [
            Market(
                symbol=m.symbol,
                asset_type=m.asset_type,
                display_payout_percent=m.display_payout_percent,
                active=m.active,
                otc=m.otc,
                payout_per_expiration=dict(m.payout_per_expiration) if m.payout_per_expiration else None,
                metadata=dict(m.metadata),
            ) for m in _PLACEHOLDER
        ]

async def refresh_catalog() -> None:
    return None

async def fetch_live_catalog(adapter: Any) -> List[Market]:  # pragma: no cover (network dependent)
    try:
        raw = await adapter.get_assets()
        markets: List[Market] = []
        iterable = raw.items() if isinstance(raw, dict) else enumerate(raw)
        for key, asset in iterable:
            symbol = asset.get("symbol") or asset.get("name") or str(key)
            payout = float(asset.get("profit", asset.get("payout", 0)) or 0)
            asset_type = asset.get("type") or asset.get("asset_type") or "Unknown"
            otc_flag = bool(asset.get("otc") or ("OTC" in symbol.upper()))
            markets.append(Market(symbol=symbol, asset_type=asset_type, display_payout_percent=payout, otc=otc_flag))
        if markets:
            global _PLACEHOLDER, _symbol_index
            _PLACEHOLDER = markets
            _symbol_index = {m.symbol: m for m in _PLACEHOLDER}
        return markets or list(_PLACEHOLDER)
    except Exception:
        return await get_market_catalog()

def get_market_by_symbol(symbol: str) -> Optional[Market]:
    return _symbol_index.get(symbol)

__all__ = ["Market", "get_market_catalog", "get_market_by_symbol", "refresh_catalog", "fetch_live_catalog"]

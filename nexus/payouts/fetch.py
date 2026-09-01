"""Payout helpers and override."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from nexus.catalog.ingest import Market, get_market_by_symbol
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.payouts.fetch")

_override_enabled = False
_override_log_path = Path("logs/payout_override.log")
_override_log_path.parent.mkdir(exist_ok=True)


def get_payout_for_market(
    market_or_symbol: Union[Market, str], expiration: Optional[str] = None
) -> float:
    """Return payout % for a Market or symbol."""
    market: Optional[Market] = None
    if isinstance(market_or_symbol, Market):
        market = market_or_symbol
    else:
        market = get_market_by_symbol(str(market_or_symbol))
    if market is None:
        logger.warning(f"Unknown market: {market_or_symbol}")
        return 0.0
    return market.effective_payout(expiration)


def is_payout_allowed(payout_percent: float, threshold: float) -> bool:
    """True if payout passes threshold or override is enabled."""
    if _override_enabled:
        return True
    return payout_percent >= threshold


def set_payout_override(enabled: bool, user: Optional[str] = None, reason: str = "") -> None:
    """Enable/disable payout override with audit logging."""
    global _override_enabled
    prev = _override_enabled
    _override_enabled = bool(enabled)
    if prev != _override_enabled:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "override": _override_enabled,
            "user": user or os.getenv("USERNAME") or os.getenv("USER") or "unknown",
            "reason": reason,
        }
        try:
            with open(_override_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed writing override log: {e}")
        logger.info(f"Payout override set to {_override_enabled} (reason='{reason}')")


def is_override_enabled() -> bool:
    return _override_enabled


__all__ = [
    "get_payout_for_market",
    "is_payout_allowed",
    "set_payout_override",
    "is_override_enabled",
]

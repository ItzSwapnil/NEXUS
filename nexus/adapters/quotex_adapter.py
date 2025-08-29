# Minimal async Quotex adapter stub used by the engine in tests
from __future__ import annotations

from typing import Any, Dict


class QuotexAdapter:
    """Lightweight stub that mimics the expected Quotex API surface.

    This avoids importing external pyquotex during tests and linting while
    providing stable async methods used by the engine.
    """

    def __init__(self, email: str, password: str, lang: str = "en") -> None:
        self.email = email
        self.password = password
        self.lang = lang
        self._connected = False

    async def connect(self) -> bool:
        """Simulate establishing a connection."""
        self._connected = True
        return True

    async def get_balance(self) -> float:
        """Return a deterministic balance value."""
        if not self._connected:
            # Mirror realistic behavior: require connection first
            raise RuntimeError("Quotex API client not initialized. Call connect() first.")
        return 10000.0

    async def buy_simple(self, asset: str, amount: float, direction: str, duration: int) -> Dict[str, Any]:
        """Simulate placing an order; returns a small confirmation payload."""
        if not self._connected:
            raise RuntimeError("Quotex API client not initialized. Call connect() first.")
        return {
            "placed": True,
            "asset": asset,
            "amount": float(amount),
            "direction": direction,
            "duration": int(duration),
        }


__all__ = ["QuotexAdapter"]

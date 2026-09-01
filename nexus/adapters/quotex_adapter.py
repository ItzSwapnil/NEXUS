# Async Quotex adapter wrapper delegating to the real integration
from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from nexus.adapters.quotex import (
    QuotexAdapter as RealQuotexAdapter,
)


class QuotexAdapter:
    """Async-compatible facade used by the engine/tests.

    This wrapper maps the lightweight engine-facing API to the full
    nexus.adapters.quotex implementation backed by pyquotex.
    """

    def __init__(
        self,
        email: str,
        password: str,
        lang: str = "en",
        demo_mode: bool = True,
        use_real: bool = True,
        retry_attempts: int = 3,
        retry_delay: int = 5,
        session_file: str = "session.json",
    ) -> None:
        self.email = email
        self.password = password
        self.lang = lang
        self.demo_mode = demo_mode
        self.use_real = use_real
        self._real: RealQuotexAdapter = RealQuotexAdapter(
            email=email,
            password=password,
            demo_mode=demo_mode,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            session_file=session_file,
        )

    @property
    def authenticated(self) -> bool:
        return getattr(self._real, "authenticated", False)

    def set_session(
        self, user_agent: str, cookies: Optional[str] = None, ssid: Optional[str] = None
    ) -> None:
        """Forward browser session details to the real adapter."""
        try:
            self._real.set_session(user_agent, cookies, ssid)
        except Exception:
            pass

    async def set_practice_mode(self, practice: bool) -> None:
        """Force broker to use PRACTICE or REAL account after connecting."""
        try:
            await self._real.set_practice_mode(bool(practice))  # type: ignore[attr-defined]
        except Exception:
            pass

    async def connect(self) -> bool:
        """Perform asynchronous login against Quotex."""
        try:
            res = await self._real.login()
            return bool(res)
        except Exception:
            return False

    async def authenticate_async(self) -> bool:
        """Perform asynchronous login against Quotex."""
        return await self.connect()

    def authenticate(self) -> bool:
        """Synchronous wrapper for authentication."""
        try:
            import asyncio

            return bool(asyncio.run(self.connect()))
        except Exception:
            return False

    async def get_balance_async(self) -> float:
        """Return the current active account balance."""
        try:
            return float(await self._real.get_balance_async())
        except Exception:
            return 0.0

    def get_balance(self) -> float:
        """Synchronous balance accessor."""
        try:
            return float(self._real.get_balance())
        except Exception:
            return 0.0

    def get_trade_outcome(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Return settlement feedback for one broker order, if available."""
        try:
            return self._real.get_trade_outcome(str(order_id))
        except Exception:
            return None

    async def get_available_assets(self) -> List[str]:
        """Return a list of available asset symbols from the broker."""
        try:
            if hasattr(self._real, "get_available_assets_async"):
                res = await self._real.get_available_assets_async()  # type: ignore[attr-defined]
                return cast(List[str], res)
            res_sync = self._real.get_available_assets()
            return cast(List[str], res_sync)
        except Exception:
            return []

    async def get_assets_with_payouts_async(self) -> List[Dict[str, Any]]:
        """Return assets with payout info when supported by the real adapter."""
        try:
            if hasattr(self._real, "get_assets_with_payouts_async"):
                res = await self._real.get_assets_with_payouts_async()  # type: ignore[attr-defined]
                return cast(List[Dict[str, Any]], res)
            # Fallback: map plain list to dicts with zero payout
            syms = await self.get_available_assets()
            return [{"symbol": s, "payout": 0.0} for s in syms]
        except Exception:
            return []

    async def get_candles_async(
        self, symbol: str, timeframe_sec: int, limit: int = 200
    ) -> Optional[List[Dict[str, float]]]:
        """Fetch recent candles for a symbol with timeframe in seconds."""
        try:
            if hasattr(self._real, "get_candles_async"):
                res = await self._real.get_candles_async(symbol, timeframe_sec, limit)  # type: ignore[attr-defined]
                return cast(Optional[List[Dict[str, float]]], res)
            if hasattr(self._real, "get_candles"):
                res_sync = self._real.get_candles(symbol, timeframe_sec, limit)  # type: ignore[attr-defined]
                return cast(Optional[List[Dict[str, float]]], res_sync)
            return None
        except Exception:
            return None

    async def buy_simple(
        self, asset: str, amount: float | str, direction: str | float = "call", duration: int = 60
    ) -> Optional[Dict[str, Any]]:
        if isinstance(direction, (int, float)) or (
            isinstance(direction, str) and direction.replace(".", "", 1).isdigit()
        ):
            amount, direction = direction, amount
        try:
            if hasattr(self._real, "buy_simple_async"):
                res_async = await self._real.buy_simple_async(
                    asset=asset,
                    direction=str(direction),
                    amount=float(amount),
                    expiration=int(duration),
                )
                return cast(Optional[Dict[str, Any]], res_async)
            if hasattr(self._real, "buy_simple"):
                fn = self._real.buy_simple
                res_sync = fn(
                    asset=asset,
                    direction=str(direction),
                    amount=float(amount),
                    expiration=int(duration),
                )
                return cast(Optional[Dict[str, Any]], res_sync)
            return None
        except Exception:
            return None

    async def buy_simple_async(
        self, asset: str, direction: str = "call", amount: float = 10.0, expiration: int = 60
    ) -> Optional[Dict[str, Any]]:
        return await self.buy_simple(
            asset=asset, amount=amount, direction=direction, duration=expiration
        )


__all__ = ["QuotexAdapter"]

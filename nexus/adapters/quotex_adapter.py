# Async Quotex adapter wrapper delegating to the real integration
from __future__ import annotations

from typing import Any, Dict, Optional

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

    def set_session(self, user_agent: str, cookies: Optional[str] = None, ssid: Optional[str] = None) -> None:
        """Forward browser session details to the real adapter."""
        try:
            self._real.set_session(user_agent, cookies, ssid)
        except Exception:
            pass

    async def connect(self) -> bool:
        try:
            ok = await self._real.login()
            return bool(ok)
        except Exception:
            return False

    async def get_balance(self) -> float:
        try:
            if hasattr(self._real, "get_balance_async"):
                bal = await self._real.get_balance_async()  # type: ignore[attr-defined]
                return float(bal or 0.0)
        except Exception:
            pass
        try:
            bal = self._real.get_balance()
            return float(bal or 0.0)
        except Exception:
            return 0.0

    async def buy_simple(self, asset: str, amount: float, direction: str, duration: int) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(self._real, "buy_simple_async"):
                return await self._real.buy_simple_async(
                    asset=asset,
                    direction=direction,
                    amount=float(amount),
                    expiration=int(duration),
                )  # type: ignore[attr-defined]
            return self._real.buy_simple(
                asset=asset,
                direction=direction,
                amount=float(amount),
                expiration=int(duration),
            )
        except Exception:
            return None


__all__ = ["QuotexAdapter"]

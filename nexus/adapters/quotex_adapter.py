import asyncio
from typing import Any, Dict

try:
    from pyquotex.stable_api import Quotex  # type: ignore
    _HAS_PYQUOTEX = True
except ImportError:  # pragma: no cover
    Quotex = object  # type: ignore
    _HAS_PYQUOTEX = False


class QuotexAdapter:
    """Thin async wrapper around pyquotex with safe sync/async handling."""

    def __init__(self, email: str, password: str, lang: str = "en"):
        self.email = email
        self.password = password
        self.lang = lang
        self.api: Any | None = None
        self.connected: bool = False

    async def _call_api(self, method_name: str, *args, **kwargs):
        """Call a pyquotex API method. Await if coroutine, thread if sync."""
        if self.api is None:
            raise RuntimeError("Quotex API client not initialized. Call connect() first.")
        func = getattr(self.api, method_name, None)
        if func is None or not callable(func):
            raise AttributeError(f"Quotex API has no callable '{method_name}'")
        try:
            result = func(*args, **kwargs)
        except TypeError:
            # Some methods are strict about positional args; run in thread
            return await asyncio.to_thread(func, *args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _ensure_connected(self) -> None:
        if not self.connected:
            await self.connect()

    async def connect(self) -> None:
        """Create client and connect, handling both sync and async connect()."""
        if not _HAS_PYQUOTEX:
            raise RuntimeError("pyquotex not installed. Install via VCS or add to PYTHONPATH.")
        self.api = Quotex(self.email, self.password, lang=self.lang)  # type: ignore[call-arg]
        connect = getattr(self.api, "connect", None)
        if connect is None or not callable(connect):
            raise RuntimeError("Quotex API has no connect method.")
        try:
            maybe_coro = connect()
        except Exception:
            await asyncio.to_thread(connect)
        else:
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
            # If sync, it's already completed
        self.connected = True

    async def get_assets(self) -> Dict[str, Any]:
        await self._ensure_connected()
        return await self._call_api("get_all_assets")

    async def get_candles(self, asset: str, interval: int, count: int = 100):
        await self._ensure_connected()
        return await self._call_api("get_candle", asset, interval, count)

    async def get_balance(self) -> float:
        """Return current account balance as float.
        Tries method get_balance(), falls back to account_balance attribute.
        """
        await self._ensure_connected()
        get_balance_fn = getattr(self.api, "get_balance", None)
        if callable(get_balance_fn):
            # Call once; if coroutine, await; if None, try attribute
            try:
                res = get_balance_fn()
            except TypeError:
                res = await asyncio.to_thread(get_balance_fn)
            if asyncio.iscoroutine(res):
                val = await res
            else:
                val = res
            if val is None:
                val = getattr(self.api, "account_balance", None)
            if val is None:
                raise RuntimeError("Balance not available after get_balance call.")
            return float(val)
        # Fallback to attribute only
        if hasattr(self.api, "account_balance"):
            val = getattr(self.api, "account_balance")
            if val is not None:
                return float(val)
        raise RuntimeError("Quotex API does not provide get_balance or account_balance.")

    async def buy_simple(self, asset: str, amount: float, direction: str, duration: int):
        await self._ensure_connected()
        return await self._call_api("buy_simple", asset, amount, direction, duration)

    async def buy_and_check_win(self, asset: str, amount: float, direction: str, duration: int):
        await self._ensure_connected()
        return await self._call_api("buy_and_check_win", asset, amount, direction, duration)

    async def get_dynamic_parameters(self) -> Dict[str, Any]:
        await self._ensure_connected()
        assets = await self.get_assets()
        balance = await self.get_balance()
        return {"assets": assets, "balance": balance}

import asyncio
from typing import Any, Dict

try:
    from pyquotex.stable_api import Quotex  # type: ignore
    _HAS_PYQUOTEX = True
except ImportError:  # pragma: no cover
    Quotex = object  # type: ignore
    _HAS_PYQUOTEX = False

class QuotexAdapter:
    def __init__(self, email: str, password: str, lang: str = "en"):
        self.email = email
        self.password = password
        self.lang = lang
        self.api = None
        self.connected = False

    async def connect(self):
        if not _HAS_PYQUOTEX:
            raise RuntimeError("pyquotex not installed. Install via VCS and use 'pip install <repo>' or add to PYTHONPATH.")
        self.api = Quotex(self.email, self.password, lang=self.lang)  # type: ignore[call-arg]
        await asyncio.to_thread(self.api.connect)  # type: ignore[attr-defined]
        self.connected = True

    async def get_assets(self) -> Dict[str, Any]:
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_all_assets)  # type: ignore[attr-defined]

    async def get_candles(self, asset: str, interval: int, count: int = 100):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_candle, asset, interval, count)  # type: ignore[attr-defined]

    async def get_balance(self):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_balance)  # type: ignore[attr-defined]

    async def buy_simple(self, asset: str, amount: float, direction: str, duration: int):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.buy_simple, asset, amount, direction, duration)  # type: ignore[attr-defined]

    async def buy_and_check_win(self, asset: str, amount: float, direction: str, duration: int):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.buy_and_check_win, asset, amount, direction, duration)  # type: ignore[attr-defined]

    async def get_dynamic_parameters(self) -> Dict[str, Any]:
        # Fetch all live parameters from Quotex
        assets = await self.get_assets()
        balance = await self.get_balance()
        # Add more as needed
        return {
            "assets": assets,
            "balance": balance,
        }

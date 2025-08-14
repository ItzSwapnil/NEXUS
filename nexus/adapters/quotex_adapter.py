import asyncio
from pyquotex.stable_api import Quotex
from typing import Any, Dict

class QuotexAdapter:
    def __init__(self, email: str, password: str, lang: str = "en"):
        self.email = email
        self.password = password
        self.lang = lang
        self.api = None
        self.connected = False

    async def connect(self):
        self.api = Quotex(self.email, self.password, lang=self.lang)
        await asyncio.to_thread(self.api.connect)
        self.connected = True

    async def get_assets(self) -> Dict[str, Any]:
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_all_assets)

    async def get_candles(self, asset: str, interval: int, count: int = 100):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_candle, asset, interval, count)

    async def get_balance(self):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.get_balance)

    async def buy_simple(self, asset: str, amount: float, direction: str, duration: int):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.buy_simple, asset, amount, direction, duration)

    async def buy_and_check_win(self, asset: str, amount: float, direction: str, duration: int):
        if not self.connected:
            await self.connect()
        return await asyncio.to_thread(self.api.buy_and_check_win, asset, amount, direction, duration)

    async def get_dynamic_parameters(self) -> Dict[str, Any]:
        # Fetch all live parameters from Quotex
        assets = await self.get_assets()
        balance = await self.get_balance()
        # Add more as needed
        return {
            "assets": assets,
            "balance": balance,
        }


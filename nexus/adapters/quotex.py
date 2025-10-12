"""
Quotex Adapter for NEXUS (stable minimal)

Provides PRACTICE/REAL login and asset listing via pyquotex with async-safe retries.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timedelta
from typing import Any, List, Optional, Dict

try:
    from pyquotex.stable_api import Quotex
    _HAS_PYQUOTEX = True
except Exception:  # pragma: no cover
    Quotex = object
    _HAS_PYQUOTEX = False

from nexus.utils.logger import get_nexus_logger, PerformanceLogger
import nexus.adapters.pyquotex_patch  # noqa: F401

logger = get_nexus_logger("nexus.adapters.quotex")
perf_logger = PerformanceLogger("quotex_adapter")
LOGIN_TIMEOUT_SECONDS = 30

COMMON_ASSETS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD", "GBPAUD",
    "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
    "Apple", "Amazon", "Google", "Microsoft", "Tesla", "Facebook",
    "Gold", "Silver", "Oil", "DAX", "S&P 500", "Dow Jones", "NASDAQ",
]


class QuotexAdapter:
    def __init__(
        self,
        email: str,
        password: str,
        demo_mode: bool = True,
        retry_attempts: int = 3,
        retry_delay: int = 5,
        session_file: str = "session.json",
    ) -> None:
        self.email = email
        self.password = password
        self.demo_mode = bool(demo_mode)
        self.retry_attempts = int(retry_attempts)
        self.retry_delay = int(retry_delay)
        self.session_file = session_file
        self.client: Optional[Any] = None
        self.authenticated = False
        self.last_action = datetime.now()
        self.login_ready = threading.Event()
        self._session_override: Optional[dict] = None

    async def _maybe_await(self, val: Any) -> Any:
        if inspect.isawaitable(val):
            return await val
        return val

    def set_session(self, user_agent: str, cookies: Optional[str] = None, ssid: Optional[str] = None) -> None:
        self._session_override = {"user_agent": user_agent, "cookies": cookies, "ssid": ssid}

    async def set_practice_mode(self, practice: bool) -> None:
        self.demo_mode = bool(practice)
        if self.client is not None and self.authenticated:
            desired = "PRACTICE" if self.demo_mode else "REAL"
            for name, args in (
                ("set_account_mode", (desired,)),
                ("set_account", (desired,)),
                ("change_account", (desired,)),
                ("set_account_type", (desired,)),
                ("set_demo_mode", (self.demo_mode,)),
                ("use_practice", ()),
                ("use_demo", ()),
                ("use_real", ()),
            ):
                try:
                    if hasattr(self.client, name):
                        fn = getattr(self.client, name)
                        await self._maybe_await(fn(*args))
                        break
                except Exception:
                    continue

    async def _ensure_authenticated(self) -> bool:
        if not self.authenticated or not self.client:
            return await self.login()
        if (datetime.now() - self.last_action) > timedelta(hours=1):
            return await self.login()
        return True

    async def login(self) -> bool:
        if not _HAS_PYQUOTEX:
            logger.error("pyquotex not installed. Install it to use QuotexAdapter.")
            return False
        try:
            with perf_logger.measure("quotex_login"):
                self.client = Quotex(email=self.email, password=self.password, lang="en")
                if self._session_override and hasattr(self.client, "set_session"):
                    try:
                        ua = self._session_override.get("user_agent") or "Quotex/1.0"
                        self.client.set_session(ua, self._session_override.get("cookies"), self._session_override.get("ssid"))
                    except Exception:
                        pass
                try:
                    if hasattr(self.client, "connect"):
                        await asyncio.wait_for(self._maybe_await(self.client.connect()), timeout=LOGIN_TIMEOUT_SECONDS)
                    if hasattr(self.client, "check_connect"):
                        ok = await asyncio.wait_for(self._maybe_await(self.client.check_connect()), timeout=LOGIN_TIMEOUT_SECONDS)
                        if isinstance(ok, bool) and not ok:
                            return False
                except Exception as e:
                    logger.error(f"Connection failure: {e}")
                    self.authenticated = False
                    self.login_ready.clear()
                    return False
                # Consider connected
                self.authenticated = True
                await self.set_practice_mode(self.demo_mode)
                self.login_ready.set()
                self.last_action = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Login error: {e}")
            self.authenticated = False
            self.login_ready.clear()
            return False

    # ---- assets ----
    def get_available_assets(self) -> List[str]:
        try:
            if not self.client:
                return list(COMMON_ASSETS)
            # Try a broader set of potential methods on pyquotex client
            methods = [
                "get_available_assets",
                "get_available_asset",
                "get_all_assets",
                "get_all_asset",
                "get_all_asset_name",
                "available_assets",
                "available_asset",
                "list_assets",
                "get_assets",
            ]
            for name in methods:
                if not hasattr(self.client, name):
                    continue
                try:
                    data = getattr(self.client, name)()
                    # If the client returns an awaitable in sync context, skip it
                    if inspect.isawaitable(data):
                        continue
                    if isinstance(data, list):
                        items = [str(x).strip() for x in data]
                        return [x for x in items if x]
                    if isinstance(data, dict):
                        result_list: List[str] = []
                        for k, v in data.items():
                            if isinstance(v, dict):
                                sym = str(v.get("symbol") or v.get("name") or k)
                            else:
                                sym = str(k if v else v or k)
                            sym = sym.strip()
                            if sym:
                                result_list.append(sym)
                        if result_list:
                            return result_list
                except Exception:
                    continue
            return list(COMMON_ASSETS)
        except Exception:
            return list(COMMON_ASSETS)

    async def get_available_assets_async(self) -> List[str]:
        try:
            if not await self._ensure_authenticated():
                return list(COMMON_ASSETS)
            methods = [
                "get_available_assets",
                "get_available_asset",
                "get_all_assets",
                "get_all_asset",
                "get_all_asset_name",
                "available_assets",
                "available_asset",
                "list_assets",
                "get_assets",
            ]
            for name in methods:
                if not hasattr(self.client, name):
                    continue
                try:
                    v = getattr(self.client, name)()
                    data = await self._maybe_await(v)
                    if isinstance(data, list):
                        items = [str(x).strip() for x in data]
                        return [x for x in items if x]
                    if isinstance(data, dict):
                        result_list2: List[str] = []
                        for k, v2 in data.items():
                            if isinstance(v2, dict):
                                sym = str(v2.get("symbol") or v2.get("name") or k)
                            else:
                                sym = str(k if v2 else v2 or k)
                            sym = sym.strip()
                            if sym:
                                result_list2.append(sym)
                        if result_list2:
                            return result_list2
                except Exception:
                    continue
            # Fallback to sync variant if nothing returned
            return self.get_available_assets()
        except Exception:
            return list(COMMON_ASSETS)

    async def get_balance_async(self) -> float:
        try:
            if not await self._ensure_authenticated():
                return 0.0
            if hasattr(self.client, "get_balance"):
                v = getattr(self.client, "get_balance")()
                v = await self._maybe_await(v)
                if isinstance(v, dict):
                    v = v.get("balance", 0.0)
                return float(v or 0.0)
            return 0.0
        except Exception:
            return 0.0

    def get_balance(self) -> float:
        try:
            if not self.client:
                return 0.0
            if hasattr(self.client, "get_balance"):
                v = getattr(self.client, "get_balance")()
                if isinstance(v, dict):
                    v = v.get("balance", 0.0)
                return float(v or 0.0)
            return 0.0
        except Exception:
            return 0.0

    async def get_assets_with_payouts_async(self) -> List[Dict[str, Any]]:
        try:
            syms = await self.get_available_assets_async()
            return [{"symbol": s, "payout": 0.0} for s in syms]
        except Exception:
            return [{"symbol": s, "payout": 0.0} for s in COMMON_ASSETS]

    # ---- market data (candles) ----
    def get_candles(self, symbol: str, timeframe_sec: int, limit: int = 200) -> Optional[List[Dict[str, float]]]:
        """Fetch recent candles for symbol.
        Returns list of dicts with keys: time, open, high, low, close, volume (volume may be 0 if not provided).
        """
        try:
            if not self.client:
                return None
            # Candidate method names and param shapes on pyquotex
            methods = [
                ("get_candles", (symbol, timeframe_sec, limit), {}),
                ("get_candles_history", (symbol, timeframe_sec, limit), {}),
                ("get_history_candles", (symbol, timeframe_sec, limit), {}),
                ("candles", (symbol, timeframe_sec, limit), {}),
                ("get_candles", (), {"asset": symbol, "period": timeframe_sec, "count": limit}),
                ("get_candles", (), {"symbol": symbol, "timeframe": timeframe_sec, "limit": limit}),
            ]
            for name, args, kwargs in methods:
                if not hasattr(self.client, name):
                    continue
                try:
                    v = getattr(self.client, name)(*args, **kwargs)
                    if inspect.isawaitable(v):
                        # In sync path, skip awaitables
                        continue
                    data = v
                    # Normalize
                    records: List[Dict[str, float]] = []
                    if isinstance(data, list):
                        for c in data:
                            try:
                                if isinstance(c, dict):
                                    o = float(c.get("open") or c.get("o") or 0.0)
                                    h = float(c.get("high") or c.get("h") or o)
                                    low = float(c.get("low") or c.get("l") or o)
                                    cl = float(c.get("close") or c.get("c") or o)
                                    t = float(c.get("time") or c.get("t") or 0.0)
                                    vlm = float(c.get("volume") or c.get("v") or 0.0)
                                else:
                                    # tuple/list format
                                    t, o, h, low, cl = c[0:5]
                                    vlm = c[5] if len(c) > 5 else 0.0
                                records.append({"time": t, "open": o, "high": h, "low": low, "close": cl, "volume": vlm})
                            except Exception:
                                continue
                        if records:
                            return records[-limit:]
                except Exception:
                    continue
            return None
        except Exception:
            return None

    async def get_candles_async(self, symbol: str, timeframe_sec: int, limit: int = 200) -> Optional[List[Dict[str, float]]]:
        try:
            if not await self._ensure_authenticated():
                return None
            methods = [
                ("get_candles", (symbol, timeframe_sec, limit), {}),
                ("get_candles_history", (symbol, timeframe_sec, limit), {}),
                ("get_history_candles", (symbol, timeframe_sec, limit), {}),
                ("candles", (symbol, timeframe_sec, limit), {}),
                ("get_candles", (), {"asset": symbol, "period": timeframe_sec, "count": limit}),
                ("get_candles", (), {"symbol": symbol, "timeframe": timeframe_sec, "limit": limit}),
            ]
            for name, args, kwargs in methods:
                if not hasattr(self.client, name):
                    continue
                try:
                    v = getattr(self.client, name)(*args, **kwargs)
                    data = await self._maybe_await(v)
                    if isinstance(data, list):
                        records: List[Dict[str, float]] = []
                        for c in data:
                            try:
                                if isinstance(c, dict):
                                    o = float(c.get("open") or c.get("o") or 0.0)
                                    h = float(c.get("high") or c.get("h") or o)
                                    low = float(c.get("low") or c.get("l") or o)
                                    cl = float(c.get("close") or c.get("c") or o)
                                    t = float(c.get("time") or c.get("t") or 0.0)
                                    vlm = float(c.get("volume") or c.get("v") or 0.0)
                                else:
                                    t, o, h, low, cl = c[0:5]
                                    vlm = c[5] if len(c) > 5 else 0.0
                                records.append({"time": t, "open": o, "high": h, "low": low, "close": cl, "volume": vlm})
                            except Exception:
                                continue
                        if records:
                            return records[-limit:]
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def buy_simple(self, asset: str, direction: str, amount: float, expiration: int) -> Optional[Dict[str, Any]]:
        try:
            if not self.client:
                return None
            # Candidate method names to try on the underlying client
            method_names = [
                "buy_and_check_win",
                "buy",
                "buy_simple",
                "open_option",
                "open_trade",
                "open_position",
                "place_order",
                "place_trade",
            ]
            exp_variants = []
            try:
                exp = int(expiration)
                exp_variants = [exp]
                if exp >= 60:
                    exp_variants.append(max(1, exp // 60))  # minutes
                exp_variants.append(exp * 60)  # seconds variant
                # de-duplicate while preserving order
                seen = set()
                exp_variants = [x for x in exp_variants if not (x in seen or seen.add(x))]
            except Exception:
                exp_variants = [expiration]
            amt = float(amount)
            # Keyword argument variants commonly used by different libs
            def _kw_variants(exp_val: int):
                return [
                    {"asset": asset, "action": direction, "expirations_times": exp_val, "amount": amt},
                    {"asset": asset, "direction": direction, "duration": exp_val, "amount": amt},
                    {"symbol": asset, "action": direction, "expiration": exp_val, "amount": amt},
                    {"symbol": asset, "direction": direction, "expiration": exp_val, "amount": amt},
                    {"asset": asset, "dir": direction, "expiration": exp_val, "investment": amt},
                    {"market": asset, "side": direction, "expiry": exp_val, "size": amt},
                ]
            # Positional argument variants (order differences)
            def _pos_variants(exp_val: int):
                return [
                    (asset, direction, amt, exp_val),
                    (asset, direction, exp_val, amt),
                    (direction, asset, amt, exp_val),
                    (direction, asset, exp_val, amt),
                ]

            for name in method_names:
                if not hasattr(self.client, name):
                    continue
                fn = getattr(self.client, name)
                for exp_val in exp_variants:
                    # Try kwargs variants first
                    for params in _kw_variants(int(exp_val)):
                        try:
                            v = fn(**params)
                            if v is not None:
                                return {"success": True, "order": v}
                        except TypeError:
                            continue
                        except Exception:
                            continue
                    # Try positional variants
                    for args in _pos_variants(int(exp_val)):
                        try:
                            v = fn(*args)
                            if v is not None:
                                return {"success": True, "order": v}
                        except TypeError:
                            continue
                        except Exception:
                            continue
            return None
        except Exception:
            return None

    async def buy_simple_async(self, asset: str, direction: str, amount: float, expiration: int) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None
            method_names = [
                "buy_and_check_win",
                "buy",
                "buy_simple",
                "open_option",
                "open_trade",
                "open_position",
                "place_order",
                "place_trade",
            ]
            exp_variants = []
            try:
                exp = int(expiration)
                exp_variants = [exp]
                if exp >= 60:
                    exp_variants.append(max(1, exp // 60))
                exp_variants.append(exp * 60)
                seen = set()
                exp_variants = [x for x in exp_variants if not (x in seen or seen.add(x))]
            except Exception:
                exp_variants = [expiration]
            amt = float(amount)
            def _kw_variants(exp_val: int):
                return [
                    {"asset": asset, "action": direction, "expirations_times": exp_val, "amount": amt},
                    {"asset": asset, "direction": direction, "duration": exp_val, "amount": amt},
                    {"symbol": asset, "action": direction, "expiration": exp_val, "amount": amt},
                    {"symbol": asset, "direction": direction, "expiration": exp_val, "amount": amt},
                    {"asset": asset, "dir": direction, "expiration": exp_val, "investment": amt},
                    {"market": asset, "side": direction, "expiry": exp_val, "size": amt},
                ]
            def _pos_variants(exp_val: int):
                return [
                    (asset, direction, amt, exp_val),
                    (asset, direction, exp_val, amt),
                    (direction, asset, amt, exp_val),
                    (direction, asset, exp_val, amt),
                ]

            for name in method_names:
                if not hasattr(self.client, name):
                    continue
                fn = getattr(self.client, name)
                for exp_val in exp_variants:
                    # Try kwargs variants first
                    for params in _kw_variants(int(exp_val)):
                        try:
                            v = fn(**params)
                            v = await self._maybe_await(v)
                            if v is not None:
                                return {"success": True, "order": v}
                        except TypeError:
                            continue
                        except Exception:
                            continue
                    # Try positional variants
                    for args in _pos_variants(int(exp_val)):
                        try:
                            v = fn(*args)
                            v = await self._maybe_await(v)
                            if v is not None:
                                return {"success": True, "order": v}
                        except TypeError:
                            continue
                        except Exception:
                            continue
            return None
        except Exception:
            return None


__all__ = ["QuotexAdapter"]

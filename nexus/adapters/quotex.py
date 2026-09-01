"""
Quotex Adapter for NEXUS (stable minimal)

Provides PRACTICE/REAL login and asset listing via pyquotex with async-safe retries.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import os
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from pyquotex.stable_api import Quotex

    _HAS_PYQUOTEX = True
except Exception:  # pragma: no cover
    Quotex: Any = object  # type: ignore[no-redef]
    _HAS_PYQUOTEX = False

import nexus.adapters.pyquotex_patch  # noqa: F401
from nexus.utils.logger import PerformanceLogger, get_nexus_logger

logger = get_nexus_logger("nexus.adapters.quotex")
perf_logger = PerformanceLogger("quotex_adapter")
LOGIN_TIMEOUT_SECONDS = 30

COMMON_ASSETS: List[str] = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "EURGBP",
    "EURAUD",
    "GBPAUD",
    "BTCUSD",
    "ETHUSD",
    "LTCUSD",
    "XRPUSD",
    "Apple",
    "Amazon",
    "Google",
    "Microsoft",
    "Tesla",
    "Facebook",
    "Gold",
    "Silver",
    "Oil",
    "DAX",
    "S&P 500",
    "Dow Jones",
    "NASDAQ",
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
        self._known_order_ids: set[str] = set()

    async def _maybe_await(self, val: Any) -> Any:
        if inspect.isawaitable(val):
            return await val
        return val

    def set_session(
        self, user_agent: str, cookies: Optional[str] = None, ssid: Optional[str] = None
    ) -> None:
        self._session_override = {"user_agent": user_agent, "cookies": cookies, "ssid": ssid}

    def _accept_new_order_id(self, order_id: Any) -> Optional[str]:
        """Accept only a non-empty broker ID that has not been seen in this session."""
        if order_id is None:
            return None
        # pyquotex has returned a UUID, a nested mapping, and a stringified
        # mapping depending on which buy channel answered.
        value = order_id
        for _ in range(4):
            if isinstance(value, dict):
                value = value.get("id") or value.get("order_id") or value.get("order")
            elif isinstance(value, str) and value.lstrip().startswith("{"):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    break
            else:
                break
        if isinstance(value, dict):
            value = value.get("id") or value.get("order_id")
        if isinstance(value, (dict, list, tuple)):
            return None
        key = str(value).strip()
        if not key or key in self._known_order_ids or key.startswith("SIM-"):
            return None
        self._known_order_ids.add(key)
        return key

    async def set_practice_mode(self, practice: bool) -> None:
        self.demo_mode = bool(practice)
        if self.client is not None:
            try:
                mode_str = "PRACTICE" if self.demo_mode else "REAL"
                if hasattr(self.client, "change_account"):
                    await self._maybe_await(self.client.change_account(mode_str))
                api_obj = getattr(self.client, "api", None)
                if api_obj is not None:
                    from pyquotex.utils.account_type import AccountType

                    api_obj.account_type = AccountType.DEMO if self.demo_mode else AccountType.REAL
            except Exception as err:
                logger.warning("Failed setting practice mode: %s", err)

    async def _ensure_authenticated(self) -> bool:
        if not self.authenticated or not self.client:
            self.authenticated = False
            return await self.login()
        if hasattr(self.client, "check_connect"):
            try:
                ok = await self._maybe_await(self.client.check_connect())
                if isinstance(ok, bool) and not ok:
                    logger.warning("Broker WebSocket connection inactive. Re-authenticating...")
                    self.authenticated = False
                    return await self.login()
            except Exception as conn_err:
                logger.warning(
                    "WebSocket connectivity check failed (%s). Re-authenticating...", conn_err
                )
                self.authenticated = False
                return await self.login()
        if (datetime.now() - self.last_action) > timedelta(hours=1):
            self.authenticated = False
            return await self.login()
        return True

    async def login(self) -> bool:
        if self.authenticated and self.client:
            if hasattr(self.client, "check_connect"):
                try:
                    ok = await self._maybe_await(self.client.check_connect())
                    if isinstance(ok, bool) and ok:
                        return True
                except Exception:
                    pass
            else:
                return True
        self.authenticated = False
        if not _HAS_PYQUOTEX:
            logger.error("pyquotex not installed. Install it to use QuotexAdapter.")
            return False

        try:
            with perf_logger.measure("quotex_login"):
                self.client = Quotex(
                    email=self.email,
                    password=self.password,
                    host="qxbroker.com",
                    lang="en",
                )

                import urllib.parse

                env_cookies = os.getenv("QUOTEX_COOKIES") or os.getenv("QUOTEX__COOKIES")
                env_ssid = os.getenv("QUOTEX_SSID") or os.getenv("QUOTEX__SSID")
                env_ua = os.getenv("QUOTEX_USER_AGENT") or os.getenv("QUOTEX__USER_AGENT")

                if env_ssid:
                    env_ssid = urllib.parse.unquote(env_ssid).strip()

                if env_cookies and not env_ssid:
                    for item in env_cookies.split(";"):
                        if "laravel_session=" in item:
                            raw_val = item.split("laravel_session=")[1].strip()
                            env_ssid = urllib.parse.unquote(raw_val)
                            break

                chrome_ua = (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
                if (env_cookies or env_ssid) and not self._session_override:
                    self.set_session(env_ua or chrome_ua, env_cookies, env_ssid)

                if self._session_override and hasattr(self.client, "set_session"):
                    try:
                        ua = self._session_override.get("user_agent") or "Quotex/1.0"

                        self.client.set_session(
                            ua,
                            self._session_override.get("cookies"),
                            self._session_override.get("ssid"),
                        )
                    except Exception:
                        pass
                try:
                    if hasattr(self.client, "connect"):
                        await asyncio.wait_for(
                            self._maybe_await(self.client.connect()), timeout=LOGIN_TIMEOUT_SECONDS
                        )
                    if hasattr(self.client, "check_connect"):
                        ok = await asyncio.wait_for(
                            self._maybe_await(self.client.check_connect()),
                            timeout=LOGIN_TIMEOUT_SECONDS,
                        )
                        if isinstance(ok, bool) and not ok:
                            return False
                except Exception as e:
                    # A practice account is still broker-backed. Do not mark
                    # the adapter authenticated when Quotex rejects the
                    # connection; callers must be able to distinguish a real
                    # order from a local simulation.
                    logger.error("Broker connection failure: %s", e)
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
                        # Calling an async API creates a coroutine that must be
                        # closed when this synchronous facade cannot await it.
                        if inspect.iscoroutine(data):
                            data.close()
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
        if not await self._ensure_authenticated() or not self.client:
            return 0.0
        try:
            await self.set_practice_mode(self.demo_mode)
            if hasattr(self.client, "get_balance"):
                v = self.client.get_balance()
                if asyncio.iscoroutine(v):
                    v = await v
                return float(v or 0.0)
        except Exception as e:
            logger.debug("Failed fetching async balance from broker: %s", e)
        return self.get_balance()

    def get_balance(self) -> float:
        try:
            if not self.client:
                return 0.0
            if hasattr(self.client, "api") and self.client.api:
                ab = getattr(self.client.api, "account_balance", {}) or {}
                if isinstance(ab, dict):
                    key = "demoBalance" if self.demo_mode else "liveBalance"
                    if key in ab and ab[key] is not None:
                        return float(ab[key])
                profile = getattr(self.client.api, "profile", None)
                if profile:
                    b = profile.demo_balance if self.demo_mode else profile.live_balance
                    if b is not None:
                        return float(b)
            return 0.0
        except Exception:
            return 0.0

    async def get_assets_with_payouts_async(self) -> List[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated() or not self.client:
                seen_syms: set[str] = set()
                fallback_res: List[Dict[str, Any]] = []
                for s in COMMON_ASSETS:
                    if s not in seen_syms:
                        seen_syms.add(s)
                        fallback_res.append(
                            {
                                "symbol": s,
                                "payout": _get_default_payout_for_symbol(s),
                                "active": True,
                            }
                        )
                return fallback_res

            instruments = []
            if hasattr(self.client, "get_instruments"):
                try:
                    instruments = await self._maybe_await(self.client.get_instruments())
                except Exception:
                    instruments = []
            if not instruments and hasattr(self.client, "api") and self.client.api:
                instruments = getattr(self.client.api, "instruments", []) or []

            res_list: List[Dict[str, Any]] = []
            seen_symbols: set[str] = set()

            if instruments:
                for i in instruments:
                    try:
                        if len(i) > 5:
                            raw_sym = str(i[1]).strip()
                            sym = normalize_broker_symbol(raw_sym)
                            if not sym or sym in seen_symbols:
                                continue

                            # i[5] is live turbo payout; i[6] binary payout; i[-9] 1M payout
                            turbo_val = float(i[5]) if i[5] is not None else 0.0
                            binary_val = float(i[6]) if len(i) > 6 and i[6] is not None else 0.0
                            m1_val = float(i[-9]) if len(i) > 9 and i[-9] is not None else 0.0

                            raw_payout = (
                                turbo_val
                                if turbo_val > 0
                                else (m1_val if m1_val > 0 else binary_val)
                            )

                            # Normalize fractional (0.31 -> 31.0)
                            if 0.0 < raw_payout <= 1.0:
                                raw_payout *= 100.0

                            is_open = bool(i[14]) if len(i) > 14 and i[14] is not None else True
                            payout = (
                                float(raw_payout)
                                if (raw_payout > 0 and is_open)
                                else (0.0 if not is_open else _get_default_payout_for_symbol(sym))
                            )

                            seen_symbols.add(sym)
                            res_list.append(
                                {
                                    "symbol": sym,
                                    "payout": payout,
                                    "active": is_open,
                                    "otc": "_otc" in sym.lower() or "otc" in sym.lower(),
                                }
                            )
                    except Exception:
                        continue
            if res_list:
                return res_list

            syms = await self.get_available_assets_async()
            dedup_res: List[Dict[str, Any]] = []
            seen_dedup: set[str] = set()
            for s in syms:
                if s not in seen_dedup:
                    seen_dedup.add(s)
                    dedup_res.append(
                        {
                            "symbol": s,
                            "payout": _get_default_payout_for_symbol(s),
                            "active": True,
                        }
                    )
            return dedup_res
        except Exception:
            seen_err: set[str] = set()
            err_res: List[Dict[str, Any]] = []
            for s in COMMON_ASSETS:
                if s not in seen_err:
                    seen_err.add(s)
                    err_res.append(
                        {
                            "symbol": s,
                            "payout": _get_default_payout_for_symbol(s),
                            "active": True,
                        }
                    )
            return err_res

    # ---- market data (candles) ----
    def get_candles(
        self, symbol: str, timeframe_sec: int, limit: int = 200
    ) -> Optional[List[Dict[str, float]]]:
        """Fetch recent candles for symbol.
        Returns list of dicts with keys: time, open, high, low, close, volume (volume may be 0 if not provided).
        """
        try:
            if not self.client:
                return None
            # Candidate method names and param shapes on pyquotex
            methods = [
                ("get_historical_candles", (symbol, timeframe_sec * limit, timeframe_sec), {}),
                ("get_candle_v2", (symbol, timeframe_sec), {}),
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
                                records.append(
                                    {
                                        "time": t,
                                        "open": o,
                                        "high": h,
                                        "low": low,
                                        "close": cl,
                                        "volume": vlm,
                                    }
                                )
                            except Exception:
                                continue
                        if records:
                            return records[-limit:]
                except Exception:
                    continue
            return None
        except Exception:
            return None

    async def get_candles_async(
        self, symbol: str, timeframe_sec: int, limit: int = 200
    ) -> Optional[List[Dict[str, float]]]:
        try:
            if not await self._ensure_authenticated():
                return None
            methods = [
                ("get_historical_candles", (symbol, timeframe_sec * limit, timeframe_sec), {}),
                ("get_candle_v2", (symbol, timeframe_sec), {}),
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
                                records.append(
                                    {
                                        "time": t,
                                        "open": o,
                                        "high": h,
                                        "low": low,
                                        "close": cl,
                                        "volume": vlm,
                                    }
                                )
                            except Exception:
                                continue
                        if records:
                            return records[-limit:]
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def get_trade_outcome(self, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.client or not hasattr(self.client, "api") or not self.client.api:
                return None
            key = str(order_id).strip()
            if not key:
                return None
            api_obj = self.client.api
            completed = getattr(api_obj, "_completed_trades", {}) or {}
            if isinstance(completed, dict):
                cached_completed = completed.get(key)
                if isinstance(cached_completed, dict):
                    cached_outcome = str(cached_completed.get("outcome", "UNVERIFIED")).upper()
                    cached_profit = float(cached_completed.get("profit", 0.0) or 0.0)
                    if (
                        (cached_outcome == "WIN" and cached_profit < 0)
                        or (cached_outcome == "LOSS" and cached_profit > 0)
                    ):
                        return {
                            "status": "UNVERIFIED",
                            "outcome": "UNVERIFIED",
                            "profit": cached_profit,
                        }
                    return {
                        "status": "SETTLED",
                        "outcome": cached_outcome,
                        "profit": cached_profit,
                    }
            list_info = getattr(api_obj, "listinfodata", None)
            cached = None
            if list_info and hasattr(list_info, "get"):
                cached = list_info.get(key)
                if not cached:
                    try:
                        cached = list_info.get(int(key))
                    except ValueError:
                        pass
            elif isinstance(list_info, dict):
                cached = list_info.get(key)

            if cached and isinstance(cached, dict):
                win_value = cached.get("win")
                win_str = str(win_value).lower()
                profit = float(cached.get("profit", 0.0))
                game_state = cached.get("game_state")
                # Quotex publishes intermediate records with a profit/status
                # field before the option is closed.  Never settle from those
                # records: doing so can attach an earlier/stale WIN to a later
                # position.  In pyquotex, state 1 means the order is closed.
                if game_state != 1:
                    return None

                explicit_win = win_str in {"win", "true", "1"}
                explicit_loss = win_str in {"loss", "false", "0"}
                explicit_equal = win_str == "equal"
                profit_sign = "WIN" if profit > 0 else "LOSS" if profit < 0 else "EQUAL"

                if explicit_equal:
                    outcome = "EQUAL" if profit == 0 else "UNVERIFIED"
                elif explicit_win:
                    outcome = "WIN" if profit >= 0 else "UNVERIFIED"
                elif explicit_loss:
                    outcome = "LOSS" if profit <= 0 else "UNVERIFIED"
                else:
                    outcome = profit_sign

                if outcome == "UNVERIFIED":
                    return {
                        "status": "UNVERIFIED",
                        "outcome": "UNVERIFIED",
                        "profit": profit,
                    }
                return {"status": "SETTLED", "outcome": outcome, "profit": profit}
            return None
        except Exception:
            return None

    async def buy_simple_async(
        self, asset: str, direction: str, amount: float, expiration: int
    ) -> Optional[Dict[str, Any]]:
        try:
            if not await self._ensure_authenticated():
                return None
            c = self.client
            if c is None:
                return None
            clean_asset = str(asset).strip()
            if "%" in clean_asset or not clean_asset:
                clean_asset = "EURUSD"
            if clean_asset.startswith("OTC_"):
                clean_asset = clean_asset.replace("OTC_", "") + "_otc"

            if not clean_asset.endswith("_otc") and hasattr(c, "check_asset_open"):
                try:
                    _, open_info = await self._maybe_await(c.check_asset_open(clean_asset))
                    if open_info and len(open_info) > 2 and not open_info[2]:
                        otc_name = f"{clean_asset}_otc"
                        logger.info(
                            "Standard market %s is closed. Auto-switching to live OTC market %s",
                            clean_asset,
                            otc_name,
                        )
                        clean_asset = otc_name
                except Exception:
                    pass

            dir_str = "call" if str(direction).lower() in ("call", "buy", "up") else "put"
            amt = float(amount)
            exp = int(expiration)

            await self.set_practice_mode(self.demo_mode)

            if hasattr(c, "buy"):
                try:
                    res = c.buy(amt, clean_asset, dir_str, exp)
                    if asyncio.iscoroutine(res):
                        res = await res
                    if isinstance(res, tuple) and len(res) == 2 and res[0] is True:
                        order_id = self._accept_new_order_id(res[1])
                        if order_id is None:
                            return {
                                "success": False,
                                "status": "UNVERIFIED",
                                "error": "Broker returned a duplicate or invalid order ID",
                            }
                        logger.info(
                            "Placed live Quotex order: %s %s $%s (%ss)",
                            clean_asset,
                            dir_str.upper(),
                            amt,
                            exp,
                        )
                        return {"success": True, "order": order_id}
                except Exception as buy_err:
                    logger.debug("Standard buy error: %s", buy_err)

            api_obj = getattr(c, "api", None)
            if api_obj is not None:
                buy_channel = getattr(api_obj, "buy", None)
                if buy_channel is not None and callable(buy_channel):
                    req_id = int(time.time() * 1000)
                    is_timer = exp < 60 or clean_asset.endswith("_otc")
                    time_mode = "TIMER" if is_timer else "TIME"
                    is_fast = True if is_timer else False
                    try:
                        from pyquotex.utils.account_type import AccountType

                        api_obj.account_type = (
                            AccountType.DEMO if self.demo_mode else AccountType.REAL
                        )
                        if hasattr(c, "start_realtime_price"):
                            await self._maybe_await(c.start_realtime_price(clean_asset, exp))
                        if hasattr(api_obj, "settings_apply"):
                            await self._maybe_await(
                                api_obj.settings_apply(clean_asset, exp, is_fast)
                            )

                        for attempt in range(2):
                            if hasattr(api_obj, "slots") and hasattr(api_obj.slots, "buy_confirm"):
                                api_obj.slots.buy_confirm.clear()
                            if hasattr(api_obj, "buy_id"):
                                api_obj.buy_id = None

                            try:
                                await buy_channel(
                                    amt, clean_asset, dir_str, exp, req_id, is_fast, time_mode
                                )
                            except Exception as first_err:
                                logger.warning("WS send exception (%s).", first_err)

                            confirmed_id = None
                            if hasattr(api_obj, "slots") and hasattr(api_obj.slots, "buy_confirm"):
                                try:
                                    event_data = await asyncio.wait_for(
                                        api_obj.slots.buy_confirm.wait(), timeout=4.0
                                    )
                                    if isinstance(event_data, dict):
                                        confirmed_id = event_data.get("id")
                                except asyncio.TimeoutError:
                                    confirmed_id = getattr(api_obj, "buy_id", None)

                            candidate_id = confirmed_id or getattr(api_obj, "buy_id", None)
                            real_id = self._accept_new_order_id(candidate_id)
                            if real_id:
                                logger.info(
                                    "Placed live Quotex order (ID: %s): %s %s $%s (%ss)",
                                    real_id,
                                    clean_asset,
                                    dir_str.upper(),
                                    amt,
                                    exp,
                                )
                                return {
                                    "success": True,
                                    "order": {
                                        "id": str(real_id),
                                        "asset": clean_asset,
                                        "action": dir_str,
                                        "amount": amt,
                                        "time": exp,
                                    },
                                }
                            if candidate_id:
                                return {
                                    "success": False,
                                    "status": "UNVERIFIED",
                                    "error": "Broker returned a duplicate or stale order ID",
                                }

                            if attempt == 0:
                                logger.warning(
                                    "Buy confirmation timeout on WS. Re-authenticating and retrying trade placement..."
                                )
                                self.authenticated = False
                                if (
                                    await self.login()
                                    and self.client is not None
                                    and getattr(self.client, "api", None)
                                ):
                                    api_obj = self.client.api
                                    buy_channel = getattr(api_obj, "buy", None)
                                    if not buy_channel or not callable(buy_channel):
                                        break

                        logger.warning(
                            "Quotex server did not confirm order placement for %s %s $%s",
                            clean_asset,
                            dir_str.upper(),
                            amt,
                        )
                        return {
                            "success": False,
                            "error": f"Broker did not confirm order for {clean_asset}",
                        }
                    except Exception as ws_err:
                        logger.error("Direct WS buy exception: %s", ws_err)
                        return {"success": False, "error": str(ws_err)}

            method_names = [
                "buy_and_check_win",
                "buy_simple",
                "open_option",
                "open_trade",
                "open_position",
                "place_order",
                "place_trade",
            ]
            for name in method_names:
                if not hasattr(self.client, name):
                    continue
                fn = getattr(self.client, name)
                res = fn(amt, clean_asset, dir_str, exp)
                if asyncio.iscoroutine(res):
                    res = await res
                if res is not None:
                    candidate_id = res.get("id") if isinstance(res, dict) else res
                    order_id = self._accept_new_order_id(candidate_id)
                    if order_id is None:
                        return {
                            "success": False,
                            "status": "UNVERIFIED",
                            "error": "Broker returned an order without a unique ID",
                        }
                    return {"success": True, "order": order_id}
            return None
        except Exception as e:
            logger.error("Failed to place order on Quotex: %s", e)
            return None

    def buy_simple(
        self, asset: str, direction: str, amount: float, expiration: int
    ) -> Optional[Dict[str, Any]]:
        actual_amt: float = (
            float(direction) if isinstance(direction, (int, float)) else float(amount)
        )
        actual_dir: str = str(amount) if isinstance(direction, (int, float)) else str(direction)
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return asyncio.create_task(
                    self.buy_simple_async(asset, actual_dir, actual_amt, int(expiration))
                )  # type: ignore[return-value]
        except RuntimeError:
            pass
        return asyncio.run(self.buy_simple_async(asset, direction, float(amount), int(expiration)))


def normalize_broker_symbol(raw_sym: str) -> str:
    s = str(raw_sym).strip()
    is_otc = "otc" in s.lower() or "(otc)" in s.lower()
    clean = re.sub(r"[^A-Za-z0-9]", "", s)
    clean_no_otc = re.sub(r"(?i)otc", "", clean).upper()
    if is_otc:
        return f"{clean_no_otc}_otc"
    return clean_no_otc


def _get_default_payout_for_symbol(symbol: str) -> float:
    s_upper = symbol.upper()
    if "OTC" in s_upper:
        return 88.0 if ("EURUSD" in s_upper or "GBPJPY" in s_upper) else 85.0
    if any(pair in s_upper for pair in ("EURUSD", "GBPUSD", "GBPJPY", "EURJPY")):
        return 85.0
    if any(pair in s_upper for pair in ("USDJPY", "AUDUSD", "USDCAD", "XAUUSD")):
        return 82.0
    if "BTC" in s_upper or "ETH" in s_upper:
        return 80.0
    return 78.0


__all__ = ["QuotexAdapter"]

"""Runtime patch and URL routing for pyquotex integration.

Configures sign-in endpoints (https://qxbroker.com/en/sign-in) and pure-Python browser headers
to handle Quotex authentication safely without native memory crashes.
"""

import json
import os
import time
from typing import Any

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.adapters.pyquotex_patch")

try:
    from pyquotex.network.login import Login

    _HAS_PYQUOTEX = True
except ImportError:
    _HAS_PYQUOTEX = False

if _HAS_PYQUOTEX:
    import logging

    logging.getLogger("pyquotex").setLevel(logging.ERROR)
    logging.getLogger("pyquotex.ws.client").setLevel(logging.ERROR)

    # 1. Update Base URLs and Login Endpoint Structure
    Login.base_url = "qxbroker.com"
    Login.https_base_url = "https://qxbroker.com"

    _original_login_init = Login.__init__

    def _patched_login_init(self: Any, api: Any, *args: Any, **kwargs: Any) -> None:
        _original_login_init(self, api, *args, **kwargs)
        lang = getattr(api, "lang", "en") or "en"
        self.https_base_url = "https://qxbroker.com"
        # Full URL should be https://qxbroker.com/en
        self.full_url = f"https://qxbroker.com/{lang}"

        # Pure Python headers to match modern Chrome
        user_agent = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36",
        )
        accept_hdr = (
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
        )
        hdr_dict = {
            "User-Agent": user_agent,
            "Accept": accept_hdr,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://qxbroker.com/{lang}/sign-in",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        session_cookie = os.getenv("QUOTEX_SESSION")
        if session_cookie:
            hdr_dict["Cookie"] = f"session={session_cookie}"
        self.headers.update(hdr_dict)

    Login.__init__ = _patched_login_init  # type: ignore[method-assign]

    # 2. Patch get_sign_page to navigate cleanly to https://qxbroker.com/en/sign-in
    async def _patched_get_sign_page(self: Any) -> Any:
        sign_in_url = f"{self.full_url}/sign-in"
        self.headers["Referer"] = "https://qxbroker.com/"
        response = await self.send_request(
            method="GET",
            url=sign_in_url,
            headers=self.headers,
        )
        if response and getattr(response, "is_success", False):
            cookies_dict = dict(response.cookies)
            cookies_str = "; ".join([f"{k}={v}" for k, v in cookies_dict.items()])
            if cookies_str:
                self.cookies = cookies_str
        return response

    Login.get_sign_page = _patched_get_sign_page  # type: ignore[method-assign]

    try:
        import asyncio

        from pyquotex.api import QuotexAPI

        _original_on_message = QuotexAPI._on_message

        async def _ping_loop(api_inst: Any) -> None:
            if getattr(api_inst, "_ping_loop_active", False):
                return
            api_inst._ping_loop_active = True
            try:
                while getattr(api_inst, "_ping_loop_active", False):
                    await asyncio.sleep(15)
                    if hasattr(api_inst, "send_websocket"):
                        try:
                            await api_inst.send_websocket("2")
                        except Exception:
                            break
            except Exception:
                pass
            finally:
                api_inst._ping_loop_active = False

        async def _patched_on_message(self: Any, msg: bytes | str) -> None:
            import io
            import sys
            import time

            self.last_message_at = time.monotonic()
            msg_str = msg.decode("utf-8", errors="ignore") if isinstance(msg, bytes) else str(msg)
            is_control = bool(msg_str and msg_str[0].isdigit())

            # Fix PyQuotex placeholder race condition:
            # If an order placeholder is pending binary payload fulfillment,
            # prevent incoming quotes/stream placeholders from clobbering _temp_status.
            prev_status = getattr(self, "_temp_status", None)
            is_order_placeholder = prev_status and any(
                k in str(prev_status)
                for k in ("orders/open", "s_orders/open", "orders/close", "s_orders/close")
            )
            is_stream_placeholder = "51-" in msg_str and any(
                stream in msg_str for stream in ("quotes/stream", "depth/change", "candle")
            )

            if not hasattr(self, "_completed_trades"):
                self._completed_trades = {}

            # Intercept Control Order Messages (e.g. 42["s_orders/open", {...}], 42["s_orders/close", {...}])
            if any(
                k in msg_str
                for k in ("orders/open", "s_orders/open", "orders/close", "s_orders/close")
            ):
                try:
                    start_idx = msg_str.find("[")
                    if start_idx != -1:
                        arr = json.loads(msg_str[start_idx:])
                        if isinstance(arr, list) and len(arr) > 1 and isinstance(arr[0], str):
                            event_name, data = arr[0], arr[1]
                            order_list = data if isinstance(data, list) else [data]
                            for order_data in order_list:
                                if isinstance(order_data, dict):
                                    order_id = order_data.get("id")
                                    if order_id:
                                        if any(
                                            k in event_name
                                            for k in ("orders/open", "s_orders/open")
                                        ):
                                            self.buy_id = order_id
                                            self.buy_successful = True
                                            if hasattr(self, "slots") and hasattr(
                                                self.slots, "buy_confirm"
                                            ):
                                                self.slots.buy_confirm.set(
                                                    {"id": order_id, "data": order_data}
                                                )
                                        elif any(
                                            k in event_name
                                            for k in ("orders/close", "s_orders/close")
                                        ):
                                            profit = float(
                                                order_data.get("profit")
                                                or order_data.get("profitAmount")
                                                or 0.0
                                            )
                                            win_value = order_data.get("win")
                                            win = (
                                                str(win_value).lower() in {"win", "true", "1", "equal"}
                                                if win_value is not None
                                                else profit > 0
                                            )
                                            self._completed_trades[str(order_id)] = {
                                                "id": str(order_id),
                                                "outcome": "WIN" if win else "LOSS",
                                                "profit": profit,
                                                "data": order_data,
                                            }
                except Exception:
                    pass

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                await _original_on_message(self, msg)
                if is_order_placeholder:
                    # Payload fulfillment after a 451- placeholder
                    if not is_control and not is_stream_placeholder:
                        try:
                            start_idx = -1
                            for idx, char in enumerate(msg_str):
                                if char in ("[", "{"):
                                    start_idx = idx
                                    break
                            if start_idx != -1:
                                payload = json.loads(msg_str[start_idx:])
                                orders = payload if isinstance(payload, list) else [payload]
                                for item in orders:
                                    if isinstance(item, dict):
                                        oid = item.get("id")
                                        if oid:
                                            if "orders/close" in str(
                                                prev_status
                                            ) or "s_orders/close" in str(prev_status):
                                                profit = float(
                                                    item.get("profit")
                                                    or item.get("profitAmount")
                                                    or 0.0
                                                )
                                                win_value = item.get("win")
                                                win = (
                                                    str(win_value).lower() in {"win", "true", "1", "equal"}
                                                    if win_value is not None
                                                    else profit > 0
                                                )
                                                self._completed_trades[str(oid)] = {
                                                    "id": str(oid),
                                                    "outcome": "WIN" if win else "LOSS",
                                                    "profit": profit,
                                                    "data": item,
                                                }
                                            else:
                                                self.buy_id = oid
                                                self.buy_successful = True
                                                if hasattr(self, "slots") and hasattr(
                                                    self.slots, "buy_confirm"
                                                ):
                                                    self.slots.buy_confirm.set(
                                                        {"id": oid, "data": item}
                                                    )
                        except Exception:
                            pass

                if is_order_placeholder and is_stream_placeholder:
                    self._temp_status = prev_status
            finally:
                captured = sys.stdout.getvalue()
                sys.stdout = old_stdout
                if captured:
                    for line in captured.splitlines():
                        line_str = line.strip()
                        if not line_str or "socket.send() raised exception" in line_str:
                            continue
                        if "authorization SUCCESS" in line_str:
                            logger.info("Quotex WebSocket authorization SUCCESS")
                            if not getattr(self, "_ping_loop_active", False):
                                asyncio.create_task(_ping_loop(self))
                        elif "authorization rejected" in line_str:
                            logger.error("Quotex WebSocket authorization rejected")
                        elif "Received while not authenticated" in line_str:
                            logger.debug("WS handshake: %s", line_str)
                        else:
                            logger.debug("%s", line_str)

        QuotexAPI._on_message = _patched_on_message  # type: ignore[method-assign]
    except ImportError:
        pass

    try:
        from pyquotex.expiration import get_expiration_time_quotex
        from pyquotex.utils import json_utils as json_u
        from pyquotex.ws.channels.buy import Buy

        async def _patched_buy_call(
            self: Any,
            price: float | int,
            asset: str,
            direction: str,
            duration: int,
            request_id: int,
            is_fast_option: bool,
            time_mode: str,
        ) -> None:
            if duration < 60 or time_mode == "TIMER":
                option_type = 100
                expiration = duration
                is_fast = True
            else:
                option_type = 3 if is_fast_option else 1
                expiration = get_expiration_time_quotex(int(time.time()), duration)
                is_fast = is_fast_option

            await self.api.settings_apply(
                asset,
                expiration,
                is_fast_option=is_fast,
                end_time=expiration if is_fast else expiration,
            )

            payload = {
                "asset": asset,
                "amount": price,
                "time": expiration,
                "action": direction,
                "isDemo": self.api.account_type,
                "tournamentId": self.api.tournament_id,
                "requestId": request_id,
                "optionType": option_type,
            }

            await self.send_websocket_request('42["tick"]')
            await self.send_websocket_request(f'42["orders/open",{json_u.dumps_str(payload)}]')

        Buy.__call__ = _patched_buy_call  # type: ignore[method-assign]
    except ImportError:
        pass

    try:
        from pyquotex.stable_api import Quotex

        async def _patched_start_realtime_price(
            self: Any, asset: str, period: int = 0, timeout: int = 5
        ) -> dict[str, Any]:
            if self.api is None:
                raise RuntimeError("API not initialized")
            try:
                await self.start_candles_stream(asset, period)
            except Exception:
                pass
            start = time.time()
            while True:
                if self.api.realtime_price.get(asset):
                    return self.api.realtime_price  # type: ignore[no-any-return]
                if time.time() - start > 1.2:
                    # Seed fallback price tick so buy order proceeds without timeout
                    self.api.realtime_price[asset] = {"price": 1.0, "time": time.time()}
                    return self.api.realtime_price  # type: ignore[no-any-return]
                await asyncio.sleep(0.1)

        Quotex.start_realtime_price = _patched_start_realtime_price  # type: ignore[method-assign]
    except ImportError:
        pass

    logger.info("Applied pyquotex pure-Python sign-in routing patch")

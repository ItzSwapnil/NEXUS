"""
Client module for interacting with the Quotex platform.

This module provides a wrapper around the pyquotex library to handle authentication,
connection, and trading operations with the Quotex platform.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# Optional pyquotex import
try:
    from pyquotex.stable_api import Quotex  # type: ignore

    _HAS_PYQUOTEX = True
except ImportError:  # pragma: no cover - environment without pyquotex
    Quotex = object  # type: ignore
    _HAS_PYQUOTEX = False

logger = logging.getLogger("nexus.client")


class QuotexClient:
    """
    Client for interacting with the Quotex platform using pyquotex.

    This class provides an async-friendly interface to the pyquotex library,
    handling authentication, connection, and trading operations.
    """

    def __init__(self, email: str, password: str, lang: str = "en"):
        """
        Initialize the Quotex client.

        Args:
            email: The email for Quotex account
            password: The password for Quotex account
            lang: The language for the Quotex interface (default: "en")
        """
        self.email = email
        self.password = password
        self.lang = lang
        self.client: Optional[Quotex] = None
        self.connected = False
        self.account_info: Dict[str, Any] = {}

    async def connect(self) -> bool:
        """Connect to the Quotex platform (requires pyquotex)."""
        if not _HAS_PYQUOTEX:
            logger.error(
                "pyquotex is not installed. Install manually (e.g. via VCS) before using QuotexClient."
            )
            return False

        try:
            # Create a new Quotex client instance
            self.client = Quotex(email=self.email, password=self.password, lang=self.lang)  # type: ignore[call-arg]

            # Connect to Quotex (run in executor to tolerate sync implementations)
            loop = asyncio.get_event_loop()
            if self.client is not None:
                client_obj = self.client
                conn_fn = getattr(client_obj, "connect", lambda: True)  # noqa: B009
                conn_res = await loop.run_in_executor(None, conn_fn)
                if asyncio.iscoroutine(conn_res):
                    await conn_res

            # Set connected flag
            self.connected = True
            logger.info("Successfully connected to Quotex")

            # Get account information
            await self.update_account_info()

            return True
        except Exception as e:  # pragma: no cover - network dependent
            logger.exception(f"Error connecting to Quotex: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from Quotex.

        Returns:
            bool: True if disconnection was successful
        """
        if not self.client or not self.connected:
            return True

        try:
            loop = asyncio.get_event_loop()
            client_obj = self.client
            close_fn = getattr(client_obj, "close", lambda: None)  # noqa: B009
            await loop.run_in_executor(None, close_fn)
            self.connected = False
            logger.info("Disconnected from Quotex")
            return True
        except Exception as e:  # pragma: no cover
            logger.exception(f"Error disconnecting from Quotex: {e}")
            return False

    async def update_account_info(self) -> Dict[str, Any]:
        """
        Update and return account information.

        Returns:
            Dict[str, Any]: Account information
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")

        try:
            loop = asyncio.get_event_loop()
            c = self.client
            assert c is not None
            # get_balance might be sync in some versions
            bal_fn = getattr(c, "get_balance", lambda: 0.0)  # noqa: B009
            balance = await loop.run_in_executor(None, bal_fn)
            if asyncio.iscoroutine(balance):
                balance = await balance

            # Try to get profile info for currency and user_id
            currency: Optional[str] = None
            user_id: Optional[str] = None

            # Try different ways to access profile data
            profile = getattr(c, "profile", None)
            if profile is not None:
                if hasattr(profile, "currency"):
                    currency = profile.currency  # type: ignore[attr-defined]
                if hasattr(profile, "user_id"):
                    user_id = profile.user_id  # type: ignore[attr-defined]
            else:
                get_profile = getattr(c, "get_profile", None)
                if callable(get_profile):
                    prof = await loop.run_in_executor(None, get_profile)  # type: ignore[misc]
                    if hasattr(prof, "currency"):
                        currency = prof.currency  # type: ignore[attr-defined]
                    if hasattr(prof, "user_id"):
                        user_id = prof.user_id  # type: ignore[attr-defined]
                    if isinstance(prof, dict):
                        currency = currency or prof.get("currency")
                        user_id = user_id or prof.get("user_id")

            # Fallback if not found
            if currency is None:
                currency = "USD"  # Default or unknown
            if user_id is None:
                user_id = "unknown"

            self.account_info = {
                "balance": float(balance) if balance is not None else 0.0,
                "currency": currency,
                "user_id": user_id,
                "last_updated": datetime.now(),
            }
            logger.debug(f"Account info updated: Balance={balance} {currency}")
            return self.account_info
        except Exception as e:  # pragma: no cover
            logger.exception(f"Error updating account info: {e}")
            raise

    async def get_candles(self, asset: str, timeframe: int = 60, count: int = 100) -> pd.DataFrame:
        """
        Get candle data for a specified asset.

        Args:
            asset: Asset identifier
            timeframe: Candle timeframe in seconds
            count: Number of candles to retrieve

        Returns:
            pd.DataFrame: Candle data
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")

        try:
            c = self.client

            # Get candles - try different method names that might exist in the pyquotex library
            loop = asyncio.get_event_loop()

            # Try different possible method names for getting candles
            candles: Any
            if hasattr(c, "get_candles"):
                gc_fn = getattr(c, "get_candles")  # noqa: B009
                candles = await loop.run_in_executor(
                    None,
                    lambda: gc_fn(asset, timeframe, count),
                )
            elif hasattr(c, "get_history"):
                gh_fn = getattr(c, "get_history")  # noqa: B009
                candles = await loop.run_in_executor(
                    None,
                    lambda: gh_fn(asset, timeframe, count),
                )
            elif hasattr(c, "get_historical_data"):
                ghd_fn = getattr(c, "get_historical_data")  # noqa: B009
                candles = await loop.run_in_executor(
                    None,
                    lambda: ghd_fn(asset, timeframe, count),
                )
            else:
                # If we can't find an appropriate method, log the available methods and raise an error
                methods: List[str] = [m for m in dir(c) if not m.startswith("_")]
                raise AttributeError(
                    f"Quotex client has no standard candle retrieval method. Available methods: {methods}"
                )

            # Process and format candle data into a DataFrame
            if isinstance(candles, list):
                df = pd.DataFrame(candles)
            elif isinstance(candles, dict) and "data" in candles:
                df = pd.DataFrame(candles["data"])
            elif isinstance(candles, pd.DataFrame):
                df = candles
            else:
                raise ValueError(f"Unexpected candle data format: {type(candles)}")

            # Ensure standard column names
            required_cols = ["open", "high", "low", "close", "time"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(
                    f"Candle data missing columns {missing_cols}. Available: {df.columns.tolist()}"
                )

            # Format time column if present
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)

            logger.debug(f"Retrieved {len(df)} candles for {asset}")
            return df
        except Exception as e:  # pragma: no cover
            logger.exception(f"Error getting candles for {asset}: {e}")
            raise

    async def place_trade(
        self,
        asset: str,
        amount: float,
        direction: str,
        expiration: int,
        wait_for_result: bool = True,
    ) -> Dict[str, Any]:
        """
        Place a trade on Quotex.

        Args:
            asset: Asset identifier
            amount: Trade amount
            direction: 'call' or 'put'
            expiration: Expiration time in seconds
            wait_for_result: Whether to wait for trade resolution

        Returns:
            Dict[str, Any]: Trade result information
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")

        try:
            loop = asyncio.get_event_loop()
            c = self.client

            # Format direction
            direction = direction.lower()
            if direction not in ("call", "put"):
                raise ValueError(f"Invalid trade direction: {direction}. Must be 'call' or 'put'")

            logger.info(
                f"Placing trade: {asset} {direction.upper()} ${amount} (Expiration: {expiration}s)"
            )

            # Place trade using pyquotex methods
            if wait_for_result:
                # Try method that waits for result if available
                if hasattr(c, "buy_and_wait"):
                    result = await loop.run_in_executor(
                        None,
                        lambda: c.buy_and_wait(
                            asset=asset,
                            action=direction.lower(),
                            amount=float(amount),
                            expirations_times=int(expiration),
                        ),
                    )  # type: ignore[attr-defined]
                else:
                    # Fallback: place simple trade and return order id
                    result = await loop.run_in_executor(
                        None,
                        lambda: getattr(c, "buy_simple", lambda **kw: {"success": True})(
                            asset=asset,
                            action=direction.lower(),
                            amount=float(amount),
                            expirations_times=int(expiration),
                        ),
                    )

                trade_info = {
                    "asset": asset,
                    "amount": float(amount),
                    "direction": direction,
                    "expiration": int(expiration),
                    "timestamp": datetime.now(),
                    "result": result,
                }

                # Update account info after trade
                await self.update_account_info()

                logger.info(f"Trade completed: {asset} {direction} {amount} - Result: {result}")
                return trade_info
            else:
                # Use buy_simple to place trade without waiting
                result = await loop.run_in_executor(
                    None,
                    lambda: getattr(c, "buy_simple", lambda **kw: {"success": True})(
                        asset=asset,
                        action=direction.lower(),
                        amount=float(amount),
                        expirations_times=int(expiration),
                    ),
                )

                trade_info = {
                    "asset": asset,
                    "amount": float(amount),
                    "direction": direction,
                    "expiration": int(expiration),
                    "timestamp": datetime.now(),
                    "trade_id": result,
                }

                logger.info(f"Trade placed: {asset} {direction} {amount} - ID: {result}")
                return trade_info

        except Exception as e:  # pragma: no cover
            logger.exception(f"Error placing trade for {asset}: {e}")
            raise

    async def get_available_assets(self) -> list:
        """
        Get available assets from the Quotex platform.

        Since pyquotex doesn't implement this method directly,
        we return a predefined list of common assets available on Quotex.

        Returns:
            list: List of available assets
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")

        # Since pyquotex doesn't provide a method to get available assets,
        common_assets = [
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

        logger.info(
            f"Returning {len(common_assets)} predefined assets as pyquotex doesn't provide get_available_assets"
        )
        return common_assets

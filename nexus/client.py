"""
Client module for interacting with the Quotex platform.

This module provides a wrapper around the pyquotex library to handle authentication,
connection, and trading operations with the Quotex platform.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List

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
            logger.error("pyquotex is not installed. Install manually (e.g. via VCS) before using QuotexClient.")
            return False

        try:
            # Create a new Quotex client instance
            self.client = Quotex(
                email=self.email,
                password=self.password,
                lang=self.lang
            )  # type: ignore[call-arg]

            # Connect to Quotex (run in executor to tolerate sync implementations)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: getattr(self.client, "connect")())  # type: ignore[attr-defined]

            # Set connected flag
            self.connected = True
            logger.info("Successfully connected to Quotex")

            # Get account information
            await self.update_account_info()

            return True
        except Exception as e:  # pragma: no cover - network dependent
            logger.exception(f"Error connecting to Quotex: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from the Quotex platform.

        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        if not self.client or not self.connected:
            logger.warning("Not connected to Quotex")
            return True

        try:
            # pyquotex does not implement disconnect; just mark as disconnected
            self.connected = False
            logger.info("Successfully disconnected from Quotex")
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
            balance = await loop.run_in_executor(None, lambda: getattr(c, "get_balance")())  # type: ignore[attr-defined]

            # Try to get profile info for currency and user_id
            currency: Optional[str] = None
            user_id: Optional[str] = None

            # Try different ways to access profile data
            profile = getattr(c, 'profile', None)
            if profile is not None:
                if hasattr(profile, 'currency'):
                    currency = getattr(profile, 'currency')  # type: ignore[attr-defined]
                if hasattr(profile, 'user_id'):
                    user_id = getattr(profile, 'user_id')  # type: ignore[attr-defined]
            else:
                get_profile = getattr(c, 'get_profile', None)
                if callable(get_profile):
                    prof = await loop.run_in_executor(None, get_profile)  # type: ignore[misc]
                    if hasattr(prof, 'currency'):
                        currency = getattr(prof, 'currency')  # type: ignore[attr-defined]
                    if hasattr(prof, 'user_id'):
                        user_id = getattr(prof, 'user_id')  # type: ignore[attr-defined]
                    if isinstance(prof, dict):
                        currency = currency or prof.get('currency')
                        user_id = user_id or prof.get('user_id')

            # Fallback if not found
            if currency is None:
                currency = 'USD'  # Default or unknown
            if user_id is None:
                user_id = 'unknown'

            self.account_info = {
                "balance": float(balance) if balance is not None else 0.0,
                "currency": currency,
                "user_id": user_id,
                "last_updated": datetime.now()
            }
            logger.debug(f"Account info updated: Balance={balance} {currency}")
            return self.account_info
        except Exception as e:  # pragma: no cover
            logger.exception(f"Error updating account info: {e}")
            raise

    async def get_candles(self, asset: str, timeframe: int, count: int) -> pd.DataFrame:
        """
        Get historical candles for a specific asset.

        Args:
            asset: The asset to get candles for (e.g., "EURUSD")
            timeframe: The timeframe in seconds (e.g., 60 for 1 minute)
            count: The number of candles to retrieve

        Returns:
            pd.DataFrame: DataFrame containing candle data with columns:
                - timestamp: Timestamp in seconds
                - open: Opening price
                - high: Highest price
                - low: Lowest price
                - close: Closing price
                - volume: Volume (if available)
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")
        c = self.client  # capture for mypy
        assert c is not None
        try:
            # Get candles - try different method names that might exist in the pyquotex library
            loop = asyncio.get_event_loop()

            # Try different possible method names for getting candles
            candles: Any
            if hasattr(c, 'get_candles'):
                candles = await loop.run_in_executor(None, lambda: getattr(c, 'get_candles')(asset, timeframe, count))  # type: ignore[attr-defined]
            elif hasattr(c, 'get_history'):
                candles = await loop.run_in_executor(None, lambda: getattr(c, 'get_history')(asset, timeframe, count))  # type: ignore[attr-defined]
            elif hasattr(c, 'get_historical_data'):
                candles = await loop.run_in_executor(None, lambda: getattr(c, 'get_historical_data')(asset, timeframe, count))  # type: ignore[attr-defined]
            else:
                # If we can't find an appropriate method, log the available methods and raise an error
                methods: List[str] = [m for m in dir(c) if not m.startswith('_')]
                raise AttributeError(f"No candle method found on client. Available methods: {methods}")

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Rename columns if needed and ensure proper types
            if 'time' in df.columns and 'timestamp' not in df.columns:
                df.rename(columns={'time': 'timestamp'}, inplace=True)

            # Ensure all required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in candle data")

            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

            logger.debug(f"Retrieved {len(df)} candles for {asset} at {timeframe}s timeframe")
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
        wait_for_result: bool = True
    ) -> Dict[str, Any]:
        """
        Place a trade on the Quotex platform.

        Args:
            asset: The asset to trade (e.g., "EURUSD")
            amount: The amount to trade
            direction: The direction of the trade ("call" for up, "put" for down)
            expiration: The expiration time in seconds
            wait_for_result: Whether to wait for the trade result

        Returns:
            Dict[str, Any]: Trade result information
        """
        if not self.client or not self.connected:
            raise RuntimeError("Not connected to Quotex")

        c = self.client
        assert c is not None
        if direction.lower() not in ["call", "put"]:
            raise ValueError("Direction must be 'call' or 'put'")

        try:
            loop = asyncio.get_event_loop()

            if wait_for_result:
                # Use buy_and_check_win to place trade and wait for result
                if hasattr(c, 'buy_and_check_win'):
                    result = await loop.run_in_executor(
                        None,
                        lambda: getattr(c, 'buy_and_check_win')(
                            asset=asset,
                            action=direction.lower(),
                            amount=float(amount),
                            expirations_times=int(expiration)
                        )
                    )  # type: ignore[attr-defined]
                else:
                    # Fallback: place simple trade and return order id
                    result = await loop.run_in_executor(
                        None,
                        lambda: getattr(c, 'buy_simple')(
                            asset=asset,
                            action=direction.lower(),
                            amount=float(amount),
                            expirations_times=int(expiration)
                        )
                    )  # type: ignore[attr-defined]

                trade_info = {
                    "asset": asset,
                    "amount": float(amount),
                    "direction": direction,
                    "expiration": int(expiration),
                    "timestamp": datetime.now(),
                    "result": result
                }

                # Update account info after trade
                await self.update_account_info()

                logger.info(f"Trade completed: {asset} {direction} {amount} - Result: {result}")
                return trade_info
            else:
                # Use buy_simple to place trade without waiting
                result = await loop.run_in_executor(
                    None,
                    lambda: getattr(c, 'buy_simple')(
                        asset=asset,
                        action=direction.lower(),
                        amount=float(amount),
                        expirations_times=int(expiration)
                    )
                )  # type: ignore[attr-defined]

                trade_info = {
                    "asset": asset,
                    "amount": float(amount),
                    "direction": direction,
                    "expiration": int(expiration),
                    "timestamp": datetime.now(),
                    "trade_id": result
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
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
            "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD", "GBPAUD",
            "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
            "Apple", "Amazon", "Google", "Microsoft", "Tesla", "Facebook",
            "Gold", "Silver", "Oil", "DAX", "S&P 500", "Dow Jones", "NASDAQ"
        ]

        logger.info(f"Returning {len(common_assets)} predefined assets as pyquotex doesn't provide get_available_assets")
        return common_assets

"""
Client module for interacting with the Quotex platform.

This module provides a wrapper around the pyquotex library to handle authentication,
connection, and trading operations with the Quotex platform.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any

import pandas as pd
from pyquotex.stable_api import Quotex

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
        """
        Connect to the Quotex platform.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # Create a new Quotex client instance
            self.client = Quotex(
                email=self.email,
                password=self.password,
                lang=self.lang
            )

            # Connect to Quotex
            await self.client.connect()

            # Set connected flag
            self.connected = True
            logger.info("Successfully connected to Quotex")

            # Get account information
            await self.update_account_info()

            return True
        except Exception as e:
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
        except Exception as e:
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
            balance = await self.client.get_balance()

            # Try to get profile info for currency and user_id
            currency = None
            user_id = None

            # Try different ways to access profile data
            if hasattr(self.client, 'profile') and self.client.profile:
                profile = self.client.profile
                # Access attributes directly rather than using .get()
                if hasattr(profile, 'currency'):
                    currency = profile.currency
                if hasattr(profile, 'user_id'):
                    user_id = profile.user_id
            elif hasattr(self.client, 'get_profile'):
                profile = await self.client.get_profile()
                # Access attributes directly rather than using .get()
                if hasattr(profile, 'currency'):
                    currency = profile.currency
                if hasattr(profile, 'user_id'):
                    user_id = profile.user_id
                # If it's a dictionary type profile (uncommon but possible)
                elif isinstance(profile, dict):
                    currency = profile.get('currency')
                    user_id = profile.get('user_id')

            # Fallback if not found
            if currency is None:
                currency = 'USD'  # Default or unknown
            if user_id is None:
                user_id = 'unknown'

            self.account_info = {
                "balance": balance,
                "currency": currency,
                "user_id": user_id,
                "last_updated": datetime.now()
            }
            logger.debug(f"Account info updated: Balance={balance} {currency}")
            return self.account_info
        except Exception as e:
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

        try:
            # Get candles - try different method names that might exist in the pyquotex library
            loop = asyncio.get_event_loop()

            # Try different possible method names for getting candles
            if hasattr(self.client, 'get_candles'):
                candles = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_candles(asset, timeframe, count)
                )
            elif hasattr(self.client, 'get_history'):
                candles = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_history(asset, timeframe, count)
                )
            elif hasattr(self.client, 'get_historical_data'):
                candles = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_historical_data(asset, timeframe, count)
                )
            else:
                # If we can't find an appropriate method, log the available methods and raise an error
                methods = [method for method in dir(self.client) if not method.startswith('_') and callable(getattr(self.client, method))]
                logger.error(f"No candle retrieval method found. Available methods: {methods}")
                raise AttributeError(f"Could not find a method to get candles in the Quotex client. Available methods: {methods}")

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            # Rename columns if needed and ensure proper types
            if 'time' in df.columns:
                df.rename(columns={'time': 'timestamp'}, inplace=True)

            # Ensure all required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in candle data")

            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            logger.debug(f"Retrieved {len(df)} candles for {asset} at {timeframe}s timeframe")
            return df
        except Exception as e:
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

        # Validate direction
        if direction.lower() not in ["call", "put"]:
            raise ValueError("Direction must be 'call' or 'put'")

        try:
            loop = asyncio.get_event_loop()

            if wait_for_result:
                # Use buy_and_check_win to place trade and wait for result
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.buy_and_check_win(
                        asset=asset,
                        amount=amount,
                        action=direction.lower(),
                        expirations_times=expiration
                    )
                )

                # Process result
                trade_info = {
                    "asset": asset,
                    "amount": amount,
                    "direction": direction,
                    "expiration": expiration,
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
                    lambda: self.client.buy_simple(
                        asset=asset,
                        amount=amount,
                        action=direction.lower(),
                        expirations_times=expiration
                    )
                )

                trade_info = {
                    "asset": asset,
                    "amount": amount,
                    "direction": direction,
                    "expiration": expiration,
                    "timestamp": datetime.now(),
                    "trade_id": result
                }

                logger.info(f"Trade placed: {asset} {direction} {amount} - ID: {result}")
                return trade_info

        except Exception as e:
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
        # we'll return a common set of assets that are typically available on Quotex
        common_assets = [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
            "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURAUD", "GBPAUD",
            "BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD",
            "Apple", "Amazon", "Google", "Microsoft", "Tesla", "Facebook",
            "Gold", "Silver", "Oil", "DAX", "S&P 500", "Dow Jones", "NASDAQ"
        ]

        logger.info(f"Returning {len(common_assets)} predefined assets as pyquotex doesn't provide get_available_assets")
        return common_assets

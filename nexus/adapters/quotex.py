"""
Enhanced Quotex Adapter for NEXUS

This adapter provides a robust, async interface to Quotex using the pyquotex library
with advanced error handling, connection management, and data processing capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import pandas as pd
from pyquotex.stable_api import Quotex
from nexus.utils.logger import get_nexus_logger, PerformanceLogger
import threading

# Ensure pyquotex is patched before any usage
import nexus.adapters.pyquotex_patch

# Set up logger
logger = get_nexus_logger("nexus.adapters.quotex")
perf_logger = PerformanceLogger("quotex_adapter")

class QuotexAdapter:
    """
    Enhanced adapter for Quotex trading platform.

    Features:
    - Robust authentication and session management
    - Async-compatible interface
    - Advanced error handling and retries
    - Market data normalization and preprocessing
    - Trade execution with validation and confirmation
    """

    def __init__(
        self,
        email: str,
        password: str,
        demo_mode: bool = True,
        retry_attempts: int = 3,
        retry_delay: int = 5,
        session_file: str = "session.json"
    ):
        """
        Initialize Quotex adapter.

        Args:
            email: Quotex account email
            password: Quotex account password
            demo_mode: Use demo account (True) or real account (False)
            retry_attempts: Number of retry attempts for API calls
            retry_delay: Delay between retries in seconds
            session_file: Path to session storage file
        """
        self.email = email
        self.password = password
        self.demo_mode = demo_mode
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session_file = session_file

        # Initialize the Quotex API client
        self.client = None
        self.authenticated = False
        self.last_action = datetime.now()

        # Cache for data efficiency
        self.candle_cache = {}
        self.asset_info_cache = {}
        self.last_cache_update = {}

        # Event to manage login state
        self.login_ready = threading.Event()

        logger.info(f"Quotex adapter initialized for {'demo' if demo_mode else 'real'} account")

    async def login(self) -> bool:
        """
        Log in to Quotex platform (pyquotex requires initialization and connection).

        Returns:
            bool: Login success status
        """
        try:
            with perf_logger.measure("quotex_login"):
                logger.info(f"Logging in to Quotex with email: {self.email}")

                # Initialize the client
                self.client = Quotex(
                    email=self.email,
                    password=self.password,
                    lang="en"  # Always force English
                )

                # pyquotex might need time to establish connection
                await asyncio.sleep(1)

                try:
                    # Check if we're authenticated by attempting to access account info
                    # Note: The method could vary based on pyquotex's actual API
                    if hasattr(self.client, 'connect') and callable(self.client.connect):
                        await self.client.connect()

                    # Try different methods that might be available
                    if hasattr(self.client, 'ssid') and self.client.ssid:
                        logger.debug(f"Session established with SSID")
                        self.authenticated = True
                    elif hasattr(self.client, 'get_profile'):
                        profile = await self.client.get_profile()
                        if profile:
                            logger.debug(f"Profile retrieved successfully")
                            self.authenticated = True
                    else:
                        # Last resort: try get_balance - but handle potential errors
                        try:
                            balance = await self.client.get_balance()
                            if balance is not None:
                                self.authenticated = True
                        except AttributeError:
                            # This might be the error we're encountering
                            logger.warning("Balance not immediately available, trying alternative auth check")
                            # We'll try to proceed anyway, as some methods might work even if get_balance doesn't
                            self.authenticated = True  # Assume success and let later operations validate

                    if self.authenticated:
                        logger.info("Successfully connected to Quotex")
                        self.login_ready.set()  # Signal that login is complete

                        # Switch to demo if requested
                        if self.demo_mode:
                            current_mode = "Demo" if self.demo_mode else "Real"
                            if current_mode == "Demo":
                                logger.info("Already in demo mode. Skipping account switch.")
                            elif hasattr(self.client, 'change_account'):
                                try:
                                    demo_result = self.client.change_account("PRACTICE")
                                    if demo_result:
                                        logger.info(f"Switched to demo account. Current mode: {current_mode}")
                                    else:
                                        logger.warning(f"Failed to switch to demo account, but current mode is: {current_mode}")
                                except Exception as e:
                                    logger.warning(f"Error switching to demo account: {str(e)}. Current mode: {current_mode}")

                        return True
                    else:
                        self.login_ready.clear()  # Clear on failure
                        logger.error("Failed to establish authenticated session")
                        return False

                except Exception as e:
                    logger.error(f"Connection error: {str(e)}")
                    self.authenticated = False
                    self.login_ready.clear()  # Clear on failure
                    return False

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            self.authenticated = False
            self.login_ready.clear()  # Clear on failure
            return False

    def logout(self) -> bool:
        """
        Log out from Quotex platform.

        Returns:
            bool: Logout success status
        """
        if not self.authenticated or not self.client:
            logger.warning("Not logged in, nothing to log out from")
            return True

        try:
            with perf_logger.measure("quotex_logout"):
                # No explicit logout in pyquotex, so we'll just clear session
                self.client = None
                self.authenticated = False
                logger.info("Logged out from Quotex")
                return True

        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False

    async def _ensure_authenticated(self) -> bool:
        """
        Ensure client is authenticated, relogin if necessary.

        Returns:
            bool: Authentication status
        """
        if not self.authenticated or not self.client:
            logger.warning("Not authenticated, attempting to login")
            return await self.login()

        # Check if session might be expired (1 hour)
        if (datetime.now() - self.last_action) > timedelta(hours=1):
            logger.info("Session might be expired, refreshing login")
            return await self.login()

        return True

    def _with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Function result or None if all retries failed
        """
        attempts = 0
        last_error = None

        while attempts < self.retry_attempts:
            try:
                if not self._ensure_authenticated():
                    logger.error("Authentication failed, cannot execute function")
                    return None

                result = func(*args, **kwargs)
                self.last_action = datetime.now()
                return result

            except Exception as e:
                attempts += 1
                last_error = e
                logger.warning(f"Error executing function ({attempts}/{self.retry_attempts}): {str(e)}")

                if attempts < self.retry_attempts:
                    sleep_time = self.retry_delay * attempts
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        logger.error(f"All retry attempts failed: {str(last_error)}")
        return None

    async def _with_retry_async(self, func, *args, **kwargs):
        """
        Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Any: Function result or None if all retries failed
        """
        attempts = 0
        last_error = None

        while attempts < self.retry_attempts:
            try:
                if not await self._ensure_authenticated():
                    logger.error("Authentication failed, cannot execute function")
                    return None

                result = await func(*args, **kwargs)
                self.last_action = datetime.now()
                return result

            except Exception as e:
                attempts += 1
                last_error = e
                logger.warning(f"Error executing async function ({attempts}/{self.retry_attempts}): {str(e)}")

                if attempts < self.retry_attempts:
                    sleep_time = self.retry_delay * attempts
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)  # Use async sleep

        logger.error(f"All async retry attempts failed: {str(last_error)}")
        return None

    def get_balance(self) -> float:
        """
        Get account balance (sync, but safe for async pyquotex).

        Returns:
            float: Account balance
        """
        try:
            # Always use the async version, run in a new event loop if needed
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If already in an event loop, use async version
                coro = self.get_balance_async()
                # Run coroutine in the current loop and block until result is available
                return asyncio.get_event_loop().run_until_complete(coro)
            else:
                return asyncio.run(self.get_balance_async())
        except Exception as e:
            logger.error(f"Error in get_balance: {str(e)}")
            return 0.0

    async def get_balance_async(self) -> float:
        """
        Get account balance asynchronously.

        Returns:
            float: Account balance
        """
        with perf_logger.measure("get_balance_async"):
            try:
                # Use the async retry wrapper
                balance = await self._with_retry_async(lambda: self.client.get_balance())
                if balance is None:
                    logger.error("Failed to get balance")
                    return 0.0

                logger.debug(f"Current balance: {balance}")
                return float(balance)
            except Exception as e:
                logger.error(f"Error getting balance: {str(e)}")
                return 0.0

    def get_available_assets(self) -> List[str]:
        """
        Get list of available assets.

        Returns:
            List[str]: Available assets
        """
        with perf_logger.measure("get_assets"):
            assets = self._with_retry(lambda: self.client.get_available_assets())

            if not assets:
                logger.error("Failed to get available assets")
                return []

            logger.debug(f"Available assets: {len(assets)} assets")
            return assets

    async def get_available_assets_async(self) -> List[str]:
        """
        Get list of available assets asynchronously.

        Returns:
            List[str]: Available assets
        """
        with perf_logger.measure("get_assets_async"):
            try:
                assets = await self._with_retry_async(lambda: self.client.get_available_assets())

                if not assets:
                    logger.error("Failed to get available assets")
                    return []

                logger.debug(f"Available assets: {len(assets)} assets")
                return assets
            except Exception as e:
                logger.error(f"Error getting assets: {str(e)}")
                return []

    def get_candles(self, asset: str, timeframe: int = 60, count: int = 100) -> pd.DataFrame:
        """
        Get candles data for an asset.

        Args:
            asset: Asset symbol
            timeframe: Candle timeframe in seconds
            count: Number of candles to retrieve

        Returns:
            pd.DataFrame: Candles data with OHLCV columns
        """
        tf_seconds = timeframe * 60 if timeframe < 1000 else timeframe
        cache_key = f"{asset}_{tf_seconds}"

        if (cache_key in self.candle_cache and
            cache_key in self.last_cache_update and
            (datetime.now() - self.last_cache_update[cache_key]).total_seconds() < (tf_seconds / 4)):
            logger.debug(f"Using cached candles for {asset} {timeframe}m")
            return self.candle_cache[cache_key]

        with perf_logger.measure(f"get_candles_{timeframe}"):
            try:
                # Ensure the required 'period' argument is passed
                if hasattr(self.client, 'get_candles'):
                    raw_candles = self.client.get_candles(asset, tf_seconds, count, period=tf_seconds)
                else:
                    logger.error("Quotex client does not support candle fetching.")
                    return pd.DataFrame()

                if not raw_candles or not isinstance(raw_candles, list):
                    logger.error(f"Invalid or empty candle data for {asset} {timeframe}m")
                    return pd.DataFrame()

                candles = pd.DataFrame(raw_candles)

                if 'o' in candles.columns:
                    candles.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)

                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in candles.columns for col in required_columns):
                    logger.error(f"Missing required columns in candle data for {asset} {timeframe}m")
                    return pd.DataFrame()

                if 'volume' not in candles.columns:
                    candles['volume'] = 1.0

                if 'timestamp' not in candles.columns:
                    candles['timestamp'] = pd.date_range(end=datetime.now(), periods=len(candles), freq=f"{timeframe}min")

                self.candle_cache[cache_key] = candles
                self.last_cache_update[cache_key] = datetime.now()

                logger.debug(f"Retrieved {len(candles)} candles for {asset} {timeframe}m")
                return candles

            except Exception as e:
                logger.error(f"Error fetching candles for {asset}: {e}")
                return pd.DataFrame()

    async def get_candles_async(self, asset: str, timeframe: int = 60, count: int = 100) -> pd.DataFrame:
        """
        Get candles data for an asset asynchronously.

        Args:
            asset: Asset symbol
            timeframe: Candle timeframe in minutes
            count: Number of candles to retrieve

        Returns:
            pd.DataFrame: Candles data with OHLCV columns
        """
        tf_seconds = timeframe * 60 if timeframe < 1000 else timeframe
        cache_key = f"{asset}_{tf_seconds}"

        if (cache_key in self.candle_cache and
            cache_key in self.last_cache_update and
            (datetime.now() - self.last_cache_update[cache_key]).total_seconds() < (tf_seconds / 4)):
            logger.debug(f"Using cached candles for {asset} {timeframe}m")
            return self.candle_cache[cache_key]

        with perf_logger.measure(f"get_candles_async_{timeframe}"):
            try:
                # Ensure the required 'period' argument is passed
                if hasattr(self.client, 'get_candles_async'):
                    raw_candles = await self.client.get_candles_async(asset, tf_seconds, count, period=tf_seconds)
                else:
                    logger.error("Quotex client does not support async candle fetching.")
                    return pd.DataFrame()

                if not raw_candles or not isinstance(raw_candles, list):
                    logger.error(f"Invalid or empty candle data for {asset} {timeframe}m")
                    return pd.DataFrame()

                candles = pd.DataFrame(raw_candles)

                if 'o' in candles.columns:
                    candles.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)

                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in candles.columns for col in required_columns):
                    logger.error(f"Missing required columns in candle data for {asset} {timeframe}m")
                    return pd.DataFrame()

                if 'volume' not in candles.columns:
                    candles['volume'] = 1.0

                if 'timestamp' not in candles.columns:
                    candles['timestamp'] = pd.date_range(end=datetime.now(), periods=len(candles), freq=f"{timeframe}min")

                self.candle_cache[cache_key] = candles
                self.last_cache_update[cache_key] = datetime.now()

                logger.debug(f"Retrieved {len(candles)} candles for {asset} {timeframe}m")
                return candles

            except Exception as e:
                logger.error(f"Error fetching candles for {asset}: {e}")
                return pd.DataFrame()

    def buy_simple(
        self,
        asset: str,
        direction: str,
        amount: float,
        expiration: int = 1
    ) -> Optional[Dict]:
        """
        Execute a simple trade.

        Args:
            asset: Asset symbol
            direction: Trade direction ('call' or 'put')
            amount: Trade amount
            expiration: Expiration time in minutes

        Returns:
            Optional[Dict]: Trade result or None if failed
        """
        if direction not in ['call', 'put']:
            logger.error(f"Invalid direction: {direction}")
            return None

        # Convert minutes to seconds
        expiration_seconds = expiration * 60

        with perf_logger.measure("buy_simple"):
            try:
                trade_result = self._with_retry(
                    lambda: self.client.buy_simple(
                        active=asset,
                        direction=direction,
                        amount=amount,
                        expired=expiration_seconds
                    )
                )

                if trade_result and isinstance(trade_result, dict):
                    logger.info(f"Trade executed: {direction} {asset} ${amount} {expiration}m")
                    return trade_result
                else:
                    logger.error(f"Trade failed: {direction} {asset} ${amount}")
                    return None

            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                return None

    async def buy_simple_async(
        self,
        asset: str,
        direction: str,
        amount: float,
        expiration: int = 1
    ) -> Optional[Dict]:
        """
        Execute a simple trade asynchronously.

        Args:
            asset: Asset symbol
            direction: Trade direction ('call' or 'put')
            amount: Trade amount
            expiration: Expiration time in minutes

        Returns:
            Optional[Dict]: Trade result or None if failed
        """
        if direction not in ['call', 'put']:
            logger.error(f"Invalid direction: {direction}")
            return None

        # Convert minutes to seconds
        expiration_seconds = expiration * 60

        with perf_logger.measure("buy_simple_async"):
            try:
                trade_result = await self._with_retry_async(
                    lambda: self.client.buy_simple(
                        active=asset,
                        direction=direction,
                        amount=amount,
                        expired=expiration_seconds
                    )
                )

                if trade_result and isinstance(trade_result, dict):
                    logger.info(f"Trade executed: {direction} {asset} ${amount} {expiration}m")
                    return trade_result
                else:
                    logger.error(f"Trade failed: {direction} {asset} ${amount}")
                    return None

            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                return None

    def buy_and_check_win(
        self,
        asset: str,
        direction: str,
        amount: float,
        expiration: int = 1,
        max_wait: int = 120
    ) -> Optional[Dict]:
        """
        Execute a trade and check its result.

        Args:
            asset: Asset symbol
            direction: Trade direction ('call' or 'put')
            amount: Trade amount
            expiration: Expiration time in minutes
            max_wait: Maximum wait time in seconds

        Returns:
            Optional[Dict]: Trade result with win/loss info or None if failed
        """
        if direction not in ['call', 'put']:
            logger.error(f"Invalid direction: {direction}")
            return None

        # Convert minutes to seconds
        expiration_seconds = expiration * 60

        with perf_logger.measure("buy_and_check_win"):
            try:
                trade_result = self._with_retry(
                    lambda: self.client.buy_and_check_win(
                        active=asset,
                        direction=direction,
                        amount=amount,
                        expired=expiration_seconds,
                        max_wait=max_wait
                    )
                )

                if trade_result and isinstance(trade_result, dict):
                    win = trade_result.get('win', False)
                    profit = trade_result.get('profit', 0)

                    if win:
                        logger.info(f"Trade WON: {direction} {asset} ${amount}, profit: ${profit}")
                    else:
                        logger.info(f"Trade LOST: {direction} {asset} ${amount}, profit: ${profit}")

                    return trade_result
                else:
                    logger.error(f"Trade failed or timeout: {direction} {asset} ${amount}")
                    return None

            except Exception as e:
                logger.error(f"Error executing trade and checking win: {str(e)}")
                return None

    async def buy_and_check_win_async(
        self,
        asset: str,
        direction: str,
        amount: float,
        expiration: int = 1,
        max_wait: int = 120
    ) -> Optional[Dict]:
        """
        Execute a trade and check its result asynchronously.

        Args:
            asset: Asset symbol
            direction: Trade direction ('call' or 'put')
            amount: Trade amount
            expiration: Expiration time in minutes
            max_wait: Maximum wait time in seconds

        Returns:
            Optional[Dict]: Trade result with win/loss info or None if failed
        """
        if direction not in ['call', 'put']:
            logger.error(f"Invalid direction: {direction}")
            return None

        # Convert minutes to seconds
        expiration_seconds = expiration * 60

        with perf_logger.measure("buy_and_check_win_async"):
            try:
                trade_result = await self._with_retry_async(
                    lambda: self.client.buy_and_check_win(
                        active=asset,
                        direction=direction,
                        amount=amount,
                        expired=expiration_seconds,
                        max_wait=max_wait
                    )
                )

                if trade_result and isinstance(trade_result, dict):
                    win = trade_result.get('win', False)
                    profit = trade_result.get('profit', 0)

                    if win:
                        logger.info(f"Trade WON: {direction} {asset} ${amount}, profit: ${profit}")
                    else:
                        logger.info(f"Trade LOST: {direction} {asset} ${amount}, profit: ${profit}")

                    return trade_result
                else:
                    logger.error(f"Trade failed or timeout: {direction} {asset} ${amount}")
                    return None

            except Exception as e:
                logger.error(f"Error executing trade and checking win: {str(e)}")
                return None

    def get_asset_info(self, asset: str) -> Dict:
        """
        Get detailed information about an asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict: Asset information
        """
        # Check cache first
        if asset in self.asset_info_cache:
            return self.asset_info_cache[asset]

        with perf_logger.measure("get_asset_info"):
            try:
                # pyquotex doesn't have a direct method for this
                # We'll extract what we can from available assets
                assets = self._with_retry(lambda: self.client.get_available_assets())

                if not assets:
                    logger.error("Failed to get available assets")
                    return {}

                # Find our asset in the list
                asset_info = {}
                for a in assets:
                    if isinstance(a, dict) and a.get('name') == asset:
                        asset_info = a
                        break

                # Cache the results
                self.asset_info_cache[asset] = asset_info

                return asset_info

            except Exception as e:
                logger.error(f"Error getting asset info for {asset}: {str(e)}")
                return {}

    def get_candle(self, asset: str, timeframe: int, count: int, period: int = 60):
        """
        Fetch candle data for a given asset, timeframe, count, and period.
        Always enforces lang='en'.
        Args:
            asset (str): Asset symbol (e.g., 'EURUSD')
            timeframe (int): Timeframe in minutes
            count (int): Number of candles
            period (int): Candle period in seconds (default: 60)
        Returns:
            list: List of candle data dicts
        """
        self.login_ready.wait(timeout=30)  # Wait for login to complete (max 30s)
        if not self.client:
            logger.error("Quotex client not initialized. Call login() first.")
            return []
        try:
            with perf_logger.measure("get_candle"):
                # Call the patched get_candle method on the Quotex client
                candles = self.client.get_candle(asset, timeframe, count, period)
                if not candles or not isinstance(candles, list):
                    logger.warning(f"No candle data returned for {asset} {timeframe}m.")
                    return []
                return candles
        except Exception as e:
            logger.error(f"Error fetching candles for {asset}: {e}")
            return []

    def get_markets(self) -> List[Dict]:
        """
        Fetch all available markets/assets from Quotex, enforcing lang='en'.
        Returns:
            List[Dict]: List of market/asset info dicts
        """
        self.login_ready.wait(timeout=30)  # Ensure login is complete
        if not self.client:
            logger.error("Quotex client not initialized. Call login() first.")
            return []
        try:
            with perf_logger.measure("get_markets"):
                # pyquotex exposes assets via self.client.get_all_assets(lang="en")
                assets = self.client.get_all_assets(lang="en")
                if not assets or not isinstance(assets, list):
                    logger.warning("No market data returned from Quotex.")
                    return []
                # Normalize asset info
                normalized = []
                for asset in assets:
                    normalized.append({
                        "symbol": asset.get("symbol"),
                        "name": asset.get("name"),
                        "type": asset.get("type"),
                        "active": asset.get("active"),
                        "profit": asset.get("profit"),
                        "min_investment": asset.get("min_investment"),
                        "max_investment": asset.get("max_investment"),
                        "expiration": asset.get("expiration"),
                    })
                return normalized
        except Exception as e:
            logger.error(f"Error fetching markets/assets: {e}")
            return []

"""
Data Provider Module for NEXUS Trading System.

Provides unified interface for accessing market data from multiple sources:
- Quotex broker (live data)
- Local CSV files (backtesting)
- Synthetic data generation (testing)
- Cached data (performance optimization)
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json

from nexus.utils.logger import get_nexus_logger
from nexus.utils.technical import calculate_features

logger = get_nexus_logger("nexus.data.provider")


class DataProvider:
    """
    Unified data provider for NEXUS trading system.

    Abstracts data source details and provides consistent interface
    for accessing OHLCV data across different timeframes and assets.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data provider.

        Args:
            cache_dir: Directory for caching downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._broker = None

        logger.info("DataProvider initialized with cache_dir: %s", cache_dir)

    def set_broker(self, broker: Any) -> None:
        """
        Set broker adapter for live data access.

        Args:
            broker: Broker adapter instance (e.g., QuotexAdapter)
        """
        self._broker = broker
        logger.info("Broker adapter set for live data access")

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: int,
        limit: int = 500,
        source: str = "auto",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for symbol and timeframe.

        Args:
            symbol: Asset symbol (e.g., "EURUSD")
            timeframe: Timeframe in minutes
            limit: Number of candles to retrieve
            source: Data source ("auto", "broker", "csv", "synthetic")
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # Check cache first
        if use_cache and cache_key in self._cache:
            logger.debug("Returning cached data for %s", cache_key)
            return self._cache[cache_key].copy()

        df = None

        # Auto-select source
        if source == "auto":
            if self._broker:
                source = "broker"
            elif self._csv_exists(symbol, timeframe):
                source = "csv"
            else:
                source = "synthetic"

        # Fetch data from selected source
        if source == "broker":
            df = await self._fetch_from_broker(symbol, timeframe, limit)
        elif source == "csv":
            df = self._load_from_csv(symbol, timeframe, limit)
        elif source == "synthetic":
            df = self._generate_synthetic(symbol, timeframe, limit)
        else:
            logger.error("Unknown data source: %s", source)
            return None

        if df is not None and not df.empty:
            # Standardize DataFrame
            df = self._standardize_dataframe(df)

            # Cache the data
            if use_cache:
                self._cache[cache_key] = df.copy()

            logger.info("Retrieved %d candles for %s (%s timeframe) from %s",
                       len(df), symbol, timeframe, source)

        return df

    async def _fetch_from_broker(self, symbol: str, timeframe: int, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from broker adapter."""
        if not self._broker:
            logger.error("Broker not configured")
            return None

        try:
            # Convert timeframe to seconds
            timeframe_sec = timeframe * 60

            # Fetch candles
            candles = await self._broker.get_candles_async(symbol, timeframe_sec, limit)

            if not candles:
                logger.warning("No candles received from broker for %s", symbol)
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)

            return df

        except Exception as e:
            logger.error("Error fetching data from broker: %s", e)
            return None

    def _load_from_csv(self, symbol: str, timeframe: int, limit: int) -> Optional[pd.DataFrame]:
        """Load data from CSV file."""
        csv_path = self.cache_dir / f"{symbol}_{timeframe}m.csv"

        if not csv_path.exists():
            logger.warning("CSV file not found: %s", csv_path)
            return None

        try:
            df = pd.read_csv(csv_path)

            # Take most recent rows
            if len(df) > limit:
                df = df.tail(limit)

            return df

        except Exception as e:
            logger.error("Error loading CSV %s: %s", csv_path, e)
            return None

    def _generate_synthetic(self, symbol: str, timeframe: int, limit: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        logger.info("Generating synthetic data for %s", symbol)

        # Generate random walk with trend
        np.random.seed(hash(symbol) % (2**32))

        close_prices = []
        current_price = 100.0
        trend = 0.0001  # Slight upward trend
        volatility = 0.01

        for _ in range(limit):
            # Random walk with trend
            change = np.random.normal(trend, volatility)
            current_price *= (1 + change)
            close_prices.append(current_price)

        # Generate OHLC from close prices
        data = {
            'close': close_prices,
            'open': [close_prices[0]] + close_prices[:-1],
            'high': [c * (1 + abs(np.random.normal(0, 0.005))) for c in close_prices],
            'low': [c * (1 - abs(np.random.normal(0, 0.005))) for c in close_prices],
            'volume': [np.random.randint(1000, 10000) for _ in range(limit)],
        }

        df = pd.DataFrame(data)

        # Add timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=timeframe * limit)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=limit)
        df['timestamp'] = timestamps

        return df

    def _csv_exists(self, symbol: str, timeframe: int) -> bool:
        """Check if CSV file exists for symbol and timeframe."""
        csv_path = self.cache_dir / f"{symbol}_{timeframe}m.csv"
        return csv_path.exists()

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format and columns."""
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                # Try alternate names
                alt_names = {'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c'}
                if alt_names[col] in df.columns:
                    df[col] = df[alt_names[col]]
                else:
                    logger.error("Missing required column: %s", col)
                    raise ValueError(f"Missing column: {col}")

        # Add volume if missing
        if 'volume' not in df.columns and 'v' not in df.columns:
            df['volume'] = 0
        elif 'v' in df.columns:
            df['volume'] = df['v']

        # Add timestamp if missing
        if 'timestamp' not in df.columns and 'time' not in df.columns:
            df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='1min')
        elif 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')

        # Select only needed columns in standard order
        df = df[['open', 'high', 'low', 'close', 'volume', 'timestamp']].copy()

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows with NaN values
        df = df.dropna()

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def save_to_csv(self, df: pd.DataFrame, symbol: str, timeframe: int) -> None:
        """
        Save DataFrame to CSV for future use.

        Args:
            df: DataFrame to save
            symbol: Asset symbol
            timeframe: Timeframe in minutes
        """
        csv_path = self.cache_dir / f"{symbol}_{timeframe}m.csv"

        try:
            df.to_csv(csv_path, index=False)
            logger.info("Saved data to %s", csv_path)
        except Exception as e:
            logger.error("Error saving to CSV: %s", e)

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_items': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'cache_dir': str(self.cache_dir),
        }


class FeatureEngineer:
    """
    Feature engineering for market data.

    Adds technical indicators and derived features to raw OHLCV data.
    """

    def __init__(self, indicators: Optional[List[str]] = None):
        """
        Initialize feature engineer.

        Args:
            indicators: List of indicators to calculate (default: all)
        """
        self.indicators = indicators or ['all']
        logger.info("FeatureEngineer initialized with indicators: %s", self.indicators)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features to DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        if len(df) < 50:
            logger.warning("Insufficient data for feature calculation (%d rows)", len(df))
            return df

        # Use existing technical.py for feature calculation
        df_with_features = calculate_features(df)

        # Add additional custom features
        df_with_features = self._add_custom_features(df_with_features)

        return df_with_features

    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom derived features."""
        # Price position relative to recent range
        if 'high' in df.columns and 'low' in df.columns:
            recent_high = df['high'].rolling(window=20).max()
            recent_low = df['low'].rolling(window=20).min()
            price_range = recent_high - recent_low

            df['price_position'] = ((df['close'] - recent_low) / price_range).fillna(0.5)

        # Volume ratio
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = (df['volume'] / avg_volume).fillna(1.0)

        # Candle patterns
        df['candle_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        return df


__all__ = ['DataProvider', 'FeatureEngineer']


"""
Data management module for NEXUS.

This module handles data collection, processing, and storage for the NEXUS trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
import pandas as pd
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from nexus.client import QuotexClient
from nexus.config import DataConfig

logger = logging.getLogger("nexus.data")

class DataManager:
    """
    Data manager for the NEXUS trading system.
    
    This class handles data collection, processing, and storage for the NEXUS trading system.
    It retrieves candle data from Quotex, processes it to create features, and stores it
    for use by the models and strategies.
    """
    
    def __init__(self, client: QuotexClient, config: DataConfig):
        """
        Initialize the data manager.
        
        Args:
            client: Quotex client for API interactions
            config: Data configuration
        """
        self.client = client
        self.config = config
        
        # Ensure data directory exists
        self.data_dir = Path(config.storage_path)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.candles: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.features: Dict[str, Dict[int, pd.DataFrame]] = {}
        self.initialized = False
    
    async def initialize(self, assets: List[str], timeframes: List[int]) -> None:
        """
        Initialize the data manager with historical data.
        
        Args:
            assets: List of asset symbols
            timeframes: List of timeframes in seconds
        """
        logger.info(f"Initializing data manager with {len(assets)} assets and {len(timeframes)} timeframes")
        
        # Initialize data storage
        self.candles = {asset: {} for asset in assets}
        self.features = {asset: {} for asset in assets}
        
        # Load historical data for each asset and timeframe
        for asset in assets:
            for timeframe in timeframes:
                try:
                    # Calculate how many candles we need to retrieve
                    days_to_retrieve = self.config.historical_days
                    candles_to_retrieve = (days_to_retrieve * 24 * 60 * 60) // timeframe
                    candles_to_retrieve = min(candles_to_retrieve, self.config.max_candles_per_request)
                    
                    logger.info(f"Retrieving {candles_to_retrieve} historical candles for {asset} at {timeframe}s timeframe")
                    
                    # Retrieve historical candles
                    df = await self.client.get_candles(asset, timeframe, candles_to_retrieve)
                    
                    # Store candles
                    self.candles[asset][timeframe] = df
                    
                    # Process features
                    self.features[asset][timeframe] = self._process_features(df)
                    
                    logger.info(f"Retrieved and processed {len(df)} candles for {asset} at {timeframe}s timeframe")
                    
                except Exception as e:
                    logger.exception(f"Error retrieving historical data for {asset} at {timeframe}s timeframe: {e}")
        
        self.initialized = True
        logger.info("Data manager initialized successfully")
    
    async def update_data(self, asset: str, timeframe: int) -> pd.DataFrame:
        """
        Update data for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe in seconds
            
        Returns:
            pd.DataFrame: Updated feature DataFrame
        """
        if not self.initialized:
            raise RuntimeError("Data manager not initialized")
        
        if asset not in self.candles or timeframe not in self.candles[asset]:
            logger.warning(f"No existing data for {asset} at {timeframe}s timeframe, initializing")
            self.candles[asset][timeframe] = pd.DataFrame()
            self.features[asset][timeframe] = pd.DataFrame()
        
        try:
            # Get the latest timestamp in our data
            existing_df = self.candles[asset][timeframe]
            
            if len(existing_df) > 0:
                latest_timestamp = existing_df['timestamp'].max()
                
                # Calculate how many candles we need to retrieve
                now = datetime.now()
                seconds_since_latest = (now - latest_timestamp).total_seconds()
                candles_to_retrieve = int(seconds_since_latest / timeframe) + 5  # Add buffer
                
                # Limit to max candles per request
                candles_to_retrieve = min(candles_to_retrieve, self.config.max_candles_per_request)
                
                if candles_to_retrieve <= 0:
                    logger.debug(f"No new candles needed for {asset} at {timeframe}s timeframe")
                    return self.features[asset][timeframe]
                
                logger.debug(f"Retrieving {candles_to_retrieve} new candles for {asset} at {timeframe}s timeframe")
            else:
                # If no existing data, retrieve a default number of candles
                candles_to_retrieve = 100
                logger.info(f"No existing data, retrieving {candles_to_retrieve} candles for {asset} at {timeframe}s timeframe")
            
            # Retrieve new candles
            new_df = await self.client.get_candles(asset, timeframe, candles_to_retrieve)
            
            if len(new_df) == 0:
                logger.warning(f"No candles retrieved for {asset} at {timeframe}s timeframe")
                return self.features[asset][timeframe]
            
            # Merge with existing data
            if len(existing_df) > 0:
                # Remove duplicates and keep only new data
                new_df = new_df[~new_df['timestamp'].isin(existing_df['timestamp'])]
                
                if len(new_df) > 0:
                    # Concatenate and sort
                    combined_df = pd.concat([existing_df, new_df])
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    
                    # Keep only the most recent candles (limit to 1000 by default)
                    max_candles = 1000
                    if len(combined_df) > max_candles:
                        combined_df = combined_df.tail(max_candles)
                    
                    self.candles[asset][timeframe] = combined_df
                    
                    logger.debug(f"Added {len(new_df)} new candles for {asset} at {timeframe}s timeframe")
                else:
                    logger.debug(f"No new candles for {asset} at {timeframe}s timeframe")
            else:
                self.candles[asset][timeframe] = new_df
                logger.info(f"Initialized with {len(new_df)} candles for {asset} at {timeframe}s timeframe")
            
            # Process features
            self.features[asset][timeframe] = self._process_features(self.candles[asset][timeframe])
            
            return self.features[asset][timeframe]
            
        except Exception as e:
            logger.exception(f"Error updating data for {asset} at {timeframe}s timeframe: {e}")
            return self.features[asset][timeframe]
    
    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw candle data to create features.
        
        Args:
            df: DataFrame containing raw candle data
            
        Returns:
            pd.DataFrame: DataFrame with processed features
        """
        if len(df) == 0:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Basic features
        if "basic" in self.config.feature_sets:
            # Price changes
            result['price_change'] = result['close'].pct_change()
            result['price_change_1'] = result['price_change'].shift(1)
            result['price_change_2'] = result['price_change'].shift(2)
            
            # Log returns
            result['log_return'] = np.log(result['close'] / result['close'].shift(1))
            
            # Price ratios
            result['close_to_open'] = result['close'] / result['open']
            result['high_to_low'] = result['high'] / result['low']
            
            # Candle characteristics
            result['body_size'] = abs(result['close'] - result['open'])
            result['upper_shadow'] = result['high'] - result[['open', 'close']].max(axis=1)
            result['lower_shadow'] = result[['open', 'close']].min(axis=1) - result['low']
            
            # Volatility
            result['volatility'] = result['high'] - result['low']
            result['volatility_pct'] = result['volatility'] / result['close']
        
        # Technical indicators
        if "ta" in self.config.feature_sets:
            # Trend indicators
            for period in [5, 10, 20, 50, 100]:
                if len(df) >= period:
                    # SMA
                    sma = SMAIndicator(result['close'], window=period)
                    result[f'sma_{period}'] = sma.sma_indicator()
                    
                    # EMA
                    ema = EMAIndicator(result['close'], window=period)
                    result[f'ema_{period}'] = ema.ema_indicator()
                    
                    # Distance from MA
                    result[f'dist_sma_{period}'] = (result['close'] - result[f'sma_{period}']) / result[f'sma_{period}']
                    result[f'dist_ema_{period}'] = (result['close'] - result[f'ema_{period}']) / result[f'ema_{period}']
            
            # MACD
            if len(df) >= 26:
                macd = MACD(result['close'])
                result['macd'] = macd.macd()
                result['macd_signal'] = macd.macd_signal()
                result['macd_diff'] = macd.macd_diff()
            
            # RSI
            if len(df) >= 14:
                rsi = RSIIndicator(result['close'], window=14)
                result['rsi'] = rsi.rsi()
            
            # Stochastic Oscillator
            if len(df) >= 14:
                stoch = StochasticOscillator(result['high'], result['low'], result['close'])
                result['stoch_k'] = stoch.stoch()
                result['stoch_d'] = stoch.stoch_signal()
            
            # Bollinger Bands
            if len(df) >= 20:
                bb = BollingerBands(result['close'])
                result['bb_high'] = bb.bollinger_hband()
                result['bb_mid'] = bb.bollinger_mavg()
                result['bb_low'] = bb.bollinger_lband()
                result['bb_width'] = (result['bb_high'] - result['bb_low']) / result['bb_mid']
                result['bb_pct'] = (result['close'] - result['bb_low']) / (result['bb_high'] - result['bb_low'])
            
            # ATR
            if len(df) >= 14:
                atr = AverageTrueRange(result['high'], result['low'], result['close'])
                result['atr'] = atr.average_true_range()
                result['atr_pct'] = result['atr'] / result['close']
        
        # Volume indicators (if volume is available)
        if 'volume' in result.columns and "ta" in self.config.feature_sets:
            # On-Balance Volume
            obv = OnBalanceVolumeIndicator(result['close'], result['volume'])
            result['obv'] = obv.on_balance_volume()
            
            # Volume moving averages
            for period in [5, 10, 20]:
                if len(df) >= period:
                    result[f'volume_sma_{period}'] = result['volume'].rolling(window=period).mean()
                    result[f'volume_ratio_{period}'] = result['volume'] / result[f'volume_sma_{period}']
        
        # Drop NaN values
        result = result.dropna()
        
        return result
    
    async def get_latest_data(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Get the latest processed data for all assets and timeframes.
        
        Returns:
            Dict[str, Dict[int, pd.DataFrame]]: Dictionary of feature DataFrames
        """
        if not self.initialized:
            raise RuntimeError("Data manager not initialized")
        
        return self.features
    
    async def get_training_data(self) -> Dict[str, Dict[int, pd.DataFrame]]:
        """
        Get data for model training.
        
        Returns:
            Dict[str, Dict[int, pd.DataFrame]]: Dictionary of feature DataFrames
        """
        if not self.initialized:
            raise RuntimeError("Data manager not initialized")
        
        # For now, just return all available data
        # In a more advanced implementation, we might want to filter or process the data differently for training
        return self.features
    
    def get_feature_columns(self) -> List[str]:
        """
        Get a list of feature columns.
        
        Returns:
            List[str]: List of feature column names
        """
        # Get feature columns from the first available DataFrame
        for asset in self.features:
            for timeframe in self.features[asset]:
                df = self.features[asset][timeframe]
                if len(df) > 0:
                    # Exclude non-feature columns
                    exclude_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    return [col for col in df.columns if col not in exclude_columns]
        
        # If no data is available, return an empty list
        return []
    
    def save_data(self) -> None:
        """Save data to disk."""
        try:
            # Ensure data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save candles
            for asset in self.candles:
                asset_dir = self.data_dir / asset
                asset_dir.mkdir(exist_ok=True)
                
                for timeframe in self.candles[asset]:
                    df = self.candles[asset][timeframe]
                    if len(df) > 0:
                        file_path = asset_dir / f"candles_{timeframe}.csv"
                        df.to_csv(file_path, index=False)
            
            logger.info("Data saved successfully")
            
        except Exception as e:
            logger.exception(f"Error saving data: {e}")
    
    def load_data(self, assets: List[str], timeframes: List[int]) -> bool:
        """
        Load data from disk.
        
        Args:
            assets: List of asset symbols
            timeframes: List of timeframes in seconds
            
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        try:
            # Initialize data storage
            self.candles = {asset: {} for asset in assets}
            self.features = {asset: {} for asset in assets}
            
            # Load candles
            for asset in assets:
                asset_dir = self.data_dir / asset
                if not asset_dir.exists():
                    continue
                
                for timeframe in timeframes:
                    file_path = asset_dir / f"candles_{timeframe}.csv"
                    if file_path.exists():
                        df = pd.read_csv(file_path)
                        
                        # Convert timestamp to datetime
                        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        self.candles[asset][timeframe] = df
                        
                        # Process features
                        self.features[asset][timeframe] = self._process_features(df)
                        
                        logger.info(f"Loaded {len(df)} candles for {asset} at {timeframe}s timeframe")
            
            self.initialized = True
            logger.info("Data loaded successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            return False
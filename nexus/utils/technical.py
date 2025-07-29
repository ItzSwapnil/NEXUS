"""
Technical Analysis Utilities for NEXUS Trading System

This module provides functions for calculating technical indicators and features
without relying on external TA libraries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """
    Calculate exponential moving average without pandas.

    Args:
        data: Input data array
        span: EMA period

    Returns:
        np.ndarray: EMA values
    """
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]  # Initialize with first value

    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

    return ema

def simple_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate simple moving average.

    Args:
        data: Input data array
        window: SMA period

    Returns:
        np.ndarray: SMA values
    """
    result = np.zeros_like(data)
    result[:] = np.nan

    for i in range(window - 1, len(data)):
        result[i] = np.mean(data[i - window + 1:i + 1])

    return result

def relative_strength_index(data: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate RSI.

    Args:
        data: Input price data array
        window: RSI period

    Returns:
        np.ndarray: RSI values
    """
    # Calculate price changes
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]

    # Separate gains and losses
    gain = np.zeros_like(delta)
    loss = np.zeros_like(delta)

    gain[delta > 0] = delta[delta > 0]
    loss[delta < 0] = -delta[delta < 0]

    # Calculate average gain and loss
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)

    # First value is simple average
    avg_gain[window] = np.mean(gain[1:window+1])
    avg_loss[window] = np.mean(loss[1:window+1])

    # Subsequent values use smoothed averages
    for i in range(window + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window

    # Calculate RS and RSI
    rs = np.zeros_like(avg_gain)
    rsi = np.zeros_like(avg_gain)

    # Avoid division by zero
    nonzero = avg_loss != 0
    rs[nonzero] = avg_gain[nonzero] / avg_loss[nonzero]
    rs[~nonzero] = 100.0

    rsi = 100 - (100 / (1 + rs))

    return rsi

def bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.

    Args:
        data: Input price data array
        window: Moving average period
        num_std: Number of standard deviations

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Middle band, upper band, lower band
    """
    # Calculate middle band (SMA)
    middle_band = simple_moving_average(data, window)

    # Calculate standard deviation
    rolling_std = np.zeros_like(middle_band)
    for i in range(window - 1, len(data)):
        rolling_std[i] = np.std(data[i - window + 1:i + 1])

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return middle_band, upper_band, lower_band

def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD.

    Args:
        data: Input price data array
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: MACD line, signal line, histogram
    """
    # Calculate fast and slow EMAs
    ema_fast = exponential_moving_average(data, fast_period)
    ema_slow = exponential_moving_average(data, slow_period)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = exponential_moving_average(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Average True Range.

    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        window: ATR period

    Returns:
        np.ndarray: ATR values
    """
    # Create shifted close prices (previous day's close)
    prev_close = np.zeros_like(close)
    prev_close[1:] = close[:-1]

    # Calculate true range
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    # True range is max of the three
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Calculate ATR
    atr = np.zeros_like(tr)

    # First value is simple average
    atr[window-1] = np.mean(tr[:window])

    # Subsequent values use smoothed average
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window-1) + tr[i]) / window

    return atr

def detect_supports_and_resistances(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 10, threshold: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect support and resistance levels.

    Args:
        high: High prices array
        low: Low prices array
        close: Close prices array
        window: Lookback window
        threshold: Price deviation threshold

    Returns:
        Tuple[np.ndarray, np.ndarray]: Support and resistance levels
    """
    supports = np.zeros_like(close)
    resistances = np.zeros_like(close)

    # We need at least window * 2 + 1 data points
    if len(close) < window * 2 + 1:
        return supports, resistances

    for i in range(window, len(close) - window):
        # Check if current low is a local minimum
        if np.all(low[i] <= low[i-window:i]) and np.all(low[i] <= low[i+1:i+window+1]):
            supports[i] = low[i]

        # Check if current high is a local maximum
        if np.all(high[i] >= high[i-window:i]) and np.all(high[i] >= high[i+1:i+window+1]):
            resistances[i] = high[i]

    # Forward fill values
    for i in range(1, len(close)):
        if supports[i] == 0:
            supports[i] = supports[i-1]
        if resistances[i] == 0:
            resistances[i] = resistances[i-1]

    return supports, resistances

def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators and features for analysis.

    Args:
        data: OHLCV DataFrame (must have 'open', 'high', 'low', 'close', 'volume' columns)

    Returns:
        pd.DataFrame: DataFrame with calculated features
    """
    # Create copy of data to avoid modifying original
    df = data.copy()

    # Check if we have enough data for calculations
    if len(df) < 50:  # Need minimum data points for reliable indicators
        return df

    # Get price arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_price = df['open'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)

    # Calculate moving averages
    df['ema_short'] = exponential_moving_average(close, 9)
    df['ema_medium'] = exponential_moving_average(close, 21)
    df['ema_long'] = exponential_moving_average(close, 50)
    df['sma_20'] = simple_moving_average(close, 20)
    df['sma_50'] = simple_moving_average(close, 50)

    # Calculate RSI
    df['rsi'] = relative_strength_index(close)

    # Calculate Bollinger Bands
    middle, upper, lower = bollinger_bands(close)
    df['bb_middle'] = middle
    df['bb_upper'] = upper
    df['bb_lower'] = lower

    # Calculate MACD
    macd_line, signal, hist = macd(close)
    df['macd'] = macd_line
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # Calculate ATR
    df['atr'] = average_true_range(high, low, close)

    # Detect support and resistance
    supports, resistances = detect_supports_and_resistances(high, low, close)
    df['support'] = supports
    df['resistance'] = resistances

    # Calculate price changes
    df['daily_return'] = np.zeros_like(close)
    df.loc[1:, 'daily_return'] = (close[1:] - close[:-1]) / close[:-1]

    # Calculate volatility (rolling 20-day standard deviation of returns)
    returns = df['daily_return'].values
    volatility = np.zeros_like(returns)
    for i in range(20, len(returns)):
        volatility[i] = np.std(returns[i-20:i])
    df['volatility'] = volatility

    # Calculate momentum (rate of change)
    df['momentum'] = np.zeros_like(close)
    df.loc[10:, 'momentum'] = (close[10:] - close[:-10]) / close[:-10]

    # Calculate trend strength
    df['trend_strength'] = np.zeros_like(close)
    # Positive values indicate bullish trend, negative values indicate bearish trend
    df.loc[50:, 'trend_strength'] = (df['ema_short'][50:] - df['ema_long'][50:]) / df['ema_long'][50:]

    # Mean reversion indicator (deviation from SMA50)
    df['mean_reversion'] = (close - df['sma_50']) / df['sma_50']

    # Volume trend
    df['volume_trend'] = np.zeros_like(volume)
    vol_sma = simple_moving_average(volume, 20)
    df.loc[20:, 'volume_trend'] = (volume[20:] - vol_sma[20:]) / vol_sma[20:]

    # Normalized close price (0-1 range in recent window)
    rolling_min = np.zeros_like(close)
    rolling_max = np.zeros_like(close)
    for i in range(20, len(close)):
        rolling_min[i] = np.min(close[i-20:i])
        rolling_max[i] = np.max(close[i-20:i])

    # Avoid division by zero
    range_diff = rolling_max - rolling_min
    range_diff[range_diff == 0] = 1.0

    df['close_norm'] = np.zeros_like(close)
    df.loc[20:, 'close_norm'] = (close[20:] - rolling_min[20:]) / range_diff[20:]

    # High-low difference relative to close
    df['high_low_diff'] = (high - low) / close

    # Price pattern detection (simple version)
    # 1 for bullish, -1 for bearish, values in between for mixed
    df['price_pattern'] = np.zeros_like(close)
    for i in range(3, len(close)):
        # Use last 3 candles to determine pattern
        o1, h1, l1, c1 = open_price[i-3], high[i-3], low[i-3], close[i-3]
        o2, h2, l2, c2 = open_price[i-2], high[i-2], low[i-2], close[i-2]
        o3, h3, l3, c3 = open_price[i-1], high[i-1], low[i-1], close[i-1]

        # Body sizes
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        body3 = abs(c3 - o3)

        # Directions
        dir1 = 1 if c1 > o1 else -1
        dir2 = 1 if c2 > o2 else -1
        dir3 = 1 if c3 > o3 else -1

        # Pattern score based on directions and body sizes
        score = (dir1 * body1 + dir2 * body2 * 2 + dir3 * body3 * 3) / (body1 + body2 * 2 + body3 * 3)

        df['price_pattern'][i] = score

    # Fill NaN values
    df = df.fillna(0)

    return df

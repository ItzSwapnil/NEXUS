"""Lightweight technical indicators without external TA libs."""

import numpy as np
import pandas as pd
from typing import Tuple


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """Compute EMA."""
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def simple_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute SMA."""
    result = np.zeros_like(data)
    result[:] = np.nan
    for i in range(window - 1, len(data)):
        result[i] = np.mean(data[i - window + 1:i + 1])
    return result


def relative_strength_index(data: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute RSI."""
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]
    gain = np.zeros_like(delta)
    loss = np.zeros_like(delta)
    gain[delta > 0] = delta[delta > 0]
    loss[delta < 0] = -delta[delta < 0]
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[window] = np.mean(gain[1:window+1])
    avg_loss[window] = np.mean(loss[1:window+1])
    for i in range(window + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
    rs = np.zeros_like(avg_gain)
    rsi = np.zeros_like(avg_gain)
    nonzero = avg_loss != 0
    rs[nonzero] = avg_gain[nonzero] / avg_loss[nonzero]
    rs[~nonzero] = 100.0
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(data: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    middle_band = simple_moving_average(data, window)
    rolling_std = np.zeros_like(middle_band)
    for i in range(window - 1, len(data)):
        rolling_std[i] = np.std(data[i - window + 1:i + 1])
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    return middle_band, upper_band, lower_band


def macd(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD line, signal, and histogram."""
    ema_fast = exponential_moving_average(data, fast_period)
    ema_slow = exponential_moving_average(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def average_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute ATR."""
    prev_close = np.zeros_like(close)
    prev_close[1:] = close[:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[window-1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
    return atr


def detect_supports_and_resistances(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 10, threshold: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Detect simple supports and resistances."""
    supports = np.zeros_like(close)
    resistances = np.zeros_like(close)
    if len(close) < window * 2 + 1:
        return supports, resistances
    for i in range(window, len(close) - window):
        if np.all(low[i] <= low[i-window:i]) and np.all(low[i] <= low[i+1:i+window+1]):
            supports[i] = low[i]
        if np.all(high[i] >= high[i-window:i]) and np.all(high[i] >= high[i+1:i+window+1]):
            resistances[i] = high[i]
    for i in range(1, len(close)):
        if supports[i] == 0:
            supports[i] = supports[i-1]
        if resistances[i] == 0:
            resistances[i] = resistances[i-1]
    return supports, resistances


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compute a feature set from OHLCV data."""
    df = data.copy()
    if len(df) < 50:
        return df
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    df['ema_short'] = exponential_moving_average(close, 9)
    df['ema_medium'] = exponential_moving_average(close, 21)
    df['ema_long'] = exponential_moving_average(close, 50)
    df['sma_20'] = simple_moving_average(close, 20)
    df['sma_50'] = simple_moving_average(close, 50)
    df['rsi'] = relative_strength_index(close)
    middle, upper, lower = bollinger_bands(close)
    df['bb_middle'] = middle
    df['bb_upper'] = upper
    df['bb_lower'] = lower
    macd_line, signal, hist = macd(close)
    df['macd'] = macd_line
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    df['atr'] = average_true_range(high, low, close)
    supports, resistances = detect_supports_and_resistances(high, low, close)
    df['support'] = supports
    df['resistance'] = resistances
    df['daily_return'] = np.zeros_like(close)
    df.loc[1:, 'daily_return'] = (close[1:] - close[:-1]) / close[:-1]
    returns = df['daily_return'].values
    volatility = np.zeros_like(returns)
    for i in range(20, len(returns)):
        volatility[i] = np.std(returns[i-20:i])
    df['volatility'] = volatility
    df['momentum'] = np.zeros_like(close)
    df.loc[10:, 'momentum'] = (close[10:] - close[:-10]) / close[:-10]
    df['trend_strength'] = np.zeros_like(close)
    df.loc[50:, 'trend_strength'] = (df['ema_short'][50:] - df['ema_long'][50:]) / df['ema_long'][50:]
    return df

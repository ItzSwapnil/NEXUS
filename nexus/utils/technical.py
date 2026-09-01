"""Lightweight technical indicators without external TA libs."""

from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd

try:  # Optional: native indicators remain available without TA-Lib.
    import talib
    from talib import abstract as talib_abstract
except ImportError:  # pragma: no cover - depends on local installation
    talib = None
    talib_abstract = None


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """Compute EMA."""
    data = np.asarray(data, dtype=float)
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def simple_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute SMA."""
    data = np.asarray(data, dtype=float)
    result = np.zeros_like(data)
    result[:] = np.nan
    for i in range(window - 1, len(data)):
        result[i] = np.mean(data[i - window + 1 : i + 1])
    return result


def relative_strength_index(data: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute RSI."""
    data = np.asarray(data, dtype=float)
    delta = np.zeros_like(data)
    delta[1:] = data[1:] - data[:-1]
    gain = np.zeros_like(delta)
    loss = np.zeros_like(delta)
    gain[delta > 0] = delta[delta > 0]
    loss[delta < 0] = -delta[delta < 0]
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[window] = np.mean(gain[1 : window + 1])
    avg_loss[window] = np.mean(loss[1 : window + 1])
    for i in range(window + 1, len(data)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window
    rs = np.zeros_like(avg_gain)
    rsi = np.zeros_like(avg_gain)
    nonzero = avg_loss != 0
    rs[nonzero] = avg_gain[nonzero] / avg_loss[nonzero]
    rs[~nonzero] = 100.0
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(
    data: np.ndarray, window: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Bollinger Bands (middle, upper, lower)."""
    data = np.asarray(data, dtype=float)
    middle_band = simple_moving_average(data, window)
    rolling_std = np.zeros_like(middle_band)
    for i in range(window - 1, len(data)):
        rolling_std[i] = np.std(data[i - window + 1 : i + 1])
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    return middle_band, upper_band, lower_band


def macd(
    data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD line, signal, and histogram."""
    data = np.asarray(data, dtype=float)
    ema_fast = exponential_moving_average(data, fast_period)
    ema_slow = exponential_moving_average(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def average_true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> np.ndarray:
    """Compute ATR."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.zeros_like(close)
    prev_close[1:] = close[:-1]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[window - 1] = np.mean(tr[:window])
    for i in range(window, len(tr)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return np.asarray(atr, dtype=np.float64)


def detect_supports_and_resistances(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 10, threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect simple supports and resistances."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    supports = np.zeros_like(close)
    resistances = np.zeros_like(close)
    if len(close) < window * 2 + 1:
        return supports, resistances
    for i in range(window, len(close) - window):
        if np.all(low[i] <= low[i - window : i]) and np.all(low[i] <= low[i + 1 : i + window + 1]):
            supports[i] = low[i]
        if np.all(high[i] >= high[i - window : i]) and np.all(
            high[i] >= high[i + 1 : i + window + 1]
        ):
            resistances[i] = high[i]
    for i in range(1, len(close)):
        if supports[i] == 0:
            supports[i] = supports[i - 1]
        if resistances[i] == 0:
            resistances[i] = resistances[i - 1]
    return supports, resistances


def stochastic_oscillator(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Stochastic Oscillator (%K, %D)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    stoch_k = np.zeros_like(close)
    for i in range(k_period - 1, len(close)):
        lowest_low = np.min(low[i - k_period + 1 : i + 1])
        highest_high = np.max(high[i - k_period + 1 : i + 1])
        denom = highest_high - lowest_low
        stoch_k[i] = ((close[i] - lowest_low) / denom * 100.0) if denom != 0 else 50.0
    stoch_d = simple_moving_average(stoch_k, d_period)
    return stoch_k, stoch_d


def commodity_channel_index(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20
) -> np.ndarray:
    """Compute Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3.0
    sma_tp = simple_moving_average(tp, window)
    cci = np.zeros_like(tp)
    for i in range(window - 1, len(tp)):
        mean_dev = np.mean(np.abs(tp[i - window + 1 : i + 1] - sma_tp[i]))
        cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mean_dev) if mean_dev != 0 else 0.0
    return cast(np.ndarray, cci)


def williams_r(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> np.ndarray:
    """Compute Williams %R."""
    wr = np.zeros_like(close)
    for i in range(window - 1, len(close)):
        hh = np.max(high[i - window + 1 : i + 1])
        ll = np.min(low[i - window + 1 : i + 1])
        denom = hh - ll
        wr[i] = ((hh - close[i]) / denom * -100.0) if denom != 0 else -50.0
    return wr


def adx_di(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADX, +DI, -DI."""
    n = len(close)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    if n < window * 2:
        return adx, plus_di, minus_di

    tr = average_true_range(high, low, close, window=1)
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = np.zeros(n)
    p_dm_smooth = np.zeros(n)
    m_dm_smooth = np.zeros(n)

    tr_smooth[window] = np.sum(tr[1 : window + 1])
    p_dm_smooth[window] = np.sum(plus_dm[:window])
    m_dm_smooth[window] = np.sum(minus_dm[:window])

    for i in range(window + 1, n):
        tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / window) + tr[i]
        p_dm_smooth[i] = p_dm_smooth[i - 1] - (p_dm_smooth[i - 1] / window) + plus_dm[i - 1]
        m_dm_smooth[i] = m_dm_smooth[i - 1] - (m_dm_smooth[i - 1] / window) + minus_dm[i - 1]

        if tr_smooth[i] > 0:
            plus_di[i] = (p_dm_smooth[i] / tr_smooth[i]) * 100.0
            minus_di[i] = (m_dm_smooth[i] / tr_smooth[i]) * 100.0

    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    nonzero = di_sum != 0
    dx[nonzero] = (np.abs(plus_di[nonzero] - minus_di[nonzero]) / di_sum[nonzero]) * 100.0

    if n > window * 2:
        adx[window * 2] = np.mean(dx[window : window * 2])
        for i in range(window * 2 + 1, n):
            adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

    return adx, plus_di, minus_di


def detect_candlestick_patterns(
    open_p: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Dict[str, np.ndarray]:
    """Detect Candlestick Patterns: Doji, Hammer, Engulfing, Morning/Evening Star."""
    n = len(close)
    patterns = {
        "doji": np.zeros(n),
        "hammer": np.zeros(n),
        "bullish_engulfing": np.zeros(n),
        "bearish_engulfing": np.zeros(n),
        "shooting_star": np.zeros(n),
    }
    body = np.abs(close - open_p)
    candle_range = high - low

    for i in range(1, n):
        cr = candle_range[i]
        if cr == 0:
            continue
        # Doji: small body <= 10% of range
        if body[i] / cr <= 0.10:
            patterns["doji"][i] = 1.0

        # Hammer: small upper wick, lower wick >= 2x body
        lower_wick = min(open_p[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_p[i], close[i])
        if lower_wick >= 2 * body[i] and upper_wick <= 0.2 * cr:
            patterns["hammer"][i] = 1.0

        # Shooting Star: long upper wick >= 2x body, tiny lower wick
        if upper_wick >= 2 * body[i] and lower_wick <= 0.2 * cr:
            patterns["shooting_star"][i] = 1.0

        # Engulfing Patterns
        if close[i - 1] < open_p[i - 1] and close[i] > open_p[i]:
            if close[i] >= open_p[i - 1] and open_p[i] <= close[i - 1]:
                patterns["bullish_engulfing"][i] = 1.0
        elif close[i - 1] > open_p[i - 1] and close[i] < open_p[i]:
            if open_p[i] >= close[i - 1] and close[i] <= open_p[i - 1]:
                patterns["bearish_engulfing"][i] = 1.0

    return patterns


def add_talib_features(data: pd.DataFrame, max_functions: int = 200) -> pd.DataFrame:
    """Append compatible TA-Lib outputs without making TA-Lib mandatory.

    TA-Lib has many multi-output functions and some require inputs not present
    in OHLCV data. Each function is isolated so one incompatible indicator
    cannot break the live feature pipeline.
    """
    if talib_abstract is None or not hasattr(talib, "get_functions"):
        return data
    result = data.copy()
    inputs = {column: result[column].to_numpy(dtype=float) for column in result.columns if column in
              {"open", "high", "low", "close", "volume"}}
    for name in list(talib.get_functions())[:max_functions]:
        try:
            fn = talib_abstract.Function(name)
            required = set(getattr(fn, "input_names", {}).keys())
            if not required.issubset(inputs):
                continue
            output = fn(inputs)
            if isinstance(output, dict):
                values = output.items()
            elif isinstance(output, (tuple, list)):
                values = ((f"output_{idx}", value) for idx, value in enumerate(output))
            else:
                values = (("value", output),)
            for suffix, value in values:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 1 or len(arr) != len(result):
                    continue
                column = f"talib_{name.lower()}_{str(suffix).lower()}"
                if column not in result:
                    result[column] = arr
        except Exception:
            continue
    return result


def available_indicator_catalog() -> Dict[str, Any]:
    """Describe the indicator search space available in this installation."""
    native = [
        "EMA", "SMA", "RSI", "Bollinger Bands", "MACD", "ATR", "ADX", "DI",
        "Stochastic", "CCI", "Williams %R", "Support/Resistance", "Candlestick Patterns",
    ]
    talib_names = list(talib.get_functions()) if talib is not None else []
    return {
        "native_indicators": native,
        "talib_available": bool(talib_names),
        "talib_count": len(talib_names),
        "talib_functions": talib_names,
        "total_available": len(native) + len(talib_names),
    }


_TRAINED_MARKET_BLUEPRINTS: Dict[str, Dict[str, Any]] = {}


def register_trained_market_blueprint(asset: str, blueprint: Dict[str, Any]) -> None:
    """Store dynamically trained market blueprint and best indicators for a specific market."""
    _TRAINED_MARKET_BLUEPRINTS[asset] = blueprint


def get_market_indicator_blueprint(asset: str) -> Dict[str, Any]:
    """Return tailored technical indicator blueprint and optimal parameters tuned for a specific market."""
    if asset in _TRAINED_MARKET_BLUEPRINTS:
        return dict(_TRAINED_MARKET_BLUEPRINTS[asset])

    s_upper = asset.upper()
    is_otc = "OTC" in s_upper

    if is_otc:
        return {
            "profile_name": f"{asset} OTC Adaptive Scalper Blueprint",
            "rsi_period": 7,
            "ema_fast": 5,
            "ema_slow": 13,
            "stoch_k": 5,
            "stoch_d": 3,
            "primary_indicators": [
                "EMA(5/13) Fast Micro-Cross",
                "RSI(7) Tick Momentum",
                "Stochastic(5,3) Oscillator",
                "Engulfing/Hammer Reversals",
            ],
            "description": f"Custom tuned for {asset} micro-scalping and rapid tick reversals.",
        }
    elif "BTC" in s_upper or "ETH" in s_upper or "SOL" in s_upper:
        return {
            "profile_name": f"{asset} Crypto Volatility Expansion Blueprint",
            "rsi_period": 14,
            "ema_fast": 9,
            "ema_slow": 21,
            "bb_std": 2.5,
            "primary_indicators": [
                "Bollinger Bands (2.5σ Breakout)",
                "ADX Trend Expansion",
                "MACD(12,26,9) Impulse",
                "ATR Volatility Channel",
            ],
            "description": f"Custom tuned for {asset} momentum surges and breakout expansions.",
        }
    elif "XAU" in s_upper or "XAG" in s_upper or "OIL" in s_upper:
        return {
            "profile_name": f"{asset} Commodity Trend & S/R Blueprint",
            "rsi_period": 14,
            "ema_fast": 10,
            "ema_slow": 30,
            "primary_indicators": [
                "Key Support & Resistance Bounces",
                "EMA(10/30) Trend Channel",
                "CCI Momentum Wave",
                "Williams %R Extremes",
            ],
            "description": f"Custom tuned for {asset} trend continuation and structural levels.",
        }
    else:
        return {
            "profile_name": f"{asset} Forex Tailored Confluence Blueprint",
            "rsi_period": 14,
            "ema_fast": 9,
            "ema_slow": 21,
            "primary_indicators": [
                f"RSI(14) Mean Reversion ({asset})",
                "EMA(9/21) Cross Confluence",
                "Stochastic(14,3) Reversals",
                "ADX Directional Strength",
            ],
            "description": f"Custom tuned for {asset} session liquidity and harmonic setups.",
        }


def calculate_features(
    data: pd.DataFrame, asset: str = "EURUSD", custom_params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Compute a rich 35+ technical indicator feature set dynamically learned and adapted for the specific asset."""
    df = data.copy()
    if len(df) < 50:
        return df

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    open_p = df["open"].to_numpy(dtype=float)

    blueprint = get_market_indicator_blueprint(asset)
    params = custom_params or blueprint.get("params", {}) or blueprint

    ema_fast_period = int(params.get("ema_fast", blueprint.get("ema_fast", 9)))
    ema_slow_period = int(params.get("ema_slow", blueprint.get("ema_slow", 21)))
    rsi_p = int(params.get("rsi_period", blueprint.get("rsi_period", 14)))

    # 1. Asset-Tuned Moving Averages & Trend
    df["ema_short"] = exponential_moving_average(close, ema_fast_period)
    df["ema_medium"] = exponential_moving_average(close, ema_slow_period)
    df["ema_long"] = exponential_moving_average(close, 50)
    df["sma_20"] = simple_moving_average(close, 20)
    df["sma_50"] = simple_moving_average(close, 50)

    # 2. Asset-Tuned Oscillators & Momentum
    df["rsi"] = relative_strength_index(close, window=rsi_p)
    stoch_k_period = int(blueprint.get("stoch_k", 14))
    stoch_d_period = int(blueprint.get("stoch_d", 3))
    stoch_k, stoch_d = stochastic_oscillator(
        high, low, close, k_period=stoch_k_period, d_period=stoch_d_period
    )
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    df["cci"] = commodity_channel_index(high, low, close)
    df["williams_r"] = williams_r(high, low, close)

    # 3. Volatility & Bands
    bb_std_val = float(blueprint.get("bb_std", 2.0))
    middle, upper, lower = bollinger_bands(close, num_std=bb_std_val)
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bollinger_pband"] = (close - lower) / (upper - lower + 1e-9)

    macd_line, signal, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    df["atr"] = average_true_range(high, low, close)

    # 4. Trend Strength (ADX, +DI, -DI)
    adx_val, p_di, m_di = adx_di(high, low, close)
    df["adx"] = adx_val
    df["plus_di"] = p_di
    df["minus_di"] = m_di

    # 5. Support & Resistance & Volatility Metrics
    supports, resistances = detect_supports_and_resistances(high, low, close)
    df["support"] = supports
    df["resistance"] = resistances

    df["daily_return"] = np.zeros_like(close)
    df.loc[1:, "daily_return"] = (close[1:] - close[:-1]) / close[:-1]
    returns = df["daily_return"].to_numpy(dtype=float)
    volatility = np.zeros_like(returns)
    for i in range(20, len(returns)):
        volatility[i] = np.std(returns[i - 20 : i])
    df["volatility"] = volatility

    df["momentum"] = np.zeros_like(close)
    df.loc[10:, "momentum"] = (close[10:] - close[:-10]) / close[:-10]
    df["trend_strength"] = np.zeros_like(close)
    df.loc[50:, "trend_strength"] = (df["ema_short"][50:] - df["ema_long"][50:]) / (
        df["ema_long"][50:] + 1e-9
    )

    # 6. Candlestick Patterns & Confluence
    patterns = detect_candlestick_patterns(open_p, high, low, close)
    for k, v in patterns.items():
        df[f"pattern_{k}"] = v

    # Expand the candidate feature space when TA-Lib is installed. The model
    # and trainer perform selection downstream; this is not a blind signal vote.
    from nexus.features.feature_engine import add_external_features

    df = add_external_features(df)

    # 7. Multi-Indicator Confluence Score (-1.0 Bearish to +1.0 Bullish)
    confluence = np.zeros_like(close)
    confluence += np.where(df["rsi"] < 35, 0.2, np.where(df["rsi"] > 65, -0.2, 0.0))
    confluence += np.where(df["macd"] > df["macd_signal"], 0.2, -0.2)
    confluence += np.where(df["ema_short"] > df["ema_medium"], 0.2, -0.2)
    confluence += np.where(df["stoch_k"] > df["stoch_d"], 0.15, -0.15)
    confluence += np.where(df["plus_di"] > df["minus_di"], 0.15, -0.15)
    confluence += patterns["bullish_engulfing"] * 0.10 - patterns["bearish_engulfing"] * 0.10

    df["confluence_score"] = np.clip(confluence, -1.0, 1.0)
    return df

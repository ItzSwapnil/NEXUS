"""Unit tests for technical indicators in nexus.utils.technical."""

import numpy as np
import pandas as pd

from nexus.utils.technical import (
    adx_di,
    average_true_range,
    bollinger_bands,
    calculate_features,
    commodity_channel_index,
    exponential_moving_average,
    macd,
    relative_strength_index,
    simple_moving_average,
    stochastic_oscillator,
    williams_r,
)


def test_basic_moving_averages():
    data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    ema = exponential_moving_average(data, span=5)
    assert len(ema) == len(data)
    assert ema[-1] > ema[0]

    sma = simple_moving_average(data, window=5)
    assert len(sma) == len(data)
    assert np.isnan(sma[0])
    assert sma[-1] == 18.0


def test_rsi():
    prices = np.linspace(100, 150, 30)
    rsi_vals = relative_strength_index(prices, window=14)
    assert len(rsi_vals) == len(prices)
    assert rsi_vals[-1] > 70.0


def test_bollinger_bands():
    data = np.sin(np.linspace(0, 10, 50)) + 100
    mid, up, low = bollinger_bands(data, window=10, num_std=2.0)
    assert len(mid) == len(data)
    assert np.all(up[9:] >= mid[9:])
    assert np.all(mid[9:] >= low[9:])


def test_macd():
    data = np.linspace(50, 100, 40)
    macd_line, sig_line, hist = macd(data, fast_period=12, slow_period=26, signal_period=9)
    assert len(macd_line) == len(data)
    assert len(sig_line) == len(data)
    assert len(hist) == len(data)


def test_atr_and_oscillators():
    high = np.array([10.5, 11.2, 12.0, 11.8, 12.5, 13.0, 12.8, 13.5, 14.0, 14.2] * 4)
    low = np.array([9.8, 10.1, 11.0, 10.9, 11.4, 12.1, 11.9, 12.5, 13.1, 13.5] * 4)
    close = np.array([10.2, 11.0, 11.5, 11.2, 12.2, 12.7, 12.3, 13.2, 13.8, 14.0] * 4)

    atr = average_true_range(high, low, close, window=14)
    assert len(atr) == len(close)
    assert np.all(atr[14:] > 0)

    stoch_k, stoch_d = stochastic_oscillator(high, low, close, k_period=14, d_period=3)
    assert len(stoch_k) == len(close)
    assert len(stoch_d) == len(close)

    cci = commodity_channel_index(high, low, close, window=14)
    assert len(cci) == len(close)

    wr = williams_r(high, low, close, window=14)
    assert len(wr) == len(close)

    adx, pdi, mdi = adx_di(high, low, close, window=14)
    assert len(adx) == len(close)


def test_calculate_features():
    n = 60
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 112, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100.5, 111, n),
            "volume": np.ones(n) * 1000,
        }
    )
    features_df = calculate_features(df, asset="EURUSD")
    assert "rsi" in features_df.columns
    assert "confluence_score" in features_df.columns
    assert len(features_df) == n

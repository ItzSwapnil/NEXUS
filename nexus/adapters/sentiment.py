"""
Lightweight sentiment adapter for NEXUS.

This module derives a synthetic sentiment score [0..1] from recent price action
without external APIs (safe for CI). You can replace this with a real news/social
feed adapter later and keep the same interface.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def compute_sentiment(candles: pd.DataFrame) -> Optional[float]:
    """
    Compute a synthetic sentiment score from OHLCV candles.
    Returns a float in [0..1], where 0.5 is neutral.
    Heuristic: combine momentum and volatility regime.
    """
    if candles is None or len(candles) < 20:
        return 0.5
    closes = candles['close'].values
    # Momentum over last 10 bars
    mom = (closes[-1] - closes[-11]) / (closes[-11] + 1e-9)
    # Volatility (std of returns) over last 20 bars
    rets = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-9)
    vol = float(np.std(rets[-20:])) if len(rets) >= 20 else float(np.std(rets))
    # Normalize to [-1..1]
    mom_norm = float(np.tanh(mom * 5))
    vol_penalty = float(np.tanh(vol * 10))
    # Sentiment prefers positive momentum and penalizes extreme volatility
    raw = mom_norm - 0.3 * vol_penalty
    # Map to [0..1]
    score = 0.5 + 0.5 * np.tanh(raw)
    return float(max(0.0, min(1.0, score)))


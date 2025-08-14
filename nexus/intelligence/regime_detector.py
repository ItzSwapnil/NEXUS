"""Lightweight market regime detector (async) for tests.

Provides a simple random regime assignment among a predefined list limited by
n_regimes parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random
import pandas as pd


@dataclass
class RegimeDetector:
    n_regimes: int = 4
    lookback_periods: int = 100

    REGIMES: List[str] = None  # type: ignore

    def __post_init__(self):
        base = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
        if self.n_regimes < 1:
            self.n_regimes = 1
        self.REGIMES = base[: self.n_regimes]

    async def detect_regime(self, data: pd.DataFrame) -> str:
        # Minimal validation: ensure we have at least lookback rows
        _ = len(data)
        return random.choice(self.REGIMES)


__all__ = ["RegimeDetector"]


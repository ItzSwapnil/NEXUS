"""Leakage-resistant research and evaluation utilities for NEXUS.

This module deliberately keeps feature selection, calibration, and promotion
inside each walk-forward training window. It is designed for replay jobs and
never mutates live model state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


@dataclass(frozen=True)
class ConfidenceBand:
    threshold: float
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    net_pnl: float


@dataclass
class RobustResearchReport:
    windows: int
    observations: int
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    max_drawdown: float
    calibration_error: float
    confidence_bands: list[ConfidenceBand] = field(default_factory=list)
    selected_features_by_window: list[list[str]] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "windows": self.windows, "observations": self.observations,
            "trades": self.trades, "wins": self.wins, "losses": self.losses,
            "win_rate": self.win_rate, "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown, "calibration_error": self.calibration_error,
            "confidence_bands": [band.__dict__ for band in self.confidence_bands],
            "selected_features_by_window": self.selected_features_by_window,
            "rejection_reasons": self.rejection_reasons,
        }


class PurgedWalkForward:
    """Create chronological train/validation/test windows with an embargo."""

    def __init__(self, train_size: int = 1000, validation_size: int = 300, test_size: int = 300, embargo: int = 1, step: int | None = None) -> None:
        self.train_size, self.validation_size, self.test_size = train_size, validation_size, test_size
        self.embargo, self.step = embargo, step or test_size

    def windows(self, observations: int) -> list[WalkForwardWindow]:
        result = []
        test_start = self.train_size + self.embargo + self.validation_size + self.embargo
        while test_start + self.test_size <= observations:
            result.append(WalkForwardWindow((0, test_start - self.embargo - self.validation_size - self.embargo), (test_start - self.validation_size - self.embargo, test_start - self.embargo), (test_start, test_start + self.test_size)))
            test_start += self.step
        return result


def select_features(frame: pd.DataFrame, target: pd.Series, train_end: int, limit: int = 32) -> list[str]:
    """Rank and de-correlate numeric features using training rows only."""
    numeric = frame.select_dtypes(include=[np.number]).iloc[:train_end]
    target = target.iloc[:train_end]
    ranked: list[tuple[float, str]] = []
    for name in numeric.columns:
        values = numeric[name].replace([np.inf, -np.inf], np.nan)
        valid = values.notna() & target.notna()
        if valid.sum() < 30 or values[valid].std() == 0:
            continue
        score = abs(float(values[valid].corr(target[valid])))
        if np.isfinite(score):
            ranked.append((score, name))
    ranked.sort(reverse=True)
    selected: list[str] = []
    for _, name in ranked:
        if len(selected) >= limit:
            break
        candidate = numeric[name].to_numpy(float)
        redundant = False
        for existing_name in selected:
            existing = numeric[existing_name].to_numpy(float)
            valid = np.isfinite(candidate) & np.isfinite(existing)
            if valid.sum() > 20 and abs(float(np.corrcoef(candidate[valid], existing[valid])[0, 1])) > 0.95:
                redundant = True
                break
        if not redundant:
            selected.append(name)
    return selected


def evaluate_predictions(predictions: Sequence[float], observations: Sequence[float], costs: float = 0.0003, initial_capital: float = 10_000.0, risk_fraction: float = 0.01) -> tuple[list[dict[str, float | bool]], float, float, float]:
    """Evaluate signed predictions with confidence-aware virtual sizing."""
    records: list[dict[str, float | bool]] = []
    capital = initial_capital
    peak = capital
    max_drawdown = 0.0
    calibration: list[float] = []
    for prediction, actual in zip(predictions, observations, strict=True):
        confidence = min(0.999, max(0.5, 0.5 + abs(float(prediction))))
        direction = 1.0 if prediction > 0 else -1.0 if prediction < 0 else 0.0
        stake = capital * risk_fraction * (confidence * 2.0 - 1.0) if direction else 0.0
        net_return = direction * float(actual) - costs if direction else 0.0
        pnl = stake * net_return
        capital += pnl
        peak = max(peak, capital)
        max_drawdown = max(max_drawdown, 1.0 - capital / peak)
        if direction:
            records.append({"confidence": confidence, "stake": stake, "pnl": pnl, "win": net_return > 0, "prediction": float(prediction), "actual": float(actual)})
            calibration.append(abs(float(actual) - float(prediction)))
    return records, capital - initial_capital, max_drawdown, float(np.mean(calibration) if calibration else 0.0)

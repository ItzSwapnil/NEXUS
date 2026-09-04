"""Bounded adaptive search over provider-generated features.

This module deliberately has no broker dependency. It searches a small,
guided candidate beam rather than enumerating the combinatorial feature space,
then evaluates every candidate using chronological walk-forward folds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResearchConfig:
    max_features: int = 32
    candidate_beam: int = 48
    min_samples: int = 120
    validation_folds: int = 3
    payout_percent: float = 80.0
    stake: float = 1.0


@dataclass
class CandidateResult:
    features: list[str]
    validation_accuracy: float
    baseline_accuracy: float
    trades: int
    net_profit: float
    profit_factor: float
    status: str


@dataclass
class ResearchResult:
    asset: str
    timeframe: int
    regime: str
    status: str
    candidates_tested: int
    selected_features: list[str] = field(default_factory=list)
    champion: CandidateResult | None = None
    candidates: list[CandidateResult] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


class AdaptiveResearchEngine:
    """Search and validate market-specific feature combinations locally."""

    def __init__(self, config: ResearchConfig | None = None):
        self.config = config or ResearchConfig()

    @staticmethod
    def _regime(frame: pd.DataFrame) -> str:
        returns = frame["close"].pct_change().dropna()
        volatility = float(returns.std()) if len(returns) else 0.0
        momentum = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
        if volatility > 0.02:
            return "VOLATILE"
        if momentum > 0.01:
            return "BULL"
        if momentum < -0.01:
            return "BEAR"
        return "SIDEWAYS"

    def _rank_features(self, frame: pd.DataFrame, target: pd.Series) -> list[str]:
        raw = {"open", "high", "low", "close", "volume", "timestamp", "time"}
        scores: list[tuple[float, str]] = []
        for name in frame.columns:
            if name in raw or not pd.api.types.is_numeric_dtype(frame[name]):
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            valid = values.replace([np.inf, -np.inf], np.nan).notna() & target.notna()
            if valid.sum() < 30 or float(values[valid].std()) == 0.0:
                continue
            correlation = abs(float(values[valid].corr(target[valid])))
            if np.isfinite(correlation):
                scores.append((correlation, name))
        scores.sort(reverse=True)

        # Remove highly redundant indicators. This keeps the beam diverse and
        # prevents a cluster of aliases from dominating every candidate.
        selected: list[str] = []
        for _, name in scores:
            if len(selected) >= self.config.max_features:
                break
            if not selected:
                selected.append(name)
                continue
            corr_values = []
            candidate = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            for selected_name in selected:
                existing = pd.to_numeric(frame[selected_name], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(existing) & np.isfinite(candidate)
                if valid.sum() < 3 or np.std(existing[valid]) == 0 or np.std(candidate[valid]) == 0:
                    continue
                corr_values.append(abs(float(np.corrcoef(existing[valid], candidate[valid])[0, 1])))
            corr = max(corr_values, default=0.0)
            if not np.isfinite(corr) or float(corr) < 0.97:
                selected.append(name)
        return selected

    def _candidate_sets(self, ranked: list[str]) -> list[list[str]]:
        if not ranked:
            return []
        candidates: list[list[str]] = [ranked[: min(4, len(ranked))]]
        for size in (6, 8, 12, 16):
            if len(ranked) >= size:
                candidates.append(ranked[:size])
        # Guided combinations: add features from successive rank bands rather
        # than testing every subset.
        for offset in range(min(12, len(ranked))):
            candidate = ranked[offset : offset + min(8, len(ranked))]
            if len(candidate) >= 3:
                candidates.append(candidate)
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique[: self.config.candidate_beam]

    @staticmethod
    def _labels(close: pd.Series) -> np.ndarray:
        forward = close.pct_change().shift(-1)
        observed = forward.dropna().abs()
        threshold = float(observed.median() * 0.5) if len(observed) else 0.001
        threshold = min(0.001, max(0.00001, threshold))
        labels = np.where(forward > threshold, 1, np.where(forward < -threshold, 2, 0))
        labels[-1] = -1  # the final candle has no known future outcome
        return labels

    def _evaluate(self, frame: pd.DataFrame, features: list[str], labels: np.ndarray) -> CandidateResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        values = frame[features].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
        x = values.to_numpy(dtype=np.float64)
        folds = max(2, self.config.validation_folds)
        fold_size = max(1, (len(x) - self.config.min_samples // 2) // folds)
        predictions: list[int] = []
        actual: list[int] = []
        for fold in range(folds):
            train_end = self.config.min_samples // 2 + fold * fold_size
            test_end = min(len(x), train_end + fold_size)
            if train_end >= test_end or len(np.unique(labels[:train_end])) < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.2),
            )
            model.fit(x[:train_end], labels[:train_end])
            predictions.extend(model.predict(x[train_end:test_end]).tolist())
            actual.extend(labels[train_end:test_end].tolist())
        if not actual:
            return CandidateResult(features, 0.0, 0.0, 0, 0.0, 0.0, "shadow")
        predicted = np.asarray(predictions)
        observed = np.asarray(actual)
        accuracy = float(np.mean(predicted == observed))
        baseline = float(np.max(np.bincount(observed)) / len(observed))
        directional = predicted != 0
        wins = ((predicted == 1) & (observed == 1)) | ((predicted == 2) & (observed == 2))
        losses = directional & ~wins
        profit = float(np.sum(np.where(wins, self.config.stake * self.config.payout_percent / 100.0, np.where(losses, -self.config.stake, 0.0))))
        gross_win = float(np.sum(wins) * self.config.stake * self.config.payout_percent / 100.0)
        gross_loss = float(np.sum(losses) * self.config.stake)
        status = "champion" if accuracy > baseline and profit > 0 else "shadow"
        return CandidateResult(
            features=features,
            validation_accuracy=round(accuracy, 4),
            baseline_accuracy=round(baseline, 4),
            trades=int(np.sum(directional)),
            net_profit=round(profit, 4),
            profit_factor=round(gross_win / gross_loss, 4) if gross_loss else 0.0,
            status=status,
        )

    def run(self, frame: pd.DataFrame, asset: str, timeframe: int = 60) -> ResearchResult:
        if len(frame) < self.config.min_samples:
            return ResearchResult(asset, timeframe, self._regime(frame), "insufficient_data", 0, reason=f"need at least {self.config.min_samples} candles")
        labels = self._labels(frame["close"])
        usable = frame.iloc[:-1].copy()
        usable_labels = labels[:-1]
        ranked = self._rank_features(usable, pd.Series(usable_labels, index=usable.index))
        candidates = self._candidate_sets(ranked)
        results = [self._evaluate(usable, candidate, usable_labels) for candidate in candidates]
        results.sort(key=lambda item: (item.status == "champion", item.net_profit, item.validation_accuracy), reverse=True)
        champion = results[0] if results and results[0].status == "champion" else None
        return ResearchResult(
            asset=asset,
            timeframe=timeframe,
            regime=self._regime(frame),
            status="champion" if champion else "shadow",
            candidates_tested=len(results),
            selected_features=champion.features if champion else (ranked[:8] if ranked else []),
            champion=champion,
            candidates=results[:10],
            reason="validated candidate selected" if champion else "no candidate beat the temporal baseline",
        )

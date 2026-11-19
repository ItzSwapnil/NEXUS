"""Bounded continual-learning orchestration for NEXUS.

The service observes frequently, settles observations only after their causal
horizon, and trains a time-ordered shadow policy candidate from replay. It
never mutates a champion or production model from one observation or trade.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import pandas as pd

from nexus.intelligence.adaptive import AdaptiveMarketIntelligence, Experience


@dataclass(frozen=True)
class PendingObservation:
    experience_id: str
    asset: str
    entry_timestamp: float
    entry_price: float
    action: str
    horizon_seconds: int


@dataclass(frozen=True)
class ContinualLearningStatus:
    running: bool
    ticks: int
    observations: int
    resolved_outcomes: int
    replay_reviews: int
    shadow_updates: int
    candidate_status: str
    candidate_id: str | None
    candidate_holdout_mse: float | None
    last_asset: str | None
    last_timestamp: float | None
    last_error: str | None


class ContinualLearningService:
    """One-second observation loop with causal settlement and shadow replay."""

    def __init__(self, intelligence: AdaptiveMarketIntelligence | None = None, replay_interval: int = 60, min_replay: int = 128) -> None:
        self.intelligence = intelligence or AdaptiveMarketIntelligence()
        self.replay_interval = max(1, replay_interval)
        self.min_replay = max(1, min_replay)
        self.ticks = 0
        self.observations = 0
        self.resolved_outcomes = 0
        self.replay_reviews = 0
        self.shadow_updates = 0
        self.candidate_status = "shadow_waiting_for_resolved_replay"
        self.candidate_id: str | None = None
        self.candidate_holdout_mse: float | None = None
        self.last_asset: str | None = None
        self.last_timestamp: float | None = None
        self.last_error: str | None = None
        self.last_indicators: list[str] = []
        self.last_regime: str | None = None
        self.last_action: str | None = None
        self.running = False
        self._last_signature: tuple[str, str] | None = None
        self._pending: dict[str, PendingObservation] = {}

    @staticmethod
    def _candle_timestamp(candles: pd.DataFrame) -> float:
        value = candles.iloc[-1].get("time", candles.iloc[-1].get("timestamp", len(candles) - 1))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(len(candles) - 1)

    def _settle(self, asset: str, candles: pd.DataFrame, current_timestamp: float) -> None:
        """Settle due observations using only prices available now."""
        current_price = float(candles.iloc[-1]["close"])
        for experience_id, pending in list(self._pending.items()):
            if pending.asset != asset or current_timestamp < pending.entry_timestamp + pending.horizon_seconds:
                continue
            signed_return = current_price / pending.entry_price - 1.0
            if pending.action == "short":
                signed_return = -signed_return
            outcome = {
                "status": "settled",
                "reward": float(signed_return),
                "win": bool(signed_return > 0.0),
                "exit_timestamp": current_timestamp,
                "exit_price": current_price,
            }
            if self.intelligence.memory.update_outcome(experience_id, outcome):
                self.resolved_outcomes += 1
            self._pending.pop(experience_id, None)

    @staticmethod
    def _replay_vector(experience: Experience) -> list[float]:
        state = experience.state or {}
        forecast = experience.forecast or {}
        probabilities = forecast.get("probabilities") or {}
        action_code = {"short": -1.0, "flat": 0.0, "long": 1.0}.get(experience.action, 0.0)
        return [
            float(forecast.get("expected_return", 0.0) or 0.0),
            float(forecast.get("volatility", 0.0) or 0.0),
            float(forecast.get("epistemic_uncertainty", 0.0) or 0.0),
            float(forecast.get("aleatoric_uncertainty", 0.0) or 0.0),
            float(state.get("novelty_score", 0.0) or 0.0),
            float(state.get("regime_probability", 0.0) or 0.0),
            float(probabilities.get("long", 0.0) or 0.0),
            float(probabilities.get("short", 0.0) or 0.0),
            action_code,
        ]

    def _train_shadow_candidate(self, asset: str | None = None) -> None:
        """Fit a replay-only ridge reward model and evaluate a time holdout."""
        replay = self.intelligence.memory.resolved_replay(limit=max(self.min_replay * 4, 512), asset=asset)
        if len(replay) < self.min_replay:
            self.candidate_status = f"shadow_replay_{len(replay)}/{self.min_replay}"
            return
        replay = sorted(replay, key=lambda item: item.timestamp)
        x = np.asarray([self._replay_vector(item) for item in replay], dtype=np.float64)
        y = np.asarray([float((item.outcome or {}).get("reward", 0.0) or 0.0) for item in replay], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0) < 1e-8, 1.0, x.std(axis=0))
        normalized = (x - mean) / scale
        split = max(1, int(len(normalized) * 0.8))
        train_x, test_x = normalized[:split], normalized[split:]
        train_y, test_y = y[:split], y[split:]
        design = np.c_[np.ones(len(train_x)), train_x]
        regularizer = np.eye(design.shape[1]) * 1e-3
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
        predictions = np.c_[np.ones(len(test_x)), test_x] @ coefficients if len(test_x) else np.empty(0)
        mse = float(np.mean((predictions - test_y) ** 2)) if len(test_y) else None
        baseline = float(np.mean((test_y - np.mean(train_y)) ** 2)) if len(test_y) else None
        candidate_payload = {
            "candidate_id": hashlib.sha256(f"{asset or 'all'}:{time.time_ns()}".encode()).hexdigest()[:16],
            "asset": asset,
            "rows": len(replay),
            "train_rows": len(train_x),
            "holdout_rows": len(test_x),
            "holdout_mse": mse,
            "baseline_holdout_mse": baseline,
            "features": ["expected_return", "volatility", "epistemic_uncertainty", "aleatoric_uncertainty", "novelty_score", "regime_probability", "p_long", "p_short", "action_code"],
            "trained_at": time.time(),
            "status": "shadow_only",
        }
        output = Path("models/adaptive")
        output.mkdir(parents=True, exist_ok=True)
        suffix = (asset or "all").replace("/", "_")
        (output / f"continual_candidate_{suffix}.json").write_text(json.dumps(candidate_payload, indent=2), encoding="utf-8")
        self.candidate_id = str(candidate_payload["candidate_id"])
        self.candidate_holdout_mse = mse
        self.shadow_updates += 1
        self.candidate_status = "shadow_candidate_trained" if mse is None or baseline is None or mse <= baseline else "shadow_candidate_rejected"

    async def tick(self, asset: str, fetch: Callable[[], Awaitable[pd.DataFrame]]) -> None:
        """Observe a market, settle due experiences, and queue new feedback."""
        self.ticks += 1
        self.last_asset = asset
        try:
            candles = await fetch()
            if candles.empty or len(candles) < 32:
                self.last_error = "insufficient candles"
                return
            current_timestamp = self._candle_timestamp(candles)
            self._settle(asset, candles, current_timestamp)
            signature = (asset, str(current_timestamp))
            if signature == self._last_signature:
                return
            self._last_signature = signature
            cycle = self.intelligence.evaluate_cycle(candles, asset)
            state = cycle["state"]
            forecast = cycle["forecast"]
            policy = cycle["policy"]
            risk = cycle["risk"]
            self.last_indicators = list(self.intelligence.representation._feature_names or ())[:12]
            self.last_regime = state.regime
            self.last_action = policy.action
            experience_id = f"observe-{asset}-{current_timestamp}"
            experience = Experience(
                experience_id=experience_id,
                timestamp=time.time(),
                asset=asset,
                state={"regime": state.regime, "regime_probability": state.regime_probability, "novelty_score": state.novelty_score, "is_ood": state.is_ood},
                action=policy.action,
                forecast={"expected_return": forecast.expected_return, "volatility": forecast.volatility, "epistemic_uncertainty": forecast.epistemic_uncertainty, "aleatoric_uncertainty": forecast.aleatoric_uncertainty, "probabilities": forecast.probabilities, "target_exposure": policy.target_exposure, "risk_allowed": risk.allowed},
                source="continuous_observation",
            )
            if policy.action in {"long", "short"} and risk.allowed:
                horizon_seconds = max(1, int(forecast.horizons[0] * 60))
                self._pending[experience_id] = PendingObservation(experience_id, asset, current_timestamp, float(candles.iloc[-1]["close"]), policy.action, horizon_seconds)
                outcome = {"status": "pending_outcome", "reward": 0.0, "entry_timestamp": current_timestamp, "entry_price": float(candles.iloc[-1]["close"]), "horizon_seconds": horizon_seconds}
            else:
                outcome = {"status": "no_trade", "reward": 0.0, "reason": risk.reason if not risk.allowed else "policy_flat"}
            self.intelligence.record_outcome(experience, outcome)
            self.observations += 1
            self.last_timestamp = time.time()
            self.last_error = None
            if self.ticks % self.replay_interval == 0:
                self.review_replay(asset)
        except Exception as exc:  # service must survive provider/broker hiccups
            self.last_error = str(exc)[:240]

    def review_replay(self, asset: str | None = None) -> None:
        """Run a shadow replay experiment; never mutate production weights."""
        self.replay_reviews += 1
        self._train_shadow_candidate(asset)

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self.running,
            "ticks": self.ticks,
            "observations": self.observations,
            "resolved_outcomes": self.resolved_outcomes,
            "pending_outcomes": len(self._pending),
            "replay_reviews": self.replay_reviews,
            "shadow_updates": self.shadow_updates,
            "candidate_status": self.candidate_status,
            "candidate_id": self.candidate_id,
            "candidate_holdout_mse": self.candidate_holdout_mse,
            "last_asset": self.last_asset,
            "last_timestamp": self.last_timestamp,
            "last_error": self.last_error,
            "last_indicators": self.last_indicators,
            "last_regime": self.last_regime,
            "last_action": self.last_action,
            "interval_seconds": 1,
            "production_model_mutated": False,
        }


__all__ = ["ContinualLearningService", "ContinualLearningStatus", "PendingObservation"]

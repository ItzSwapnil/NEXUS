"""Adaptive market-intelligence primitives.

This module is the intelligence spine for NEXUS.  Technical indicators are
inputs to the representation, never the decision rule.  The public objects
make the causal boundary explicit:

    observations -> representation -> state -> forecast -> policy -> risk
    -> execution -> outcome -> replay/research

No production model is updated by :meth:`record_outcome`; learning is an
explicit, replay-based operation owned by research/training jobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from nexus.features.feature_engine import FeatureRegistry

try:  # ML is optional so feature/state/risk services work on a small host.
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on minimal installations
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment,misc]
    HAS_TORCH = False


Action = Literal["short", "flat", "long"]


@dataclass(frozen=True)
class RepresentationConfig:
    lookback: int = 64
    embedding_dim: int = 64
    max_features: int = 256
    seed: int = 7


@dataclass(frozen=True)
class FeatureBundle:
    values: np.ndarray
    names: tuple[str, ...]
    schema_hash: str
    source_rows: int


@dataclass(frozen=True)
class MarketState:
    regime: str
    regime_probability: float
    novelty_score: float
    is_ood: bool
    embedding: np.ndarray
    feature_schema: str


@dataclass(frozen=True)
class ProbabilisticForecast:
    horizons: tuple[int, ...]
    quantiles: dict[str, tuple[float, ...]]
    expected_return: float
    volatility: float
    probabilities: dict[str, float]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    calibration_error: float


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    target_exposure: float
    expected_utility: float
    exploration: bool
    rationale: str


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    exposure: float
    reason: str
    hard_limits_applied: tuple[str, ...] = ()


@dataclass(frozen=True)
class Experience:
    experience_id: str
    timestamp: float
    asset: str
    state: dict[str, Any]
    action: str
    forecast: dict[str, Any]
    reward: float | None = None
    outcome: dict[str, Any] | None = None
    source: str = "paper"


@dataclass(frozen=True)
class ResearchResult:
    status: Literal["promote", "reject", "insufficient_data"]
    candidate_id: str
    champion_id: str | None
    walk_forward_splits: int
    out_of_sample_score: float
    calibration_error: float
    max_drawdown: float
    reason: str


def _schema_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


class MarketRepresentation:
    """Causal feature/representation builder with a modern sequence encoder."""

    def __init__(self, config: RepresentationConfig | None = None) -> None:
        self.config = config or RepresentationConfig()
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._feature_names: tuple[str, ...] | None = None
        self.encoder: Any | None = None
        if HAS_TORCH:
            torch.manual_seed(self.config.seed)

    def features(self, candles: pd.DataFrame) -> FeatureBundle:
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(candles.columns)
        if missing:
            raise ValueError(f"market data missing columns: {sorted(missing)}")
        frame = FeatureRegistry.compute(candles.copy())
        numeric = frame.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        # Forward-fill only: backfilling would copy future indicator values
        # into earlier timestamps and leak information into training/replay.
        numeric = numeric.ffill().fillna(0.0)
        numeric = numeric.iloc[:, : self.config.max_features]
        names = tuple(str(name) for name in numeric.columns)
        values = np.asarray(numeric.to_numpy(dtype=np.float64), dtype=np.float64)
        # Some indicator implementations emit extreme intermediate values on
        # short/ill-conditioned windows.  Keep the representation finite and
        # bounded before normalization; otherwise a single overflow poisons a
        # complete training run.
        values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
        values = np.clip(values, -1e6, 1e6).astype(np.float32)
        return FeatureBundle(values, names, _schema_hash(names), len(frame))

    def fit(self, bundle: FeatureBundle) -> None:
        """Fit normalization on training data only."""
        self._mean = bundle.values.mean(axis=0)
        scale = bundle.values.std(axis=0)
        self._scale = np.where(scale < 1e-8, 1.0, scale)
        self._feature_names = bundle.names

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise RuntimeError("representation normalizer is unfitted")
        if self._feature_names is not None and self._feature_names != bundle.names:
            positions = {name: index for index, name in enumerate(bundle.names)}
            aligned = np.zeros((len(bundle.values), len(self._feature_names)), dtype=np.float32)
            for target_index, name in enumerate(self._feature_names):
                source_index = positions.get(name)
                if source_index is not None:
                    aligned[:, target_index] = bundle.values[:, source_index]
            values = aligned
        else:
            values = bundle.values
        transformed = (values - self._mean) / self._scale
        return np.asarray(transformed, dtype=np.float32)

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        """Return a deterministic baseline embedding when no trained encoder exists."""
        if sequence.ndim != 2 or len(sequence) == 0:
            raise ValueError("sequence must have shape [time, features]")
        tail = sequence[-self.config.lookback :]
        if self.encoder is not None and HAS_TORCH:
            self.encoder.eval()
            with torch.no_grad():
                tensor = torch.as_tensor(tail, dtype=torch.float32).unsqueeze(0)
                return np.asarray(
                    self.encoder(tensor)["embedding"][0].cpu().numpy(), dtype=np.float32
                )
        summary = np.concatenate((tail[-1], tail.mean(axis=0), tail.std(axis=0)))
        result = np.zeros(self.config.embedding_dim, dtype=np.float32)
        result[: min(len(summary), len(result))] = summary[: len(result)]
        return result


if HAS_TORCH:

    class AdaptiveSequenceEncoder(nn.Module):
        """Transformer encoder with masked reconstruction and multi-task heads."""

        def __init__(self, input_dim: int, embedding_dim: int = 64, layers: int = 3) -> None:
            super().__init__()
            self.input = nn.Linear(input_dim, embedding_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 4,
                batch_first=True,
                norm_first=True,
                dropout=0.1,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.reconstruction = nn.Linear(embedding_dim, input_dim)
            self.return_head = nn.Linear(embedding_dim, 3)  # location, log scale, tail
            self.direction_head = nn.Linear(embedding_dim, 2)  # future down/up
            self.regime_head = nn.Linear(embedding_dim, 4)
            self.volatility_head = nn.Linear(embedding_dim, 1)

        def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            encoded = self.encoder(self.input(x))
            pooled = encoded[:, -1]
            return {
                "embedding": pooled,
                "reconstruction": self.reconstruction(encoded),
                "return": self.return_head(pooled),
                "direction": self.direction_head(pooled),
                "regime": self.regime_head(pooled),
                "volatility": nn.functional.softplus(self.volatility_head(pooled)),
            }

else:

    class AdaptiveSequenceEncoder:  # type: ignore[no-redef] # pragma: no cover - compatibility fallback
        """Compatibility fallback for installations without torch."""

        def __init__(self, input_dim: int, embedding_dim: int = 64, layers: int = 3) -> None:
            self.input_dim, self.embedding_dim, self.layers = input_dim, embedding_dim, layers

        def __call__(self, x: Any) -> dict[str, Any]:
            raise RuntimeError("AdaptiveSequenceEncoder requires the 'ai' extra (torch)")


class RegimeAndNoveltyDetector:
    """Online robust state detector; fitting is separate from observation."""

    REGIMES = ("trend_up", "trend_down", "range", "stress")

    def __init__(self, novelty_threshold: float = 4.0) -> None:
        self.novelty_threshold = novelty_threshold
        self._center: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        self._center = np.median(embeddings, axis=0)
        mad = np.median(np.abs(embeddings - self._center), axis=0)
        self._scale = np.where(mad < 1e-6, 1.0, 1.4826 * mad)

    def detect(self, embedding: np.ndarray, returns: Iterable[float]) -> MarketState:
        if self._center is None or self._scale is None:
            raise RuntimeError("regime detector is unfitted")
        z = np.abs((embedding - self._center) / self._scale)
        novelty = float(np.sqrt(np.mean(z * z)))
        values = np.asarray(list(returns), dtype=float)
        slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
        vol = float(np.std(values)) if len(values) else 0.0
        if vol > max(0.02, float(np.quantile(np.abs(values), 0.9)) if len(values) else 0.02):
            regime = "stress"
        elif slope > 0:
            regime = "trend_up"
        elif slope < 0:
            regime = "trend_down"
        else:
            regime = "range"
        return MarketState(
            regime, 1.0 / (1.0 + novelty), novelty, novelty >= self.novelty_threshold, embedding, ""
        )


class ForecastCalibrator:
    """Small conformal-style calibration layer fitted only on held-out data."""

    def __init__(self) -> None:
        self._error = 0.0

    def fit(self, predictions: np.ndarray, observations: np.ndarray) -> None:
        self._error = (
            float(np.quantile(np.abs(predictions - observations), 0.9)) if len(predictions) else 0.0
        )

    def apply(self, forecast: ProbabilisticForecast) -> ProbabilisticForecast:
        spread = self._error
        quantiles = {
            k: tuple(
                float(x + (spread if k == "0.9" else -spread if k == "0.1" else 0.0))
                for x in values
            )
            for k, values in forecast.quantiles.items()
        }
        return ProbabilisticForecast(
            **{**asdict(forecast), "quantiles": quantiles, "calibration_error": spread}
        )


class RiskPolicy:
    """Deterministic risk boundary. AI may propose exposure, never override limits."""

    def __init__(
        self, max_exposure: float = 1.0, max_drawdown: float = 0.2, max_novelty: float = 8.0
    ) -> None:
        self.max_exposure, self.max_drawdown, self.max_novelty = (
            max_exposure,
            max_drawdown,
            max_novelty,
        )

    def authorize(
        self, proposal: PolicyDecision, state: MarketState, drawdown: float
    ) -> RiskDecision:
        if drawdown >= self.max_drawdown:
            return RiskDecision(False, 0.0, "drawdown limit", ("max_drawdown",))
        if state.is_ood or state.novelty_score > self.max_novelty:
            return RiskDecision(False, 0.0, "novelty/OOD limit", ("novelty",))
        exposure = float(np.clip(proposal.target_exposure, -self.max_exposure, self.max_exposure))
        limits = ("exposure_clamp",) if exposure != proposal.target_exposure else ()
        return RiskDecision(True, exposure, "authorized", limits)


class ExperienceStore:
    """Durable experience/failure/pattern memory with explicit replay sampling."""

    def __init__(self, path: str | Path = "data/experience.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS experiences (id TEXT PRIMARY KEY, ts REAL, asset TEXT, payload TEXT, reward REAL, outcome TEXT)"
            )

    def append(self, experience: Experience) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO experiences VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experience.experience_id,
                    experience.timestamp,
                    experience.asset,
                    json.dumps(
                        asdict(experience),
                        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                    ),
                    experience.reward,
                    json.dumps(experience.outcome),
                ),
            )

    def replay(self, limit: int = 128, *, asset: str | None = None) -> list[Experience]:
        query = (
            "SELECT payload FROM experiences"
            + (" WHERE asset = ?" if asset else "")
            + " ORDER BY ts DESC LIMIT ?"
        )
        args: tuple[Any, ...] = (asset, limit) if asset else (limit,)
        with sqlite3.connect(self.path) as db:
            rows = db.execute(query, args).fetchall()
        return [Experience(**json.loads(row[0])) for row in rows]

    def update_outcome(self, experience_id: str, outcome: dict[str, Any]) -> bool:
        """Settle one observation without rewriting its original state/action."""
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT payload FROM experiences WHERE id = ?", (experience_id,)
            ).fetchone()
            if row is None:
                return False
            original = Experience(**json.loads(row[0]))
            reward = float(outcome.get("reward", 0.0) or 0.0)
            updated = Experience(**{**asdict(original), "outcome": outcome, "reward": reward})
            db.execute(
                "UPDATE experiences SET payload = ?, reward = ?, outcome = ? WHERE id = ?",
                (
                    json.dumps(
                        asdict(updated),
                        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
                    ),
                    reward,
                    json.dumps(outcome),
                    experience_id,
                ),
            )
        return True

    def resolved_replay(self, limit: int = 128, *, asset: str | None = None) -> list[Experience]:
        """Return only observations with causal outcome feedback."""
        candidates = self.replay(limit=max(limit * 3, limit), asset=asset)
        resolved = [
            item
            for item in candidates
            if item.outcome and item.outcome.get("status") not in {"pending_outcome", "unresolved"}
        ]
        return resolved[:limit]


class ResearchLoop:
    """Promotion gate for observe/diagnose/hypothesize/experiment/validate."""

    def __init__(self, min_splits: int = 3, minimum_score_delta: float = 0.02) -> None:
        self.min_splits, self.minimum_score_delta = min_splits, minimum_score_delta
        self.champion_id: str | None = None
        self.champion_score = -np.inf

    def compare(
        self,
        candidate_id: str,
        scores: Sequence[float],
        calibration_error: float,
        max_drawdown: float,
    ) -> ResearchResult:
        if len(scores) < self.min_splits:
            return ResearchResult(
                "insufficient_data",
                candidate_id,
                self.champion_id,
                len(scores),
                0.0,
                calibration_error,
                max_drawdown,
                "not enough walk-forward splits",
            )
        score = float(np.mean(scores))
        valid = np.isfinite(score) and calibration_error <= 0.15 and max_drawdown <= 0.2
        promote = valid and (
            self.champion_id is None or score >= self.champion_score + self.minimum_score_delta
        )
        if promote:
            self.champion_id, self.champion_score = candidate_id, score
        return ResearchResult(
            "promote" if promote else "reject",
            candidate_id,
            self.champion_id,
            len(scores),
            score,
            calibration_error,
            max_drawdown,
            "passed" if promote else "failed promotion gates",
        )


class WalkForwardEvaluator:
    """Leakage-resistant expanding-window evaluator for candidate policies."""

    def __init__(self, min_train: int = 128, test_size: int = 32, step: int | None = None) -> None:
        self.min_train, self.test_size, self.step = min_train, test_size, step or test_size

    def splits(self, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        result = []
        train_end = self.min_train
        while train_end + self.test_size <= n_rows:
            # The test interval starts strictly after the train interval.
            result.append(
                (np.arange(0, train_end), np.arange(train_end, train_end + self.test_size))
            )
            train_end += self.step
        return result


if HAS_TORCH:

    class AdaptiveTrainer:
        """Self-supervised + multi-task trainer used only by controlled jobs."""

        def __init__(self, model: AdaptiveSequenceEncoder, learning_rate: float = 3e-4) -> None:
            self.model = model
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-4
            )

        def loss(
            self,
            output: dict[str, Any],
            target: "torch.Tensor",
            returns: "torch.Tensor",
            regimes: "torch.Tensor",
            volatility: "torch.Tensor",
            directions: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            # Masked reconstruction forces representation learning from raw
            # sequences; auxiliary heads learn returns, state, and volatility.
            mask = torch.rand(target.shape[:2], device=target.device) < 0.15
            reconstruction = output["reconstruction"][mask]
            target_masked = target[mask]
            recon_loss = (
                nn.functional.smooth_l1_loss(reconstruction, target_masked)
                if len(reconstruction)
                else target.sum() * 0.0
            )
            ret = output["return"]
            return_loss = nn.functional.smooth_l1_loss(ret[:, 0], returns)
            scale_loss = nn.functional.smooth_l1_loss(
                torch.exp(ret[:, 1]).clamp(max=10), returns.abs()
            )
            regime_loss = nn.functional.cross_entropy(output["regime"], regimes)
            vol_loss = nn.functional.smooth_l1_loss(output["volatility"].squeeze(-1), volatility)
            direction_loss = (
                nn.functional.cross_entropy(output["direction"], directions)
                if directions is not None
                else target.sum() * 0.0
            )
            return (
                recon_loss
                + return_loss
                + 0.5 * scale_loss
                + direction_loss
                + regime_loss
                + 0.5 * vol_loss
            )

        def train_batch(
            self,
            sequences: "torch.Tensor",
            returns: "torch.Tensor",
            regimes: "torch.Tensor",
            volatility: "torch.Tensor",
            directions: "torch.Tensor | None" = None,
        ) -> float:
            self.model.train()
            output = self.model(sequences)
            value = self.loss(output, sequences, returns, regimes, volatility, directions)
            self.optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return float(value.detach().cpu())

else:

    class AdaptiveTrainer:  # type: ignore[no-redef] # pragma: no cover - compatibility fallback
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("AdaptiveTrainer requires the 'ai' extra (torch)")


class AdaptiveMarketIntelligence:
    """Single pipeline facade shared by backtest, paper, and live callers."""

    def __init__(self, memory_path: str | Path = "data/experience.sqlite3") -> None:
        self.representation = MarketRepresentation()
        self.regimes = RegimeAndNoveltyDetector()
        self.calibrator = ForecastCalibrator()
        self.risk = RiskPolicy()
        self.memory = ExperienceStore(memory_path)
        self.research = ResearchLoop()

        # Automatically bind pretrained universal indicator foundation encoder if present
        default_encoder = Path("models/adaptive/universal_indicator_encoder.pt")
        if default_encoder.is_file() and HAS_TORCH:
            try:
                state = torch.load(default_encoder, map_location="cpu", weights_only=True)
                input_dim = state.get("input_dim", 256)
                cfg = state.get("config", {})
                emb_dim = cfg.get("embedding_dim", 128)
                self.representation.config = RepresentationConfig(
                    lookback=cfg.get("lookback", 64),
                    embedding_dim=emb_dim,
                    max_features=cfg.get("max_features", 256),
                )
                model = AdaptiveSequenceEncoder(input_dim=input_dim, embedding_dim=emb_dim)
                model.load_state_dict(state["model"])
                self.representation.encoder = model
            except Exception:
                pass

    def load_encoder(self, checkpoint: str | Path, input_dim: int) -> None:
        """Load a previously trained encoder; weights are never changed here."""
        if not HAS_TORCH:
            raise RuntimeError("loading an adaptive encoder requires torch")
        model = AdaptiveSequenceEncoder(
            input_dim=input_dim, embedding_dim=self.representation.config.embedding_dim
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(
            state["model"] if isinstance(state, dict) and "model" in state else state
        )
        self.representation.encoder = model

    def observe(self, candles: pd.DataFrame, asset: str) -> tuple[FeatureBundle, MarketState]:
        bundle = self.representation.features(candles)
        if self.representation._mean is None:
            self.representation.fit(bundle)
        sequence = self.representation.transform(bundle)
        embedding = self.representation.encode(sequence)
        if self.regimes._center is None:
            self.regimes.fit(np.vstack([embedding, embedding + 1e-3]))
        state = self.regimes.detect(embedding, candles["close"].pct_change().dropna().to_numpy())
        return bundle, MarketState(
            state.regime,
            state.regime_probability,
            state.novelty_score,
            state.is_ood,
            state.embedding,
            bundle.schema_hash,
        )

    def forecast(
        self, candles: pd.DataFrame, state: MarketState | None = None
    ) -> ProbabilisticForecast:
        """Produce a probabilistic forecast from the loaded encoder or fallback.

        A loaded checkpoint is read-only here.  Its location, scale, and
        regime head drive the shared forecast contract; the empirical return
        distribution remains an explicit fallback when no candidate has been
        promoted.
        """
        if self.representation.encoder is not None and HAS_TORCH:
            bundle = self.representation.features(candles)
            if self.representation._mean is None:
                self.representation.fit(bundle)
            sequence = self.representation.transform(bundle)[-self.representation.config.lookback :]
            if len(sequence) >= self.representation.config.lookback:
                model = self.representation.encoder
                model.eval()
                device = next(model.parameters()).device
                with torch.no_grad():
                    output = model(
                        torch.as_tensor(sequence, dtype=torch.float32, device=device).unsqueeze(0)
                    )
                    location = float(output["return"][0, 0].cpu())
                    scale = float(torch.exp(output["return"][0, 1]).clamp(min=1e-7, max=0.1).cpu())
                    direction_probability = float(
                        torch.softmax(output["direction"], dim=-1)[0, 1].cpu()
                    )
                    regime_probability = float(torch.softmax(output["regime"], dim=-1).max().cpu())
                    predicted_volatility = float(output["volatility"][0, 0].cpu())
                horizons = (1, 3, 6)
                probabilities_long = direction_probability
                probabilities_long = float(np.clip(probabilities_long, 0.01, 0.99))
                quantiles = {
                    str(q): tuple(
                        float(location * horizon + z * scale * math.sqrt(horizon))
                        for horizon in horizons
                    )
                    for q, z in ((0.1, -1.2815516), (0.5, 0.0), (0.9, 1.2815516))
                }
                return ProbabilisticForecast(
                    horizons=horizons,
                    quantiles=quantiles,
                    expected_return=location,
                    volatility=max(scale, predicted_volatility, 1e-8),
                    probabilities={
                        "long": probabilities_long,
                        "flat": 0.0,
                        "short": 1.0 - probabilities_long,
                    },
                    epistemic_uncertainty=float(
                        state.novelty_score if state else 1.0 - regime_probability
                    ),
                    aleatoric_uncertainty=max(scale, predicted_volatility, 1e-8),
                    calibration_error=self.calibrator._error,
                )
        returns = candles["close"].pct_change().dropna().to_numpy(dtype=float)
        if len(returns) == 0:
            returns = np.zeros(1, dtype=float)
        recent = returns[-min(32, len(returns)) :]
        location = float(np.mean(recent))
        scale = float(np.std(recent))
        quantiles = {str(q): (float(np.quantile(recent, q)),) * 3 for q in (0.1, 0.5, 0.9)}
        # Convert standardized directional edge into an explicit probability.
        # This is still a forecast, not a guarantee; calibration_error is
        # reported separately and must be fitted on held-out outcomes.
        edge_z = abs(location) / max(scale, 1e-8)
        directional_confidence = min(0.99, 0.5 * (1.0 + math.erf(edge_z / math.sqrt(2.0))))
        probabilities = np.asarray(
            [
                directional_confidence if location > 0 else 1.0 - directional_confidence,
                0.0,
                directional_confidence if location < 0 else 1.0 - directional_confidence,
            ]
        )
        probabilities /= probabilities.sum() or 1.0
        return ProbabilisticForecast(
            horizons=(1, 3, 6),
            quantiles=quantiles,
            expected_return=location,
            volatility=max(scale, 1e-8),
            probabilities={
                "long": float(probabilities[0]),
                "flat": float(probabilities[1]),
                "short": float(probabilities[2]),
            },
            epistemic_uncertainty=float(state.novelty_score if state else 1.0),
            aleatoric_uncertainty=scale,
            calibration_error=self.calibrator._error,
        )

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    @staticmethod
    def decide(
        forecast: ProbabilisticForecast, *, cost: float = 0.0, exploration: bool = False
    ) -> PolicyDecision:
        """Choose exposure using expected utility, variance, and trading costs."""
        edge = forecast.expected_return - cost
        risk = max(forecast.volatility, 1e-8) ** 2 + forecast.aleatoric_uncertainty**2
        utility = edge / risk**0.5
        exposure = float(np.clip(utility, -1.0, 1.0))
        action: Action = "long" if exposure > 0.1 else "short" if exposure < -0.1 else "flat"
        return PolicyDecision(
            action, exposure, float(utility), exploration, "risk-adjusted expected utility"
        )

    def evaluate_cycle(
        self, candles: pd.DataFrame, asset: str, drawdown: float = 0.0
    ) -> dict[str, Any]:
        """Run the same causal pipeline used by all execution modes."""
        _, state = self.observe(candles, asset)
        forecast = self.calibrator.apply(self.forecast(candles, state))
        proposal = self.decide(forecast)
        risk = self.risk.authorize(proposal, state, drawdown)
        return {"state": state, "forecast": forecast, "policy": proposal, "risk": risk}

    def record_outcome(self, experience: Experience, outcome: dict[str, Any]) -> None:
        """Persist feedback only; training requires a controlled replay job."""
        self.memory.append(
            Experience(
                **{
                    **asdict(experience),
                    "outcome": outcome,
                    "reward": float(outcome.get("reward", 0.0)),
                }
            )
        )


class AdaptiveBacktestRunner:
    """Market-replay runner using the exact intelligence/risk contracts."""

    def __init__(
        self,
        intelligence: AdaptiveMarketIntelligence | None = None,
        cost_bps: float = 2.0,
        slippage_bps: float = 1.0,
    ) -> None:
        self.intelligence = intelligence or AdaptiveMarketIntelligence(":memory:")
        self.cost_bps, self.slippage_bps = cost_bps, slippage_bps

    def run(self, candles: pd.DataFrame, asset: str, start_at: int = 2) -> dict[str, Any]:
        if len(candles) < 3:
            return {"trades": 0, "net_return": 0.0, "max_drawdown": 0.0, "calibration_error": 0.0}
        # Compute provider features once. Recomputing every expanding window
        # is both unnecessarily expensive and can make a long replay appear
        # hung when pandas-ta executes its full catalog.
        bundle = self.intelligence.representation.features(candles)
        if self.intelligence.representation._mean is None:
            self.intelligence.representation.fit(bundle)
        transformed = self.intelligence.representation.transform(bundle)
        equity = [1.0]
        virtual_equity = 10_000.0
        peak_virtual_equity = virtual_equity
        calibration: list[float] = []
        trade_records: list[dict[str, Any]] = []
        winning_trades = 0
        trades = 0
        for end in range(max(2, start_at), len(candles) - 1):
            embedding = self.intelligence.representation.encode(transformed[: end + 1])
            if self.intelligence.regimes._center is None:
                self.intelligence.regimes.fit(np.vstack([embedding, embedding + 1e-3]))
            state = self.intelligence.regimes.detect(
                embedding, candles["close"].iloc[: end + 1].pct_change().dropna()
            )
            forecast = self.intelligence.forecast(candles.iloc[: end + 1], state)
            policy = self.intelligence.decide(forecast)
            risk = self.intelligence.risk.authorize(policy, state, max(0.0, 1.0 - equity[-1]))
            horizon = 1 if forecast.volatility > 0.003 else 3 if forecast.volatility > 0.001 else 5
            exit_index = min(end + horizon, len(candles) - 1)
            next_return = float(
                candles.iloc[exit_index]["close"] / candles.iloc[end]["close"] - 1.0
            )
            turnover_cost = abs(risk.exposure) * (self.cost_bps + self.slippage_bps) / 10_000.0
            pnl = risk.exposure * next_return - turnover_cost if risk.allowed else 0.0
            equity.append(equity[-1] * (1.0 + pnl))
            is_trade = bool(risk.allowed and abs(risk.exposure) > 0.1)
            trades += int(is_trade)
            confidence = float(max(forecast.probabilities.values()))
            win = bool(is_trade and risk.exposure * next_return > 0.0)
            winning_trades += int(win)
            stake_amount = (
                virtual_equity * min(0.05, 0.05 * abs(risk.exposure)) if is_trade else 0.0
            )
            money_return = (
                (np.sign(risk.exposure) * next_return - turnover_cost) if is_trade else 0.0
            )
            money_pnl = stake_amount * money_return
            virtual_equity += money_pnl
            peak_virtual_equity = max(peak_virtual_equity, virtual_equity)
            if is_trade:
                trade_records.append(
                    {
                        "index": end,
                        "asset": asset,
                        "confidence": confidence,
                        "action": policy.action,
                        "exposure": risk.exposure,
                        "trade_time_seconds": horizon * 60,
                        "virtual_stake_fraction": abs(risk.exposure),
                        "virtual_stake_amount": stake_amount,
                        "expected_return": forecast.expected_return,
                        "realized_return": next_return,
                        "cost_return": turnover_cost,
                        "money_pnl": money_pnl,
                        "equity_after": virtual_equity,
                        "net_return": pnl,
                        "win": win,
                        "regime": state.regime,
                    }
                )
            calibration.append(abs(next_return - forecast.expected_return))
        curve = np.asarray(equity)
        drawdown = float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
        return {
            "trades": trades,
            "winning_trades": winning_trades,
            "losing_trades": trades - winning_trades,
            "win_rate": winning_trades / trades if trades else 0.0,
            "starting_virtual_capital": 10_000.0,
            "final_virtual_capital": virtual_equity,
            "virtual_pnl": virtual_equity - 10_000.0,
            "virtual_max_drawdown": 1.0
            - (min([virtual_equity, peak_virtual_equity]) / peak_virtual_equity),
            "net_return": float(curve[-1] - 1.0),
            "max_drawdown": drawdown,
            "calibration_error": float(np.mean(calibration) if calibration else 0.0),
            "cost_bps": self.cost_bps,
            "slippage_bps": self.slippage_bps,
            "trade_records": trade_records,
        }


__all__ = [
    "AdaptiveBacktestRunner",
    "AdaptiveMarketIntelligence",
    "AdaptiveSequenceEncoder",
    "AdaptiveTrainer",
    "Experience",
    "ExperienceStore",
    "FeatureBundle",
    "ForecastCalibrator",
    "MarketRepresentation",
    "MarketState",
    "PolicyDecision",
    "ProbabilisticForecast",
    "ResearchLoop",
    "ResearchResult",
    "RiskDecision",
    "RiskPolicy",
    "RegimeAndNoveltyDetector",
    "WalkForwardEvaluator",
]

"""Research-only multi-market walk-forward evaluation; never submits orders."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from nexus.core.engine import NexusEngine
from nexus.features.feature_engine import (
    get_feature_provider_catalog,
    get_synthesized_indicator_catalog,
)
from nexus.intelligence.adaptive import (
    HAS_TORCH,
    AdaptiveSequenceEncoder,
    AdaptiveTrainer,
    MarketRepresentation,
    RepresentationConfig,
)
from nexus.intelligence.robust_research import PurgedWalkForward, WalkForwardWindow
from nexus.utils.config import load_runtime_settings

if HAS_TORCH:
    import torch


def _seed_torch(seed: int) -> None:
    """Make candidate training as reproducible as the installed backend allows."""
    if not HAS_TORCH:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


async def _refresh_broker_session(engine: NexusEngine) -> bool:
    """Close stale pyquotex state before reconnecting the authenticated feed."""
    broker = getattr(engine, "_broker", None)
    real_broker = getattr(broker, "_real", broker)
    close = getattr(real_broker, "close", None)
    if close is not None:
        try:
            await close()
        except Exception:
            pass
    else:
        real_broker.authenticated = False
    return await engine.login_broker()


async def _refresh_login_browser(root: Path) -> bool:
    """Refresh the broker session once when credentials/session auth fails."""
    if os.getenv("NEXUS_AUTO_BROWSER_REFRESH", "1").lower() in {"0", "false", "no"}:
        return False
    process = await asyncio.create_subprocess_exec(
        "uv", "run", "python", str(root / "scripts" / "login_browser.py"),
        cwd=root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    await process.communicate()
    return process.returncode == 0


def _regime(frame: pd.DataFrame, index: int) -> str:
    # State at t may only use candles strictly before t.  In particular, do
    # not take a median over the full frame: that leaked future volatility
    # into every historical regime label in the earlier evaluator.
    all_returns = frame["close"].pct_change()
    returns = all_returns.iloc[max(0, index - 48):index].dropna()
    if len(returns) < 10:
        return "unknown"
    drift, volatility = float(returns.mean()), float(returns.std())
    direction = "bull" if drift > volatility * 0.08 else "bear" if drift < -volatility * 0.08 else "sideways"
    history = all_returns.iloc[:index].dropna()
    rolling_volatility = history.rolling(240, min_periods=20).std().dropna()
    long_vol = float(rolling_volatility.median()) if len(rolling_volatility) else volatility
    level = "high_vol" if long_vol and volatility > long_vol * 1.5 else "normal_vol"
    return f"{direction}_{level}"


def _chronological(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize broker output; Quotex may return newest candles first."""
    result = frame.copy()
    time_column = "time" if "time" in result.columns else "timestamp" if "timestamp" in result.columns else None
    if time_column:
        result[time_column] = pd.to_numeric(result[time_column], errors="coerce")
        result = result.dropna(subset=[time_column]).sort_values(time_column)
    return result.drop_duplicates(subset=[time_column] if time_column else None).reset_index(drop=True)


def _observed_timeframe_seconds(frame: pd.DataFrame, fallback: int) -> int:
    time_column = "time" if "time" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    if not time_column:
        return fallback
    spacing = pd.to_numeric(frame[time_column], errors="coerce").sort_values().diff().dropna()
    if spacing.empty:
        return fallback
    observed = float(spacing.median())
    return max(1, int(round(observed))) if np.isfinite(observed) and observed > 0 else fallback


def _candidate_horizons(timeframe_seconds: int, asset: str) -> list[int]:
    if "otc" in asset.lower() and timeframe_seconds <= 10:
        # 5-second data supports the requested sub-minute precision.
        return [1, 25, 29, 41, 56, 60]  # 5s, 2m05s, 2m25s, 3m25s, 4m40s, 5m
    # Non-OTC decisions remain aligned to one-minute candle boundaries.
    return [1, 2, 3, 5, 7]


def _verified_history_limit(requested: int) -> int:
    """Keep broker history requests inside the verified stable range."""
    return min(3200, max(3000, int(requested)))


def _walk_forward_windows(
    observations: int, requested_folds: int | None = None
) -> list[WalkForwardWindow]:
    """Use a bounded set of expanding folds as authentic history grows."""
    requested_folds = max(
        2,
        min(6, requested_folds if requested_folds is not None else int(os.getenv("NEXUS_WALK_FORWARD_FOLDS", "3"))),
    )
    if observations <= 5000:
        windows = PurgedWalkForward(
            train_size=1200, validation_size=400, test_size=500, embargo=10, step=500
        ).windows(observations)
        return windows[-requested_folds:]
    test_size = max(500, int(observations * 0.08))
    validation_size = max(400, int(observations * 0.08))
    train_size = max(1200, int(observations * 0.30))
    windows = PurgedWalkForward(
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        embargo=10,
        step=test_size,
    ).windows(observations)
    return windows[-requested_folds:]


def _candidate_timeframes(source_seconds: int, asset: str) -> list[int]:
    """Return decision frames supported by the observed candle resolution."""
    if source_seconds <= 10 and "otc" in asset.lower():
        return [source_seconds, 60, 300, 900]
    if source_seconds <= 60:
        return [60, 300, 900]
    return [source_seconds]


def _choose_timeframes(
    frame: pd.DataFrame,
    validation_start: int,
    validation_end: int,
    asset: str,
    source_seconds: int,
    *,
    direction_by_index: dict[int, str] | None = None,
    payout_percent: float = 0.0,
) -> tuple[int, dict[str, int]]:
    """Choose analysis frames using validation-only model directions/outcomes.

    Coarser frames are represented by completed groups of source candles; this
    function never creates a finer timestamp than the source feed provides.
    """
    candidates = _candidate_timeframes(source_seconds, asset)
    close = frame["close"].to_numpy(float)
    payout_rate = payout_percent / 100.0
    scores: dict[str, dict[int, list[float]]] = {}
    overall: dict[int, list[float]] = {candidate: [] for candidate in candidates}
    for index in range(validation_start, validation_end):
        regime = _regime(frame, index)
        scores.setdefault(regime, {candidate: [] for candidate in candidates})
        direction = (direction_by_index or {}).get(index)
        for candidate in candidates:
            bars = max(1, int(round(candidate / source_seconds)))
            exit_index = index + bars
            if exit_index >= len(close):
                continue
            move = close[exit_index] / close[index] - 1.0
            if direction == "short":
                move = -move
            if direction in {"long", "short"} and payout_rate > 0:
                score = payout_rate if move > 0 else -1.0
            else:
                score = move
            scores[regime][candidate].append(float(score))
            overall[candidate].append(float(score))
    default = max(candidates, key=lambda candidate: float(np.mean(overall[candidate])) if overall[candidate] else -np.inf)
    selected: dict[str, int] = {}
    for regime, values in scores.items():
        selected[regime] = max(
            candidates,
            key=lambda candidate: float(np.mean(values[candidate])) if len(values[candidate]) >= 20 else float(np.mean(overall[candidate])) if overall[candidate] else -np.inf,
        )
    return default, selected


def _choose_horizons(
    frame: pd.DataFrame,
    validation_start: int,
    validation_end: int,
    asset: str,
    timeframe_seconds: int,
    *,
    direction_by_index: dict[int, str] | None = None,
    payout_percent: float = 0.0,
) -> tuple[int, dict[str, int]]:
    candidates = _candidate_horizons(timeframe_seconds, asset)
    scores: dict[str, dict[int, list[float]]] = {}
    overall: dict[int, list[float]] = {h: [] for h in candidates}
    close = frame["close"].to_numpy(float)
    payout_rate = payout_percent / 100.0
    for index in range(validation_start, validation_end):
        regime = _regime(frame, index)
        scores.setdefault(regime, {h: [] for h in candidates})
        for horizon in candidates:
            exit_index = index + horizon
            if exit_index >= len(close):
                continue
            direction = (direction_by_index or {}).get(index)
            if direction in {"long", "short"} and payout_rate > 0:
                signed_return = close[exit_index] / close[index] - 1.0
                if direction == "short":
                    signed_return = -signed_return
                result = payout_rate if signed_return > 0 else -1.0
                result -= 0.00025
            else:
                result = (close[exit_index] / close[index] - 1.0) - 0.00025
            scores[regime][horizon].append(float(result))
            overall[horizon].append(float(result))
    default = max(candidates, key=lambda h: float(np.mean(overall[h])) if overall[h] else -np.inf)
    selected: dict[str, int] = {}
    for regime, values in scores.items():
        selected[regime] = max(
            candidates,
            key=lambda h: float(np.mean(values[h])) if len(values[h]) >= 20 else float(np.mean(overall[h])) if overall[h] else -np.inf,
        )
    return default, selected


def _calibration_map(probability: np.ndarray, actual: np.ndarray) -> dict[int, float]:
    """Calibrate P(long) with shrinkage and bounded probabilities.

    A sparse validation bucket must never become an apparent 0%/100% oracle.
    The old fallback could make a wrong short forecast appear as 100%
    confidence after taking ``1 - probability``.
    """
    labels = np.digitize(probability, [0.55, 0.65, 0.75, 0.85, 0.95])
    prior = 0.5
    strength = 20.0
    result: dict[int, float] = {}
    for bucket in range(6):
        observed = actual[labels == bucket]
        # Empirical-Bayes shrinkage prevents tiny samples from becoming
        # certainty, while retaining information from well-populated bins.
        value = (float(observed.sum()) + strength * prior) / (len(observed) + strength)
        result[bucket] = float(np.clip(value, 0.01, 0.99))
    return result


def _expected_calibration_error(records: list[dict[str, object]]) -> float:
    """Return reliability error, not per-trade classification error.

    Mean absolute error between a binary outcome and its probability is about
    0.5 even for a perfectly calibrated 50% forecast.  The promotion gate
    needs expected calibration error: the gap between predicted probability
    and observed win rate inside out-of-sample confidence buckets.
    """
    buckets: list[list[tuple[float, float]]] = [[] for _ in range(5)]
    for record in records:
        try:
            confidence = float(record["confidence"])
            win = float(bool(record["win"]))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(confidence) and 0.0 <= confidence <= 1.0:
            buckets[min(4, max(0, int((confidence - 0.5) / 0.1)))].append((confidence, win))
    total = sum(len(bucket) for bucket in buckets)
    if not total:
        return 1.0
    return float(sum(
        len(bucket) / total * abs(
            float(np.mean([confidence for confidence, _ in bucket]))
            - float(np.mean([win for _, win in bucket]))
        )
        for bucket in buckets if bucket
    ))


class _ConservativeIsotonicCalibrator:
    """Monotonic calibration with shrinkage and no optimistic extrapolation."""

    def __init__(self, prior_strength: float = 40.0) -> None:
        self.prior_strength = prior_strength
        self._model: IsotonicRegression | None = None
        self._constant = 0.5
        self._minimum = 0.5
        self._maximum = 0.5

    def fit(self, scores: np.ndarray, wins: np.ndarray) -> "_ConservativeIsotonicCalibrator":
        scores = np.asarray(scores, dtype=float)
        wins = np.asarray(wins, dtype=float)
        valid = np.isfinite(scores) & np.isfinite(wins)
        scores, wins = scores[valid], wins[valid]
        if not len(scores):
            return self
        order = np.argsort(scores, kind="mergesort")
        scores, wins = scores[order], wins[order]
        groups = np.array_split(np.arange(len(scores)), min(8, len(scores)))
        x, y, weights = [], [], []
        for group in groups:
            if len(group) == 0:
                continue
            count = len(group)
            x.append(float(np.mean(scores[group])))
            y.append(float((np.sum(wins[group]) + self.prior_strength * 0.5) / (count + self.prior_strength)))
            weights.append(float(count + self.prior_strength))
        self._constant = float(np.average(y, weights=weights)) if y else 0.5
        self._minimum = float(min(y, default=0.5))
        self._maximum = float(max(y, default=0.5))
        if len(x) >= 2 and len(set(x)) >= 2:
            self._model = IsotonicRegression(y_min=0.5, y_max=0.99, out_of_bounds="clip")
            self._model.fit(x, y, sample_weight=weights)
        return self

    def predict(self, values: list[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if self._model is None:
            result = np.full(values.shape, self._constant, dtype=float)
        else:
            result = np.asarray(self._model.predict(values), dtype=float)
        # Out-of-range raw scores are clipped to the evidence-supported range;
        # they never become a 99% claim merely because they exceed validation.
        return np.clip(result, max(0.5, self._minimum), min(0.99, self._maximum))


def _prepare_representation(frame: pd.DataFrame, train_end: int) -> tuple[MarketRepresentation, np.ndarray, tuple[str, ...]]:
    """Build and normalize the sequence input using the current fold only."""
    representation = MarketRepresentation(
        RepresentationConfig(lookback=64, embedding_dim=64, max_features=256, seed=7)
    )
    bundle = representation.features(frame)
    train_bundle = type(bundle)(bundle.values[:train_end], bundle.names, bundle.schema_hash, train_end)
    representation.fit(train_bundle)
    return representation, representation.transform(bundle), bundle.names


def _regime_label(frame: pd.DataFrame, index: int) -> int:
    name = _regime(frame, index)
    return {"bull_normal_vol": 0, "bull_high_vol": 0, "bear_normal_vol": 1,
            "bear_high_vol": 1, "sideways_normal_vol": 2,
            "sideways_high_vol": 3, "unknown": 2}.get(name, 2)


def _sequence_rows(
    values: np.ndarray,
    frame: pd.DataFrame,
    starts: range | list[int] | np.ndarray,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create causal sequence examples whose labels are strictly in the future."""
    close = frame["close"].to_numpy(float)
    lookback = 64
    sequences: list[np.ndarray] = []
    returns: list[float] = []
    regimes: list[int] = []
    volatility: list[float] = []
    indices: list[int] = []
    for end in starts:
        end = int(end)
        if end < lookback or end + horizon >= len(close):
            continue
        sequence = values[end - lookback:end]
        if not np.isfinite(sequence).all():
            continue
        sequence_returns = pd.Series(close).pct_change().iloc[max(1, end - 32):end].dropna().to_numpy(float)
        sequences.append(sequence)
        returns.append(float(close[end + horizon] / close[end] - 1.0))
        regimes.append(_regime_label(frame, end))
        volatility.append(float(np.std(sequence_returns)) if len(sequence_returns) else 1e-6)
        indices.append(end)
    if not sequences:
        return (np.empty((0, lookback, values.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64))
    return (np.asarray(sequences, dtype=np.float32), np.asarray(returns, dtype=np.float32),
            np.asarray(regimes, dtype=np.int64), np.asarray(volatility, dtype=np.float32),
            np.asarray(indices, dtype=np.int64))


def _train_sequence_model(
    values: np.ndarray,
    frame: pd.DataFrame,
    train_start: int,
    train_end: int,
    seed: int,
) -> tuple[object | None, dict[str, float]]:
    """Train the representation/world-model candidate inside one fold."""
    if not HAS_TORCH:
        return None, {"status": "torch_unavailable"}
    training_indices = np.arange(train_start, max(train_start, train_end - 1), dtype=int)
    max_training_rows = max(256, int(os.getenv("NEXUS_SEQUENCE_TRAIN_ROWS", "6000")))
    if len(training_indices) > max_training_rows:
        # Preserve the full chronological span so older and newer regimes
        # both contribute without changing the causal boundary.
        training_indices = np.unique(
            np.linspace(training_indices[0], training_indices[-1], max_training_rows, dtype=int)
        )
    train_data = _sequence_rows(values, frame, training_indices)
    sequences, returns, regimes, volatility, _ = train_data
    if len(sequences) < 256:
        return None, {"status": "insufficient_sequence_rows", "rows": float(len(sequences))}
    _seed_torch(seed)
    np.random.seed(seed)
    requested_device = os.getenv("NEXUS_SEQUENCE_DEVICE", "cpu").lower()
    device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    model = AdaptiveSequenceEncoder(sequences.shape[-1], embedding_dim=64).to(device)
    trainer = AdaptiveTrainer(model)
    x = torch.as_tensor(sequences, dtype=torch.float32, device=device)
    y = torch.as_tensor(returns, dtype=torch.float32, device=device)
    regime_y = torch.as_tensor(regimes, dtype=torch.long, device=device)
    vol_y = torch.as_tensor(volatility, dtype=torch.float32, device=device)
    direction_y = torch.as_tensor((returns > 0.0).astype(np.int64), dtype=torch.long, device=device)
    losses: list[float] = []
    epochs = max(1, int(os.getenv("NEXUS_SEQUENCE_EPOCHS", "3")))
    batch_size = max(64, int(os.getenv("NEXUS_SEQUENCE_BATCH_SIZE", "256")))
    for _ in range(epochs):
        epoch_losses: list[float] = []
        # Chronological order is retained; shuffling could mix observations
        # across regimes and makes replay diagnostics less reproducible.
        for start in range(0, len(x), batch_size):
            epoch_losses.append(
                trainer.train_batch(
                    x[start:start + batch_size],
                    y[start:start + batch_size],
                    regime_y[start:start + batch_size],
                    vol_y[start:start + batch_size],
                    direction_y[start:start + batch_size],
                )
            )
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    return model, {"status": "trained", "rows": float(len(sequences)), "epochs": float(epochs), "device": str(device), "deterministic_backend": True, "final_loss": losses[-1]}


def _sequence_forecast(model: object | None, values: np.ndarray, indices: list[int] | np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    """Return (location, scale, regime confidence, P(up)) for causal rows."""
    if model is None or not HAS_TORCH:
        return {int(index): (0.0, 1e-4, 0.0, 0.5) for index in indices}
    valid = [int(index) for index in indices if 64 <= int(index) < len(values)]
    if not valid:
        return {}
    sequences = np.asarray([values[index - 64:index] for index in valid], dtype=np.float32)
    with torch.no_grad():
        output = model(torch.as_tensor(sequences, dtype=torch.float32, device=next(model.parameters()).device))
        location = output["return"][:, 0].detach().cpu().numpy()
        scale = torch.exp(output["return"][:, 1]).clamp(min=1e-7, max=0.1).detach().cpu().numpy()
        direction_probability = torch.softmax(output["direction"], dim=-1)[:, 1].detach().cpu().numpy()
        regime_confidence = torch.softmax(output["regime"], dim=-1).max(dim=-1).values.detach().cpu().numpy()
    return {index: (float(loc), float(scl), float(regime), float(probability)) for index, loc, scl, regime, probability in zip(valid, location, scale, regime_confidence, direction_probability, strict=True)}


def _normal_cdf_abs(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(abs(value) / math.sqrt(2.0))))


def evaluate(
    frame: pd.DataFrame,
    *,
    indicator_scope: str | None = None,
    search_depth: str | None = None,
    folds: int | None = None,
) -> dict[str, object]:
    """Train and evaluate the adaptive sequence/world model causally.

    Every fold owns its normalizer, sequence model, horizon policy, confidence
    calibrator, abstention threshold, and risk budget.  The final test slice is
    never used to choose any of those values.  The monetary replay models a
    fixed-payout digital contract because these are Quotex market records;
    price direction determines the outcome, while payout and costs determine
    the account result.
    """
    # Providers may attach a synthetic datetime index (needed by pandas-ta);
    # restore positional alignment before joining features to future labels.
    indicator_scope = (indicator_scope or os.getenv("NEXUS_INDICATOR_SCOPE", "core")).lower()
    search_depth = (search_depth or os.getenv("NEXUS_SEARCH_DEPTH", "deep")).lower()
    # Core is intentionally compact and fast; full preserves a broader
    # provider-derived feature set for the same causal walk-forward test.
    feature_budget = 64 if indicator_scope == "core" else 256
    base_representation = MarketRepresentation(RepresentationConfig(lookback=64, embedding_dim=64, max_features=feature_budget, seed=7))
    bundle = base_representation.features(frame)
    synthesized_formulas = {
        name: formula for name, formula in get_synthesized_indicator_catalog().items()
        if name in bundle.names
    }
    close = frame["close"].to_numpy(float)
    windows = _walk_forward_windows(len(frame), folds)
    records: list[dict[str, object]] = []
    baseline_records: list[dict[str, float]] = []
    folds: list[dict[str, object]] = []
    selected_by_window: list[list[str]] = []
    portfolio_capital = 10_000.0
    payout_percent = float(frame.attrs.get("payout_percent", 0.0) or 0.0)
    timeframe_seconds = int(frame.attrs.get("timeframe_seconds", 300) or 300)
    asset = str(frame.attrs.get("asset", "unknown"))
    fixed_cost_rate = 0.00025
    payout_rate = payout_percent / 100.0
    calibration_rows = 0
    model_rows = 0
    model_artifacts: list[dict[str, object]] = []

    def money_outcome(index: int, horizon: int, direction: str, amount: float) -> tuple[bool, float, float, float]:
        exit_index = index + horizon
        signed_move = (close[exit_index] / close[index] - 1.0) * (1.0 if direction == "long" else -1.0)
        won = bool(signed_move > 0.0)
        gross = amount * payout_rate if won else -amount
        costs = amount * fixed_cost_rate
        return won, gross, costs, gross - costs

    def raw_confidence(index: int, forecast: tuple[float, float, float, float], horizon: int) -> float:
        """Combine direct direction and return-head evidence conservatively."""
        location, scale, regime_confidence, direction_probability = forecast
        recent_returns = pd.Series(close).pct_change().iloc[max(1, index - 64):index].dropna().to_numpy(float)
        observed_scale = float(np.std(recent_returns)) if len(recent_returns) > 1 else 0.0
        effective_scale = max(scale, observed_scale, 1e-7)
        z_score = location * math.sqrt(horizon) / effective_scale
        gaussian_confidence = _normal_cdf_abs(z_score)
        classifier_confidence = max(direction_probability, 1.0 - direction_probability)
        combined = 0.65 * classifier_confidence + 0.35 * gaussian_confidence
        # A weakly identified regime should pull the probability toward an
        # honest 50% rather than allow the return head to claim certainty.
        return float(np.clip(0.5 + (combined - 0.5) * np.clip(regime_confidence, 0.25, 1.0), 0.5, 0.99))

    def calibrated_policy(
        indices: list[int],
        forecasts: dict[int, tuple[float, float, float, float]],
        horizon_by_regime: dict[str, int],
        calibrator: _ConservativeIsotonicCalibrator,
    ) -> tuple[float, float, float, list[dict[str, float | bool]]]:
        """Choose abstention threshold and max risk from validation only."""
        candidates: list[tuple[float, float, float, list[dict[str, float | bool]]]] = []
        thresholds = {
            "rapid": (0.50, 0.60, 0.70, 0.80, 0.90, 0.99),
            "deep": (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99),
            "exhaustive": tuple(round(value, 2) for value in np.arange(0.50, 1.00, 0.03)),
        }.get(search_depth, (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99))
        break_even = 1.0 / (1.0 + payout_rate) if payout_rate > 0 else 1.0
        for threshold in thresholds:
            simulated: list[dict[str, float | bool]] = []
            for index in indices:
                forecast = forecasts.get(index)
                if forecast is None:
                    continue
                location, _, _, direction_probability = forecast
                horizon = horizon_by_regime.get(_regime(frame, index), 1)
                direction = "long" if direction_probability >= 0.5 else "short"
                pre_calibration_confidence = raw_confidence(index, forecast, horizon)
                confidence = float(np.clip(calibrator.predict([pre_calibration_confidence])[0], 0.5, 0.99))
                payout_edge = confidence * (1.0 + payout_rate) - 1.0 - fixed_cost_rate
                if confidence < threshold or payout_edge <= 0.0:
                    continue
                won, gross, costs, net = money_outcome(index, horizon, direction, 1.0)
                simulated.append({"confidence": confidence, "win": won, "net_per_dollar": net, "gross_per_dollar": gross, "cost_per_dollar": costs})
            if len(simulated) < 20:
                continue
            win_rate = float(np.mean([bool(row["win"]) for row in simulated]))
            expectancy = win_rate * payout_rate - (1.0 - win_rate) - fixed_cost_rate
            _emit_telemetry("STRATEGY_TRIAL", {
                "asset": asset, "fold": fold_id, "total_folds": total_folds,
                "stage": "parameter_sweep",
                "indicators": {"features": list(synthesized_formulas) + list(bundle.names[:8]), "formulas": synthesized_formulas, "scope": indicator_scope, "feature_budget": feature_budget},
                "threshold": round(threshold, 4), "win_rate": round(win_rate, 4),
                "expectancy": round(expectancy, 6),
                "status": "candidate_validated" if expectancy > 0 else "candidate_rejected",
            })
            # Select a threshold by expected value and stability, not by a
            # single fortunate result.  The break-even margin is deliberately
            # strict enough that a candidate with no real edge abstains.
            if expectancy <= max(0.0, break_even - win_rate) + 0.005:
                continue
            candidates.append((expectancy, threshold, win_rate, simulated))
        if not candidates:
            # No threshold produced enough validation evidence for a positive
            # policy.  Do not label this as a viable strategy merely because
            # the break-even probability itself is positive.
            return 0.99, 0.0025, 0.0, []
        expectancy, threshold, win_rate, simulated = max(candidates, key=lambda item: (item[0], item[1]))
        # Higher validation edge earns more risk, but always remains under
        # the deterministic 1% research budget.  The lower bound is a risk
        # fraction, not the $1 executable-stake minimum; keeping it below
        # max_risk lets uncertainty and confidence change allocation.
        risk = min(0.01, max(0.00005, 0.01 * max(0.05, min(1.0, (win_rate - break_even) / 0.15))))
        return threshold, risk, expectancy, simulated

    total_folds = len(windows)
    for fold_id, window in enumerate(windows, 1):
        train_start, train_end = window.train
        val_start, val_end = window.validation
        test_start, test_end = window.test
        representation = MarketRepresentation(RepresentationConfig(lookback=64, embedding_dim=64, max_features=feature_budget, seed=7 + fold_id))
        train_bundle = type(bundle)(bundle.values[:train_end], bundle.names, bundle.schema_hash, train_end)
        representation.fit(train_bundle)
        values = representation.transform(bundle)
        selected_by_window.append(list(bundle.names))
        _emit_telemetry("STRATEGY_TRIAL", {
            "asset": asset, "fold": fold_id, "total_folds": total_folds,
            "stage": "training", "indicators": {"features": list(synthesized_formulas) + list(bundle.names[:12]), "formulas": synthesized_formulas, "scope": indicator_scope, "feature_budget": feature_budget},
            "feature_count": len(bundle.names),
            "train_rows": int(train_end - train_start),
            "status": "fitting_model",
        })
        model, model_metrics = _train_sequence_model(values, frame, train_start, train_end, seed=7 + fold_id)
        if model is None:
            continue
        model_rows += int(model_metrics.get("rows", 0.0))
        if HAS_TORCH:
            artifact_dir = Path("models/adaptive")
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_base = artifact_dir / f"{asset.replace('/', '_')}_robust_fold{fold_id}"
            torch.save({"model": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, artifact_base.with_suffix(".pt"))
            metadata = {"asset": asset, "fold": fold_id, "feature_schema": bundle.schema_hash, "input_dim": len(bundle.names), "lookback": 64, "model": "AdaptiveSequenceEncoder", "training": model_metrics, "status": "shadow_candidate", "production_promoted": False}
            artifact_base.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            model_artifacts.append({"fold": fold_id, "checkpoint": str(artifact_base.with_suffix(".pt")), "metadata": str(artifact_base.with_suffix(".json"))})
        policy_val_end = val_start + max(80, int((val_end - val_start) * 0.6))
        validation_indices = list(range(policy_val_end, val_end))
        validation_forecasts = _sequence_forecast(model, values, validation_indices)
        horizon_forecasts = _sequence_forecast(model, values, list(range(val_start, val_end)))
        horizon_directions = {
            index: "long" if forecast[3] >= 0.5 else "short"
            for index, forecast in horizon_forecasts.items()
        }
        default_horizon, horizon_by_regime = _choose_horizons(
            frame,
            val_start,
            val_end,
            asset,
            timeframe_seconds,
            direction_by_index=horizon_directions,
            payout_percent=payout_percent,
        )
        default_timeframe, timeframe_by_regime = _choose_timeframes(
            frame,
            val_start,
            val_end,
            asset,
            timeframe_seconds,
            direction_by_index=horizon_directions,
            payout_percent=payout_percent,
        )
        raw_validation_confidence: list[float] = []
        validation_wins: list[bool] = []
        for index in validation_indices:
            forecast = validation_forecasts.get(index)
            if forecast is None:
                continue
            location, _, _, direction_probability = forecast
            horizon = horizon_by_regime.get(_regime(frame, index), default_horizon)
            direction = "long" if direction_probability >= 0.5 else "short"
            raw_validation_confidence.append(raw_confidence(index, forecast, horizon))
            validation_wins.append(money_outcome(index, horizon, direction, 1.0)[0])
        if len(raw_validation_confidence) < 20 or len(set(validation_wins)) < 2:
            continue
        calibrator = _ConservativeIsotonicCalibrator().fit(
            np.asarray(raw_validation_confidence), np.asarray(validation_wins, dtype=float)
        )
        calibration_rows += len(raw_validation_confidence)
        trade_threshold, max_risk, validation_expectancy, validation_policy = calibrated_policy(validation_indices, validation_forecasts, horizon_by_regime, calibrator)
        # Emit telemetry for this fold's validated policy
        fold_val_wins = sum(bool(v) for v in validation_wins)
        fold_val_rate = fold_val_wins / len(validation_wins) if validation_wins else 0.0
        _emit_telemetry("STRATEGY_TRIAL", {
            "asset": asset, "fold": fold_id, "total_folds": total_folds,
            "stage": "validated",
            "indicators": {
                "features": list(synthesized_formulas) + list(bundle.names[:8]),
                "formulas": synthesized_formulas,
                "lookback": 64, "embedding_dim": 64,
                "horizon_default": default_horizon,
                "horizon_by_regime": horizon_by_regime,
                "timeframe_default": default_timeframe,
            },
            "win_rate": round(fold_val_rate, 4),
            "threshold": round(trade_threshold, 4),
            "max_risk": round(max_risk, 6),
            "expectancy": round(validation_expectancy, 6),
            "calibration_rows": len(raw_validation_confidence),
            "status": "candidate_validated" if validation_policy and validation_expectancy > 0 else "candidate_rejected",
        })
        test_indices = list(range(test_start, test_end))
        test_forecasts = _sequence_forecast(model, values, test_indices)
        fold_pnl = 0.0
        fold_wins = 0
        next_available = test_start
        for index in test_indices:
            if index < next_available:
                continue
            forecast = test_forecasts.get(index)
            if forecast is None:
                continue
            location, scale, regime_confidence, direction_probability = forecast
            regime = _regime(frame, index)
            horizon_bars = horizon_by_regime.get(regime, default_horizon)
            decision_timeframe_seconds = timeframe_by_regime.get(regime, default_timeframe)
            exit_index = index + horizon_bars
            if exit_index >= len(close):
                continue
            direction = "long" if direction_probability >= 0.5 else "short"
            raw_directional_probability = raw_confidence(index, forecast, horizon_bars)
            confidence = float(np.clip(calibrator.predict([raw_directional_probability])[0], 0.5, 0.99))
            expected_edge = confidence * (1.0 + payout_rate) - 1.0 - fixed_cost_rate
            # Independent policy: it may abstain, and it cannot enter while a
            # prior selected duration is still open in this market.
            if confidence < trade_threshold or expected_edge <= 0.0:
                continue
            uncertainty_scale = float(np.clip(1.0 / (1.0 + scale * 10_000.0), 0.05, 1.0))
            edge_scale = float(np.clip((confidence - trade_threshold) / max(1.0 - trade_threshold, 1e-6), 0.05, 1.0))
            # Learned allocation: the fold-local validation policy supplies
            # max_risk; confidence edge and forecast uncertainty determine
            # how much of that budget is used on this particular decision.
            # The additive base prevents the product from collapsing into a
            # single floor, while the deterministic cap remains untouched.
            risk_fraction = float(np.clip(
                max_risk * (0.05 + 0.95 * edge_scale * uncertainty_scale),
                0.00005,
                max_risk,
            ))
            balance_before = portfolio_capital
            amount_before = balance_before * risk_fraction
            # Research allocations are executable whole-dollar stakes: never
            # below $1 and never rounded to a sub-dollar increment.
            amount = float(max(1, round(amount_before)))
            won, gross_pnl, costs, pnl = money_outcome(index, horizon_bars, direction, amount)
            fold_pnl += pnl
            portfolio_capital += pnl
            fold_wins += int(won)
            next_available = exit_index + 1
            timestamp = str(frame.iloc[index].get("time", frame.iloc[index].get("timestamp", index)))
            records.append({
                "timestamp": timestamp, "asset": asset, "fold": fold_id,
                "regime": regime, "timeframe_seconds": timeframe_seconds, "decision_timeframe_seconds": decision_timeframe_seconds, "entry_timestamp": timestamp,
                "exit_timestamp": str(frame.iloc[exit_index].get("time", frame.iloc[exit_index].get("timestamp", exit_index))),
                "entry_offset_seconds": 0, "trade_duration_seconds": horizon_bars * timeframe_seconds,
                "trade_duration_minutes": horizon_bars * timeframe_seconds / 60, "direction": direction, "confidence": confidence,
                "probability_long": confidence if direction == "long" else 1.0 - confidence, "probability_flat": 1.0 - confidence,
                "probability_short": confidence if direction == "short" else 1.0 - confidence,
                "directional_probability": float(raw_directional_probability),
                "amount_allocated": amount, "trade_amount_before": amount_before, "trade_amount_after": amount, "account_balance_before": balance_before, "risk_fraction": risk_fraction,
                "entry_price": float(close[index]), "exit_price": float(close[exit_index]),
                "raw_directional_probability": float(raw_directional_probability), "regime_model_confidence": regime_confidence,
                "fees": costs / 2.0, "spread": costs / 2.0, "slippage": 0.0,
                "gross_pnl": gross_pnl, "net_pnl": pnl, "account_balance_after": balance_before + pnl,
                "position_size": amount, "payout_percent": payout_percent, "payout_amount": amount * payout_rate if won else 0.0, "payment_after_payout": amount * (1.0 + payout_rate) if won else 0.0, "action_metadata": "adaptive_sequence_world_model_validation_policy", "win": won,
            })
        baseline_pnl = 0.0
        baseline_trades = baseline_wins = 0
        for index in range(test_start, test_end):
            sign = np.sign(close[index] / close[max(0, index - 8)] - 1.0)
            if sign == 0:
                continue
            horizon = min(1, len(close) - index - 1)
            if horizon <= 0:
                continue
            direction = "long" if sign > 0 else "short"
            won, _, _, binary_net = money_outcome(index, horizon, direction, 100.0)
            baseline_pnl += binary_net
            baseline_trades += 1
            baseline_wins += int(won)
            baseline_records.append({"pnl": binary_net, "win": float(won)})
        folds.append({"fold": fold_id, "train": list(window.train), "validation": list(window.validation), "test": list(window.test), "trades": sum(r["fold"] == fold_id for r in records), "wins": fold_wins, "net_pnl": fold_pnl, "baseline_pnl": baseline_pnl, "baseline_trades": baseline_trades, "baseline_win_rate": baseline_wins / baseline_trades if baseline_trades else 0.0, "feature_combo": {**model_metrics, "feature_count": float(len(bundle.names))}, "default_horizon_bars": default_horizon, "horizon_by_regime": horizon_by_regime, "default_decision_timeframe_seconds": default_timeframe, "decision_timeframe_by_regime": timeframe_by_regime, "policy_threshold": trade_threshold, "policy_max_risk": max_risk, "validation_expectancy_per_dollar": validation_expectancy, "validation_policy_trades": len(validation_policy)})
    capital = portfolio_capital
    peak = capital
    max_drawdown = 0.0
    for record in records:
        capital = float(record["account_balance_after"])
        peak = max(peak, capital)
        max_drawdown = max(max_drawdown, 1.0 - capital / peak)
    wins = sum(bool(r["win"]) for r in records)
    returns = np.array([float(r["net_pnl"]) / float(r["amount_allocated"]) for r in records if float(r["amount_allocated"]) > 0])
    downside = returns[returns < 0]
    gross_profit = sum(float(r["net_pnl"]) for r in records if float(r["net_pnl"]) > 0)
    gross_loss = abs(sum(float(r["net_pnl"]) for r in records if float(r["net_pnl"]) < 0))
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(len(returns))) if len(downside) > 1 and np.std(downside) > 0 else 0.0
    calibration_error = _expected_calibration_error(records)
    baseline_pnl = float(sum(r["pnl"] for r in baseline_records))
    baseline_wins = int(sum(r["win"] for r in baseline_records))
    bands = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90, 0.99):
        chosen = [r for r in records if float(r["confidence"]) >= threshold]
        band_wins = sum(bool(r["win"]) for r in chosen)
        bands.append({"threshold": threshold, "trades": len(chosen), "wins": band_wins, "losses": len(chosen) - band_wins, "win_rate": band_wins / len(chosen) if chosen else None, "average_amount": float(np.mean([r["amount_allocated"] for r in chosen])) if chosen else 0.0, "total_amount_allocated": sum(float(r["amount_allocated"]) for r in chosen), "net_pnl": sum(float(r["net_pnl"]) for r in chosen)})
    confidence_buckets = []
    for low in np.arange(0.0, 1.0, 0.1):
        high = round(float(low + 0.1), 1)
        chosen = [r for r in records if low <= float(r["confidence"]) <= high] if high == 1.0 else [r for r in records if low <= float(r["confidence"]) < high]
        bucket_wins = sum(bool(r["win"]) for r in chosen)
        net = sum(float(r["net_pnl"]) for r in chosen)
        gross = sum(float(r["gross_pnl"]) for r in chosen)
        allocated = sum(float(r["amount_allocated"]) for r in chosen)
        confidence_buckets.append({"range": f"{low:.0%}-{high:.0%}", "trades": len(chosen), "wins": bucket_wins, "losses": len(chosen) - bucket_wins, "win_rate": bucket_wins / len(chosen) if chosen else None, "average_confidence": float(np.mean([r["confidence"] for r in chosen])) if chosen else None, "average_amount": allocated / len(chosen) if chosen else 0.0, "total_amount_allocated": allocated, "gross_pnl": gross, "net_pnl": net, "roi": net / allocated if allocated else 0.0, "expectancy": net / len(chosen) if chosen else 0.0, "average_payout": float(np.mean([r["gross_pnl"] for r in chosen if float(r["gross_pnl"]) > 0])) if any(float(r["gross_pnl"]) > 0 for r in chosen) else 0.0, "average_loss": float(np.mean([r["gross_pnl"] for r in chosen if float(r["gross_pnl"]) < 0])) if any(float(r["gross_pnl"]) < 0 for r in chosen) else 0.0})
    reasons = []
    if len(records) < 300:
        reasons.append("fewer than 300 out-of-sample trades")
    if not folds or any(float(f["net_pnl"]) <= 0 for f in folds):
        reasons.append("not profitable in every walk-forward test fold")
    if capital - 10_000.0 <= baseline_pnl:
        reasons.append("does not beat fixed-size one-candle momentum payout baseline")
    if max_drawdown >= 0.10:
        reasons.append("maximum drawdown exceeds 10% gate")
    if calibration_error > 0.15:
        reasons.append("calibration error exceeds 0.15 gate")
    evidence_floor = max(0.60, 1.0 / (1.0 + payout_rate) + 0.03)
    if sum(float(r["confidence"]) >= evidence_floor for r in records) < 20:
        reasons.append(f"insufficient held-out evidence at or above {evidence_floor:.0%} confidence")
    # Emit feature ranking telemetry for the dashboard discovery scene
    all_feature_names: list[str] = []
    for window_features in selected_by_window:
        all_feature_names.extend(window_features)
    feature_frequency: dict[str, int] = {}
    for name in all_feature_names:
        feature_frequency[name] = feature_frequency.get(name, 0) + 1
    ranked_features = sorted(feature_frequency.items(), key=lambda item: item[1], reverse=True)[:15]
    if ranked_features:
        max_freq = max(freq for _, freq in ranked_features)
        _emit_telemetry("FEATURE_RANK", {
            "asset": asset,
            "top_features": [
                {"name": name, "score": round(freq / max(1, max_freq), 4), "frequency": freq}
                for name, freq in ranked_features
            ],
            "total_folds": total_folds,
            "promotion": "promote" if not reasons else "reject",
        })
    return {"trades": len(records), "wins": wins, "losses": len(records) - wins, "win_rate": wins / len(records) if records else 0.0, "starting_virtual_capital": 10_000.0, "final_virtual_capital": capital, "net_pnl": capital - 10_000.0, "roi": capital / 10_000.0 - 1.0, "average_trade_pnl": (capital - 10_000.0) / len(records) if records else 0.0, "profit_factor": gross_profit / gross_loss if gross_loss else None, "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown, "capital_utilization": float(sum(float(r["amount_allocated"]) for r in records) / (10_000.0 * len(records))) if records else 0.0, "calibration_error": calibration_error, "calibration_metric": "confidence-bucket expected calibration error", "confidence_evidence_floor": evidence_floor, "confidence_floor": min((float(r["confidence"]) for r in records), default=0.5), "confidence_thresholds": bands, "confidence_buckets": confidence_buckets, "baseline": {"name": "fixed-size one-candle momentum payout", "trades": len(baseline_records), "wins": baseline_wins, "win_rate": baseline_wins / len(baseline_records) if baseline_records else 0.0, "net_pnl": baseline_pnl}, "folds": folds, "selected_features_by_window": selected_by_window, "sequence_model_rows": model_rows, "calibration_rows": calibration_rows, "model_artifacts": model_artifacts, "promotion_status": "promote" if not reasons else "reject", "promotion_rejection_reasons": reasons, "trade_records": records}


class BrokerAuthorizationError(RuntimeError):
    """The broker rejected the session even after one refresh attempt."""


class MarketDataUnavailableError(RuntimeError):
    """The authenticated broker could not provide candles for this market."""


def _looks_like_auth_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("unauthor", "not authenticated", "unauthenticated", "session", "login", "authorization", "token"))


async def _fetch_candles_with_reconnect(engine: NexusEngine, asset: str, timeframe_seconds: int, limit: int, timeout: float) -> list[dict[str, float]]:
    """Fetch authentic candles and retry once after a stale-session reconnect."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            if not getattr(engine._broker, "authenticated", False):
                raise RuntimeError("broker unauthenticated")
            raw = await asyncio.wait_for(
                engine._broker.get_candles_async(asset, timeframe_seconds, limit), timeout=timeout
            )
            if raw:
                return raw
            raise RuntimeError("broker returned no candles")
        except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt == 0:
                # QuotexAdapter exposes authenticated read-only; reset the
                # underlying real adapter before asking the engine to login.
                if not await _refresh_broker_session(engine):
                    if not getattr(engine._broker, "authenticated", False) or _looks_like_auth_failure(exc):
                        raise BrokerAuthorizationError(f"login refresh rejected while fetching {asset}") from exc
                    break
                await asyncio.sleep(1)
    real_broker = getattr(engine._broker, "_real", engine._broker)
    close = getattr(real_broker, "close", None)
    if close is not None:
        await close()
    else:
        real_broker.authenticated = False
    if not getattr(engine._broker, "authenticated", False) or _looks_like_auth_failure(last_error or RuntimeError()):
        raise BrokerAuthorizationError(f"login refresh failed while fetching {asset}") from last_error
    raise MarketDataUnavailableError(f"market candle data unavailable for {asset}: {type(last_error).__name__ if last_error else 'unknown'}")


async def _fetch_best_supported_resolution(engine: NexusEngine, asset: str, preferred_seconds: int, limit: int, timeout: float) -> tuple[list[dict[str, float]], int]:
    """Try honest resolutions in order; never resample a coarser candle feed."""
    candidates = [preferred_seconds]
    if preferred_seconds <= 60:
        candidates.extend([300, 900])
    last_error: Exception | None = None
    for resolution in candidates:
        try:
            return await _fetch_candles_with_reconnect(engine, asset, resolution, limit, timeout), resolution
        except (RuntimeError, MarketDataUnavailableError) as exc:
            last_error = exc
            if not getattr(engine._broker, "authenticated", False):
                break
            if resolution != candidates[-1]:
                if not await _refresh_broker_session(engine):
                    continue
    if isinstance(last_error, BrokerAuthorizationError):
        raise last_error
    raise MarketDataUnavailableError(f"no supported candle resolution for {asset}: {type(last_error).__name__ if last_error else 'unknown'}")


async def _fetch_historical_range(
    engine: NexusEngine,
    asset: str,
    timeframe_seconds: int,
    history_seconds: int,
    timeout: float,
) -> tuple[list[dict[str, float]], int]:
    """Fetch a genuine broker time range when the authenticated API supports it."""
    broker = getattr(engine, "_broker", None)
    real_broker = getattr(broker, "_real", broker)
    client = getattr(real_broker, "client", None)
    method = getattr(client, "get_historical_candles", None)
    if method is None:
        raise RuntimeError("authenticated historical-range API unavailable")
    try:
        raw = await asyncio.wait_for(
            method(asset, int(history_seconds), int(timeframe_seconds), timeout=max(30, int(timeout)), max_workers=5),
            timeout=max(float(timeout) * 5.0, 120.0),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if _looks_like_auth_failure(exc) or not getattr(broker, "authenticated", False):
            if await _refresh_broker_session(engine):
                try:
                    raw = await asyncio.wait_for(
                        method(asset, int(history_seconds), int(timeframe_seconds), timeout=max(30, int(timeout)), max_workers=5),
                        timeout=max(float(timeout) * 5.0, 120.0),
                    )
                except Exception as retry_exc:
                    raise BrokerAuthorizationError(f"login refresh rejected while fetching {asset}") from retry_exc
            else:
                raise BrokerAuthorizationError(f"login refresh failed while fetching {asset}") from exc
        else:
            raise MarketDataUnavailableError(f"historical candle data unavailable for {asset}") from exc
    if not isinstance(raw, list):
        raise MarketDataUnavailableError(f"historical-range API returned invalid data for {asset}")
    normalized: list[dict[str, float]] = []
    for candle in raw:
        if not isinstance(candle, dict):
            continue
        try:
            normalized.append({
                "time": float(candle.get("time", candle.get("t", 0.0))),
                "open": float(candle.get("open", candle.get("o", 0.0))),
                "high": float(candle.get("high", candle.get("h", 0.0))),
                "low": float(candle.get("low", candle.get("l", 0.0))),
                "close": float(candle.get("close", candle.get("c", 0.0))),
                "volume": float(candle.get("volume", candle.get("v", 0.0))),
            })
        except (TypeError, ValueError):
            continue
    if len(normalized) < 3000:
        raise MarketDataUnavailableError(f"historical-range API returned only {len(normalized)} candles for {asset}")
    return normalized, timeframe_seconds


def _emit_telemetry(tag: str, payload: dict[str, object]) -> None:
    """Emit a structured telemetry line for the dashboard to parse."""
    print(f"[{tag}] {json.dumps(payload, default=str)}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-only market replay")
    parser.add_argument("--min-payout", type=float, default=None, help="Only include active broker markets with payout >= this percent")
    parser.add_argument("--assets", type=str, default=None, help="Comma-separated list of specific assets to research")
    parser.add_argument("--search-depth", type=str, default=None, choices=["rapid", "deep", "exhaustive"], help="Strategy search depth: rapid (6), deep (18), exhaustive (36+)")
    parser.add_argument("--indicator-scope", type=str, default=None, choices=["core", "full"], help="Indicator scope: core (momentum/trend) or full (all 750+ providers)")
    parser.add_argument("--folds", type=int, default=None, help="Number of walk-forward folds (2-6)")
    args = parser.parse_args()
    min_payout = args.min_payout if args.min_payout is not None else float(os.getenv("NEXUS_MIN_PAYOUT", "0"))
    if not 0.0 <= min_payout <= 100.0:
        raise ValueError("minimum payout must be between 0 and 100 percent")
    search_depth = args.search_depth or os.getenv("NEXUS_SEARCH_DEPTH", "deep")
    indicator_scope = args.indicator_scope or os.getenv("NEXUS_INDICATOR_SCOPE", "core")
    max_folds = args.folds or int(os.getenv("NEXUS_WALK_FORWARD_FOLDS", "3"))
    max_folds = max(2, min(6, max_folds))
    selected_assets_csv = args.assets or os.getenv("NEXUS_RESEARCH_ASSETS", "")
    engine = NexusEngine(load_runtime_settings(), demo_mode=True, auto_login=False)
    if not await engine.login_broker():
        root = Path(__file__).resolve().parents[1]
        if not await _refresh_login_browser(root) or not await engine.login_broker():
            raise BrokerAuthorizationError("broker login failed after login refresh")
    catalog = await engine._broker.get_assets_with_payouts_async()
    catalog_assets = sorted({str(x.get("symbol") or x.get("name")) for x in catalog or [] if isinstance(x, dict) and x.get("active", True)})
    payout_map = {str(item.get("symbol", "")).upper(): float(item.get("payout") or 0.0) for item in catalog if isinstance(item, dict)}
    excluded_low_payout = sorted(asset for asset in catalog_assets if payout_map.get(asset.upper(), 0.0) < min_payout)
    catalog_assets = [asset for asset in catalog_assets if asset not in excluded_low_payout]
    # If specific assets were requested, filter to only those
    if selected_assets_csv.strip():
        requested = [a.strip() for a in selected_assets_csv.split(",") if a.strip()]
        assets = [a for a in catalog_assets if a in requested]
    else:
        assets = list(catalog_assets)
    limit = int(os.getenv("NEXUS_RESEARCH_LIMIT", "0"))
    if limit > 0:
        assets = assets[:limit]
    # A shard is a disjoint worker lane.  It lets a launcher use several CPU
    # and CUDA streams without allowing workers to overwrite one checkpoint.
    shard_total = max(1, int(os.getenv("NEXUS_RESEARCH_SHARDS", "1")))
    shard_index = int(os.getenv("NEXUS_RESEARCH_SHARD_INDEX", "0"))
    if not 0 <= shard_index < shard_total:
        raise ValueError(f"NEXUS_RESEARCH_SHARD_INDEX must be in [0, {shard_total})")
    if shard_total > 1:
        assets = assets[shard_index::shard_total]
    history_dir = Path("data/historical")
    history_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the authenticated historical-range API so the model sees weeks
    # of genuine candles.  The 3,200-candle value remains a safe fallback for
    # clients that do not expose range retrieval.
    history_days = max(7, int(os.getenv("NEXUS_HISTORY_DAYS", "7")))
    history_seconds = history_days * 86400
    history_limit = _verified_history_limit(int(os.getenv("NEXUS_HISTORY_LIMIT", "3200")))
    report = Path(os.getenv("NEXUS_RESEARCH_REPORT", "reports/robust_market_research.json"))
    report.parent.mkdir(exist_ok=True)
    progress_path = Path(os.getenv("NEXUS_RESEARCH_PROGRESS", str(report.with_suffix(".progress.json"))))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    def write_progress(**updates: object) -> None:
        payload = {
            "status": "running",
            "stage": "starting",
            "market_index": 0,
            "markets_total": 0,
            "completed_markets": 0,
            "error_markets": 0,
            "progress": 0.0,
            "orders_placed": 0,
            "started_at": started_at,
            "updated_at": time.time(),
            **updates,
        }
        temporary = progress_path.with_suffix(progress_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    previous = {}
    if report.exists():
        try:
            previous = {str(item.get("asset")): item for item in json.loads(report.read_text(encoding="utf-8")).get("markets", [])}
        except (OSError, ValueError, TypeError):
            previous = {}
    only_errors = os.getenv("NEXUS_RESEARCH_ONLY_ERRORS", "0").lower() in {"1", "true", "yes"}
    if only_errors:
        assets = [asset for asset in assets if previous.get(asset, {}).get("status") == "error"]
    results = [item for asset, item in previous.items() if only_errors and asset not in assets]
    provider_catalog = get_feature_provider_catalog()
    force_retrain = os.getenv("NEXUS_FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    write_progress(
        stage="ready",
        markets_total=len(assets),
        detail=f"{len(assets)} markets selected; forced_refit={force_retrain}",
    )

    def write_checkpoint() -> None:
        report.write_text(json.dumps({"mode": "research_only_local_replay", "catalog_source": "refreshed_broker", "assets": len(catalog_assets), "run_assets": len(assets), "minimum_payout_percent": min_payout, "excluded_low_payout_markets": len(excluded_low_payout), "completed_or_recorded": len(results), "timing_policy": "OTC uses measured five-second candles when returned; non-OTC uses measured one-minute candles; no resampling fabricates resolution", "history_days_requested": history_days, "history_seconds_requested": history_seconds, "legacy_history_limit_per_asset": history_limit, "history_span_note": "Uses the authenticated historical-range API when available; falls back to the verified 3,200-candle request only when range retrieval is unavailable", "walk_forward": "chronological train-validation-test folds with ten-bar purge/embargo and at most six expanding test folds as history grows", "calibration": "fold-local monotonic isotonic calibration on validation only, never fitted on test", "feature_provider_catalog": provider_catalog, "orders_placed": 0, "markets": results}, indent=2), encoding="utf-8")

    for i, asset in enumerate(assets, 1):
        completed = sum(item.get("status") == "complete" for item in results)
        errors = sum(item.get("status") == "error" for item in results)
        write_progress(
            stage="collecting",
            market_index=i,
            markets_total=len(assets),
            asset=asset,
            completed_markets=completed,
            error_markets=errors,
            progress=round((i - 1) / max(1, len(assets)), 4),
            detail="fetching historical broker candles",
        )
        preferred_seconds = int(os.getenv("NEXUS_OTC_TIMEFRAME_SECONDS", "60")) if "otc" in asset.lower() else 60
        path = history_dir / f"{asset.replace('/', '_')}_{preferred_seconds}s.csv"
        cached_result = previous.get(asset)
        expected_candles = max(3000, int(history_seconds / preferred_seconds * 0.90))
        if not force_retrain and cached_result and cached_result.get("status") == "complete" and path.exists() and len(pd.read_csv(path, nrows=expected_candles + 1)) >= expected_candles:
            results.append(cached_result)
            print(f"[{i}/{len(assets)}] {asset}: reused checkpoint", flush=True)
            continue
        try:
            if not force_retrain and path.exists() and len(pd.read_csv(path, nrows=expected_candles + 1)) >= expected_candles:
                frame, source = pd.read_csv(path), "persisted_historical_cache"
            else:
                timeout = float(os.getenv("NEXUS_HISTORY_TIMEOUT", "30"))
                try:
                    raw, fetched_seconds = await _fetch_historical_range(engine, asset, preferred_seconds, history_seconds, timeout)
                    source = "refreshed_broker_historical_range"
                except (MarketDataUnavailableError, asyncio.TimeoutError):
                    raw, fetched_seconds = await _fetch_best_supported_resolution(engine, asset, preferred_seconds, history_limit, timeout)
                    source = "refreshed_broker_legacy_limit"
                frame = _chronological(pd.DataFrame(raw).dropna().reset_index(drop=True))
                if len(frame) >= 3000:
                    path = history_dir / f"{asset.replace('/', '_')}_{fetched_seconds}s.csv"
                    frame.to_csv(path, index=False)
            frame = _chronological(frame)
            observed_seconds = _observed_timeframe_seconds(frame, preferred_seconds)
            # Never claim five-second timing if the broker actually returned
            # coarser candles. Use the measured interval for all decisions.
            if "otc" in asset.lower() and preferred_seconds <= 10 and not 4 <= observed_seconds <= 6:
                # Some broker implementations answer a 5-second request with
                # irregular tick records. Those are not 5-second candles and
                # must not be treated as such. Fall back to genuine minute
                # candles rather than fabricating sub-minute resolution.
                observed_seconds = 60
                fallback_path = history_dir / f"{asset.replace('/', '_')}_{observed_seconds}s.csv"
                if fallback_path.exists() and len(pd.read_csv(fallback_path)) >= 3000:
                    frame = _chronological(pd.read_csv(fallback_path))
                else:
                    raw = await _fetch_candles_with_reconnect(engine, asset, observed_seconds, history_limit, float(os.getenv("NEXUS_HISTORY_TIMEOUT", "30")))
                    frame = _chronological(pd.DataFrame(raw).dropna().reset_index(drop=True))
                    if len(frame) >= 3000:
                        frame.to_csv(fallback_path, index=False)
                source = f"{source}; measured_resolution_fallback"
            if len(frame) < 3000:
                results.append({"asset": asset, "status": "insufficient_data", "candles": len(frame), "history_file": str(path)})
                write_checkpoint()
                continue
            frame.attrs["asset"] = asset
            frame.attrs["timeframe_seconds"] = observed_seconds
            frame.attrs["payout_percent"] = payout_map.get(asset.upper(), 0.0)
            write_progress(
                stage="training_evaluation",
                market_index=i,
                markets_total=len(assets),
                asset=asset,
                completed_markets=completed,
                error_markets=errors,
                progress=round((i - 0.5) / max(1, len(assets)), 4),
                detail=f"walk-forward, calibration, costs, and promotion gates ({len(frame):,} candles)",
            )
            result = evaluate(
                frame,
                indicator_scope=indicator_scope,
                search_depth=search_depth,
                folds=max_folds,
            )
            # Emit market summary telemetry for the visual dashboard
            _emit_telemetry("MARKET_SUMMARY", {
                "asset": asset,
                "trades": result.get("trades", 0),
                "wins": result.get("wins", 0),
                "win_rate": round(float(result.get("win_rate", 0)), 4),
                "net_pnl": round(float(result.get("net_pnl", 0)), 2),
                "promotion": result.get("promotion_status", "unknown"),
                "sharpe": round(float(result.get("sharpe", 0)), 4),
                "max_drawdown": round(float(result.get("max_drawdown", 0)), 4),
                "search_depth": search_depth,
                "indicator_scope": indicator_scope,
            })
            results.append({"asset": asset, "status": "complete", "candles": len(frame), "history_file": str(path), "history_source": source, **result})
            write_checkpoint()
            write_progress(
                stage="market_complete",
                market_index=i,
                markets_total=len(assets),
                asset=asset,
                completed_markets=sum(item.get("status") == "complete" for item in results),
                error_markets=sum(item.get("status") == "error" for item in results),
                progress=round(i / max(1, len(assets)), 4),
                detail=f"{result['trades']} OOS trades; promotion={result['promotion_status']}",
            )
            print(f"[{i}/{len(assets)}] {asset}: {result['trades']} OOS trades, win={float(result['win_rate']):.2%}, pnl=${float(result['net_pnl']):.2f}, {result['promotion_status']}", flush=True)
        except BrokerAuthorizationError as exc:
            results.append({"asset": asset, "status": "error", "error_type": "authentication", "login_refresh_attempted": True, "login_refresh_succeeded": False, "candles": 0, "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
            write_checkpoint()
            write_progress(stage="market_error", market_index=i, markets_total=len(assets), asset=asset, completed_markets=sum(item.get("status") == "complete" for item in results), error_markets=sum(item.get("status") == "error" for item in results), progress=round(i / max(1, len(assets)), 4), detail=str(exc)[:240])
            print(f"[{i}/{len(assets)}] {asset}: AUTHENTICATION_REFRESH_FAILED {exc}", flush=True)
            continue
        except MarketDataUnavailableError as exc:
            results.append({"asset": asset, "status": "error", "error_type": "market_data_unavailable", "login_refresh_attempted": False, "login_refresh_succeeded": False, "candles": 0, "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
            write_checkpoint()
            write_progress(stage="market_error", market_index=i, markets_total=len(assets), asset=asset, completed_markets=sum(item.get("status") == "complete" for item in results), error_markets=sum(item.get("status") == "error" for item in results), progress=round(i / max(1, len(assets)), 4), detail=str(exc)[:240])
            print(f"[{i}/{len(assets)}] {asset}: MARKET_DATA_UNAVAILABLE {exc}", flush=True)
            continue
        except Exception as exc:
            results.append({"asset": asset, "status": "error", "error_type": "unexpected", "login_refresh_attempted": False, "login_refresh_succeeded": False, "candles": 0, "error": f"{type(exc).__name__}: {str(exc)[:220]}"})
            write_checkpoint()
            print(f"[{i}/{len(assets)}] {asset}: ERROR {type(exc).__name__}: {exc}", flush=True)
            # Refresh the stateful WebSocket before moving to the next market.
            # A failed request must not poison every later asset in the sweep.
            try:
                await _refresh_broker_session(engine)
                await asyncio.sleep(1)
            except Exception as refresh_error:
                print(f"[{i}/{len(assets)}] session refresh failed: {refresh_error}", flush=True)
    report.write_text(json.dumps({"mode": "research_only_local_replay", "catalog_source": "refreshed_broker", "assets": len(catalog_assets), "run_assets": len(assets), "minimum_payout_percent": min_payout, "excluded_low_payout_markets": len(excluded_low_payout), "timing_policy": "OTC uses measured five-second candles when returned; non-OTC uses measured one-minute candles; no resampling fabricates resolution", "history_days_requested": history_days, "history_seconds_requested": history_seconds, "legacy_history_limit_per_asset": history_limit, "history_span_note": "Uses the authenticated historical-range API when available; falls back to the verified 3,200-candle request only when range retrieval is unavailable", "walk_forward": "chronological train-validation-test folds with ten-bar purge/embargo and at most six expanding test folds as history grows", "calibration": "fold-local monotonic isotonic calibration on validation only, never fitted on test", "orders_placed": 0, "markets": results}, indent=2), encoding="utf-8")
    write_progress(stage="complete", market_index=len(assets), markets_total=len(assets), completed_markets=sum(item.get("status") == "complete" for item in results), error_markets=sum(item.get("status") == "error" for item in results), progress=1.0, detail="report finalized")
    close = getattr(getattr(engine, "_broker", None), "close", None)
    if close is not None:
        await close()
    print(json.dumps({"report": str(report), "assets": len(catalog_assets), "run_assets": len(assets), "minimum_payout_percent": min_payout, "excluded_low_payout_markets": len(excluded_low_payout), "completed": sum(x.get("status") == "complete" for x in results), "promoted": sum(x.get("promotion_status") == "promote" for x in results), "orders_placed": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

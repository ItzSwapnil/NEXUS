"""AI Ensemble Manager for NEXUS.

Combines predictions from multiple models using adaptive fitness weighting:
- Amazon Chronos 2 (pretrained foundation model, zero-shot forecasting)
- Market Transformer (multi-head self-attention time-series predictor)
- Attention-LSTM (bidirectional LSTM with temporal attention)
- Deep RL Agent (Dueling Double DQN with Prioritized Experience Replay)
- Market-specific gradient boosting classifier
- Market Regime Detector
"""

from typing import Any, Dict, Optional

import numpy as np

from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.ai.ensemble_manager")


class AIEnsembleManager:
    """Manages deep learning model ensembles and dynamically weights model predictions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        default_weights = {
            "chronos": 0.35,  # pretrained foundation model (highest prior)
            "transformer": 0.25,
            "lstm": 0.20,
            "rl_agent": 0.12,
            "market_model": 0.10,
            "regime": 0.08,
        }
        self.weights = weights if weights is not None else default_weights
        # Ensure all keys are present when loading older checkpoints
        self.weights.setdefault("chronos", 0.35)
        self.weights.setdefault("market_model", 0.10)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def update_model_weight(self, model_name: str, performance_score: float) -> None:
        """Dynamically adjust model weights based on recent performance."""
        if model_name in self.weights:
            current = self.weights[model_name]
            self.weights[model_name] = max(0.05, current * 0.9 + performance_score * 0.1)
            self._normalize_weights()
            logger.info("Updated weight for %s: %.4f", model_name, self.weights[model_name])

    def combine_predictions(
        self,
        transformer_pred: Optional[Dict[str, Any]] = None,
        lstm_pred: Optional[Dict[str, Any]] = None,
        rl_pred: Optional[Dict[str, Any]] = None,
        market_model_pred: Optional[Dict[str, Any]] = None,
        chronos_pred: Optional[Dict[str, Any]] = None,
        regime: str = "SIDEWAYS",
    ) -> Dict[str, Any]:
        """Aggregate model predictions using ensemble weights."""
        scores = {"call": 0.0, "put": 0.0, "hold": 0.0}
        total_weight = 0.0

        # --- Chronos 2 (pretrained foundation model) ---
        if chronos_pred and "probabilities" in chronos_pred:
            w = self.weights.get("chronos", 0.35)
            probs = chronos_pred["probabilities"]
            for k in scores:
                scores[k] += probs.get(k, 1.0 / 3.0) * w
            total_weight += w

        if transformer_pred and "probabilities" in transformer_pred:
            w = self.weights.get("transformer", 0.4)
            probs = transformer_pred["probabilities"]
            for k in scores:
                scores[k] += probs.get(k, 0.33) * w
            total_weight += w

        if lstm_pred and "probabilities" in lstm_pred:
            w = self.weights.get("lstm", 0.3)
            probs = lstm_pred["probabilities"]
            for k in scores:
                scores[k] += probs.get(k, 0.33) * w
            total_weight += w

        if rl_pred and "action" in rl_pred:
            w = self.weights.get("rl_agent", 0.2)
            action = rl_pred.get("action", "hold")
            if action in scores:
                # Never turn an action into an arbitrary 80% probability.  The
                # RL agent supplies a confidence estimate; absent that, use a
                # neutral vote so the ensemble cannot manufacture certainty.
                rl_confidence = float(rl_pred.get("confidence", 1.0 / 3.0))
                scores[action] += max(0.0, min(1.0, rl_confidence)) * w
            total_weight += w

        if market_model_pred and "probabilities" in market_model_pred:
            w = self.weights.get("market_model", 0.15)
            probs = market_model_pred["probabilities"]
            for k in scores:
                scores[k] += probs.get(k, 0.0) * w
            total_weight += w

        if regime == "VOLATILE":
            scores["hold"] += 0.2
        elif regime == "BULL":
            scores["call"] += 0.1
        elif regime == "BEAR":
            scores["put"] += 0.1

        sum_scores = sum(scores.values())
        if sum_scores > 0:
            probs = {k: v / sum_scores for k, v in scores.items()}
        else:
            probs = {"call": 0.33, "put": 0.33, "hold": 0.34}

        best_signal = max(probs, key=lambda k: probs[k])
        dir_sum = probs.get("call", 0.0) + probs.get("put", 0.0)
        if best_signal in ("call", "put") and dir_sum > 0.0:
            dir_prob = probs[best_signal] / dir_sum
            # In binary options, directional confidence is the conditional win probability [0.50, 0.99].
            # Only apply slight damping if hold exceeds neutral baseline (1/3).
            hold_mass = float(probs.get("hold", 0.0))
            penalty = max(0.0, hold_mass - (1.0 / 3.0))
            confidence = float(np.clip(dir_prob - penalty * 0.5, 0.50, 0.99))
        else:
            confidence = min(0.99, float(probs[best_signal]))

        return {
            "signal": best_signal,
            "confidence": round(confidence, 4),
            "probabilities": probs,
            "regime": regime,
            "weights": self.weights,
        }


__all__ = ["AIEnsembleManager"]

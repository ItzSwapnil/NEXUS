"""Exploration / exploitation controller (Spec §3).

Implements epsilon computation blending confidence + uncertainty metrics and payout.

Formula (adapted from spec):
  confidence = normalized(Sharpe, stability, win_rate)
  uncertainty = f(ATR, disagreement, spread, OTC_flag)
  epsilon = clamp(base_epsilon*(1 - confidence) + k*uncertainty, min, max)
  Reward higher payout by reducing exploration slightly when payout is strong.
"""
from __future__ import annotations
from typing import Dict
from nexus.utils.config import NexusSettings

class ExplorationController:
    def __init__(self, settings: NexusSettings):
        self.settings = settings
        self.cfg = settings.exploration

    # ------------------------------ Normalizers --------------------------- #
    @staticmethod
    def _norm(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        x = (value - lo) / (hi - lo)
        return 0.0 if x < 0 else 1.0 if x > 1 else x

    def _confidence_score(self, metrics: Dict[str, float]) -> float:
        sharpe = self._norm(metrics.get("sharpe", 0.0), -1.0, 3.0)
        stability = self._norm(metrics.get("stability", 0.0), 0.0, 1.0)
        win_rate = self._norm(metrics.get("win_rate", 0.0), 0.3, 0.75)
        # simple mean
        return (sharpe + stability + win_rate) / 3.0

    def _uncertainty_score(self, metrics: Dict[str, float]) -> float:
        atr = self._norm(metrics.get("atr", 0.0), 0.0, 2.0)
        disagreement = self._norm(metrics.get("disagreement", 0.0), 0.0, 1.0)
        spread = self._norm(metrics.get("spread", 0.0), 0.0, 0.0005)
        otc = 1.0 if metrics.get("otc", False) else 0.0
        return min(1.0, 0.35*atr + 0.35*disagreement + 0.2*spread + 0.1*otc)

    # ------------------------------ Epsilon ------------------------------- #
    def compute_epsilon(self, confidence_metrics: Dict[str, float], uncertainty_metrics: Dict[str, float], payout: float) -> float:
        c = self._confidence_score(confidence_metrics)
        u = self._uncertainty_score(uncertainty_metrics)
        base_eps = self.cfg.base_epsilon * (1 - c)
        uncertainty_component = self.cfg.k_uncertainty * u
        # Higher payout reduces exploration modestly (encourage exploitation)
        payout_norm = self._norm(payout, 60.0, 95.0)  # typical payout band
        payout_modifier = 0.15 * (payout_norm - 0.5)  # in [-0.075, +0.075]
        eps = base_eps + uncertainty_component - payout_modifier
        if eps < self.cfg.min_epsilon:
            eps = self.cfg.min_epsilon
        if eps > self.cfg.max_epsilon:
            eps = self.cfg.max_epsilon
        return round(eps, 4)

__all__ = ["ExplorationController"]


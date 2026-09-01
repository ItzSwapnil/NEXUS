"""Candidate lifecycle promotion/demotion logic (Spec §10).

Integrates with fitness computation and exploration controller thresholds.

Lifecycle states:
  shadow -> micro-live -> champion
Demotions:
  micro-live -> shadow on consecutive failures
  champion -> micro-live on underperformance (simplified placeholder)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from nexus.utils.config import NexusSettings

from .fitness import CandidateState

PromotionCallback = Callable[[CandidateState, str, str], None]
ViolationCallback = Callable[[CandidateState, str], None]


class PromotionManager:
    def __init__(
        self,
        settings: NexusSettings,
        on_promotion: Optional[PromotionCallback] = None,
        on_demotion: Optional[PromotionCallback] = None,
        on_violation: Optional[ViolationCallback] = None,
    ):
        self.settings = settings
        self.on_promotion = on_promotion
        self.on_demotion = on_demotion
        self.on_violation = on_violation
        self._demotion_streaks: Dict[str, int] = {}

    # -------------------------- Core Update ------------------------------- #
    def update_lifecycle(self, candidate: CandidateState, violation: Optional[str] = None) -> None:
        prev = candidate.lifecycle
        cfg = self.settings.exploration
        threshold = cfg.fitness_promotion_threshold

        # Handle violations (immediate demotion except shadow)
        if violation:
            if candidate.lifecycle in ("micro-live", "champion"):
                self._demote(candidate, "shadow", reason=f"violation:{violation}")
            if self.on_violation:
                self.on_violation(candidate, violation)
            return

        fit_ok = candidate.fitness >= threshold
        # Promotion windows counter
        if fit_ok:
            candidate.promotion_windows_ok += 1
        else:
            candidate.promotion_windows_ok = 0

        # Shadow -> Micro-Live
        if (
            candidate.lifecycle == "shadow"
            and candidate.promotion_windows_ok >= cfg.promotion_windows
        ):
            self._promote(candidate, "micro-live")

        # Micro-Live -> Champion (simplified: sustain 2x windows threshold + high fitness)
        elif (
            candidate.lifecycle == "micro-live"
            and candidate.promotion_windows_ok >= (cfg.promotion_windows * 2)
            and candidate.fitness >= (threshold + 0.1)
        ):
            self._promote(candidate, "champion")

        # Demotions for under-performance
        elif candidate.lifecycle == "micro-live" and not fit_ok:
            streak = self._demotion_streaks.get(candidate.name, 0) + 1
            self._demotion_streaks[candidate.name] = streak
            if streak >= 2:  # two consecutive fails
                self._promote(candidate, "shadow", demotion=True)
                self._demotion_streaks[candidate.name] = 0
        elif candidate.lifecycle == "champion" and not fit_ok:
            # single window drop demotes champion to micro-live
            self._promote(candidate, "micro-live", demotion=True)

        # Reset demotion streak if recovered
        if fit_ok and candidate.name in self._demotion_streaks:
            self._demotion_streaks[candidate.name] = 0

        # Callback triggers handled inside _promote/_demote
        if prev != candidate.lifecycle:
            return

    # -------------------------- Helpers ---------------------------------- #
    def _promote(self, c: CandidateState, new_state: str, demotion: bool = False) -> None:
        prev = c.lifecycle
        c.lifecycle = new_state
        if not demotion and self.on_promotion:
            self.on_promotion(c, prev, new_state)
        if demotion and self.on_demotion:
            self.on_demotion(c, prev, new_state)

    def _demote(self, c: CandidateState, new_state: str, reason: str = "") -> None:
        prev = c.lifecycle
        c.lifecycle = new_state
        if self.on_demotion:
            self.on_demotion(c, prev, new_state)


__all__ = ["PromotionManager"]

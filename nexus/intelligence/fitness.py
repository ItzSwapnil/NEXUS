"""Composite fitness function implementation (Spec §4).

Provides a light-weight, dependency-minimal scorer used by the exploration
controller and promotion logic. All metrics are expected to be supplied as raw
values; this module performs simple min/max style normalization with basic
safeguards so tests do not require historical buffers.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class FitnessWeights:
    alpha_sharpe: float
    alpha_sortino: float
    alpha_profit_factor: float
    alpha_payout: float
    beta_mdd: float
    beta_ulcer: float
    beta_turnover: float
    gamma_slippage: float
    gamma_constraint: float

@dataclass
class CandidateMetrics:
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 1.0
    payout: float = 80.0
    max_drawdown: float = 0.0
    ulcer_index: float = 0.0
    turnover: float = 0.0
    slippage: float = 0.0
    constraint_violations: float = 0.0

@dataclass
class CandidateState:
    name: str
    lifecycle: str = "shadow"  # shadow -> micro-live -> champion
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
    fitness: float = 0.0
    promotion_windows_ok: int = 0  # consecutive windows passing threshold

# Simple clamp for general use
def _clamp(v, lo, hi):
    return hi if v > hi else lo if v < lo else v

def _norm_positive(x: float, max_ref: float, min_ref: float = 0.0) -> float:
    if max_ref <= min_ref:
        return 0.0
    return _clamp((x - min_ref) / (max_ref - min_ref), 0.0, 1.0)

def compute_composite_fitness(m: CandidateMetrics, w: FitnessWeights) -> float:
    """Compute fitness as defined in Spec §4.

    Normalization strategy (bounded / heuristic for tests):
    - sharpe, sortino: assume range [-2, 4]; clip and map to [0,1]
    - profit_factor: map log( pf ) where pf in [0.1, 10]
    - payout: assume [50, 100]
    - max_drawdown: penalize; assume [0, 60]% drawdown -> norm
    - ulcer_index: assume [0, 50]
    - turnover: assume [0, 20]
    - slippage: assume [0, 5]%
    - constraint violations: assume [0, 10]
    """
    sharpe_n = _norm_positive(m.sharpe + 2, 6)  # shift to [0,6]
    sortino_n = _norm_positive(m.sortino + 2, 6)
    profit_factor_n = _norm_positive((m.profit_factor), 10, 0.1)
    payout_n = _norm_positive(m.payout, 100, 50)

    mdd_n = _norm_positive(m.max_drawdown, 60)
    ulcer_n = _norm_positive(m.ulcer_index, 50)
    turnover_n = _norm_positive(m.turnover, 20)
    slippage_n = _norm_positive(m.slippage, 5)
    constraint_n = _norm_positive(m.constraint_violations, 10)

    fitness = (
        w.alpha_sharpe * sharpe_n +
        w.alpha_sortino * sortino_n +
        w.alpha_profit_factor * profit_factor_n +
        w.alpha_payout * payout_n -
        w.beta_mdd * mdd_n -
        w.beta_ulcer * ulcer_n -
        w.beta_turnover * turnover_n -
        w.gamma_slippage * slippage_n -
        w.gamma_constraint * constraint_n
    )
    return float(_clamp(fitness, 0.0, 1.0))


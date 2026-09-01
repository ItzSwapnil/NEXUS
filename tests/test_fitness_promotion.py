from nexus.intelligence.fitness import (
    CandidateMetrics,
    CandidateState,
    FitnessWeights,
    compute_composite_fitness,
)
from nexus.intelligence.promotion import PromotionManager
from nexus.utils.config import NexusSettings, QuotexSettings, TradingSettings


def _make_settings():
    return NexusSettings(
        quotex=QuotexSettings(email="a@b.com", password="pw"), trading=TradingSettings()
    )


def test_fitness_ordering():
    s = _make_settings()
    w = FitnessWeights(
        alpha_sharpe=s.fitness.alpha_sharpe,
        alpha_sortino=s.fitness.alpha_sortino,
        alpha_profit_factor=s.fitness.alpha_profit_factor,
        alpha_payout=s.fitness.alpha_payout,
        beta_mdd=s.fitness.beta_mdd,
        beta_ulcer=s.fitness.beta_ulcer,
        beta_turnover=s.fitness.beta_turnover,
        gamma_slippage=s.fitness.gamma_slippage,
        gamma_constraint=s.fitness.gamma_constraint,
    )
    strong = CandidateMetrics(
        sharpe=1.5, sortino=2.0, profit_factor=2.0, payout=90.0, max_drawdown=5.0
    )
    weak = CandidateMetrics(
        sharpe=0.2, sortino=0.5, profit_factor=1.1, payout=82.0, max_drawdown=20.0
    )
    f_strong = compute_composite_fitness(strong, w)
    f_weak = compute_composite_fitness(weak, w)
    assert f_strong > f_weak
    assert 0.0 <= f_strong <= 1.0


def test_promotion_lifecycle():
    s = _make_settings()
    c = CandidateState(name="policy_X")
    pm = PromotionManager(s)

    # Promote shadow -> micro-live
    for _ in range(s.exploration.promotion_windows):
        c.fitness = s.exploration.fitness_promotion_threshold + 0.01
        pm.update_lifecycle(c)
    assert c.lifecycle == "micro-live"

    # Promote micro-live -> champion
    for _ in range(s.exploration.promotion_windows):
        c.fitness = s.exploration.fitness_promotion_threshold + 0.11
        pm.update_lifecycle(c)
    assert c.lifecycle == "champion"

    # Demote champion on a single poor window
    c.fitness = s.exploration.fitness_promotion_threshold - 0.2
    pm.update_lifecycle(c)
    assert c.lifecycle == "micro-live"

    # Demote micro-live after 2 consecutive fails
    c.fitness = s.exploration.fitness_promotion_threshold - 0.2
    pm.update_lifecycle(c)
    c.fitness = s.exploration.fitness_promotion_threshold - 0.25
    pm.update_lifecycle(c)
    assert c.lifecycle == "shadow"

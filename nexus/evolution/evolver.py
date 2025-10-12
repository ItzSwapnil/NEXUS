"""Evolutionary weight optimization for MetaStrategy.

This module provides a lightweight evolutionary loop that:
 1. Generates an initial population of ensemble weight candidates.
 2. Evaluates each candidate via the Backtester on synthetic OHLC data.
 3. Computes a composite fitness using existing fitness module heuristics.
 4. Selects top-k elites, mutates the rest, and repeats for N generations.
 5. Persists generation summaries to evolution/ directory.

Design Goals:
 - Keep runtime fast for tests (small population & rows).
 - Avoid external ML dependencies; operate purely on weight vectors.
 - Use existing Backtester metrics (profit, win_rate, drawdown, profit_factor).

Scoring Heuristic Mapping to Fitness Metrics:
 - profit_factor -> profit_factor
 - total_profit normalized -> sharpe & sortino proxies (simple transform)
 - max_drawdown -> max_drawdown
 - win_rate -> payout proxy (scaled to 50-100 range)

Future Extensions:
 - Integrate real Sharpe / Sortino calculations from equity curve.
 - Pluggable mutation strategies.
 - Parallel evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
import random
from pathlib import Path
import math
import pandas as pd

from nexus.strategies.meta_strategy import MetaStrategy, StrategyConfig
from nexus.backtest.backtester import Backtester
from nexus.core.engine import NexusEngine
from nexus.utils.config import NexusSettings
from nexus.intelligence.fitness import (
    CandidateMetrics,
    FitnessWeights,
    compute_composite_fitness,
)
CHAMPION_PATH = Path("models/meta_strategy_champion.json")
HOF_PATH = Path("evolution/hall_of_fame.json")

# ---------------------------- Data Structures ---------------------------- #

@dataclass
class EvolutionConfig:
    population_size: int = 6
    generations: int = 3
    elite_fraction: float = 0.3
    mutation_rate: float = 0.25         # probability a weight is mutated
    mutation_strength: float = 0.15     # gaussian stddev scale
    backtest_rows: int = 240
    backtest_window: int = 50
    timeframe: int = 1
    random_seed: int = 42

@dataclass
class Candidate:
    weights: Dict[str, float]
    fitness: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationResult:
    generation: int
    candidates: List[Candidate]
    best_fitness: float
    best_weights: Dict[str, float]
    path: Path

# ---------------------------- Helper Functions --------------------------- #

def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: max(v, 0.0) / total for k, v in raw.items()}

def _synthetic_ohlc(rows: int) -> pd.DataFrame:
    # Simple upward drift with minor oscillation
    base = [i + math.sin(i/10.0)*0.5 for i in range(rows)]
    data = {
        'open': [float(x) for x in base],
        'close': [float(x) + 0.2 for x in base],
        'high': [float(x) + 0.4 for x in base],
        'low': [float(x) - 0.4 for x in base],
        'volume': [1000.0 for _ in range(rows)],
    }
    return pd.DataFrame(data)

def _candidate_metrics_from_backtest(bt: Any) -> CandidateMetrics:
    # Map backtester result to expected metrics (heuristic approximations)
    # profit_factor already present; use total_profit to derive pseudo-sharpe/sortino
    profit_factor = bt.profit_factor if bt.profit_factor > 0 else 0.1
    total_profit = bt.total_profit
    win_rate = bt.win_rate
    max_dd = bt.max_drawdown
    # Sharpe/Sortino proxies scale profit by win rate to approximate risk-adjusted edge
    pseudo_sharpe = max(-2.0, min(4.0, total_profit / 100.0 + (win_rate - 0.5)))
    pseudo_sortino = pseudo_sharpe * 0.9
    payout_equiv = 50.0 + (win_rate * 50.0)  # map [0,1] win_rate -> [50,100]
    return CandidateMetrics(
        sharpe=pseudo_sharpe,
        sortino=pseudo_sortino,
        profit_factor=profit_factor,
        payout=payout_equiv,
        max_drawdown=max_dd,
    )

# ---------------------------- Evolution Runner --------------------------- #

class EvolutionRunner:
    def __init__(self, engine: NexusEngine, config: EvolutionConfig, settings: NexusSettings):
        self.engine = engine
        self.config = config
        self.settings = settings
        self.random = random.Random(config.random_seed)
        self.output_dir = Path('evolution')
        self.output_dir.mkdir(exist_ok=True)
        self.fitness_weights = FitnessWeights(
            alpha_sharpe=settings.fitness.alpha_sharpe,
            alpha_sortino=settings.fitness.alpha_sortino,
            alpha_profit_factor=settings.fitness.alpha_profit_factor,
            alpha_payout=settings.fitness.alpha_payout,
            beta_mdd=settings.fitness.beta_mdd,
            beta_ulcer=settings.fitness.beta_ulcer,
            beta_turnover=settings.fitness.beta_turnover,
            gamma_slippage=settings.fitness.gamma_slippage,
            gamma_constraint=settings.fitness.gamma_constraint,
        )

    def _init_population(self, base_weights: Dict[str, float]) -> List[Candidate]:
        pop: List[Candidate] = []
        for _ in range(self.config.population_size):
            perturbed = {
                k: max(0.01, w + self.random.gauss(0, 0.1)) for k, w in base_weights.items()
            }
            pop.append(Candidate(weights=_normalize_weights(perturbed)))
        return pop

    def _mutate(self, cand: Candidate, keys: List[str]) -> Candidate:
        new_w = cand.weights.copy()
        for k in keys:
            if self.random.random() < self.config.mutation_rate:
                new_w[k] = max(0.01, new_w[k] + self.random.gauss(0, self.config.mutation_strength))
        return Candidate(weights=_normalize_weights(new_w))

    async def _evaluate_candidate(self, candidate: Candidate, df: pd.DataFrame) -> Candidate:
        # Build a MetaStrategy instance with candidate weights
        strategy_cfg = StrategyConfig()
        strategy_cfg.ensemble_weights = candidate.weights.copy()
        meta = MetaStrategy(config=strategy_cfg)  # models omitted for speed
        bt = Backtester(window=self.config.backtest_window, expiration=60)
        result = await bt.run(meta, self.engine, df, asset="EURUSD", timeframe=self.config.timeframe)
        metrics = _candidate_metrics_from_backtest(result)
        fitness = compute_composite_fitness(metrics, self.fitness_weights)
        candidate.fitness = fitness
        candidate.metrics = {
            'total_profit': result.total_profit,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown,
            'profit_factor': result.profit_factor,
        }
        return candidate

    async def run(self) -> List[GenerationResult]:
        df = _synthetic_ohlc(self.config.backtest_rows)
        base_cfg = StrategyConfig()
        population = self._init_population(base_cfg.ensemble_weights)
        generations: List[GenerationResult] = []
        elite_count = max(1, int(self.config.elite_fraction * self.config.population_size))
        keys = list(base_cfg.ensemble_weights.keys())

        for g in range(self.config.generations):
            evaluated: List[Candidate] = []
            for cand in population:
                evaluated.append(await self._evaluate_candidate(cand, df))
            # Sort by fitness descending
            evaluated.sort(key=lambda c: c.fitness, reverse=True)
            best = evaluated[0]
            gen_path = self.output_dir / f"generation_{g}.json"
            with open(gen_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'generation': g,
                    'best_fitness': best.fitness,
                    'best_weights': best.weights,
                    'candidates': [
                        {
                            'fitness': c.fitness,
                            'weights': c.weights,
                            'metrics': c.metrics,
                        } for c in evaluated
                    ]
                }, f, indent=2)
            generations.append(GenerationResult(
                generation=g,
                candidates=evaluated,
                best_fitness=best.fitness,
                best_weights=best.weights,
                path=gen_path,
            ))
            # Prepare next population
            elites = evaluated[:elite_count]
            next_pop: List[Candidate] = [Candidate(weights=e.weights.copy(), fitness=e.fitness, metrics=e.metrics) for e in elites]
            # Fill rest with mutated offspring
            while len(next_pop) < self.config.population_size:
                parent = self.random.choice(elites)
                next_pop.append(self._mutate(parent, keys))
            population = next_pop

        # Persist final champion & update hall of fame
        if generations:
            champion = generations[-1].candidates[0]
            try:
                CHAMPION_PATH.parent.mkdir(exist_ok=True, parents=True)
                CHAMPION_PATH.write_text(json.dumps({
                    'weights': champion.weights,
                    'fitness': champion.fitness,
                    'generation': generations[-1].generation,
                }, indent=2), encoding='utf-8')
            except Exception:
                pass
            # Hall of fame aggregation (top 10 by fitness)
            hof: List[Dict[str, Any]] = []
            if HOF_PATH.exists():
                try:
                    hof = json.loads(HOF_PATH.read_text(encoding='utf-8')) or []
                except Exception:
                    hof = []
            hof.append({
                'generation': generations[-1].generation,
                'fitness': champion.fitness,
                'weights': champion.weights,
            })
            # Deduplicate by weights signature
            seen = set()
            filtered: List[Dict[str, Any]] = []
            for entry in sorted(hof, key=lambda x: x.get('fitness', 0), reverse=True):
                sig = tuple(sorted((k, round(v,4)) for k,v in entry.get('weights', {}).items()))
                if sig in seen:
                    continue
                seen.add(sig)
                filtered.append(entry)
                if len(filtered) >= 10:
                    break
            try:
                HOF_PATH.write_text(json.dumps(filtered, indent=2), encoding='utf-8')
            except Exception:
                pass
        return generations

__all__ = [
    'EvolutionConfig',
    'EvolutionRunner',
    'GenerationResult',
    'Candidate',
]

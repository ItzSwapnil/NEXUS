# Adaptive intelligence

NEXUS contains an experimental adaptive research pipeline in `nexus/intelligence/adaptive.py`. It separates market representation and forecasting from policy proposals, risk checks, experience storage, and candidate evaluation.

## Flow

```text
OHLCV observations
  → feature bundle and market representation
  → market state, regime, and novelty estimate
  → probabilistic forecast
  → policy proposal
  → risk authorization
  → backtest or broker workflow
  → settled experience and replay evaluation
```

This describes the components and data flow; it does not mean every entry point runs every step or that a candidate is production-ready.

## Main components

| Component | Role |
|---|---|
| `MarketRepresentation` | Builds and normalizes feature bundles. |
| `AdaptiveSequenceEncoder` and `AdaptiveTrainer` | Encode sequences and provide a training routine for return, regime, and volatility objectives. |
| `RegimeAndNoveltyDetector` | Estimates state and unfamiliarity from the available market features. |
| `ForecastCalibrator` | Adjusts forecast uncertainty using observed calibration data. |
| `RiskPolicy` | Applies exposure, drawdown, and novelty limits independently of the policy proposal. |
| `ExperienceStore` | Persists observations and later outcome feedback in SQLite. |
| `WalkForwardEvaluator` and `ResearchLoop` | Create time-ordered evaluation splits and apply candidate promotion gates. |
| `ContinualLearningService` | Coordinates observations, outcome settlement, and replay-based shadow candidate updates. |

`AdaptiveMarketIntelligence` composes the representation, state detector, calibrator, risk policy, experience store, and research gate. It can load a trained encoder checkpoint through `load_encoder`.

These implementations are experimental and may evolve. Check the source docstrings and script options when integrating them.

## Learning boundary

`AdaptiveMarketIntelligence.record_outcome` stores outcome feedback. It does not immediately train or replace a model. Training and candidate evaluation are separate operations so a single result cannot directly update the active model.

The included training utility defaults to deterministic synthetic data. A local CSV can be supplied with `--csv`; `--broker` fetches recent candles through a configured broker session. The script writes checkpoint and metadata files under `models/adaptive/`, which are intentionally ignored by Git.

```bash
uv run python scripts/train_adaptive.py --help
uv run python scripts/train_adaptive.py --csv path/to/candles.csv
```

## Limitations

- Synthetic-data training is useful for exercising the pipeline, not for evidence of trading performance.
- Walk-forward evaluation reduces some forms of leakage but cannot establish future returns.
- A risk authorization is one software control and does not remove market, execution, or broker risk.
- Do not treat a model artifact as approved for live trading merely because it trained or passed a local check.

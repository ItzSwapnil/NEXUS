# Local model artifacts

`models/` holds locally generated weights, checkpoints, and model metadata. These files are ignored by Git because they can be large, change during research, and may include derived training data or results. This README is kept in the repository to document the directory.

## Model code

NEXUS contains several generations of model code:

- The adaptive research components are in `nexus/intelligence/`. They include feature representation, market state, probabilistic forecasts, risk-aware policy decisions, replay memory, and candidate evaluation.
- The earlier LSTM, DQN, and ensemble components remain in `nexus/ai/` for existing workflows and checkpoint compatibility.

See [Adaptive intelligence](../docs/ADAPTIVE_INTELLIGENCE.md) for the newer research pipeline and [Technical overview](../docs/OVERVIEW.md) for the code map.

## Training and checkpoints

The legacy training entry point can be started with:

```bash
uv run python -m nexus.ai.train_models
```

It uses synthetic data in its example training-data path and may write model checkpoints into `models/`. Review the training code and output paths before running it. Do not assume the resulting checkpoints are suitable for live trading.

Other research and training scripts are listed in [scripts/README.md](../scripts/README.md). Script options can change; check `--help` or the source before a long-running job.

## Managing artifacts

- Keep checkpoints, serialized weights, and generated model metadata out of Git.
- Back up artifacts you need; deleting this directory can remove local research results.
- Record the code revision and data provenance alongside any model you intend to reproduce.
- Evaluate candidates on held-out, time-ordered data before considering promotion.

# NEXUS technical overview

NEXUS is a Python research and trading-system project. The codebase includes market-data providers, forecasting and adaptive research components, backtesting, risk controls, broker adapters, and web and CLI entry points. The implementation is experimental; this document describes the current source tree rather than a production readiness claim.

## Start the project

The project currently requires CPython 3.15 (`>=3.15,<3.16`). From the repository root:

```bash
uv sync --all-extras
uv run python run.py
```

Copy `.env.example` to `.env` for local configuration. Keep account credentials and tokens out of Git. See [Getting Started](../GETTING_STARTED.md) for platform-specific setup.

## Main components

| Area | Location | Responsibility |
|---|---|---|
| Entry points | `run.py`, `nexus/main.py`, `nexus/cli.py` | Interactive launcher, web app, and command-line interface |
| Broker integration | `nexus/adapters/` | Quotex adapter and compatibility code |
| Market data | `nexus/data/`, `nexus/catalog/` | OHLCV retrieval, caching, and market catalog management |
| Adaptive intelligence | `nexus/intelligence/` | Representation, forecasts, policy and risk decisions, replay memory, and research evaluation |
| Legacy AI | `nexus/ai/` | LSTM, DQN, ensemble, and training components retained alongside newer work |
| Backtesting | `nexus/backtest/` | Historical replay and result aggregation |
| Web interface | `nexus/web/` | Dashboard, API, and static client assets |
| Tests | `tests/` | Unit and integration coverage |

## Adaptive research flow

The newer adaptive components are organized around a causal research pipeline:

```text
market observations
  → features and representation
  → market state and probabilistic forecast
  → policy proposal
  → independent risk decision
  → execution or paper observation
  → settled outcome and replay research
```

An outcome is stored as experience; it does not by itself update a production model. Replay and evaluation are separate steps. See [Adaptive intelligence](ADAPTIVE_INTELLIGENCE.md) for the component-level description.

## Run modes

```bash
uv run python -m nexus.main --web --demo             # Web dashboard
uv run python -m nexus.main --cli --demo --stats     # CLI with statistics
uv run python -m nexus.main --cli --backtest --demo  # Backtest
uv run python -m nexus.cli                           # CLI module
```

The main entry point defaults to the web dashboard and demo mode. Broker-backed workflows still require valid account setup; demo mode can contact the broker and should not be mistaken for a fully offline simulation.

## Configuration and generated files

Configuration is loaded from environment variables and `.env`; supported settings are defined in `nexus/utils/config.py` and the related settings models. Start from `.env.example`, and check the code before relying on a setting name or default.

Caches, logs, reports, session data, local databases, and trained weights are runtime artifacts and are ignored by Git. The repository keeps small README files in `data/` and `models/` to explain those directories.

## Development

```bash
uv sync --all-extras
uv run pytest
```

Lint and type-check commands are documented in [Contributing](../CONTRIBUTING.md). Test counts are intentionally omitted because they change as the suite changes.

## Scope and limitations

NEXUS includes experimental research and broker-connected code. Backtests, model outputs, and demo results are not evidence of future returns. Review [Security](../SECURITY.md), the broker workflow, and applicable terms before connecting an account.

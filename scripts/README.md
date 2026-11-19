# Script guide

Scripts are convenience entry points, and their options may change. Check `--help` or review the source before starting a long-running job. Unless a section says otherwise, run Python scripts from the repository root with `uv run python scripts/<name>.py`.

> **Broker access:** Research and account utilities may log in to Quotex or download broker data. Trade-test and autonomous demo scripts may submit PRACTICE orders. Confirm the script’s behavior and account mode before running it.

## Setup and maintenance

| Script | Purpose |
|---|---|
| `verify_project.py` | Imports selected dependencies and initializes selected models and the engine. It does not check broker access or require a GPU. |
| `clean_workspace.py` | Removes Python and tool caches, local log files, and build artifacts. Review the script before use. |
| `download_models.py` | Downloads the configured Chronos forecasting model and runs a sample inference. Requires network access and can use significant disk space. |

```bash
uv run python scripts/verify_project.py
uv run python scripts/clean_workspace.py
```

## Local research and training

These scripts use local or synthetic data and produce local reports or model artifacts:

| Script | Purpose |
|---|---|
| `run_local_market_sweep.py` | Trains and evaluates the adaptive core using synthetic data; writes a report under `reports/`. |
| `train_adaptive.py` | Trains the adaptive sequence encoder. Defaults to synthetic data; accepts `--csv` for a local CSV or `--broker` to fetch candles. Writes artifacts under `models/adaptive/`. |
| `train_indicators.py` | Runs the indicator training/selection workflow. |
| `train_indicator_foundation.py` | Builds the indicator foundation artifact. |
| `run_batched_research.py` | Logs in to the broker to refresh markets and fetch candles, then performs local replay. The script records `orders_placed: 0`. |
| `run_robust_research.py` | Robust research workflow used by the batched runner; review its options before running it directly. |
| `run_parallel_research.py` | Splits broker-data research across workers. It can start a browser login refresh, then fetches broker data and writes local reports; it is documented in code as submitting no orders. |
| `enrich_research_report.py`, `normalize_research_report.py`, `generate_confidence_chart.py` | Transform or visualize existing research output. |

Example commands:

```bash
uv run python scripts/run_local_market_sweep.py
uv run python scripts/train_adaptive.py --help
uv run python scripts/run_batched_research.py --min-payout 80
```

## Broker-connected utilities

| Script | Effect |
|---|---|
| `run_broker_market_backtest.py` | Fetches broker market data and runs local backtesting; it is intended to place no orders. |
| `fetch_balance.py` | Logs in to the configured broker account and reads its balance. |
| `login_browser.py` | Performs browser-assisted login and may save session material locally. Keep resulting files private. |
| `test_otc_trades.py`, `test_eurjpy_ai_trade.py`, `run_1h_trade_test.py` | Trade-test workflows that may place broker-backed PRACTICE orders. Inspect options and confirm demo account mode first. |
| `start_1h_test.sh` | Starts the one-hour trade-test workflow. |
| `run_autonomous_demo_loop.sh` | Repeats research and broker-backed PRACTICE workflows; may place demo orders under its configured conditions. |
| `run_unattended_research.sh` | Resumable broker-data research screen; inspect the runner’s environment settings before use. |
| `start_appdeploy_tunnel.sh` | Exposes the configured local web service through Cloudflare Tunnel; requires API token protection. See [deployment notes](../docs/APPDEPLOY.md). |

Do not confuse a broker-backed research or demo script with an offline simulation. Store credentials only in local configuration and review sensitive output before sharing.

For project architecture, see [Technical overview](../docs/OVERVIEW.md) and [Adaptive intelligence](../docs/ADAPTIVE_INTELLIGENCE.md). For the main setup, see [Getting Started](../GETTING_STARTED.md).

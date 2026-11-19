<div align="center">

# NEXUS

### Adaptive market intelligence, research, and risk-aware trading tools

[![Python 3.15](https://img.shields.io/badge/Python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-7A5?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Author: Swapnil](https://img.shields.io/badge/Author-Swapnil%20De%20Sarkar-6C63FF)](https://github.com/ItzSwapnil)

**Research carefully. Manage risk. Learn from the market.**

[Getting started](GETTING_STARTED.md) · [Quick install](QUICK_INSTALL.md) · [Architecture](docs/OVERVIEW.md) · [Security](SECURITY.md) · [Contributing](CONTRIBUTING.md)

</div>

---

NEXUS is an experimental Python platform for market research and trading-system development. It brings together adaptive forecasting, walk-forward evaluation, risk controls, backtesting, and a Quotex adapter in one project.

## ✨ What’s inside

| | Capability | What it does |
|---|---|---|
| 🧠 | **Adaptive intelligence** | Learns sequence representations, estimates outcomes probabilistically, and tracks market regimes and unfamiliar conditions. |
| 🔬 | **Research workflows** | Runs backtests and walk-forward studies, then evaluates candidates through controlled promotion gates. |
| 🛡️ | **Risk controls** | Applies exposure and drawdown limits outside the model’s predictions. |
| 📊 | **Trading tools** | Includes a command-line interface, optional desktop dashboard, and Quotex integration. |

> **Experimental software:** Model and backtest results do not predict future performance. Start in demo mode and review the code and broker requirements before use.

## 🚀 Get started

### Requirements

- CPython **3.15** (`pyproject.toml` requires `>=3.15,<3.16`)
- [`uv`](https://docs.astral.sh/uv/)

### Install

```bash
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS
uv sync --all-extras
cp .env.example .env       # Windows PowerShell: copy .env.example .env
```

Add your local settings to `.env`. The checked-in `.env.example` is a placeholder template; never put broker credentials or tokens in source control.

### Launch

```bash
uv run python run.py
```

<details>
<summary><strong>Other ways to run NEXUS</strong></summary>

```bash
uv run python -m nexus.cli                         # Command-line interface
uv run python -m nexus.main --web --demo           # Web dashboard in demo mode
uv run python -m nexus.main --cli --backtest --demo # Demo backtest
```

</details>

## 🧭 Project guide

| Guide | Find here |
|---|---|
| [Getting started](GETTING_STARTED.md) | Installation and first steps |
| [Technical overview](docs/OVERVIEW.md) | Modules and system architecture |
| [Adaptive intelligence](docs/ADAPTIVE_INTELLIGENCE.md) | Forecasting, replay, and evaluation pipeline |
| [AppDeploy notes](docs/APPDEPLOY.md) | Deployment boundaries and API access |
| [Utility scripts](scripts/README.md) | Research, training, and maintenance commands |
| [Security policy](SECURITY.md) | Credential handling and vulnerability reporting |
| [Contributing](CONTRIBUTING.md) | How to propose and prepare changes |

## 🔬 Research and demo runs

Run a bounded payout-filtered research screen:

```bash
uv run python scripts/run_batched_research.py --min-payout 80
```

This runner logs in to Quotex and downloads broker data, then evaluates it locally; it records zero orders. Generated reports and local research data are kept out of Git. The optional autonomous demo loop can submit broker-backed **PRACTICE** trades when its configured conditions allow. Review [the runner](scripts/run_autonomous_demo_loop.sh) and its requirements before starting it. The standard launch does not start this loop.

## 🧑‍💻 Development

Install the project and development extras with `uv sync --all-extras`, then run the suite:

```bash
uv run pytest
```

## 🔒 Local files

`.env`, browser sessions, caches, reports, databases, logs, and trained weights are local runtime artifacts and are ignored by Git. See [data/README.md](data/README.md) and [models/README.md](models/README.md) for details. Keep `.env.example` limited to safe placeholders.

## ⚠️ Disclaimer

NEXUS is intended for software development and research; it is not financial advice. Trading involves risk and can result in loss. Backtests and model outputs do not guarantee future results. Follow applicable laws and broker terms.

## 📄 License

MIT License. See [LICENSE](LICENSE).

<div align="center">

**Made by [Swapnil De Sarkar](https://github.com/ItzSwapnil)**

</div>

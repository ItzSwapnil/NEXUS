# Getting started

This guide takes you from a fresh checkout to a local NEXUS run. For a shorter overview, see [README.md](README.md); for advanced install options, see [QUICK_INSTALL.md](QUICK_INSTALL.md).

## Requirements

- CPython 3.15 (`pyproject.toml` requires `>=3.15,<3.16`)
- [`uv`](https://docs.astral.sh/uv/)
- Windows, macOS, or Linux

## Install

Install `uv` using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/), then clone and sync the project:

```bash
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS
uv sync --all-extras
```

`uv sync --all-extras` creates or updates the local `.venv` and installs all declared project extras.

## Configure local settings

Create a private local configuration file:

```bash
cp .env.example .env
```

In Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Edit `.env` for your setup. The template contains placeholders only. Never commit broker credentials, cookies, session IDs, or API tokens. NEXUS defaults to demo mode, but broker-backed demo features can still connect to your account.

## Run NEXUS

Launch the interactive menu:

```bash
uv run python run.py
```

Or start a mode directly:

```bash
uv run python -m nexus.main --web --demo             # Web dashboard
uv run python -m nexus.main --cli --demo --stats     # CLI and statistics
uv run python -m nexus.main --cli --backtest --demo  # Backtest
```

The web dashboard is the default for `nexus.main`; the `--demo` flag is also the default. A demo run is not necessarily offline: workflows that use broker data or place PRACTICE trades still communicate with the broker.

## Verify the setup

Run the project’s readiness script and test suite:

```bash
uv run python scripts/verify_project.py
uv run pytest
```

The verification script imports the core dependencies and initializes selected models and the engine. It does not validate broker credentials or guarantee that every optional feature works on every host.

## Data and generated files

Local caches, logs, reports, browser sessions, databases, and trained weights are ignored by Git. They can contain sensitive or account-related information, so inspect and sanitize them before sharing. See [data/README.md](data/README.md) and [models/README.md](models/README.md).

The workspace cleaner removes caches and temporary files. Review its behavior before using it:

```bash
uv run python scripts/clean_workspace.py
```

## Troubleshooting

### Recreate the environment

```bash
uv sync --all-extras --reinstall
```

If needed, remove `.venv` first (`Remove-Item .venv -Recurse -Force` in PowerShell, or `rm -rf .venv` on macOS/Linux), then run the sync command again.

### Find local logs

NEXUS writes local log files under `logs/`. Review them before sharing because errors or account-related details may be included.

### Get help

- Read the [technical overview](docs/OVERVIEW.md) and [script guide](scripts/README.md).
- Check the [security policy](SECURITY.md) before reporting security issues.
- Review [Contributing](CONTRIBUTING.md) before proposing a code change.

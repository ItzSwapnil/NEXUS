# Quick install

NEXUS uses [`uv`](https://docs.astral.sh/uv/) to manage its Python environment and dependencies.

## Requirements

- CPython 3.15 (`>=3.15,<3.16`)
- `uv`

Install `uv` with the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS
uv sync --all-extras
cp .env.example .env
```

On Windows PowerShell, replace the last command with `Copy-Item .env.example .env`. Edit `.env` locally and keep all real credentials and tokens out of Git.

Start NEXUS:

```bash
uv run python run.py
```

## Common commands

```bash
uv run python -m nexus.main --web --demo             # Web dashboard
uv run python -m nexus.main --cli --demo --stats     # CLI and statistics
uv run python -m nexus.main --cli --backtest --demo  # Backtest
uv run python scripts/verify_project.py              # Readiness checks
uv run pytest                                        # Test suite
```

`uv sync --all-extras` installs all optional dependencies declared by the project, including GUI, AI, and development/testing extras. If you need a smaller installation, review the extras in `pyproject.toml`; the core dependency set still includes several data and web packages.

## Environment maintenance

To update dependencies within the project’s declared version ranges:

```bash
uv sync --all-extras --upgrade
```

To reinstall the declared environment:

```bash
uv sync --all-extras --reinstall
```

See [Getting Started](GETTING_STARTED.md) for configuration and troubleshooting, and [Contributing](CONTRIBUTING.md) for development guidance.

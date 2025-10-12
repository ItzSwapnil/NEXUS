# Getting Started with NEXUS

Quick setup guide for the autonomous AI trading system.

## Prerequisites

- Python 3.12 or higher
- Windows, Linux, or macOS
- UV package manager (recommended) or pip

## Installation

### Quick Setup

```bash
# 1. Install UV package manager
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS

# 3. Create environment and install
uv venv
.venv\Scripts\activate  # Windows
uv pip install -e ".[gui,ta]"

# 4. Configure
copy .env.example .env  # Edit with your broker credentials

# 5. Run
python run.py
```

# Create .env from template
copy .env.example .env    # Windows
cp .env.example .env      # Linux/macOS
```

## Configuration

Edit `.env` file with your settings:

```env
# Broker Credentials
QUOTEX_EMAIL=your_quotex_email
QUOTEX_PASSWORD=your_password

# Trading Mode (always start with demo!)
TRADING_MODE=demo

# Trading Settings
DEFAULT_ASSET=EURUSD
BASE_TRADE_AMOUNT=10.0
DEFAULT_EXPIRATION=60

# AI/ML Settings
ENABLE_AI_MODELS=true
MODEL_TRAINING_ENABLED=false

# Risk Management
MAX_DAILY_LOSS=500.0
MAX_DRAWDOWN_PERCENT=20.0
USE_KELLY_CRITERION=true
```

## First Steps

### 1. Test the Installation

```bash
# Run tests to verify everything works
pytest -v
```

### 2. Launch Demo Mode

```bash
# Using launcher (interactive menu)
python run.py

# Or directly
python -m nexus.main --demo
```

### 3. Run a Backtest

```bash
# Test strategies on historical data
python -m nexus.main --backtest
```

### 4. Train AI Models (Optional)

```bash
# Train the deep learning models
python -m nexus.ai.train_models
```

## Usage Modes

### GUI Mode
```bash
python -m nexus.main --demo
```
- Visual dashboard
- Real-time monitoring
- Easy to use

### CLI Mode
```bash
python -m nexus.cli
```
- Command-line interface
- Performance statistics
- Lightweight

### Backtest Mode
```bash
python -m nexus.main --backtest
```
- Test strategies on historical data
- Performance analysis
- No real trading

## Common Tasks

### Update Dependencies
```bash
uv pip install --upgrade -e ".[ai,gui,ta]"
```

### View Logs
```bash
# Latest log file
# Windows (PowerShell):
Get-Content logs\nexus_*.log -Tail 50

# Linux/macOS:
tail -f logs/nexus_*.log
```

### Clean Project
```bash
# Using utility script
python scripts/clean_workspace.py

# Or via launcher
python run.py  # Select clean option
```

## Safety First

⚠️ **IMPORTANT SAFETY GUIDELINES:**

1. **Always start with DEMO mode**
2. **Never trade with money you can't afford to lose**
3. **Test strategies thoroughly before live trading**
4. **Set appropriate risk limits in .env**
5. **Monitor trades regularly**
6. **Keep credentials secure**

## Troubleshooting

### Virtual Environment Issues
```bash
# Recreate virtual environment
# Windows (PowerShell):
Remove-Item .venv -Recurse -Force

# Linux/macOS:
rm -rf .venv

# Reinstall
python setup.py
```

### Import Errors
```bash
# Reinstall in development mode
uv pip install -e ".[ai,gui,ta]"
```

### Test Failures
```bash
# Run tests with verbose output
pytest -v --tb=short
```

### UV Not Found
```bash
# Install UV
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Next Steps

1. Read the [README.md](README.md) for detailed features
2. Explore the [docs/OVERVIEW.md](docs/OVERVIEW.md) to understand the system
3. Start with demo mode and small trade amounts
4. Monitor performance and adjust settings
5. Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Getting Help

- **Documentation**: See [docs/](docs/) directory
- **Overview**: [docs/OVERVIEW.md](docs/OVERVIEW.md)
- **Security**: [SECURITY.md](SECURITY.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

For issues or questions:
1. Check existing documentation
2. Review test output for errors
3. Check log files in `logs/` directory
4. Ensure .env is properly configured
5. Open an issue on GitHub

---

**Welcome to NEXUS!** Start with demo mode and explore the features safely.


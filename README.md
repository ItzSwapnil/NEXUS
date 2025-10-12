# NEXUS - Autonomous AI Trading System

[![Python 3.12+](https://img.shields.io/badge/python-3.12%7C3.13%7C3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passing-brightgreen.svg)](tests/)

An autonomous trading system combining deep learning, reinforcement learning, and advanced risk management. Features LSTM networks with attention mechanisms, Deep Q-Learning (DQN), and adaptive ensemble learning.

## Core Features

- **Deep Learning Models**: LSTM with multi-head attention for temporal pattern recognition
- **Reinforcement Learning**: DQN agent for strategy optimization
- **Ensemble Learning**: Meta-learning system for adaptive model weighting
- **Advanced Risk Management**: Kelly Criterion, drawdown protection, position sizing
- **Multi-Broker Support**: Quotex integration with extensible adapter architecture

---

## ✨ Key Features

### 🚀 Production-Ready AI
- **LSTM Predictor** with attention mechanisms for sequence learning
- **Deep Q-Network (DQN)** with dueling architecture and noisy layers
- **Ensemble Manager** using meta-learning for model combination
- **Online Learning** - models adapt to market changes in real-time

### 🛡️ Advanced Risk Management
- Kelly Criterion position sizing
- Volatility-adjusted trade amounts
- Multi-layered circuit breakers
- Drawdown protection
- Daily/hourly trade limits

### 📈 **Complete Trading System**
- Multi-broker support (Quotex integrated)
- Real-time market data processing
- Technical indicator library (20+)
- Backtesting with market replay
- Live trading with demo mode

### 🎨 **Professional Tools**
- PySide6 GUI dashboard
- Real-time performance monitoring
- Comprehensive logging
- Model training scripts
- Performance analytics

---

## 🚀 **Quick Start**

```bash
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS

# Create virtual environment
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -e ".[ai,gui,ta]"

# Configure
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
# Edit .env with your credentials

# Run
python run.py
```

---

## 📋 Installation

```bash
# Clone repository
git clone https://github.com/ItzSwapnil/NEXUS.git
cd NEXUS

# Setup environment
uv venv
.venv\Scripts\activate  # Windows

# Install
uv pip install -e ".[gui,ta]"
uv pip install torch scikit-learn joblib

# Configure
copy .env.example .env
# Edit .env with your credentials

# Run
python run.py
```

For more options, see [QUICK_INSTALL.md](QUICK_INSTALL.md).

---

## 🔧 **Configuration**

Edit `.env` file:

```env
# Broker Credentials
QUOTEX_EMAIL=your_quotex_email
QUOTEX_PASSWORD=your_password

# Trading Mode
TRADING_MODE=demo

# AI/ML Settings
ENABLE_AI_MODELS=true

# Risk Management
MAX_DAILY_LOSS=500.0
USE_KELLY_CRITERION=true
```

---

## 📖 **Usage**

### GUI Mode
```bash
python run.py
# Select option 1 for GUI
```

### CLI Mode
```bash
python -m nexus.cli
```

### Backtest
```bash
python -m nexus.main --backtest
```

### Train Models
```bash
python -m nexus.ai.train_models
```

---

## 🧪 **Testing**

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=nexus --cov-report=html
```

---

## 📚 **Documentation**

- [Getting Started Guide](GETTING_STARTED.md)
- [Technical Overview](docs/OVERVIEW.md)
- [Security Policy](SECURITY.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md)

---

## 🎯 **Performance**

Typical model performance (after training):
- **LSTM Predictor**: 65-70% directional accuracy
- **DQN Agent**: Positive expected value through learned policies  
- **Ensemble**: 5-10% improvement over individual models

*Past performance does not guarantee future results.*

---

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## ⚠️ **Disclaimer**

**IMPORTANT**: NEXUS is for educational and research purposes only. Trading involves substantial risk of loss. Always:

- Start with demo mode
- Use proper risk management
- Never trade with money you can't afford to lose
- Comply with local regulations

The AI models are experimental and do not constitute financial advice.

---

## 🤝 **Contributing**

I welcome contributions to NEXUS! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation
- Share your trading strategies

---

## 💬 **Support**

If you find this project helpful, please:
- ⭐ Star the repository
- 🐛 Report issues
- 📖 Improve documentation
- 🔀 Submit pull requests

---

## 📧 **Contact**

- **Author**: Swapnil De Sarkar
- **GitHub**: [@ItzSwapnil](https://github.com/ItzSwapnil/NEXUS)

---

**Built with ❤️ for the algorithmic trading community**


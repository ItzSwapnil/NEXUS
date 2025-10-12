# NEXUS Documentation Overview

Comprehensive overview of the NEXUS autonomous AI trading system architecture and implementation.

## 1. Purpose & Scope
NEXUS is a production-ready trading system focusing on:
- Deep learning AI models (LSTM with attention, DQN agent, ensemble learning)
- Advanced risk management and position sizing
- Multi-broker integration architecture
- Real-time market data processing and technical analysis
- Comprehensive backtesting framework
- PySide6 GUI dashboard with real-time monitoring
- 51+ unit tests covering core functionality

## 2. Quick Start
```bash
uv venv .venv
.venv\Scripts\activate   # Windows
uv pip install -e ".[gui,ta]"
pytest -v              # Run test suite
python run.py          # Launch GUI
```

## 3. Configuration
Key .env configuration fields:
- QUOTEX_EMAIL / QUOTEX_PASSWORD - Broker credentials
- TRADING_MODE - demo or live
- DEFAULT_ASSET / DEFAULT_EXPIRATION / BASE_TRADE_AMOUNT
- MAX_RISK_PER_TRADE_PERCENT - Risk management
- ENABLE_AI_MODELS - Enable/disable AI predictions
- USE_KELLY_CRITERION - Advanced position sizing

## 4. Core Modules
| Module | Description |
|--------|-------------|
| catalog.ingest | Static placeholder markets & payouts |
| ai.lstm_predictor | LSTM with attention for price prediction |
| ai.deep_rl_agent | Deep Q-Network agent for strategy learning |
| ai.ensemble_manager | Meta-learning model combination |
| core.engine | Trading engine with risk management |
| strategies.meta_strategy | Strategy orchestration |
| adapters.quotex_adapter | Quotex broker integration |
| gui.main_window | PySide6 dashboard implementation |
| intelligence.* | Regime detection, fitness evaluation, exploration |
## 5. Data Shapes
- Market: `symbol, asset_type, payout_per_expiration, display_payout_percent, otc`
- Trade result: `{success, profit, asset, direction, expiration, real_executed}`
- Engine stats: `{total_trades, winning_trades, losing_trades, total_profit}`
- Emotion state: greed/fear/confidence ∈ [0,1]

## 6. Safety Controls
| Control | Effect |
|---------|-------|
| Payout Threshold | Blocks (future real) trades if payout < threshold |
| Override | Bypass guard (audit log) |
| Demo Mode | All trades simulated |
| PANIC STOP | Forces demo mode + disables override |

## 7. GUI (Implemented Elements)
- Markets table + payout filter
- Demo mode toggle, override button, panic stop
- Refresh catalog, test trade button
- Autonomy (epsilon base) slider
- Stats + epsilon labels
(Deferred: live balance, charts, asset selection, strategy cards.)

## 8. Epsilon Formula
```
confidence = mean(norm(sharpe), norm(stability), norm(win_rate))
uncertainty = weighted(atr, disagreement, spread, otc)
modifier from payout reduces exploration for higher payouts
clamped to [min_epsilon, max_epsilon]
```
Display only (no routing yet).

## 9. Roadmap (Condensed)
1. Real pyquotex adapter integration
2. Live catalog ingestion
3. Persistence (DuckDB) for trades & payouts
4. Backtester (replay + metrics parity)
5. Predictive model baseline (statistical / ML)
6. Strategy lifecycle (shadow → micro-live → champion)
7. Risk extensions (drawdown & daily loss guards)
8. GUI expansions (asset selection, history panel, charts, balance)
9. Structured JSON logging & async task manager
10. Incremental RL / evolution features

## 10. Out of Scope (Now)
- Credential encryption
- Multi-broker abstraction
- Complex RL/evolution pipelines
- Distributed or high-performance compute optimizations

## 11. Testing
Run: `pytest -q` (23 passing). GUI not covered by tests yet.
Run: `pytest -v` (51+ passing tests covering core functionality).
## 12. Contributing
Small, focused PRs; update docs only when a feature is implemented. See CONTRIBUTING.md.

## 13. Glossary
| Term | Definition |
|------|------------|
| OTC | Over-the-counter (higher uncertainty) |
| Override | Manual guard bypass (audited) |
| Epsilon | Exploration rate figure (display) |

## 14. Disclaimer
Prototype / simulation. Not suitable for real trading decisions without further development.

---
This unified overview replaces fragmented earlier docs; keep it tight & factual.

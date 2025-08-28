# NEXUS Documentation Overview

This single document unifies the previously scattered specs (SPEC.md, ARCHITECTURE.md, GUI.md, USAGE.md) into one concise reference. The older files now just point here.

## 1. Purpose & Current Scope
Lightweight prototype focusing on:
- Placeholder in‑memory market catalog (payouts + OTC flag)
- Payout guard (threshold + audited override)
- Minimal NexusEngine (registries, emotional state, simple risk sizing, simulated trades)
- Exploration epsilon calculation (display only)
- PySide6 GUI dashboard with basic controls & stats
- 22 unit tests covering core logic

## 2. Quick Start
```bash
uv venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
test -f .venv/bin/activate && source .venv/bin/activate  # (optional POSIX)
uv pip install -e .
pytest -q              # Expect 22 passed
python main.py         # or: python -m nexus.main
```
If console script is installed: `nexus`.

## 3. Configuration (Used Today)
Key config.yaml fields in active use:
- quotex.email / password (stored only, not used by simulator)
- trading.payout_threshold (default 80.0)
- trading.default_asset / default_expiration / base_trade_amount
- trading.max_risk_per_trade_percent
- trading.payout_poll_interval_seconds
- exploration.base_epsilon (+ other exploration bounds)

## 4. Core Modules
| Module | Description |
|--------|-------------|
| catalog.ingest | Static placeholder markets & payouts |
| payouts.fetch | Payout lookup, threshold check, override audit |
| core.engine | Simulated trade execution + stats & emotions |
| intelligence.exploration | Epsilon computation function |
| intelligence.fitness / promotion | Tested stubs only |
| gui.main_window | PySide6 main window implementation |

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
Run: `pytest -q` (22 passing). GUI not covered by tests yet.

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

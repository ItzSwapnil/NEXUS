# NEXUS Software Requirements Specification (SRS)

Version: 0.1 (Aligned with current codebase – lightweight alpha)
Status: Living document (update only when implementation changes)

## 1. Introduction
### 1.1 Purpose
This SRS describes the currently implemented functionality and near-term roadmap of the NEXUS trading research sandbox. It aims to replace prior sprawling docs with a precise, auditable reference for contributors.

### 1.2 Scope
Implemented scope (alpha):
- Deterministic in‑memory market catalog (placeholder) with payouts & OTC flag
- Payout guard with configurable threshold & audited manual override
- Minimal NexusEngine (registries, emotional state, simulated trade execution, basic risk sizing)
- Exploration epsilon computation (confidence + uncertainty + payout)
- Fitness & promotion stubs (unit tested scaffolds – no live lifecycle routing)
- PySide6 GUI dashboard for core controls / monitoring
- Logging (payout override audit)

Out-of-scope (yet): live broker execution, real-time data ingestion, persistence, predictive modeling, strategy lifecycle orchestration.

### 1.3 Definitions, Acronyms, Abbreviations
| Term | Definition |
|------|------------|
| OTC | Over-the-counter instrument (flagged higher uncertainty) |
| Payout Guard | Rule preventing real trades below configured payout unless override active |
| Override | Manual bypass of payout guard (logged) |
| Epsilon | Exploration rate indicator (display only at this stage) |
| Candidate | Strategy entity placeholder for future lifecycle management |

### 1.4 References
- Code modules in `nexus/` tree (see Traceability §10)
- Tests in `tests/` (21 passing at time of writing)

### 1.5 Overview
Sections 2–5 describe the system context, features, data, and constraints; Sections 6–10 address non-functional concerns, future scope, and traceability.

## 2. Overall Description
### 2.1 Product Perspective
Standalone local Python application using simulated logic with optional GUI. No external dependencies for live data in current operation (catalog mocked). No database persistence yet.

### 2.2 Product Functions (Implemented)
- Load config (Pydantic settings + fallback creation)
- Provide market catalog (placeholder deterministic list)
- Compute payouts per expiration & enforce threshold
- Record simulated trades and maintain stats/emotional state
- Compute exploration epsilon (display value)
- Basic fitness & promotion scoring for future expansion
- GUI interaction: toggles, filter, override, panic stop, test trade, epsilon display

### 2.3 User Classes & Characteristics
| User Class | Need |
|-----------|------|
| Researcher / Dev | Extend engine, prototype strategies |
| Tester | Validate guard logic & UI behavior |
| Future Trader | (Deferred) Real execution once adapter integrated |

### 2.4 Operating Environment
- Python 3.13.6
- Windows / cross-platform (no OS-specific code in core) – GUI requires Qt runtime.

### 2.5 Design & Implementation Constraints
- Must keep deterministic placeholder data for test stability.
- No network dependency for unit tests.
- GUI optional; test suite excludes Qt.

### 2.6 Assumptions & Dependencies
Assumptions:
- Users run in a virtual environment.
- Payout threshold enforcement will be extended (real adapter) without breaking current API.
Dependencies:
- PySide6 (GUI), Pydantic, pytest, numpy/pandas (risk module), logging.

## 3. System Features
### 3.1 Market Catalog
Description: Provide a static list of markets with per-expiration payouts.
Inputs: None (internal constant).
Outputs: List[Market]; each market has `effective_payout(expiration)`.
Errors: Unknown symbol returns None / payout 0.0.
Modules: `nexus.catalog.ingest`
Tests: `test_catalog.py`

### 3.2 Payout Guard & Override
Description: Block real trade execution when payout < threshold unless override enabled.
Data: `payout_override.log` (JSON lines).
Modules: `nexus.payouts.fetch`
Tests: `test_payouts.py`

### 3.3 NexusEngine (Core Simulation)
Responsibilities: registries, emotional state update, simple risk sizing, trade execution simulation, stats.
Modules: `nexus.core.engine`
Tests: `test_engine.py`

### 3.4 Exploration Epsilon
Computes epsilon from confidence + uncertainty metrics + payout (formula in code).
Modules: `nexus.intelligence.exploration`
Tests: `test_intelligence.py`

### 3.5 Fitness & Promotion Stubs
Simple composite fitness function + state transition logic (shadow/micro-live/champion) for future use.
Modules: `nexus.intelligence.fitness`, `nexus.intelligence.promotion`
Tests: `test_fitness_promotion.py`

### 3.6 GUI Dashboard
Controls: Demo Mode, Payout Filter, Override toggle, Panic Stop, Refresh, Test Trade, Autonomy slider, Stats/Epsilon display.
Modules: `nexus.gui.main_window`, `nexus.gui.launch_gui`
Tests: Not automated (manual). Future: add smoke test.

### 3.7 Dynamic Trading Engine Scaffold (Experimental)
Module: `nexus.core.dynamic_engine` – currently hardened stub (no test coverage) providing future adaptation points.

## 4. External Interface Requirements
| Interface | Description |
|----------|-------------|
| Config File (`config.yaml`) | Auto-created; Pydantic settings loader |
| CLI / Entry | `python main.py` or `python -m nexus.main` or console script `nexus` |
| Log File | `logs/payout_override.log` (append-only audit) |
| GUI | Desktop Qt window (manual run) |

## 5. Data Requirements
### 5.1 Data Entities
- Market: symbol, asset_type, payout map, otc
- Trade Stats: total_trades, winning_trades, losing_trades, total_profit
- Emotion State: greed, fear, confidence (float 0–1)

### 5.2 Data Retention
Currently in memory only; override audit persisted to file.

## 6. Non-Functional Requirements
| Category | Requirement |
|----------|-------------|
| Performance | Epsilon computation O(1); catalog load < 50 ms local |
| Reliability | Override logging best-effort (error logged if fail) |
| Portability | Pure Python except GUI dependency |
| Testability | 21 deterministic unit tests; no network I/O |
| Security | No credential storage beyond plaintext config (local) |
| Maintainability | Small, modular files; minimal cross-module coupling |

## 7. Constraints & Limitations
- Simulation only; live adapter intentionally disconnected.
- Risk & dynamic engine modules partially implemented / not integrated.
- GUI lacks headless test harness.

## 8. Future Enhancements (Roadmap Summary)
1. Real adapter integration & live payouts
2. Trade persistence (DuckDB)
3. Backtesting engine
4. Predictive modeling (baseline ML → advanced)
5. Strategy lifecycle orchestration
6. Advanced risk controls (drawdown, VaR/Kelly integration)
7. GUI enhancements (history, charts, balance)
8. Structured JSON logging + task manager
9. RL / evolution modules & performance profiling

## 9. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Scope creep | Diluted stability | SRS acts as boundary; add features only after tests |
| Overfitting to placeholders | Misleading expectations | Explicit marking of simulated modules |
| Missing GUI tests | UI regressions | Planned smoke test addition |
| Logging failure (override audit) | Loss of audit trail | Error log + user notification (future) |

## 10. Traceability Matrix
| Feature | Module(s) | Test(s) |
|---------|-----------|---------|
| Catalog retrieval | catalog.ingest | test_catalog.py |
| Payout guard & override | payouts.fetch | test_payouts.py |
| Engine trade simulation | core.engine | test_engine.py |
| Epsilon computation | intelligence.exploration | test_intelligence.py |
| Fitness scoring | intelligence.fitness | test_fitness_promotion.py |
| Promotion logic | intelligence.promotion | test_fitness_promotion.py |
| Config loading | utils.config (and legacy config) | test_config.py |

## 11. Outstanding Gaps
- Duplicate config concepts: `nexus.config` vs `nexus.utils.config` (consolidation planned)
- Unused dynamic engine & risk modules not wired into GUI/engine
- No persistence or backtesting logic active

## 12. Approval & Change Control
Changes to core behavior MUST:
1. Update this SRS (feature & traceability sections)
2. Include or update unit tests
3. Link PR to SRS diff

---
End of SRS (v0.1)


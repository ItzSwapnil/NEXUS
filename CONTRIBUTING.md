# Contributing to NEXUS

Contribution guidelines for the NEXUS autonomous AI trading system.

## Quick Start
```bash
uv venv .venv
.venv\Scripts\activate   # Windows
uv pip install -e .
pytest -q   # ensure tests pass
```
Target Python: **3.12+**.

## Contribution Guidelines
- Keep PRs small & focused
- Add/adjust tests when changing logic
- Preserve deterministic behaviour for existing tests
- Avoid adding heavy dependencies without discussion
- Update documentation for implemented features

## Code Style
- PEP 8 + type hints
- Clear, small functions
- Minimal logging for tests

## Tests
Run full test suite before opening a PR:
```bash
pytest -v
```

## Commit Messages
Readable imperative style (e.g., `engine: clamp emotion values`, `docs: prune outdated roadmap`). Squash fixups prior to merge.

## What *Not* To Add (Yet)
- Large RL / evolution frameworks
- Complex backtesting engines
- Cryptic meta-programming abstractions
- Unused stubs or speculative doc sections

## Reporting Issues
Open a GitHub issue with:
- Summary
- Reproduction steps (if bug)
- Expected vs actual
- Environment (OS, Python, commit hash)

## Security
No secrets should be committed. Current code does not persist credentials; see SECURITY.md for reporting process.

## License
By contributing you agree your work is licensed under the project MIT License.

---
Focus on making the existing small core *solid* before expanding scope.

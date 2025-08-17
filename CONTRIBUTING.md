# Contributing to NEXUS (Lightweight Alpha)

Thanks for your interest. The project is a minimal sandbox right now; focus contributions on **implemented scope** (catalog, payouts, engine, exploration, GUI polish) or small incremental roadmap steps.

## Quick Start
```bash
uv venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell
uv pip install -e .
pytest -q   # ensure 21 tests pass
```
Target Python: **3.13.6**.

## Contribution Guidelines
- Keep PRs small & focused (easier review).
- Add / adjust tests when changing logic.
- Preserve deterministic behaviour for existing tests.
- Avoid adding heavy dependencies without discussion.
- Update README / SPEC only for *implemented* features.

## Code Style
- PEP 8 + type hints.
- Prefer clear, small functions.
- Log sparingly; keep console noise low for tests.

## Tests
Run full test suite before opening a PR:
```bash
pytest -q
```
If adding GUI features, consider adding a lightweight non-blocking smoke test (future enhancement—currently none).

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

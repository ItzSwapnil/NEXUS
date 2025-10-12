# NEXUS Utility Scripts

This directory contains Python utility scripts for managing NEXUS.

## Available Scripts

### 1. Clean Workspace (`clean_workspace.py`)

Cleans up temporary files and build artifacts.

```bash
python scripts/clean_workspace.py
```

**What it removes:**
- `__pycache__/` directories
- `.pyc` files
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `build/` directory
- `*.egg-info/` directories
- Temporary log files (optional)

### 2. Verify Project (`verify_project.py`)

Verifies project structure and dependencies.

```bash
python scripts/verify_project.py
```

**What it checks:**
- Required directories exist
- Python version compatibility
- Key files are present
- Virtual environment status
- Dependency installation

## Usage

All scripts can be run directly:

```bash
# Clean the workspace
python scripts/clean_workspace.py

# Verify project structure
python scripts/verify_project.py
```

Or via the main launcher:

```bash
python run.py
# Select the appropriate option from the menu
```

## Notes

- Scripts are cross-platform (Windows, Linux, macOS)
- No external dependencies required (uses Python standard library)
- Safe to run anytime - they won't delete your code or data

---

For more information, see the main [README.md](../README.md)


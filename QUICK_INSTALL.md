# Quick Install Guide

Fast installation for NEXUS using UV package manager.

## Installation Method

**Using UV** - Fast, modern Python package installer.

Install UV:
```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## ✅ Faster Installation Options

### Option 1: Install Core First (RECOMMENDED - Fast!)
```bash
# Install just the basics (fast - under 2 minutes)
uv pip install -e .

# Then add what you need later
uv pip install -e ".[gui,ta]"      # Add GUI and technical analysis
```

### Option 2: Install GUI + Analysis (Medium - 5-10 minutes) ✅ Python 3.14
```bash
# Everything that works on Python 3.14
uv pip install -e ".[gui,ta]"
uv pip install scikit-learn joblib
```

### Option 3: Full Install - ⚠️ Requires Python 3.13
```bash
# PyTorch not available for Python 3.14 yet
# Use Python 3.13 or install PyTorch nightly
uv pip install -e ".[full]"
```

### Option 4: Using uv sync
```bash
# Sync with Python 3.14 compatible extras
uv sync --extra gui --extra ta
```

## 🚀 Recommended Quick Start

## 🚀 Recommended Quick Start (Python 3.14)
# 1. Install core dependencies (FAST - 1-2 minutes)
uv pip install -e .

# 2. Add GUI and broker (5-10 minutes)
uv pip install -e ".[gui,quotex]"
# 2. Add GUI and technical analysis
uv pip install -e ".[gui,ta]"
python -c "import nexus; print(f'NEXUS {nexus.__version__} installed successfully!')"
# 3. Add scikit-learn for ML
uv pip install scikit-learn joblib

# 4. Test it works
# 4. Run tests (optional)
uv pip install -e ".[test]"
# 5. Run tests (optional)

# 5. Run NEXUS
python run.py
# 6. Run NEXUS

## 🛑 If Installation is Stuck

**Note**: PyTorch and PyQuotex have compatibility issues with Python 3.14.  
See **PYTHON_314_COMPATIBILITY.md** for solutions.

**Symptom**: `uv pip install` or `uv sync` stuck on "Preparing packages..." or "Building nexus"

**Solution - Install Step by Step**:
```bash
# 1. Press Ctrl+C to stop the stuck process

# 2. Clear UV cache
uv cache clean

# 3. Install setuptools first (required for building)
uv pip install setuptools wheel

# 4. Install core dependencies one by one (shows progress better)
uv pip install pydantic pydantic-settings
uv pip install numpy pandas
uv pip install python-dotenv rich scipy

# 5. Install NEXUS without dependencies
uv pip install -e . --no-deps

# 6. Verify installation
python -c "import nexus; print(f'NEXUS {nexus.__version__} installed!')"
```

**Alternative - Install in parts**:
```bash
# Core first
uv pip install -e .

# Then add extras one at a time
uv pip install PySide6 matplotlib
uv pip install pyquotex
```

## 💡 Install AI Models Later (Optional)

If you want AI features later:
```bash
# Install PyTorch only
uv pip install torch torchvision

# Or install all AI dependencies
uv pip install -e ".[ai]"

# Or install scikit-learn only (lighter than PyTorch)
uv pip install scikit-learn joblib
```

## ⚠️ If Installation Gets Stuck

```bash
# Kill the process
Ctrl+C

# Clear UV cache
uv cache clean

# Try installing in steps
uv pip install -e .                    # Core first
uv pip install PySide6 matplotlib      # GUI second  
uv pip install pyquotex                # Broker third
uv pip install torch                   # AI last (optional)
```

## 📊 Installation Time Estimates

- **Core only** (`uv pip install -e .`): 1-2 minutes ⚡
- **Core + GUI + Broker** (`.[gui,quotex]`): 5-10 minutes 🏃
- **Full with AI** (`.[full]`): 30-60 minutes 🐌 (PyTorch is huge!)

Start with core, test the system, then add AI if needed!

---

## 🆕 What's New in This Setup

- ✅ **Python 3.14 Compatible** - Fully tested with the latest Python
- ✅ **Modern pyproject.toml** - PEP 621 compliant configuration
- ✅ **UV Package Manager** - Fast, modern Python package installer
- ✅ **Flexible Extras** - Install only what you need with `uv pip install -e ".[gui,quotex]"`
- ✅ **Better Documentation** - Clearer installation instructions
- ✅ **Optional Lock File** - Use `uv lock` if desired, not required

## 🎯 Why UV?

- **Fast**: UV is 10-100x faster than pip
- **Reliable**: Better dependency resolution
- **Modern**: Built for Python 3.12+
- **Compatible**: Works with standard pyproject.toml

---

**Ready to install?** Start with: `uv pip install -e .`

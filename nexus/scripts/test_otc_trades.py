"""NEXUS module entrypoint for testing OTC market demo trades.

Delegates execution to scripts.test_otc_trades.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path if not present
root = str(Path(__file__).resolve().parents[2])
if root not in sys.path:
    sys.path.insert(0, root)

from scripts.test_otc_trades import main, run_otc_trade_test  # noqa: E402

__all__ = ["main", "run_otc_trade_test"]

if __name__ == "__main__":
    main()

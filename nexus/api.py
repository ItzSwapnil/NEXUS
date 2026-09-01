"""
This module previously exposed a FastAPI app. All web server functionality has
been removed per project requirements. Keep this module as a lightweight stub
for backwards compatibility. No external web framework is used here.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["health", "metrics"]

# Optional rich printer
try:
    from rich import print as rprint  # type: ignore[assignment]
except Exception:  # pragma: no cover
    rprint = print  # type: ignore[assignment]


def health() -> Dict[str, str]:
    """Return a simple health payload similar to the old /health endpoint."""
    return {"status": "ok", "lang": "en", "version": "2.0.0"}


def metrics() -> Dict[str, int | float]:
    """
    Return a minimal metrics payload. Previously served as /metrics.
    Implementations that need richer stats should call into internal
    components directly without going through a web API.
    """
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_profit": 0.0,
    }


if __name__ == "__main__":
    # No server to run; print guidance instead.
    rprint(
        "[bold yellow]NEXUS web API has been removed. Use internal Python APIs instead.[/bold yellow]"
    )

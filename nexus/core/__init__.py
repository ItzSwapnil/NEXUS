"""
Core module for the NEXUS trading system (lightweight init).

Avoid importing heavy optional dependencies (faiss, etc.) at package import time.
Import components explicitly where needed, e.g.:
    from nexus.core.engine import NexusEngine
"""

try:
    from .engine import NexusEngine  # noqa: F401
except Exception:  # pragma: no cover
    # Provide a placeholder to prevent import explosions if dependencies missing
    class NexusEngine:  # type: ignore
        pass


__all__ = ["NexusEngine"]

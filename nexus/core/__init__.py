"""
Core module for the NEXUS trading system (lightweight init).

Avoid importing heavy optional dependencies (faiss, etc.) at package import time.
Import components explicitly where needed, e.g.:
    from nexus.core.engine import NexusEngine
"""

from .engine import NexusEngine

__all__ = ["NexusEngine"]

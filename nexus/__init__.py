"""Lightweight NEXUS package initializer.

Heavy imports have been removed to avoid pulling optional / large dependency
chains during simple imports (e.g. unit tests needing only NexusEngine).

Access core components explicitly, e.g.:
    from nexus.core.engine import NexusEngine

Version metadata is kept here.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]

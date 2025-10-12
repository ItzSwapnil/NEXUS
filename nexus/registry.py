"""Lightweight global registry for strategies and models.

This module exposes a simple runtime registry used by CLI and tooling to
list and resolve strategies or models by name without importing heavy
subsystems eagerly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable


class _Registry:
    def __init__(self) -> None:
        # Store callables/classes; keep lightweight to avoid heavy imports at module import time
        self.strategies: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}

    # -------- strategies --------
    def register_strategy(self, name: str, obj: Any) -> None:
        self.strategies[str(name)] = obj

    def unregister_strategy(self, name: str) -> None:
        self.strategies.pop(str(name), None)

    def list_strategies(self) -> List[str]:
        return sorted(self.strategies.keys())

    def get_strategy(self, name: str) -> Optional[Callable[..., Any]]:
        obj = self.strategies.get(str(name))
        return obj

    # -------- models --------
    def register_model(self, name: str, obj: Any) -> None:
        self.models[str(name)] = obj

    def unregister_model(self, name: str) -> None:
        self.models.pop(str(name), None)

    def list_models(self) -> List[str]:
        return sorted(self.models.keys())

    def get_model(self, name: str) -> Optional[Callable[..., Any]]:
        obj = self.models.get(str(name))
        return obj


# Singleton instance used by the rest of the app
registry = _Registry()

__all__ = ["registry", "_Registry"]

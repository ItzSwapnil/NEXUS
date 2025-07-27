"""
NEXUS Plugin Registry System

This module enables dynamic registration, loading, and management of strategies and models.
Supports hot-swapping, experimentation, and upgrades for extensibility.
"""

import importlib
import pkgutil
import logging
from typing import Dict, Type, Any, Callable, List

logger = logging.getLogger("nexus.registry")

class PluginRegistry:
    """
    Registry for strategies and models.
    """
    def __init__(self):
        self.strategies: Dict[str, Type] = {}
        self.models: Dict[str, Type] = {}

    def register_strategy(self, name: str, cls: Type):
        logger.info(f"Registering strategy: {name}")
        self.strategies[name] = cls

    def register_model(self, name: str, cls: Type):
        logger.info(f"Registering model: {name}")
        self.models[name] = cls

    def get_strategy(self, name: str) -> Type:
        return self.strategies.get(name)

    def get_model(self, name: str) -> Type:
        return self.models.get(name)

    def list_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def load_plugins(self, package: str):
        """
        Dynamically discover and load plugins from a package.
        """
        for _, modname, ispkg in pkgutil.iter_modules([package]):
            if not ispkg:
                module = importlib.import_module(f"{package}.{modname}")
                if hasattr(module, "register_plugin"):
                    module.register_plugin(self)

registry = PluginRegistry()

# Example: auto-register built-in strategies and models
# from nexus.strategies import meta_strategy
# registry.register_strategy("meta_strategy", meta_strategy.MetaStrategy)
# from nexus.intelligence import transformer
# registry.register_model("transformer", transformer.MarketTransformer)

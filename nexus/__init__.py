"""
NEXUS - A Free-Form, Self-Evolving AI Trader for Quotex

This is the next-generation autonomous trading system that combines:
- Reinforcement Learning with continuous adaptation
- Transformer-based market analysis
- Evolutionary strategy optimization
- Vector memory for pattern recognition
- Real-time regime detection
- Self-modifying neural architectures
"""

from typing import Dict, Any, Optional, List, Union
import asyncio
import logging
from pathlib import Path

from nexus.core.engine import NexusEngine
from nexus.core.evolution import EvolutionEngine
from nexus.core.memory import VectorMemory
from nexus.adapters.quotex import QuotexAdapter
from nexus.intelligence.transformer import MarketTransformer
from nexus.intelligence.rl_agent import RLAgent
from nexus.intelligence.regime_detector import RegimeDetector
from nexus.strategies.meta_strategy import MetaStrategy
from nexus.utils.logger import setup_nexus_logging

__version__ = "1.0.0"
__all__ = [
    "NexusEngine",
    "EvolutionEngine",
    "VectorMemory",
    "QuotexAdapter",
    "MarketTransformer",
    "RLAgent",
    "RegimeDetector",
    "MetaStrategy",
    "setup_nexus_logging"
]

# Global logger
logger = logging.getLogger("nexus")

class NEXUS:
    """
    Main NEXUS system orchestrator.

    This class brings together all components of the self-evolving AI trader,
    managing the complete lifecycle from initialization to autonomous trading.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config.yaml")
        self.engine: Optional[NexusEngine] = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the complete NEXUS system."""
        try:
            setup_nexus_logging()
            logger.info("🚀 Initializing NEXUS - Self-Evolving AI Trader")

            # Initialize the core engine
            self.engine = NexusEngine(self.config_path)
            await self.engine.initialize()

            self.is_initialized = True
            logger.info("✅ NEXUS initialization complete")
            return True

        except Exception as e:
            logger.error(f"❌ NEXUS initialization failed: {e}")
            return False

    async def run(self) -> None:
        """Run the NEXUS trading system."""
        if not self.is_initialized or not self.engine:
            raise RuntimeError("NEXUS must be initialized before running")

        logger.info("🎯 Starting NEXUS autonomous trading")
        await self.engine.run()

    async def shutdown(self) -> None:
        """Gracefully shutdown NEXUS."""
        if self.engine:
            await self.engine.shutdown()
        logger.info("🛑 NEXUS shutdown complete")

def create_nexus(config_path: Optional[Path] = None) -> NEXUS:
    """Factory function to create a NEXUS instance."""
    return NEXUS(config_path)

"""
Core module for the NEXUS trading system.

This package contains the core components of the NEXUS system,
including the main engine, memory management, and evolutionary algorithms.
"""

from nexus.core.engine import NexusEngine
from nexus.core.memory import VectorMemory
from nexus.core.evolution import EvolutionEngine

__all__ = ['NexusEngine', 'VectorMemory', 'EvolutionEngine']

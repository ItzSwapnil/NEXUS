"""
NEXUS Trading Strategies

This package contains the various trading strategies used by the NEXUS system:
- Meta-strategy for adaptive strategy selection
- Individual trading strategies
- Strategy optimization and adaptation logic
"""

from .meta_strategy import MetaStrategy, SignalType

# Allow importing the meta strategy
__all__ = ['MetaStrategy']


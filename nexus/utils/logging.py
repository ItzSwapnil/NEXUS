"""
This module is deprecated. Please use nexus.utils.logger instead.

For backward compatibility, this module re-exports the functionality from logger.py.
"""

# Re-export specific items from logger instead of using star import
from nexus.utils.logger import (
    LogConfig,
    setup_nexus_logging,
    get_nexus_logger,
    PerformanceLogger,
    TradeLogger,
    MetricsLogger
)

import warnings

warnings.warn(
    "The nexus.utils.logging module is deprecated. Please use nexus.utils.logger instead.",
    DeprecationWarning,
    stacklevel=2
)

# Make the imported items available
__all__ = [
    'LogConfig',
    'setup_nexus_logging',
    'get_nexus_logger',
    'PerformanceLogger',
    'TradeLogger',
    'MetricsLogger'
]

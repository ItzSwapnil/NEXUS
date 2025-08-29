"""Deprecated; use nexus.utils.logger instead (re-exports kept for compat)."""

from nexus.utils.logger import (
    LogConfig,
    setup_nexus_logging,
    get_nexus_logger,
    PerformanceLogger,
    TradeLogger,
    MetricsCollector,
)

import warnings

warnings.warn(
    "nexus.utils.logging is deprecated; use nexus.utils.logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'LogConfig',
    'setup_nexus_logging',
    'get_nexus_logger',
    'PerformanceLogger',
    'TradeLogger',
    'MetricsCollector',
]

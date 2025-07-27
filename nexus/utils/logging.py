"""
This module is deprecated. Please use nexus.utils.logger instead.

For backward compatibility, this module re-exports the functionality from logger.py.
"""

# Re-export all from logger
from nexus.utils.logger import *

import warnings

warnings.warn(
    "The nexus.utils.logging module is deprecated. Please use nexus.utils.logger instead.",
    DeprecationWarning,
    stacklevel=2
)

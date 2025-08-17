"""NEXUS root package initialization"""

# This file ensures that the NEXUS directory is treated as a package
# for Python module discovery. This is particularly important for
# local development where the package is installed in development mode.

import os
import sys
from pathlib import Path

# Add project root to Python path if not already there
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

__version__ = "2.0.0"
__all__ = ["nexus"]
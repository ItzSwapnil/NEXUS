"""NEXUS core package initialization"""

# This file ensures that the nexus directory is treated as a package
# for Python module discovery. This is particularly important for
# local development where the package is installed in development mode.

__version__ = "2.0.0"
__all__ = [
    "core",
    "gui",
    "intelligence",
    "strategies",
    "utils",
    "adapters",
    "data",
    "settings",
    "models"
]
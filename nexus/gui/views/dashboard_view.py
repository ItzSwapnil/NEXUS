import logging
import importlib.util

logger = logging.getLogger("nexus.gui.dashboard")

# Check if pyqtgraph is available without importing it
PYQTGRAPH_AVAILABLE = importlib.util.find_spec("pyqtgraph") is not None

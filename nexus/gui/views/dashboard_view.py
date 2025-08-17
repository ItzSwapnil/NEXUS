from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, QComboBox, QTabWidget, QTextEdit, QSizePolicy, QTableView, QLabel, QCheckBox, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject, QModelIndex
from nexus.adapters.quotex import QuotexAdapter
from nexus.intelligence.regime_detector import RegimeDetector
from nexus.data.trade_history import TradeHistory, AdvancedDataStore
import threading
import asyncio
import logging

logger = logging.getLogger("nexus.gui.dashboard")

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
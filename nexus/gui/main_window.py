from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from nexus.adapters.quotex import QuotexAdapter
from nexus.gui.views.dashboard_view import DashboardView

class NexusMainWindow(QMainWindow):
    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("NEXUS AI Trader")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("background-color: #181c20; color: #e0e0e0;")
        # Use engine's QuotexAdapter if available
        if self.engine and hasattr(self.engine, 'quotex'):
            self.quotex_adapter = self.engine.quotex
        else:
            from nexus.utils.config import load_config
            config = load_config()
            self.quotex_adapter = QuotexAdapter(
                email=config.quotex.email,
                password=config.quotex.password,
                demo_mode=config.quotex.demo_mode
            )
        self._init_ui()

    def _init_ui(self):
        # Sidebar navigation
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(180)
        self.sidebar.setStyleSheet("background: #23272b; border: none; font-size: 16px;")
        for name in ["Dashboard", "Trading", "Analytics", "Settings"]:
            item = QListWidgetItem(name)
            self.sidebar.addItem(item)
        self.sidebar.setCurrentRow(0)

        # Central stacked widget
        self.stack = QStackedWidget()
        self.stack.addWidget(DashboardView(self.quotex_adapter))
        self.stack.addWidget(self._make_placeholder("Trading"))
        self.stack.addWidget(self._make_placeholder("Analytics"))
        self.stack.addWidget(self._make_placeholder("Settings"))

        # Layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stack, 1)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect navigation
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

    def _make_placeholder(self, name):
        w = QWidget()
        layout = QVBoxLayout()
        label = QLabel(f"{name} view coming soon...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 32px; color: #7fd1b9;")
        layout.addWidget(label)
        w.setLayout(layout)
        return w

"""NEXUS GUI Main Window (Spec §7 implementation + enhancements).

(Recovered implementation after prior file corruption.)
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import List

from nexus.utils.config import NexusSettings
from nexus.utils.logger import get_nexus_logger
from nexus.payouts.fetch import set_payout_override, is_override_enabled
from nexus.catalog.ingest import get_market_catalog, Market
from nexus.core.engine import NexusEngine

logger = get_nexus_logger("nexus.gui.main_window")

try:  # pragma: no cover - GUI only
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        QTableWidget, QTableWidgetItem, QCheckBox, QSlider, QMessageBox, QGroupBox, QFormLayout, QListWidget
    )
    from PySide6.QtCore import Qt, QTimer
except Exception:  # pragma: no cover
    QApplication = object  # type: ignore
    QMainWindow = object  # type: ignore
    logger.error("PySide6 not installed; GUI unavailable.")

PAYOUT_COLORS = {"ok": "#2e7d32", "warn": "#f9a825", "bad": "#c62828"}


class NexusMainWindow(QMainWindow):  # pragma: no cover - UI heavy
    def __init__(self, engine: NexusEngine):
        super().__init__()
        self.engine = engine
        self.settings: NexusSettings = engine.settings
        self.setWindowTitle("NEXUS – Self-Evolving Trader")
        self.resize(1180, 760)
        self._panic = False
        self._payout_filter_enabled = True
        self._markets_cache: List[Market] = []

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # Controls
        controls_layout = QHBoxLayout()
        self.demo_checkbox = QCheckBox("Demo Mode")
        self.demo_checkbox.setChecked(self.engine.demo_mode)
        self.demo_checkbox.stateChanged.connect(self._toggle_demo_mode)

        self.filter_checkbox = QCheckBox(f"Only payout ≥ {self.settings.trading.payout_threshold:.0f}%")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.stateChanged.connect(self._toggle_payout_filter)

        self.override_btn = QPushButton("Disable Payout Override" if is_override_enabled() else "Enable Payout Override")
        self.override_btn.clicked.connect(self._handle_override)

        self.panic_btn = QPushButton("PANIC STOP")
        self.panic_btn.setStyleSheet("background:#c62828;color:#fff;font-weight:bold;")
        self.panic_btn.clicked.connect(self._panic_stop)

        self.refresh_btn = QPushButton("Refresh Catalog")
        self.refresh_btn.clicked.connect(lambda: asyncio.ensure_future(self._refresh_catalog()))

        self.trade_btn = QPushButton("Execute Test Trade")
        self.trade_btn.clicked.connect(lambda: asyncio.ensure_future(self._execute_test_trade()))

        self.autonomy_slider = QSlider(Qt.Orientation.Horizontal)
        self.autonomy_slider.setMinimum(0)
        self.autonomy_slider.setMaximum(100)
        self.autonomy_slider.setValue(int(self.engine.exploration_controller.cfg.base_epsilon * 100))
        self.autonomy_slider.valueChanged.connect(self._autonomy_changed)
        self.autonomy_label = QLabel(f"Autonomy: {self.engine.exploration_controller.cfg.base_epsilon:.2f}")

        for w in [self.demo_checkbox, self.filter_checkbox, self.override_btn, self.panic_btn, self.refresh_btn,
                  self.trade_btn, self.autonomy_label, self.autonomy_slider]:
            controls_layout.addWidget(w)
        controls_layout.addStretch(1)
        root_layout.addLayout(controls_layout)

        # Markets table
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Symbol", "Type", "Payout %", "OTC"])
        self.table.setSortingEnabled(True)
        root_layout.addWidget(QLabel("Markets"))
        root_layout.addWidget(self.table)

        # Trades log
        root_layout.addWidget(QLabel("Recent Trades"))
        self.trade_log = QListWidget()
        root_layout.addWidget(self.trade_log)

        # Strategy & stats box
        group = QGroupBox("Strategy & Engine Snapshot")
        form = QFormLayout()
        self.epsilon_label = QLabel("ε: --")
        self.stats_label = QLabel("Stats: --")
        form.addRow("Exploration Epsilon", self.epsilon_label)
        form.addRow("Performance", self.stats_label)
        group.setLayout(form)
        root_layout.addWidget(group)

        self.balance_label = QLabel("Balance: --")
        self.status = self.statusBar()
        self.status.addPermanentWidget(self.balance_label)
        self.status.showMessage("Initializing…")

        # Timers
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(self.settings.trading.payout_poll_interval_seconds * 1000)
        self.refresh_timer.timeout.connect(lambda: asyncio.ensure_future(self._refresh_catalog()))
        self.refresh_timer.start()

        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(3000)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start()

        # Kick off initial load
        asyncio.ensure_future(self._refresh_catalog())
        self._update_stats()

    # ------------------------ UI Event Handlers ------------------------ #
    def _toggle_demo_mode(self):
        self.engine.demo_mode = self.demo_checkbox.isChecked()
        mode = "DEMO" if self.engine.demo_mode else "REAL"
        self.status.showMessage(f"Mode switched to {mode}")

    def _toggle_payout_filter(self):
        self._payout_filter_enabled = self.filter_checkbox.isChecked()
        self._populate_markets_table()

    def _handle_override(self):
        enabled = is_override_enabled()
        set_payout_override(not enabled, user="gui", reason="user toggle")
        self.override_btn.setText("Disable Payout Override" if not enabled else "Enable Payout Override")
        self.status.showMessage(f"Payout override {'ENABLED' if not enabled else 'DISABLED'}")
        self._populate_markets_table()

    def _panic_stop(self):
        self._panic = True
        self.engine.demo_mode = True
        self.demo_checkbox.setChecked(True)
        set_payout_override(False)
        QMessageBox.warning(self, "PANIC STOP", "Real trading halted. Demo mode enforced.")

    def _autonomy_changed(self):
        val = self.autonomy_slider.value() / 100.0
        self.engine.exploration_controller.cfg.base_epsilon = val
        self.autonomy_label.setText(f"Autonomy: {val:.2f}")
        self._update_stats()

    # ------------------------ Data / Async Ops ------------------------- #
    async def _refresh_catalog(self):
        try:
            self._markets_cache = await get_market_catalog()
            self._populate_markets_table()
            self.status.showMessage(f"Catalog refreshed @ {datetime.utcnow().strftime('%H:%M:%S')} (markets={len(self._markets_cache)})")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed refreshing catalog: {e}")

    async def _execute_test_trade(self):
        asset = self.settings.trading.default_asset
        result = await self.engine.execute_trade(asset, "call", self.settings.trading.base_trade_amount, str(self.settings.trading.default_expiration))
        stamp = datetime.utcnow().strftime('%H:%M:%S')
        if result.get("success"):
            self.trade_log.addItem(f"[{stamp}] {asset} WIN +{result['profit']:.2f}")
        else:
            self.trade_log.addItem(f"[{stamp}] {asset} BLOCKED: {result.get('error')}")
        self.trade_log.scrollToBottom()
        self._update_stats()

    # ------------------------ Rendering Helpers ------------------------- #
    def _populate_markets_table(self):
        markets = self._markets_cache
        threshold = self.settings.trading.payout_threshold
        if self._payout_filter_enabled:
            markets = [m for m in markets if m.effective_payout("60") >= threshold]
        self.table.setRowCount(len(markets))
        for row, m in enumerate(markets):
            payout = m.effective_payout("60")
            payout_str = f"{payout:.1f}"
            if payout >= threshold:
                color = PAYOUT_COLORS["ok"]
            elif payout >= threshold - 5:
                color = PAYOUT_COLORS["warn"]
            else:
                color = PAYOUT_COLORS["bad"]
            for col, value in enumerate([m.symbol, m.asset_type, payout_str, "Yes" if m.otc else "No"]):
                item = QTableWidgetItem(str(value))
                if col == 2:
                    item.setForeground(Qt.GlobalColor.white)
                    item.setBackground(self._qcolor(color))
                self.table.setItem(row, col, item)

    def _qcolor(self, hex_color: str):  # pragma: no cover - minor UI
        from PySide6.QtGui import QColor
        return QColor(hex_color)

    def _update_stats(self):
        stats = self.engine.get_performance_stats()
        payout_sample = 85.0
        epsilon = self.engine.exploration_controller.compute_epsilon(
            confidence_metrics={"sharpe": 1.0, "stability": 0.6, "win_rate": 0.55},
            uncertainty_metrics={"atr": 0.5, "disagreement": 0.2, "spread": 0.0002, "otc": False},
            payout=payout_sample,
        )
        self.epsilon_label.setText(f"ε: {epsilon}")
        self.stats_label.setText(
            f"Trades {stats['total_trades']} | W {stats['winning_trades']} | L {stats['losing_trades']} | PnL {stats['total_profit']:.2f}"
        )

__all__ = ["NexusMainWindow"]

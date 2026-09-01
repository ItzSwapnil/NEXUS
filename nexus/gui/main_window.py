"""NEXUS GUI Main Window.

Provides live market catalog display, interactive trade execution, real-time balance tracking,
a live interactive Matplotlib price chart, and active position monitors with trade countdowns.
"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from nexus.analysis.market_analyzer import MarketAnalysisResult, MarketAnalyzer
from nexus.catalog.ingest import Market
from nexus.core.engine import NexusEngine
from nexus.data.trade_history import TradeHistory
from nexus.payouts.fetch import is_override_enabled, set_payout_override
from nexus.utils.config import NexusSettings
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.gui.main_window")

try:  # pragma: no cover - GUI only
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QSpinBox,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # pragma: no cover
    QApplication = object  # type: ignore
    QMainWindow = object  # type: ignore
    logger.error("PySide6 not installed; GUI unavailable.")

try:
    import numpy as np
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

PAYOUT_COLORS = {"ok": "#2e7d32", "warn": "#f9a825", "bad": "#c62828"}


class MarketChartWidget(QWidget):
    """Live interactive price chart widget with technical indicators & AI future price trajectory projections."""

    def __init__(self, symbol: str = "EURUSD", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.prices: List[float] = []
        self.ai_prediction: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if _HAS_MATPLOTLIB:
            self.figure = Figure(figsize=(5, 3), facecolor="#181818")
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            layout.addWidget(self.canvas)
            self._init_series()
        else:
            self.placeholder_lbl = QLabel("Live Market Chart (Matplotlib unavailable)")
            self.placeholder_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.placeholder_lbl)

    def set_ai_prediction(self, pred: Optional[Dict[str, Any]]) -> None:
        """Set latest AI prediction dictionary to render dynamic prediction overlay & future path."""
        self.ai_prediction = pred
        self.draw_chart()

    def _init_series(self) -> None:
        if not _HAS_MATPLOTLIB:
            return
        base = (
            1.0850
            if "EUR" in self.symbol
            else 150.25
            if "JPY" in self.symbol
            else 65000.0
            if "BTC" in self.symbol
            else 2400.0
            if "XAU" in self.symbol
            else 100.0
        )
        np.random.seed(abs(hash(self.symbol)) % (2**32))
        step_scale = 0.0003 if base < 10 else 0.15
        walk = np.cumsum(np.random.randn(120) * step_scale) + base

        self.candles = []
        for i in range(0, len(walk) - 3, 4):
            c_vals = walk[i : i + 4]
            open_p = float(c_vals[0])
            close_p = float(c_vals[-1])
            high_p = float(max(c_vals.max(), open_p, close_p))
            low_p = float(min(c_vals.min(), open_p, close_p))
            self.candles.append((open_p, high_p, low_p, close_p))
        self._tick_count = 0
        self.draw_chart()

    def update_real_candles(self, raw_candles: List[dict[str, float]]) -> None:
        if not _HAS_MATPLOTLIB or not raw_candles:
            return
        parsed = []
        for c in raw_candles:
            try:
                o = float(c["open"])
                h = float(c["high"])
                low_val = float(c["low"])
                cl = float(c["close"])
                parsed.append((o, h, low_val, cl))
            except Exception:
                continue
        if parsed:
            self.candles = parsed[-35:]
            self.draw_chart()

    def set_symbol(self, symbol: str) -> None:
        if self.symbol != symbol and symbol:
            self.symbol = symbol
            self.ai_prediction = None
            self._init_series()

    def update_price(self, new_price: float) -> None:
        if not hasattr(self, "candles") or not self.candles:
            self._init_series()
            return

        last_o, last_h, last_l, _ = self.candles[-1]
        self._tick_count += 1

        if self._tick_count % 4 == 0:
            # Start a new candle
            self.candles.append((new_price, new_price, new_price, new_price))
        else:
            # Update existing candle
            new_h = max(last_h, new_price)
            new_l = min(last_l, new_price)
            self.candles[-1] = (last_o, new_h, new_l, new_price)

        if len(self.candles) > 35:
            self.candles.pop(0)

        self.draw_chart()

    def draw_chart(self) -> None:
        if not _HAS_MATPLOTLIB or not hasattr(self, "ax") or not getattr(self, "candles", None):
            return
        self.ax.clear()
        self.ax.set_facecolor("#121212")
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#333333")

        closes = [c[3] for c in self.candles]
        n_candles = len(self.candles)

        # 1. Render Candlesticks
        for i, (open_p, high_p, low_p, close_p) in enumerate(self.candles):
            color = "#00e676" if close_p >= open_p else "#ff5252"
            self.ax.vlines(i, low_p, high_p, color=color, linewidth=1.2)
            body_bottom = min(open_p, close_p)
            body_top = max(open_p, close_p)
            height = max(body_top - body_bottom, (high_p - low_p) * 0.02)
            self.ax.vlines(i, body_bottom, body_bottom + height, color=color, linewidth=5.5)

        # 2. Render Moving Averages (EMA-9 & SMA-20)
        if len(closes) >= 9:
            ema9 = pd.Series(closes).ewm(span=9).mean().to_numpy()
            self.ax.plot(
                np.arange(len(closes)), ema9, color="#ff9800", linewidth=1.2, label="EMA-9"
            )
        if len(closes) >= 20:
            sma20 = pd.Series(closes).rolling(20).mean().to_numpy()
            std20 = pd.Series(closes).rolling(20).std().to_numpy()
            self.ax.plot(
                np.arange(len(closes)), sma20, color="#00bcd4", linewidth=1.2, label="SMA-20"
            )
            bb_upper = sma20 + 2.0 * std20
            bb_lower = sma20 - 2.0 * std20
            self.ax.fill_between(
                np.arange(len(closes)),
                bb_lower,
                bb_upper,
                color="#00bcd4",
                alpha=0.07,
                label="Bollinger Bands",
            )

        last_close = closes[-1]
        fmt = ".5f" if last_close < 10 else ".2f"
        val_str = f"{last_close:{fmt}}"

        # 3. Dynamic Support & Resistance Levels
        highs = [c[1] for c in self.candles]
        lows = [c[2] for c in self.candles]
        res_level = max(highs)
        sup_level = min(lows)
        self.ax.axhline(
            res_level,
            color="#ff5252",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"Res: {res_level:{fmt}}",
        )
        self.ax.axhline(
            sup_level,
            color="#00e676",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label=f"Sup: {sup_level:{fmt}}",
        )

        # 4. Live Per-Market Dynamic AI Indicator Learning & Signal Overlay (Top-Right)
        if len(closes) >= 14:
            from nexus.utils.technical import (
                get_market_indicator_blueprint,
                relative_strength_index,
            )

            blueprint = get_market_indicator_blueprint(self.symbol)
            params = blueprint.get("params", {}) or blueprint
            gen = blueprint.get("generation", 1)
            p_name = blueprint.get("profile_name", "AI Dynamic Blueprint")
            rsi_p = params.get("rsi_period", 14)
            ema_f = params.get("ema_fast", 9)
            ema_s = params.get("ema_slow", 21)

            rsi_val = relative_strength_index(np.array(closes), window=rsi_p)[-1]
            rsi_str = f"RSI({rsi_p}): {rsi_val:.1f} {'(Oversold BUY)' if rsi_val < 35 else '(Overbought SELL)' if rsi_val > 65 else '(Neutral)'}"

            active_info = f"[AI] DYNAMIC ENGINE: {self.symbol} (Gen {gen})\n* Strategy: {p_name}\n* Evolved: EMA({ema_f}/{ema_s}) | RSI({rsi_p})\n* {rsi_str}"

            if self.ai_prediction and isinstance(self.ai_prediction, dict):
                act_sigs = self.ai_prediction.get("active_signals", [])
                if act_sigs:
                    active_info += "\n* Active Learned Signals:"
                    for s in act_sigs[:3]:
                        clean_s = (
                            str(s)
                            .replace("🟢", "(BUY)")
                            .replace("🔴", "(SELL)")
                            .replace("⬆", "^")
                            .replace("⬇", "v")
                            .replace("🕯️", "[Candle]")
                            .replace("🔨", "[Hammer]")
                        )
                        active_info += f"\n  - {clean_s}"

            self.ax.text(
                0.98,
                0.95,
                active_info,
                transform=self.ax.transAxes,
                fontsize=7.2,
                fontfamily="monospace",
                verticalalignment="top",
                horizontalalignment="right",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "#181818",
                    "edgecolor": "#00bcd4",
                    "alpha": 0.88,
                },
                color="#ffffff",
            )

        # 5. Render AI Future Trajectory Projection Line & Shaded Cone
        if self.ai_prediction and isinstance(self.ai_prediction, dict):
            sig = str(self.ai_prediction.get("signal", "call")).lower()
            conf = float(self.ai_prediction.get("confidence", 0.75))
            exp = int(self.ai_prediction.get("recommended_expiration", 60))
            regime = str(self.ai_prediction.get("regime", "BULL")).upper()

            # Compute future slope direction
            price_delta = (last_close * 0.0015 if last_close < 10 else 0.5) * (
                1.0 if sig == "call" else -1.0
            )
            future_steps = 6
            future_x = np.arange(n_candles - 1, n_candles - 1 + future_steps)
            slope = np.linspace(0, price_delta, future_steps)
            future_y = last_close + slope

            pred_color = "#00e676" if sig == "call" else "#ff5252"
            self.ax.plot(
                future_x,
                future_y,
                color=pred_color,
                linestyle="--",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=f"AI Target ({sig.upper()} {conf * 100:.0f}%)",
            )

            # Draw prediction uncertainty cone
            uncertainty = np.linspace(0, abs(price_delta) * 0.4, future_steps)
            self.ax.fill_between(
                future_x,
                future_y - uncertainty,
                future_y + uncertainty,
                color=pred_color,
                alpha=0.18,
            )

            # Badge text overlay
            badge_text = (
                f"AI TARGET: {sig.upper()} ^" if sig == "call" else f"AI TARGET: {sig.upper()} v"
            )
            badge_text += f"\nConfidence: {conf * 100:.1f}%\nTimeframe: {exp}s | Regime: {regime}"
            self.ax.text(
                0.02,
                0.95,
                badge_text,
                transform=self.ax.transAxes,
                fontsize=8,
                fontweight="bold",
                verticalalignment="top",
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "#1e1e1e",
                    "edgecolor": pred_color,
                    "alpha": 0.9,
                },
                color="#ffffff",
            )

        self.ax.set_title(
            f"LIVE CANDLESTICK CHART: {self.symbol} (${val_str})",
            color="#ffffff",
            fontsize=10,
            fontweight="bold",
        )
        self.ax.grid(True, color="#262626", linestyle=":", alpha=0.6)
        self.ax.legend(
            loc="lower left",
            facecolor="#1e1e1e",
            edgecolor="#333333",
            labelcolor="#cccccc",
            fontsize=7,
        )
        self.canvas.draw_idle()


class NexusMainWindow(QMainWindow):  # pragma: no cover - UI heavy
    login_done_signal = Signal(bool, float, object)
    catalog_refreshed_signal = Signal(bool, object, object)
    balance_done_signal = Signal(bool, float, object)
    trade_executed_signal = Signal(bool, object, object)
    status_signal = Signal(str)
    candles_loaded_signal = Signal(str, object)
    chart_ai_prediction_signal = Signal(dict)
    market_trained_signal = Signal(dict)
    market_training_progress_signal = Signal(dict)
    market_analysis_signal = Signal(object)

    def __init__(self, engine: NexusEngine):
        super().__init__()
        self.engine = engine
        self.settings: NexusSettings = engine.settings
        self.setWindowTitle("NEXUS – Autonomous AI Trading Terminal")
        self.resize(1600, 1100)
        self._panic = False
        self._payout_filter_enabled = False
        self._markets_cache: List[Market] = []
        self._auto_trade_rotation_idx = 0
        self._last_auto_traded_symbol = ""
        self._latest_market_analysis: Optional[MarketAnalysisResult] = None
        self._trained_market_symbols: set[str] = set()
        self._training_excluded_symbols: set[str] = set()
        self._balance_refresh_lock = threading.Lock()
        self._auto_trade_lock = threading.Lock()
        self._positions_lock = threading.RLock()
        self._trade_store = TradeHistory()
        # The visible history is rebuilt from durable records, not only from
        # objects that survived in this process.
        self.engine.trade_history = self._trade_store.get_trade_history(limit=500)

        # Connect thread-safe signals to main-thread UI update slots
        if hasattr(self.login_done_signal, "connect"):
            self.login_done_signal.connect(self._on_login_done_main_thread)
            self.catalog_refreshed_signal.connect(self._on_catalog_refreshed_main_thread)
            self.balance_done_signal.connect(self._on_balance_done_main_thread)
            self.trade_executed_signal.connect(self._on_trade_executed_main_thread)
            self.status_signal.connect(self._update_status_main_thread)
            self.candles_loaded_signal.connect(self._on_candles_loaded_main_thread)
            self.chart_ai_prediction_signal.connect(self._on_chart_ai_prediction_main_thread)
            self.market_trained_signal.connect(self._on_market_trained_main_thread)
            self.market_training_progress_signal.connect(self._on_market_training_progress_main_thread)
            self.market_analysis_signal.connect(self._on_market_analysis_main_thread)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ------------------- Top Control Bar -------------------
        top_bar = QHBoxLayout()

        self.balance_card = QLabel("Balance: $10,000.00 (DEMO)")
        self.balance_card.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #00e676; padding: 6px 12px; background: #1e1e1e; border-radius: 6px;"
        )

        self.demo_checkbox = QCheckBox("Demo Mode")
        self.demo_checkbox.setChecked(self.engine.demo_mode)
        self.demo_checkbox.stateChanged.connect(self._toggle_demo_mode)

        self.filter_checkbox = QCheckBox("Filter Min Payout:")
        self.filter_checkbox.setChecked(True)
        self.filter_checkbox.stateChanged.connect(self._toggle_payout_filter)
        self._payout_filter_enabled = True

        self.payout_threshold_spin = QSpinBox()
        self.payout_threshold_spin.setRange(0, 100)
        self.payout_threshold_spin.setValue(int(self.settings.trading.payout_threshold))
        self.payout_threshold_spin.setSuffix("%")
        self.payout_threshold_spin.setStyleSheet("font-weight: bold; padding: 2px 6px;")
        self.payout_threshold_spin.valueChanged.connect(self._on_payout_threshold_changed)

        self.override_btn = QPushButton(
            "Disable Payout Override" if is_override_enabled() else "Enable Payout Override"
        )
        self.override_btn.clicked.connect(self._handle_override)

        self.panic_btn = QPushButton("PANIC STOP")
        self.panic_btn.setStyleSheet(
            "background:#c62828;color:#fff;font-weight:bold;padding:6px 12px;"
        )
        self.panic_btn.clicked.connect(self._panic_stop)

        self.refresh_btn = QPushButton("🔄 Refresh Markets & Balance")
        self.refresh_btn.clicked.connect(self._on_refresh_button_click)

        self.train_btn = QPushButton("⚡ Train All Markets AI")
        self.train_btn.setStyleSheet(
            "background: #00bcd4; color: #000000; font-weight: bold; font-size: 12px; padding: 6px 12px; border-radius: 6px;"
        )
        self.train_btn.setToolTip(
            "Independently train AI models and optimize indicators for ALL markets"
        )
        self.train_btn.clicked.connect(self._train_market_threaded)

        self.autonomy_slider = QSlider(Qt.Orientation.Horizontal)
        self.autonomy_slider.setMinimum(0)
        self.autonomy_slider.setMaximum(100)
        self.autonomy_slider.setValue(
            int(self.engine.exploration_controller.cfg.base_epsilon * 100)
        )
        self.autonomy_slider.valueChanged.connect(self._autonomy_changed)
        self.autonomy_label = QLabel(
            f"Autonomy: {self.engine.exploration_controller.cfg.base_epsilon:.2f}"
        )

        min_conf_pct = int(self.settings.trading.min_confidence * 100)
        self.confidence_label = QLabel(f"Min Conf: {min_conf_pct}%")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setMinimum(50)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(min_conf_pct)
        self.confidence_slider.setToolTip("Minimum AI Confidence % required for trades")
        self.confidence_slider.valueChanged.connect(self._confidence_changed)

        self.auto_trade_btn = QPushButton("🤖 START AI AUTO-TRADER")
        self.auto_trade_btn.setStyleSheet(
            "background: #00c853; color: #ffffff; font-weight: bold; font-size: 13px; padding: 6px 14px; border-radius: 6px;"
        )
        self.auto_trade_btn.clicked.connect(self._toggle_auto_trading)
        self._auto_trading_active = False

        top_bar.addWidget(self.balance_card)
        top_bar.addWidget(self.demo_checkbox)
        top_bar.addWidget(self.auto_trade_btn)
        top_bar.addWidget(self.train_btn)
        top_bar.addWidget(self.filter_checkbox)
        top_bar.addWidget(self.payout_threshold_spin)
        top_bar.addWidget(self.override_btn)
        top_bar.addWidget(self.panic_btn)
        top_bar.addWidget(self.refresh_btn)
        top_bar.addWidget(self.autonomy_label)
        top_bar.addWidget(self.autonomy_slider)
        top_bar.addWidget(self.confidence_label)
        top_bar.addWidget(self.confidence_slider)
        top_bar.addStretch(1)
        root_layout.addLayout(top_bar)

        # ------------------- Market-Wide AI Analysis -------------------
        analysis_box = QGroupBox("AI Market Lab — Settings, Price Gates & Scenario Estimates")
        analysis_layout = QVBoxLayout(analysis_box)
        analysis_controls = QHBoxLayout()
        self.price_gate_checkbox = QCheckBox("Use AI price gate")
        self.price_gate_checkbox.setChecked(True)
        self.price_gate_checkbox.setToolTip(
            "Suggests a better entry zone from live candles; it does not guarantee a price or profit."
        )
        self.ai_timeframe_checkbox = QCheckBox("AI chooses timeframe per trade")
        self.ai_timeframe_checkbox.setChecked(self.settings.trading.ai_select_timeframe)
        self.ai_timeframe_checkbox.setToolTip(
            "Use each market's AI-recommended expiration; when disabled, use the selected/default timeframe."
        )
        self.analyze_markets_btn = QPushButton("🔎 Analyze All Open Markets")
        self.analyze_markets_btn.clicked.connect(self._analyze_all_markets_threaded)
        self.analysis_summary = QLabel("No market analysis yet. Refresh the live catalog first.")
        self.analysis_summary.setWordWrap(True)
        analysis_controls.addWidget(self.analyze_markets_btn)
        analysis_controls.addWidget(self.price_gate_checkbox)
        analysis_controls.addWidget(self.ai_timeframe_checkbox)
        analysis_controls.addWidget(self.analysis_summary, 1)
        analysis_layout.addLayout(analysis_controls)

        self.analysis_table = QTableWidget(0, 9)
        self.analysis_table.setHorizontalHeaderLabels(
            [
                "Market",
                "1m Payout",
                "Payout Source",
                "Signal",
                "Confidence",
                "Regime",
                "Expected EV/$1",
                "AI Entry Gate",
                "AI Timeframe",
            ]
        )
        self.analysis_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.analysis_table.setMinimumHeight(170)
        self.analysis_table.itemSelectionChanged.connect(self._on_analysis_market_selected)
        analysis_layout.addWidget(self.analysis_table)
        self.training_status_table = QTableWidget(0, 3)
        self.training_status_table.setHorizontalHeaderLabels(["Market", "Training status", "Generation"])
        self.training_status_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.training_status_table.setMinimumHeight(100)
        analysis_layout.addWidget(QLabel("Per-market training progress"))
        analysis_layout.addWidget(self.training_status_table)

        self.scenario_table = QTableWidget(0, 13)
        self.scenario_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Autonomy",
                "Min Conf.",
                "Eligible Markets",
                "10m Trades",
                "10m Exp. P&L",
                "15m Trades",
                "15m Exp. P&L",
                "30m Trades",
                "30m Exp. P&L",
                "1h Trades",
                "1h Exp. P&L",
                "Scope",
            ]
        )
        self.scenario_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scenario_table.setMinimumHeight(150)
        analysis_layout.addWidget(QLabel("Aggregate settings scenarios across all eligible markets"))
        analysis_layout.addWidget(self.scenario_table)
        root_layout.addWidget(analysis_box)

        # ------------------- Main Content Splitter -------------------
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # The legacy market table has been removed; give its space to the
        # chart and trade panel instead.

        # Center Panel: Live Price Chart
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        chart_box = QGroupBox("Real-Time Market Price Graph")
        chart_box_layout = QVBoxLayout()
        self.chart_widget = MarketChartWidget(symbol="EURUSD")
        chart_box_layout.addWidget(self.chart_widget)
        chart_box.setLayout(chart_box_layout)
        chart_box.setMinimumHeight(540)
        center_layout.addWidget(chart_box)

        # Right Panel: Manual Order Placement & Live Positions Tracker
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 1. Trade Execution Box
        trade_box = QGroupBox("Manual Order Placement")
        trade_form = QFormLayout()

        self.asset_selector = QComboBox()
        self.asset_selector.setEditable(True)
        self.asset_selector.currentTextChanged.connect(self._on_asset_changed)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["CALL ⬆ (BUY)", "PUT ⬇ (SELL)"])

        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(1.0, 10000.0)
        self.amount_spin.setValue(10.0)
        self.amount_spin.setPrefix("$ ")

        self.expiration_combo = QComboBox()
        self.expiration_combo.addItems(
            [
                "5s (OTC Micro)",
                "10s (OTC Micro)",
                "15s (OTC Micro)",
                "30s (OTC Micro)",
                "60s (1 min)",
                "120s (2 min)",
                "180s (3 min)",
                "300s (5 min)",
                "600s (10 min)",
                "900s (15 min)",
            ]
        )

        self.place_trade_btn = QPushButton("PLACE TRADE NOW")
        self.place_trade_btn.setStyleSheet(
            "background: #2e7d32; color: #ffffff; font-weight: bold; font-size: 14px; padding: 8px;"
        )
        self.place_trade_btn.clicked.connect(self._execute_manual_trade_threaded)

        trade_form.addRow("Target Asset:", self.asset_selector)
        trade_form.addRow("Direction:", self.direction_combo)
        trade_form.addRow("Trade Amount:", self.amount_spin)
        trade_form.addRow("Expiration:", self.expiration_combo)
        trade_form.addRow(self.place_trade_btn)
        trade_box.setLayout(trade_form)
        right_layout.addWidget(trade_box)

        # 2. Live Active Open Trades Table
        active_box = QGroupBox("Live Active Open Positions (Real-time Tracker)")
        active_layout = QVBoxLayout()
        self.active_table = QTableWidget(0, 8)
        self.active_table.setHorizontalHeaderLabels(
            ["ID", "Time", "Asset", "Direction", "Amount", "Entry", "Current", "Countdown"]
        )
        self.active_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        active_layout.addWidget(self.active_table)
        active_box.setLayout(active_layout)
        right_layout.addWidget(active_box)

        # 3. Completed Trade History Table
        history_box = QGroupBox("Completed Trade History")
        history_layout = QVBoxLayout()
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels(
            ["ID", "Time", "Asset", "Direction", "Amount", "Outcome", "Profit/Loss"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        history_layout.addWidget(self.history_table)
        history_box.setLayout(history_layout)
        right_layout.addWidget(history_box)
        self._update_history_table()

        main_splitter.addWidget(center_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setSizes([900, 520])
        root_layout.addWidget(main_splitter)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Initializing NEXUS Terminal...")

        # Timers
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(self.settings.trading.payout_poll_interval_seconds * 1000)
        self.refresh_timer.timeout.connect(self._refresh_catalog_threaded)
        self.refresh_timer.start()

        self.balance_timer = QTimer(self)
        self.balance_timer.setInterval(5000)
        self.balance_timer.timeout.connect(self._update_balance_threaded)
        self.balance_timer.start()

        # 1-second live trade update timer
        self.trade_ticker = QTimer(self)
        self.trade_ticker.setInterval(1000)
        self.trade_ticker.timeout.connect(self._tick_live_trades)
        self.trade_ticker.start()

        # AI Auto-Trader Timer (runs every 10 seconds when enabled)
        self.ai_auto_timer = QTimer(self)
        self.ai_auto_timer.setInterval(10000)
        self.ai_auto_timer.timeout.connect(self._run_ai_auto_trade_tick)

        # Initial load
        self._refresh_catalog_threaded()
        self._update_balance_threaded()
        if self.engine.auto_login:
            self._auto_login_threaded()

    # ------------------------ UI Handlers ------------------------ #
    def _toggle_auto_trading(self) -> None:
        self._auto_trading_active = not self._auto_trading_active
        if self._auto_trading_active:
            self.auto_trade_btn.setText("⏸️ PAUSE AI AUTO-TRADER")
            self.auto_trade_btn.setStyleSheet(
                "background: #d50000; color: #ffffff; font-weight: bold; font-size: 13px; padding: 6px 14px; border-radius: 6px;"
            )
            self.status.showMessage(
                "🤖 AI Auto-Trader ACTIVATED (Scanning markets & executing SOTA trades...)"
            )
            self.ai_auto_timer.start()
            self._run_ai_auto_trade_tick()
        else:
            self.auto_trade_btn.setText("🤖 START AI AUTO-TRADER")
            self.auto_trade_btn.setStyleSheet(
                "background: #00c853; color: #ffffff; font-weight: bold; font-size: 13px; padding: 6px 14px; border-radius: 6px;"
            )
            self.status.showMessage("⏸️ AI Auto-Trader PAUSED")
            self.ai_auto_timer.stop()

    def _run_ai_auto_trade_tick(self) -> None:
        if self._panic or not self._auto_trading_active:
            return
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(self._run_ai_auto_trade_sync)

    def _run_ai_auto_trade_sync(self) -> None:
        if not self._auto_trade_lock.acquire(blocking=False):
            return
        try:
            self._run_ai_auto_trade_sync_locked()
        finally:
            self._auto_trade_lock.release()

    def _run_ai_auto_trade_sync_locked(self) -> None:
        broker = getattr(self.engine, "_broker", None)
        if broker is None or not getattr(broker, "authenticated", False):
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit("⚠️ AI Auto-Trader paused: live Quotex session required.")
            return
        live_symbols = {m.symbol for m in self._markets_cache if m.active}
        untrained = live_symbols - self._trained_market_symbols - self._training_excluded_symbols
        if untrained:
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit(
                    f"🤖 AI Auto-Trader waiting: train all live markets first ({len(untrained)} remaining)."
                )
            return
        with self._positions_lock:
            open_count = len(self.engine.active_positions)
        if open_count >= int(self.settings.trading.max_open_trades):
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit(
                    f"🤖 AI Auto-Trader: max open trades reached ({open_count}); waiting for settlement."
                )
            return
        threshold = self.settings.trading.payout_threshold
        candidates = [
            m
            for m in self._markets_cache
            if m.active and m.display_payout_percent >= threshold and m.display_payout_percent > 0
        ]
        if not candidates:
            candidates = [
                m for m in self._markets_cache if m.active and m.display_payout_percent > 0
            ]
        if not candidates:
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit(
                    "⚠️ AI Auto-Trader: No active live markets meeting payout threshold."
                )
            return

        # 1. Multi-Market Scanning & Pair Rotation
        top_candidates = sorted(candidates, key=lambda m: m.display_payout_percent, reverse=True)[
            :10
        ]

        min_conf = float(self.settings.trading.min_confidence)
        selected_market = None
        selected_prediction: dict = {}
        best_conf = 0.0

        num_candidates = len(top_candidates)
        for offset in range(num_candidates):
            idx = (self._auto_trade_rotation_idx + offset) % num_candidates
            candidate = top_candidates[idx]
            is_cand_otc = getattr(candidate, "otc", False) or ("otc" in candidate.symbol.lower())
            try:
                ai_res = self.engine.run_async(
                    self.engine.get_ai_prediction(candidate.symbol, is_otc=is_cand_otc)
                )
                conf = float(ai_res.get("confidence", 0.0))
                sig = str(ai_res.get("signal", "hold")).lower()
                if sig in ("call", "put") and conf >= min_conf:
                    if candidate.symbol != self._last_auto_traded_symbol or conf > (
                        best_conf + 0.05
                    ):
                        selected_market = candidate
                        selected_prediction = ai_res
                        best_conf = conf
                        self._auto_trade_rotation_idx = (idx + 1) % num_candidates
                        break
            except Exception as scan_err:
                logger.debug("Market scan error for %s: %s", candidate.symbol, scan_err)

        if not selected_market:
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit(
                    f"🤖 AI Auto-Trader: Scanning markets... No setup meets min confidence threshold ({min_conf * 100:.0f}%)."
                )
            return

        symbol = selected_market.symbol
        self._last_auto_traded_symbol = symbol
        is_market_otc = getattr(selected_market, "otc", False) or ("otc" in symbol.lower())

        if hasattr(self.chart_ai_prediction_signal, "emit"):
            self.chart_ai_prediction_signal.emit(selected_prediction)

        signal = str(selected_prediction.get("signal", "call")).lower()
        conf = float(selected_prediction.get("confidence", 0.65))

        if signal in ("call", "put") and conf >= min_conf:
            ai_engine = getattr(self.engine, "ai_engine", None)
            if ai_engine is not None and hasattr(ai_engine, "live_risk_gate"):
                allowed, reason = ai_engine.live_risk_gate(symbol, selected_prediction, min_conf)
                if not allowed:
                    logger.info("AI Auto-Trader skipped %s: %s", symbol, reason)
                    if hasattr(self.status_signal, "emit"):
                        self.status_signal.emit(f"🤖 AI skipped {symbol}: {reason}")
                    return
            payout = selected_market.display_payout_percent

            # 2. Dynamic Money Management (Position Sizing)
            try:
                bal_val = self.engine.run_async(self.engine.get_account_balance())
                balance = float(bal_val)
            except Exception as balance_err:
                logger.warning("AI Auto-Trader paused: broker balance unavailable: %s", balance_err)
                if hasattr(self.status_signal, "emit"):
                    self.status_signal.emit(
                        "⚠️ AI Auto-Trader paused: live broker balance unavailable."
                    )
                return

            if balance <= 0:
                logger.warning("AI Auto-Trader paused: broker returned non-positive balance")
                return

            # Scale risk (1.5% base) by confidence (0.50 -> 1.0x, 0.90 -> 1.8x)
            risk_fraction = 0.015 * (0.5 + min(1.0, max(0.0, conf)))
            calc_stake = balance * risk_fraction
            max_stake = max(10.0, balance * 0.05)
            stake = round(max(10.0, min(calc_stake, max_stake)), 2)

            # 3. Dynamic Time Management (AI Expiration: OTC 5s-900s, Real 60s-900s)
            ai_rec_exp = int(selected_prediction.get("recommended_expiration", 60))
            min_bound = 5 if is_market_otc else 60
            max_bound = 900  # 15 minutes max
            exp_seconds = (
                max(min_bound, min(max_bound, ai_rec_exp))
                if self.ai_timeframe_checkbox.isChecked()
                else max(
                    min_bound,
                    min(max_bound, int(self.settings.trading.default_expiration)),
                )
            )

            payout_map = selected_market.payout_per_expiration or {}
            # Fallback if specific expiration has significantly lower payout
            if str(exp_seconds) in payout_map and payout_map[str(exp_seconds)] < (payout - 15):
                valid_exps = [
                    int(k)
                    for k in payout_map.keys()
                    if k.isdigit() and min_bound <= int(k) <= max_bound
                ]
                if valid_exps:
                    exp_seconds = max(valid_exps, key=lambda k: payout_map.get(str(k), 0.0))

            regime = str(selected_prediction.get("regime", "neutral")).upper()
            if hasattr(self.status_signal, "emit"):
                self.status_signal.emit(
                    f"🤖 AI Auto-Trader: [{symbol}] {signal.upper()} (${stake:.2f}, {exp_seconds}s) | Payout: {payout:.0f}% | Conf: {conf * 100:.0f}% | Regime: {regime}"
                )

            try:
                ok, pos, err = self._execute_trade_sync(
                    asset=symbol,
                    direction=signal,
                    amount=stake,
                    seconds=exp_seconds,
                    analysis=selected_prediction,
                )
                if ok and pos:
                    logger.info("AI Auto-Trader position logged: %s", pos["id"])
                elif err:
                    logger.warning("AI Auto-Trader trade rejected: %s", err)
                    # A broker timeout is ambiguous: the order may have been
                    # accepted even though the response was lost. Do not place
                    # another order until the operator reviews the account.
                    err_lower = str(err).lower()
                    if "confirm" in err_lower or "unverified" in err_lower or "timeout" in err_lower:
                        self._auto_trading_active = False
                        logger.error(
                            "AI Auto-Trader paused after ambiguous broker confirmation: %s", err
                        )
                self._update_balance_threaded()
            except Exception as trd_err:
                logger.warning("Auto-trader trade execution error: %s", trd_err)
                self._auto_trading_active = False

    def _toggle_demo_mode(self) -> None:
        self.engine.demo_mode = self.demo_checkbox.isChecked()
        mode = "DEMO" if self.engine.demo_mode else "REAL"
        self.status.showMessage(f"Account mode switched to {mode}")
        self._update_balance_threaded()

    def _toggle_payout_filter(self) -> None:
        self._payout_filter_enabled = self.filter_checkbox.isChecked()
        self._populate_markets_table()

    def _on_payout_threshold_changed(self, val: int) -> None:
        self.settings.trading.payout_threshold = float(val)
        self.status.showMessage(f"Minimum Payout Threshold set to {val}%")
        self._populate_markets_table()

    def _handle_override(self) -> None:
        enabled = is_override_enabled()
        set_payout_override(not enabled, user="gui", reason="user toggle")
        self.override_btn.setText(
            "Disable Payout Override" if not enabled else "Enable Payout Override"
        )
        self.status.showMessage(f"Payout override {'ENABLED' if not enabled else 'DISABLED'}")
        self._populate_markets_table()

    def _panic_stop(self) -> None:
        self._panic = True
        self.engine.demo_mode = True
        self.demo_checkbox.setChecked(True)
        set_payout_override(False)
        QMessageBox.warning(self, "PANIC STOP", "Trading halted. Demo mode enforced.")

    def _autonomy_changed(self) -> None:
        val = self.autonomy_slider.value() / 100.0
        self.engine.exploration_controller.cfg.base_epsilon = val
        self.autonomy_label.setText(f"Autonomy: {val:.2f}")

    def _confidence_changed(self) -> None:
        val = self.confidence_slider.value() / 100.0
        self.settings.trading.min_confidence = val
        self.confidence_label.setText(f"Min Conf: {int(val * 100)}%")

    def _on_chart_ai_prediction_main_thread(self, pred: dict) -> None:
        if hasattr(self, "chart_widget") and self.chart_widget is not None:
            self.chart_widget.set_ai_prediction(pred)

    def _update_chart_ai_prediction_threaded(self, symbol: str) -> None:
        if not symbol:
            return
        is_otc = "otc" in symbol.lower()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._fetch_ai_prediction_sync, symbol, is_otc)
        future.add_done_callback(
            lambda f: (
                self.chart_ai_prediction_signal.emit(f.result())
                if not f.exception() and hasattr(self.chart_ai_prediction_signal, "emit")
                else None
            )
        )

    def _fetch_ai_prediction_sync(self, symbol: str, is_otc: bool) -> dict:
        try:
            res = self.engine.run_async(self.engine.get_ai_prediction(symbol, is_otc=is_otc))
            return cast(dict, res) if isinstance(res, dict) else {}
        except Exception:
            return {}

    def _on_market_selected(self) -> None:
        """Retained for compatibility with older UI integrations."""
        return None

    def _on_analysis_market_selected(self) -> None:
        """Load the selected AI-market row into the chart and asset selector."""
        row = self.analysis_table.currentRow()
        if row < 0:
            return
        item = self.analysis_table.item(row, 0)
        if item is not None:
            self._on_asset_changed(item.text().strip())

    def _on_asset_changed(self, text: str) -> None:
        sym = text.strip()
        if sym:
            self.chart_widget.set_symbol(sym)
            self._fetch_real_candles_threaded(sym)
            self._update_chart_ai_prediction_threaded(sym)

    # ------------------------ Thread-Safe Main-Thread Slots ----------------- #
    def _update_status_main_thread(self, message: str) -> None:
        self.status.showMessage(message)

    def _on_candles_loaded_main_thread(self, symbol: str, raw_candles: object) -> None:
        if self.chart_widget.symbol == symbol and isinstance(raw_candles, list) and raw_candles:
            self.chart_widget.update_real_candles(raw_candles)

    def _on_login_done_main_thread(self, success: bool, balance: float, error: str | None) -> None:
        mode_str = "DEMO" if self.engine.demo_mode else "REAL"
        if success:
            self.status.showMessage(f"Connected to Quotex Broker ({mode_str})")
            self.balance_card.setText(f"Balance: ${balance:,.2f} ({mode_str})")
        else:
            self.status.showMessage(f"Quotex broker unavailable ({mode_str}): {error or 'unknown error'}")
            self.balance_card.setText(f"Balance: unavailable ({mode_str})")

    def _on_catalog_refreshed_main_thread(
        self, success: bool, markets: List[Market] | None, error: str | None
    ) -> None:
        if success and markets:
            seen_syms: set[str] = set()
            dedup_markets: List[Market] = []
            for m in markets:
                if m.symbol not in seen_syms:
                    seen_syms.add(m.symbol)
                    dedup_markets.append(m)
            self._markets_cache = dedup_markets
            self._populate_markets_table()
            self._update_asset_selector()
            stamp = datetime.now(UTC).strftime("%H:%M:%S")
            self.status.showMessage(f"Refreshed {len(dedup_markets)} markets @ {stamp}")
            self._update_balance_threaded()
        elif error:
            logger.error("Catalog refresh error: %s", error)

    def _on_balance_done_main_thread(
        self, success: bool, balance: float, error: str | None
    ) -> None:
        mode_str = "DEMO" if self.engine.demo_mode else "REAL"
        if not success:
            self.balance_card.setText(f"Balance: unavailable ({mode_str})")
            self.status.showMessage(f"Balance refresh failed: {error or 'broker unavailable'}")
            return
        color_str = "#00e676" if self.engine.demo_mode else "#ff9100"
        self.balance_card.setText(f"Balance: ${balance:,.2f} ({mode_str})")
        self.balance_card.setStyleSheet(
            f"font-size: 15px; font-weight: bold; color: {color_str}; padding: 6px 12px; background: #1e1e1e; border: 1px solid {color_str}; border-radius: 6px;"
        )

    def _on_trade_executed_main_thread(
        self, success: bool, pos: dict | None, error: str | None
    ) -> None:
        if success and pos:
            self.status.showMessage(
                f"Trade placed: {pos['id']} {pos['asset']} {pos['direction']} (${pos['amount']:.2f})"
            )
            self._update_active_trades_table()
            self._update_balance_threaded()
        else:
            QMessageBox.warning(self, "Trade Error", f"Failed placing trade: {error}")

    def _on_market_trained_main_thread(self, res: dict) -> None:
        self.train_btn.setEnabled(True)
        if not res.get("success"):
            err = res.get("error", "Unknown error")
            self.status.showMessage(f"⚠️ Training failed: {err}")
            return
        results = res.get("results", [])
        count = len(results)
        if count > 0:
            trained = [item for item in results if item.get("status") == "trained" or item.get("accuracy")]
            skipped = count - len(trained)
            last = trained[-1] if trained else results[-1]
            last_sym = last.get("symbol", "Market")
            last_acc = last.get("accuracy", 80.0)
            last_inds = last.get("best_indicators", [])
            ind_str = ", ".join(last_inds[:2]) if last_inds else "Confluence"
            self.status.showMessage(
                f"⚡ Training complete: {len(trained)} trained, {skipped} skipped. "
                f"Latest: {last_sym} Acc: {last_acc}%, Indicators: {ind_str}"
            )
            current_symbol = self.asset_selector.currentText().strip() or "EURUSD"
            self._update_chart_ai_prediction_threaded(current_symbol)
            if hasattr(self, "chart_widget"):
                self.chart_widget.symbol = current_symbol
                self.chart_widget.draw_chart()

    def _on_market_training_progress_main_thread(self, progress: dict) -> None:
        symbol = str(progress.get("symbol", ""))
        if not symbol:
            return
        row = next(
            (r for r in range(self.training_status_table.rowCount())
             if self.training_status_table.item(r, 0)
             and self.training_status_table.item(r, 0).text() == symbol),
            -1,
        )
        if row < 0:
            row = self.training_status_table.rowCount()
            self.training_status_table.insertRow(row)
        status = str(progress.get("status", "queued"))
        generation = progress.get("generation", "—")
        if status == "trained":
            self._trained_market_symbols.add(symbol)
            self._training_excluded_symbols.discard(symbol)
        elif status.startswith("skipped"):
            self._training_excluded_symbols.add(symbol)
            self._trained_market_symbols.discard(symbol)
        elif status == "queued":
            self._training_excluded_symbols.discard(symbol)
        self.training_status_table.setItem(row, 0, QTableWidgetItem(symbol))
        self.training_status_table.setItem(row, 1, QTableWidgetItem(status))
        self.training_status_table.setItem(row, 2, QTableWidgetItem(str(generation)))
        self.status.showMessage(f"AI training {symbol}: {status}")

    def _analyze_all_markets_threaded(self) -> None:
        markets = list(self._markets_cache)
        if not markets:
            self.analysis_summary.setText("No markets loaded. Refresh the live catalog first.")
            return
        self.analyze_markets_btn.setEnabled(False)
        self.analysis_summary.setText(f"Analyzing {len(markets)} markets with live candles and payouts…")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._analyze_all_markets_sync, markets)
        future.add_done_callback(lambda f: self._on_market_analysis_done(f))

    def _analyze_all_markets_sync(self, markets: List[Market]) -> MarketAnalysisResult:
        from nexus.ai.engine_ai import RealAITradingEngine

        ai_engine = self.engine.ai_engine or RealAITradingEngine()
        broker = getattr(self.engine, "_broker", None)

        async def fetch_candles(symbol: str):
            if broker is None or not hasattr(broker, "get_candles_async"):
                return None
            try:
                return await broker.get_candles_async(symbol, timeframe_sec=60, limit=100)
            except Exception:
                return None

        analyzer = MarketAnalyzer(
            base_stake=self.settings.trading.base_trade_amount,
            cycle_seconds=self.settings.trading.auto_trade_interval_seconds,
            max_markets_per_cycle=self.settings.trading.max_open_trades * 20,
        )
        return self.engine.run_async(
            analyzer.analyze(
                markets,
                ai_engine,
                candles_fetcher=fetch_candles,
                min_confidence=self.settings.trading.min_confidence,
                autonomy=self.autonomy_slider.value() / 100.0,
                use_price_gate=self.price_gate_checkbox.isChecked(),
            )
        )

    def _on_market_analysis_done(self, future: concurrent.futures.Future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            result = exc
        if hasattr(self.market_analysis_signal, "emit"):
            self.market_analysis_signal.emit(result)

    def _on_market_analysis_main_thread(self, result: object) -> None:
        self.analyze_markets_btn.setEnabled(True)
        if not isinstance(result, MarketAnalysisResult):
            self.analysis_summary.setText(f"Market analysis failed: {result}")
            return
        self._latest_market_analysis = result
        self.analysis_table.setRowCount(len(result.opportunities))
        for row, item in enumerate(result.opportunities):
            values = [
                item.symbol,
                f"{item.payout_1m:.1f}%",
                item.payout_source,
                item.signal.upper(),
                f"{item.confidence * 100:.1f}%",
                item.regime,
                f"${item.expected_value_per_unit:.3f}",
                f"{item.recommended_entry_price:g}" if item.recommended_entry_price else item.price_gate,
                f"{item.recommended_expiration}s",
            ]
            for col, value in enumerate(values):
                self.analysis_table.setItem(row, col, QTableWidgetItem(value))
        ranked_scenarios = sorted(
            result.scenarios,
            key=lambda scenario: scenario.expected_profit_1h,
            reverse=True,
        )
        self.scenario_table.setRowCount(len(ranked_scenarios))
        for row, scenario in enumerate(ranked_scenarios, 1):
            is_best = result.best_scenario == scenario
            values = [
                "BEST" if is_best else str(row),
                f"{scenario.autonomy:.0%}",
                f"{scenario.min_confidence:.0%}",
                str(scenario.eligible_markets),
                str(scenario.trades_10m),
                f"${scenario.expected_profit_10m:.2f}",
                str(scenario.trades_15m),
                f"${scenario.expected_profit_15m:.2f}",
                str(scenario.trades_30m),
                f"${scenario.expected_profit_30m:.2f}",
                str(scenario.trades_1h),
                f"${scenario.expected_profit_1h:.2f}",
                "All eligible markets",
            ]
            for col, value in enumerate(values):
                self.scenario_table.setItem(row - 1, col, QTableWidgetItem(value))
        best = result.best_scenario
        if best:
            self.analysis_summary.setText(
                f"Best scenario (estimate): autonomy {best.autonomy:.0%}, min confidence "
                f"{best.min_confidence:.0%} → {best.trades_10m}/{best.trades_15m}/{best.trades_30m}/"
                f"{best.trades_1h} trades in 10m/15m/30m/1h; expected P&L "
                f"${best.expected_profit_10m:.2f}/${best.expected_profit_15m:.2f}/"
                f"${best.expected_profit_30m:.2f}/${best.expected_profit_1h:.2f}. "
                "These are EV estimates, not guaranteed returns."
            )
        else:
            self.analysis_summary.setText("No eligible positive-EV markets met the current confidence threshold.")

    # ------------------------ Threaded Operations ------------------------- #
    def _on_refresh_button_click(self) -> None:
        self._refresh_catalog_threaded()
        self._update_balance_threaded()

    def _train_market_threaded(self) -> None:
        symbols = [m.symbol for m in self._markets_cache] if self._markets_cache else []
        if not symbols:
            cur = self.asset_selector.currentText().strip()
            symbols = [cur] if cur else ["EURUSD"]
        self.train_btn.setEnabled(False)
        self.training_status_table.setRowCount(0)
        for symbol in symbols:
            self._on_market_training_progress_main_thread({"symbol": symbol, "status": "queued"})
        self.status.showMessage(
            f"⚡ Training AI models for {len(symbols)} markets... Finding optimal indicators..."
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._train_market_sync, symbols)
        future.add_done_callback(lambda f: self._on_market_trained(f))

    def _train_market_sync(self, symbols: List[str]) -> dict:
        try:
            broker = getattr(self.engine, "_broker", None)
            ai_engine = self.engine.ai_engine
            if broker is None or not getattr(broker, "authenticated", False):
                return {"success": False, "error": "Live Quotex session required for training"}
            if ai_engine is None:
                from nexus.ai.engine_ai import RealAITradingEngine

                ai_engine = RealAITradingEngine()
                self.engine.ai_engine = ai_engine
            results = []
            for symbol in symbols:
                self.market_training_progress_signal.emit(
                    {"symbol": symbol, "status": "training"}
                )
                candles = self.engine.run_async(broker.get_candles_async(symbol, 60, 150))
                if candles is None or len(candles) < 30:
                    results.append(
                        {"symbol": symbol, "status": "skipped: insufficient live candles"}
                    )
                    self.market_training_progress_signal.emit(
                        {"symbol": symbol, "status": "skipped: insufficient live candles"}
                    )
                    continue
                train_data = candles if hasattr(candles, "columns") else pd.DataFrame(candles)
                result = ai_engine.train_market(symbol, train_data)
                results.append(result)
                self.market_training_progress_signal.emit(
                    {
                        "symbol": symbol,
                        "status": "trained",
                        "generation": result.get("generation", "—"),
                    }
                )
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _on_market_trained(self, future: concurrent.futures.Future) -> None:
        res = future.result()
        if hasattr(self.market_trained_signal, "emit"):
            self.market_trained_signal.emit(res)

    def _fetch_real_candles_threaded(self, symbol: str) -> None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(self._fetch_real_candles_sync, symbol)

    def _fetch_real_candles_sync(self, symbol: str) -> None:
        try:
            broker = getattr(self.engine, "_broker", None)
            if broker and hasattr(broker, "get_candles_async"):
                candles = self.engine.run_async(broker.get_candles_async(symbol, 60, 50))
                if candles and hasattr(self.candles_loaded_signal, "emit"):
                    self.candles_loaded_signal.emit(symbol, candles)
        except Exception as e:
            logger.debug("Candle fetch exception for %s: %s", symbol, e)

    def _auto_login_threaded(self) -> None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._auto_login_sync)
        future.add_done_callback(lambda f: self._on_login_done(f))

    def _auto_login_sync(self) -> tuple[bool, float, str | None]:
        try:
            ok = self.engine.run_async(self.engine.login_broker())
            balance = self.engine.run_async(self.engine.get_account_balance())
            return (
                ok,
                balance,
                None if ok else "Not logged into live broker (Simulated Mode active)",
            )
        except Exception as e:
            return (False, 10000.0, str(e))

    def _on_login_done(self, future: concurrent.futures.Future) -> None:
        res = future.result()
        if hasattr(self.login_done_signal, "emit"):
            self.login_done_signal.emit(*res)

    def _refresh_catalog_threaded(self) -> None:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._refresh_catalog_sync)
        future.add_done_callback(lambda f: self._on_catalog_refreshed(f))

    def _refresh_catalog_sync(self) -> tuple[bool, List[Market] | None, str | None]:
        try:
            from nexus.catalog.ingest import fetch_live_catalog, get_market_catalog

            broker = getattr(self.engine, "_broker", None)
            if broker is not None and getattr(broker, "authenticated", False):
                markets = self.engine.run_async(fetch_live_catalog(broker))
            else:
                markets = self.engine.run_async(get_market_catalog())
            return (True, markets, None)
        except Exception as e:
            return (False, None, str(e))

    def _on_catalog_refreshed(self, future: concurrent.futures.Future) -> None:
        res = future.result()
        if hasattr(self.catalog_refreshed_signal, "emit"):
            self.catalog_refreshed_signal.emit(*res)

    def _update_asset_selector(self) -> None:
        current = self.asset_selector.currentText()
        self.asset_selector.clear()
        symbols: List[str] = []
        for m in self._markets_cache:
            if m.symbol not in symbols:
                symbols.append(m.symbol)
        self.asset_selector.addItems(symbols)
        if current in symbols:
            self.asset_selector.setCurrentText(current)

    def _execute_manual_trade_threaded(self) -> None:
        asset = self.asset_selector.currentText().strip() or "EURUSD"
        dir_text = self.direction_combo.currentText()
        direction = "call" if "CALL" in dir_text else "put"
        amount = self.amount_spin.value()
        exp_text = self.expiration_combo.currentText()

        import re

        is_otc = "otc" in asset.lower()
        min_bound = 5 if is_otc else 60
        max_bound = 900

        m_num = re.search(r"(\d+)", exp_text)
        raw_sec = int(m_num.group(1)) if m_num else 60
        if "min" in exp_text.lower() and "s" not in exp_text.split("min")[0]:
            raw_sec *= 60

        seconds = max(min_bound, min(max_bound, raw_sec))

        # Manual orders can opt into the latest per-market AI recommendation.
        if self.ai_timeframe_checkbox.isChecked() and self._latest_market_analysis:
            for opportunity in self._latest_market_analysis.opportunities:
                if opportunity.symbol == asset:
                    seconds = max(
                        min_bound,
                        min(max_bound, int(opportunity.recommended_expiration)),
                    )
                    break

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._execute_trade_sync, asset, direction, amount, seconds)
        future.add_done_callback(lambda f: self._on_trade_executed(f))

    def _execute_trade_sync(
        self,
        asset: str,
        direction: str,
        amount: float,
        seconds: int,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, dict | None, str | None]:
        try:
            trade_id = f"TRD-{uuid.uuid4().hex[:12].upper()}"
            payout = 85.0
            for m in self._markets_cache:
                if m.symbol == asset:
                    payout = m.effective_payout(str(seconds))
                    break

            base_price = (
                1.0850
                if "EUR" in asset
                else 150.25
                if "JPY" in asset
                else 65000.0
                if "BTC" in asset
                else 2400.0
                if "XAU" in asset
                else 100.0
            )
            entry_price = round(base_price * (1.0 + random.uniform(-0.001, 0.001)), 5)

            try:
                res = self.engine.run_async(
                    self.engine.execute_trade(
                        asset=asset, signal_type=direction, amount=amount, expiration=seconds
                    )
                )
                if not res or not res.get("success"):
                    err_msg = (
                        res.get("error", "Broker rejected trade placement")
                        if res
                        else "Broker trade placement failed"
                    )
                    logger.warning("Broker trade rejected: %s", err_msg)
                    return (False, None, err_msg)
                logger.info("Real broker trade confirmed by Quotex server: %s", res)
            except Exception as trd_err:
                logger.warning("Broker dispatch error: %s", trd_err)
                return (False, None, str(trd_err))

            real_order_id = (
                str(res.get("order_id") or res.get("order", {}).get("id") or "")
                if isinstance(res, dict)
                else ""
            )

            position = {
                "id": trade_id,
                "broker_order_id": real_order_id,
                "timestamp": datetime.now(UTC).strftime("%H:%M:%S"),
                "asset": asset,
                "direction": direction.upper(),
                "amount": amount,
                "expiration": seconds,
                "entry_price": entry_price,
                "current_price": entry_price,
                "payout": payout,
                "time_remaining": seconds,
                "status": "ACTIVE",
                "analysis": analysis or {},
            }
            # Write the placement before adding it to the in-memory list. A
            # GUI crash, duplicate broker response, or missing settlement can
            # therefore never make a broker order disappear from the app.
            self._trade_store.record_placement(
                {
                    "local_id": trade_id,
                    "broker_order_id": real_order_id,
                    "asset": asset,
                    "direction": direction.upper(),
                    "amount": amount,
                    "expiration": seconds,
                    "status": "PLACED" if real_order_id else "UNVERIFIED",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": None if real_order_id else "Broker confirmed without an order id",
                }
            )
            if self.engine.demo_mode:
                self.engine.virtual_balance -= amount

            with self._positions_lock:
                self.engine.active_positions.append(position)
            return (True, position, None)
        except Exception as e:
            return (False, None, str(e))

    def _on_trade_executed(self, future: concurrent.futures.Future) -> None:
        res = future.result()
        if hasattr(self.trade_executed_signal, "emit"):
            self.trade_executed_signal.emit(*res)

    def _update_balance_threaded(self) -> None:
        if not self._balance_refresh_lock.acquire(blocking=False):
            return
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._update_balance_sync)
        future.add_done_callback(self._on_balance_done)

    def _update_balance_sync(self) -> tuple[bool, float, str | None]:
        try:
            broker = getattr(self.engine, "_broker", None)
            if broker is None or not getattr(broker, "authenticated", False):
                return (False, self.engine.virtual_balance, "Quotex broker is not authenticated")
            balance = self.engine.run_async(self.engine.get_account_balance())
            return (True, balance, None)
        except Exception as e:
            return (False, self.engine.virtual_balance, str(e))

    def _on_balance_done(self, future: concurrent.futures.Future) -> None:
        try:
            res = future.result()
        except Exception as exc:
            res = (False, self.engine.virtual_balance, str(exc))
        finally:
            self._balance_refresh_lock.release()
        if hasattr(self.balance_done_signal, "emit"):
            self.balance_done_signal.emit(*res)

    # ------------------------ Real-Time Position Ticker ------------------------- #
    def _tick_live_trades(self) -> None:
        if not hasattr(self.engine, "active_positions"):
            return

        current_asset = self.asset_selector.currentText().strip()
        to_remove = []
        broker = getattr(self.engine, "_broker", None)
        with self._positions_lock:
            positions = list(self.engine.active_positions)
        for pos in positions:
            pos["time_remaining"] -= 1

            step = random.uniform(-0.0003, 0.0003) * pos["entry_price"]
            pos["current_price"] = round(pos["current_price"] + step, 5)

            if pos["asset"] == current_asset:
                self.chart_widget.update_price(pos["current_price"])

            if pos["time_remaining"] <= 0:
                broker_id = pos.get("broker_order_id")
                real_result = None
                if broker and hasattr(broker, "get_trade_outcome") and broker_id:
                    real_result = broker.get_trade_outcome(str(broker_id))

                if real_result and real_result.get("status") == "SETTLED" and str(
                    real_result.get("outcome", "")
                ).upper() in {"WIN", "LOSS", "EQUAL"}:
                    pos["outcome"] = str(real_result["outcome"]).upper()
                    pos["profit"] = float(real_result.get("profit", 0.0))
                    self._trade_store.record_settlement(
                        str(pos["id"]), pos["outcome"], pos["profit"]
                    )
                    ai_engine = getattr(self.engine, "ai_engine", None)
                    if ai_engine is not None and pos.get("analysis"):
                        try:
                            self.engine.run_async(
                                ai_engine.learn_and_evolve(
                                    asset=pos["asset"],
                                    signal_type=str(pos["direction"]).lower(),
                                    success=pos["outcome"] == "WIN",
                                    profit=pos["profit"],
                                    analysis=pos["analysis"],
                                )
                            )
                        except Exception as learn_err:
                            logger.warning("AI feedback update failed for %s: %s", pos["id"], learn_err)
                    if self.engine.demo_mode:
                        payout_gain = (
                            pos["amount"] + pos["profit"]
                            if pos["profit"] > 0
                            else (pos["amount"] if pos["profit"] == 0 else 0.0)
                        )
                        self.engine.virtual_balance += payout_gain
                    to_remove.append(pos)
                elif broker is not None and getattr(broker, "authenticated", False):
                    # Keep it pending while the broker's settlement feed catches
                    # up. Never turn missing feedback into a loss or a win.
                    pos["status"] = "PENDING SETTLEMENT"
                    if pos["time_remaining"] <= -30:
                        pos["outcome"] = "UNVERIFIED"
                        pos["profit"] = 0.0
                        pos["report_error"] = "Broker settlement feedback unavailable"
                        self._trade_store.record_settlement(
                            str(pos["id"]), "UNVERIFIED", 0.0,
                            status="UNVERIFIED", error=pos["report_error"]
                        )
                        to_remove.append(pos)
                elif (
                    pos["time_remaining"] <= -10
                    or broker is None
                    or not getattr(broker, "authenticated", False)
                ):
                    win = (
                        pos["current_price"] >= pos["entry_price"]
                        if pos["direction"] == "CALL"
                        else pos["current_price"] <= pos["entry_price"]
                    )
                    if win:
                        payout_gain = round(pos["amount"] * (1.0 + pos["payout"] / 100.0), 2)
                        profit = round(pos["amount"] * (pos["payout"] / 100.0), 2)
                        pos["outcome"] = "WIN"
                        pos["profit"] = profit
                        if self.engine.demo_mode:
                            self.engine.virtual_balance += payout_gain
                    else:
                        profit = -round(pos["amount"], 2)
                        pos["outcome"] = "LOSS"
                        pos["profit"] = profit
                    to_remove.append(pos)

        for pos in to_remove:
            with self._positions_lock:
                if pos in self.engine.active_positions:
                    self.engine.active_positions.remove(pos)
            self.engine.trade_history.insert(0, pos)

        if to_remove:
            self._update_balance_threaded()
            self._update_history_table()

        self._update_active_trades_table()

    def _populate_markets_table(self) -> None:
        """Compatibility no-op; the AI Market Lab owns market presentation."""
        return None

    def _update_active_trades_table(self) -> None:
        self.active_table.setRowCount(len(self.engine.active_positions))
        for row, pos in enumerate(self.engine.active_positions):
            self.active_table.setItem(row, 0, QTableWidgetItem(str(pos["id"])))
            self.active_table.setItem(row, 1, QTableWidgetItem(str(pos["timestamp"])))
            self.active_table.setItem(row, 2, QTableWidgetItem(str(pos["asset"])))

            dir_item = QTableWidgetItem(str(pos["direction"]))
            dir_item.setForeground(QColor("#00e676" if pos["direction"] == "CALL" else "#ff5252"))
            self.active_table.setItem(row, 3, dir_item)

            self.active_table.setItem(row, 4, QTableWidgetItem(f"${pos['amount']:.2f}"))
            self.active_table.setItem(row, 5, QTableWidgetItem(f"{pos['entry_price']:.5f}"))
            self.active_table.setItem(row, 6, QTableWidgetItem(f"{pos['current_price']:.5f}"))

            time_item = QTableWidgetItem(f"{pos['time_remaining']}s")
            time_item.setForeground(QColor("#ffb74d"))
            self.active_table.setItem(row, 7, time_item)

    def _update_history_table(self) -> None:
        self.history_table.setRowCount(len(self.engine.trade_history))
        for row, pos in enumerate(self.engine.trade_history):
            self.history_table.setItem(row, 0, QTableWidgetItem(str(pos["id"])))
            self.history_table.setItem(row, 1, QTableWidgetItem(str(pos["timestamp"])))
            self.history_table.setItem(row, 2, QTableWidgetItem(str(pos["asset"])))

            dir_item = QTableWidgetItem(str(pos["direction"]))
            dir_item.setForeground(QColor("#00e676" if pos["direction"] == "CALL" else "#ff5252"))
            self.history_table.setItem(row, 3, dir_item)

            self.history_table.setItem(row, 4, QTableWidgetItem(f"${pos['amount']:.2f}"))

            outcome_item = QTableWidgetItem(str(pos["outcome"]))
            outcome = str(pos.get("outcome", pos.get("result", "UNVERIFIED"))).upper()
            outcome_item.setForeground(
                QColor("#00e676" if outcome == "WIN" else "#ffb74d" if outcome in {"UNVERIFIED", "PENDING"} else "#ff5252")
            )
            self.history_table.setItem(row, 5, outcome_item)

            p_sign = "+" if pos["profit"] > 0 else ""
            profit_item = QTableWidgetItem(f"{p_sign}${pos['profit']:.2f}")
            profit_item.setForeground(QColor("#00e676" if pos["profit"] > 0 else "#ff5252"))
            self.history_table.setItem(row, 6, profit_item)

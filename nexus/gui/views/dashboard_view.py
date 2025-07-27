from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QFrame, QGraphicsView, QGraphicsScene, QPushButton, QComboBox, QTabWidget, QTextEdit, QSizePolicy
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QObject
from nexus.adapters.quotex import QuotexAdapter
from nexus.intelligence.regime_detector import RegimeDetector
from nexus.data.trade_history import TradeHistory, AdvancedDataStore
import threading
import asyncio
import pyqtgraph as pg

class CandleFetcher(QObject):
    candles_fetched = Signal(list)

    def __init__(self, quotex_adapter, asset, timeframe, count):
        super().__init__()
        self.quotex_adapter = quotex_adapter
        self.asset = asset
        self.timeframe = timeframe
        self.count = count
        self._running = True

    def stop(self):
        self._running = False

    def fetch_candles(self):
        try:
            if not self._running:
                return
            candles = self.quotex_adapter.get_candle(self.asset, self.timeframe, self.count, period=60)
            if self._running:
                self.candles_fetched.emit(candles)
        except Exception as e:
            print(f"Error fetching candles: {e}")

class DashboardView(QWidget):
    update_info_signal = Signal(str)
    update_regime_signal = Signal(str)
    update_trades_signal = Signal(list)

    def __init__(self, quotex_adapter: QuotexAdapter):
        super().__init__()
        self.quotex_adapter = quotex_adapter
        self.regime_detector = RegimeDetector(sensitivity=0.5)
        self.trade_history = TradeHistory()
        self.data_store = AdvancedDataStore()
        self.setStyleSheet("font-size: 18px; color: #e0e0e0;")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Info label
        self.info_label = QLabel("Loading account info...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label)

        # Tabs for analytics
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Performance tab
        self.performance_tab = QWidget()
        self.performance_layout = QVBoxLayout()
        self.performance_tab.setLayout(self.performance_layout)
        self.tabs.addTab(self.performance_tab, "Performance")

        # Live PnL chart
        self.pnl_plot = pg.PlotWidget(title="PnL Over Time")
        self.performance_layout.addWidget(self.pnl_plot)
        self.pnl_curve = self.pnl_plot.plot(pen=pg.mkPen('g', width=2))

        # Drawdown chart
        self.drawdown_plot = pg.PlotWidget(title="Drawdown")
        self.performance_layout.addWidget(self.drawdown_plot)
        self.drawdown_curve = self.drawdown_plot.plot(pen=pg.mkPen('r', width=2))

        # Regime tab
        self.regime_tab = QWidget()
        self.regime_layout = QVBoxLayout()
        self.regime_tab.setLayout(self.regime_layout)
        self.tabs.addTab(self.regime_tab, "Regime")
        self.regime_label = QLabel("Current Regime: Unknown")
        self.regime_label.setAlignment(Qt.AlignCenter)
        self.regime_layout.addWidget(self.regime_label)
        self.regime_plot = pg.PlotWidget(title="Regime State Over Time")
        self.regime_layout.addWidget(self.regime_plot)
        self.regime_curve = self.regime_plot.plot(pen=pg.mkPen('b', width=2))

        # Trades tab
        self.trades_tab = QWidget()
        self.trades_layout = QVBoxLayout()
        self.trades_tab.setLayout(self.trades_layout)
        self.tabs.addTab(self.trades_tab, "Trades")
        self.trades_text = QTextEdit()
        self.trades_text.setReadOnly(True)
        self.trades_layout.addWidget(self.trades_text)

        # Controls tab
        self.controls_tab = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_tab.setLayout(self.controls_layout)
        self.tabs.addTab(self.controls_tab, "Controls")
        self.strategy_combo = QComboBox()
        self.strategy_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.strategy_combo.addItem("meta_strategy")
        try:
            from nexus.registry import registry
            for name in registry.list_strategies():
                if name != "meta_strategy":
                    self.strategy_combo.addItem(name)
        except Exception:
            pass
        self.controls_layout.addWidget(QLabel("Switch Strategy:"))
        self.controls_layout.addWidget(self.strategy_combo)
        self.switch_btn = QPushButton("Switch Strategy")
        self.controls_layout.addWidget(self.switch_btn)
        self.switch_btn.clicked.connect(self.switch_strategy)
        self.risk_combo = QComboBox()
        self.risk_combo.addItems(["kelly", "var", "emotional"])
        self.controls_layout.addWidget(QLabel("Select Risk Model:"))
        self.controls_layout.addWidget(self.risk_combo)
        self.live_toggle = QPushButton("Switch to Paper Trading")
        self.live_toggle.setCheckable(True)
        self.controls_layout.addWidget(self.live_toggle)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.controls_layout.addWidget(QLabel("System Log:"))
        self.controls_layout.addWidget(self.log_output)

        # Candle chart placeholder
        self.chart_view = QGraphicsView()
        self.chart_scene = QGraphicsScene()
        self.chart_view.setScene(self.chart_scene)
        self.layout.addWidget(self.chart_view)

        self.update_info_signal.connect(self._set_info_label)
        self.update_regime_signal.connect(self._set_regime_label)
        self.update_trades_signal.connect(self._set_trades_label)

        self._refresh_info()
        self._refresh_regime()
        self._refresh_trades()

        # Refresh every 10 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh_info)
        self.timer.timeout.connect(self._refresh_regime)
        self.timer.timeout.connect(self._refresh_trades)
        self.timer.start(10000)

        # Candle fetching: only start after login is ready, and always from the main thread
        def start_candle_thread():
            # This must run in the main thread
            self.candle_fetcher = CandleFetcher(self.quotex_adapter, "EURUSD", 1, 100)
            self.candle_fetcher.candles_fetched.connect(self._update_chart)
            self.candle_thread = QThread(self)
            self.candle_fetcher.moveToThread(self.candle_thread)
            self.candle_thread.started.connect(self.candle_fetcher.fetch_candles)
            self.candle_thread.start()

        def wait_and_start():
            self.quotex_adapter.login_ready.wait()
            # Use Qt's singleShot to ensure this runs in the main thread
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, start_candle_thread)

        threading.Thread(target=wait_and_start, daemon=True).start()

    def _set_info_label(self, text):
        self.info_label.setText(text)

    def _set_regime_label(self, text):
        self.regime_label.setText(f"Market Regime: {text}")

    def _set_trades_label(self, trades):
        trades_text = "\n".join([
            f"{trade['timestamp']}: {trade['asset']} {trade['direction']} ${trade['amount']} ({trade['result']}, ${trade['profit']})"
            for trade in trades
        ])
        self.trades_label.setText(f"Recent Trades:\n{trades_text}")

    def _refresh_info(self):
        def fetch():
            try:
                # Create a new event loop for the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Login if not already
                if not self.quotex_adapter.authenticated:
                    loop.run_until_complete(self.quotex_adapter.login())

                # Fetch balance using the new event loop
                balance = loop.run_until_complete(self.quotex_adapter.get_balance_async())
                email = self.quotex_adapter.email
                mode = "Demo" if self.quotex_adapter.demo_mode else "Real"
                html = f"<b>Account:</b> {email}<br><b>Mode:</b> {mode}<br><b>Balance:</b> ${balance:,.2f}"
                self.update_info_signal.emit(html)
            except Exception as e:
                self.update_info_signal.emit(f"Error loading account info: {e}")
            finally:
                asyncio.set_event_loop(None)
        threading.Thread(target=fetch, daemon=True).start()

    def _refresh_regime(self):
        def fetch():
            try:
                # Use only the sync candle fetching method
                candles = self.quotex_adapter.get_candle("EURUSD", 1, 100, period=60)
                if not candles or not isinstance(candles, list) or len(candles) == 0:
                    self.update_regime_signal.emit("No candle data available.")
                    return
                regime = self.regime_detector.detect(candles)
                self.update_regime_signal.emit(regime)
            except Exception as e:
                self.update_regime_signal.emit(f"Error detecting regime: {e}")
        threading.Thread(target=fetch, daemon=True).start()

    def _refresh_trades(self):
        def fetch():
            try:
                trades = self.trade_history.get_trade_history(limit=10)
                self.update_trades_signal.emit(trades)
            except Exception as e:
                self.update_trades_signal.emit([])

    def _update_chart(self, candles):
        if not candles:
            self.info_label.setText("No candle data available.")
            return
        # Update the chart with the fetched candles
        self.chart_scene.clear()
        # Add logic to render candles on the chart

    def update_dashboard(self):
        # Update account info
        try:
            balance = self.quotex_adapter.get_balance()
            self.info_label.setText(f"Balance: ${balance:.2f}")
        except Exception as e:
            self.info_label.setText(f"Error: {e}")
        # Update trades
        trades = self.trade_history.get_trade_history(limit=100)
        self.trades_text.setPlainText("\n".join([
            f"{t['timestamp']} | {t['asset']} | {t['direction']} | {t['amount']} | {t['result']} | {t['profit']}" for t in trades
        ]))
        # Update PnL and drawdown charts
        pnl = [t['profit'] for t in trades]
        cum_pnl = [sum(pnl[:i+1]) for i in range(len(pnl))]
        self.pnl_curve.setData(cum_pnl)
        drawdown = [max(cum_pnl[:i+1]) - cum_pnl[i] for i in range(len(cum_pnl))]
        self.drawdown_curve.setData(drawdown)
        # Update regime info
        try:
            asset = trades[0]['asset'] if trades else "EURUSD"
            candles = self.quotex_adapter.get_candle(asset, 1, 200, period=60)
            regime = self.regime_detector.detect_regime(candles)
            self.regime_label.setText(f"Current Regime: {regime}")
            regime_states = [self.regime_detector.detect_regime(candles[max(0, i-50):i+1]) for i in range(len(candles))]
            regime_numeric = [self.regime_detector.REGIMES.index(r) if r in self.regime_detector.REGIMES else 0 for r in regime_states]
            self.regime_curve.setData(regime_numeric)
        except Exception as e:
            self.regime_label.setText(f"Regime Error: {e}")
        # Update log output (stub: could tail a log file or use a logger handler)
        try:
            with open("logs/nexus_20250717.log", "r") as f:
                lines = f.readlines()[-50:]
                self.log_output.setPlainText("".join(lines))
        except Exception:
            pass

    def switch_strategy(self):
        selected = self.strategy_combo.currentText()
        try:
            from nexus.registry import registry
            if selected in registry.strategies:
                # In production, switch strategy in engine
                print(f"Switched to strategy: {selected}")
        except Exception as e:
            print(f"Error switching strategy: {e}")

    def closeEvent(self, event):
        # Gracefully stop candle thread
        if hasattr(self, 'candle_fetcher') and hasattr(self, 'candle_thread'):
            self.candle_fetcher.stop()
            self.candle_thread.quit()
            self.candle_thread.wait()
        super().closeEvent(event)

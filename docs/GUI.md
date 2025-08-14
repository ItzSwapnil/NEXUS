# NEXUS GUI (PySide6)

The desktop GUI provides live control and monitoring for NEXUS.

Highlights
- Language: English only.
- Live account panel: email, mode (Demo/Real), and live balance.
- Dynamic controls: switch strategies, toggle paper vs. live trading, select risk model.
- Live metrics: PnL curve, drawdown, and recent trades stream.
- Regime view: current regime and recent regime trajectory.
- Candle preview: background thread fetches candles via the adapter and updates the chart.

Runtime dynamism
- Assets, timeframes, and intervals can be injected at launch from CLI or switched at runtime within the GUI (placeholder controls ready; wire to engine as needed).
- All adapter calls enforce lang="en" and run with robust retry and caching.

Mockup (wireframe)
- Sidebar: Dashboard, Trading, Analytics, Settings
- Dashboard tabs: Performance, Regime, Trades, Controls
- Controls: strategy dropdown, risk model dropdown, live/paper toggle, log tail panel

Notes
- The GUI uses signals/threads to avoid blocking the main Qt loop.
- Candle fetching waits until login is ready; all data is processed on the UI thread safely via Qt signals.
- You can launch it from CLI: `python -m nexus.gui.launch_gui` or `nexus --gui` if installed.


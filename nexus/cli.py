"""
NEXUS CLI Interface

Provides command-line management, monitoring, and control for the NEXUS AI trading system.
Supports live trading, paper trading, strategy switching, analytics, and registry management.
"""

import argparse
import asyncio
import os
import sys

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nexus.backtest.backtester import Backtester
from nexus.backtest.report_aggregator import aggregate_reports
from nexus.core.engine import NexusEngine
from nexus.evolution.evolver import EvolutionConfig, EvolutionRunner
from nexus.registry import registry
from nexus.strategies.meta_strategy import MetaStrategy
from nexus.utils.config import NexusSettings, load_runtime_settings

console = Console()


def _build_engine(settings: NexusSettings, demo: bool) -> NexusEngine:
    return NexusEngine(settings=settings, demo_mode=demo)


def _synthetic_df(rows: int) -> "pd.DataFrame":  # type: ignore
    data = {
        "open": [float(i) for i in range(rows)],
        "close": [float(i) + 0.25 for i in range(rows)],
        "high": [float(i) + 0.5 for i in range(rows)],
        "low": [float(i) - 0.5 for i in range(rows)],
        "volume": [1000.0 for _ in range(rows)],
    }
    return pd.DataFrame(data)


async def _fetch_candles_df(
    engine: NexusEngine, asset: str, timeframe_min: int, rows: int
) -> "pd.DataFrame":  # type: ignore
    """Login if needed, pull candles via broker adapter, and return a pandas DataFrame."""
    ok = await engine.login_broker()
    if not ok:
        raise RuntimeError(
            "Failed to login to Quotex; provide QUOTEX_EMAIL/QUOTEX_PASSWORD in env or settings"
        )
    # Ensure practice/real mode consistency
    try:
        if getattr(engine, "_broker", None) is not None:
            await engine._broker.set_practice_mode(bool(engine.demo_mode))  # type: ignore[attr-defined]
    except Exception:
        pass
    broker = getattr(engine, "_broker", None)
    if broker is None:
        raise RuntimeError("Broker adapter unavailable after login")
    candles = await broker.get_candles_async(asset, int(timeframe_min) * 60, max(100, int(rows)))
    if not candles:
        raise RuntimeError("No candle data received; asset/timeframe may be unsupported")
    # Normalize to DataFrame
    df = pd.DataFrame(candles)
    # Ensure columns present
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise RuntimeError("Candle response missing required fields")
    return df[["open", "high", "low", "close"]]


async def main():
    parser = argparse.ArgumentParser(description="NEXUS Autonomous AI Trader CLI")
    parser.add_argument("--email", type=str, help="Quotex account email")
    parser.add_argument("--password", type=str, help="Quotex account password")
    parser.add_argument("--demo", action="store_true", help="Use demo account")
    parser.add_argument(
        "--assets",
        type=str,
        nargs="+",
        default=["EURUSD"],
        help="Assets to trade or 'auto' to fetch live",
    )
    parser.add_argument("--timeframe", type=int, default=5, help="Timeframe in minutes")
    parser.add_argument(
        "--mode", type=str, choices=["live", "paper"], default="paper", help="Trading mode"
    )
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--switch-strategy", type=str, help="Switch to a different strategy")
    parser.add_argument("--train", action="store_true", help="Trigger model retraining")
    parser.add_argument("--stats", action="store_true", help="Show performance stats")
    parser.add_argument("--start", action="store_true", help="Start trading loop")
    parser.add_argument("--gui", action="store_true", help="Launch NEXUS GUI dashboard")
    # New commands
    parser.add_argument("--backtest", action="store_true", help="Run synthetic backtest")
    parser.add_argument(
        "--live-backtest",
        action="store_true",
        help="Run backtest on live OHLC via Quotex (uses demo/practice mode)",
    )
    # Place a single trade on the broker (uses demo if --demo is set)
    parser.add_argument(
        "--place-trade", action="store_true", help="Place a single trade on Quotex (demo if --demo)"
    )
    parser.add_argument("--direction", choices=["call", "put"], help="Direction for --place-trade")
    parser.add_argument("--amount", type=float, help="Amount for --place-trade (e.g., 5.0)")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Expiration in seconds for --place-trade (default 60)",
    )
    # Minimal live trading loop (demo/practice by default)
    parser.add_argument(
        "--live-trade",
        action="store_true",
        help="Run a minimal live trading loop (demo by default)",
    )
    parser.add_argument("--max-trades", type=int, default=3, help="Max trades for --live-trade")
    parser.add_argument(
        "--interval", type=int, default=30, help="Seconds between signal checks for --live-trade"
    )
    parser.add_argument("--rows", type=int, default=500, help="Rows of OHLC data for backtest")
    parser.add_argument("--window", type=int, default=50, help="Sliding window size for backtest")
    parser.add_argument(
        "--aggregate-reports",
        action="store_true",
        help="Aggregate backtest reports and show leaderboard",
    )
    parser.add_argument(
        "--force-epsilon", type=float, help="Force exploration epsilon (0-1) during backtest"
    )
    parser.add_argument(
        "--evolve", action="store_true", help="Run evolutionary optimization over ensemble weights"
    )
    parser.add_argument("--generations", type=int, default=3, help="Generations for evolution run")
    parser.add_argument("--population", type=int, default=6, help="Population size")
    args = parser.parse_args()

    settings = load_runtime_settings()
    # Allow credential override via CLI
    if args.email:
        settings.quotex.email = args.email  # type: ignore[attr-defined]
    if args.password:
        settings.quotex.password = args.password  # type: ignore[attr-defined]
    # Auto-login is enabled by default in settings; demo mode selection:
    engine = _build_engine(settings, demo=args.demo or args.mode == "paper")

    # Dynamic asset discovery if requested
    if len(args.assets) == 1 and args.assets[0].lower() == "auto":
        console.print(
            Panel("Live asset auto-fetch not implemented in lightweight prototype", style="yellow")
        )

    if args.list_strategies:
        table = Table(title="Available Strategies", box=box.SIMPLE)
        table.add_column("Name")
        for name in registry.list_strategies():
            table.add_row(name)
        console.print(table)
        sys.exit(0)

    if args.list_models:
        table = Table(title="Available Models", box=box.SIMPLE)
        table.add_column("Name")
        for name in registry.list_models():
            table.add_row(name)
        console.print(table)
        sys.exit(0)

    if args.switch_strategy:
        if args.switch_strategy in registry.strategies:
            engine.meta_strategy = registry.get_strategy(args.switch_strategy)()
            console.print(
                Panel(f"Switched to strategy: {args.switch_strategy}", style="bold green")
            )
        else:
            console.print(Panel(f"Strategy not found: {args.switch_strategy}", style="bold red"))
        sys.exit(0)

    if args.train:
        console.print(Panel("Model retraining stub (not implemented).", style="yellow"))
        sys.exit(0)

    if args.stats:
        stats = engine.get_performance_stats()
        table = Table(title="Performance Stats", box=box.SIMPLE)
        for k, v in stats.items():
            table.add_row(str(k), str(v))
        console.print(table)
        sys.exit(0)

    if args.aggregate_reports:
        entries = aggregate_reports()
        if not entries:
            console.print(Panel("No backtest reports found.", style="yellow"))
            sys.exit(0)
        table = Table(title="Backtest Leaderboard", box=box.MINIMAL_DOUBLE_HEAD)
        table.add_column("Rank", justify="right")
        table.add_column("Asset")
        table.add_column("Score")
        table.add_column("Trades")
        table.add_column("Profit")
        table.add_column("Win%")
        table.add_column("MDD")
        table.add_column("PF")
        for idx, e in enumerate(entries, start=1):
            table.add_row(
                str(idx),
                e.asset,
                f"{e.score:.4f}",
                str(e.total_trades),
                f"{e.total_profit:.2f}",
                f"{e.win_rate * 100:.1f}%",
                f"{e.max_drawdown:.2f}",
                f"{e.profit_factor:.2f}",
            )
        console.print(table)
        sys.exit(0)

    if args.live_backtest:
        try:
            asset = args.assets[0]
            # Fetch real OHLC candles from Quotex
            df = await _fetch_candles_df(
                engine, asset=asset, timeframe_min=args.timeframe, rows=args.rows
            )
            if "NEXUS_FORCE_EPSILON" not in os.environ:
                os.environ["NEXUS_FORCE_EPSILON"] = "0.20"
            meta = MetaStrategy()
            bt = Backtester(window=args.window, expiration=60)
            result = await bt.run(
                meta, engine, df, asset=asset, timeframe=args.timeframe, mode="market"
            )
            table = Table(title="Live Market Backtest Summary", box=box.SIMPLE_HEAVY)
            table.add_column("Metric")
            table.add_column("Value")
            summary_rows = {
                "Trades": result.total_trades,
                "Profit": result.total_profit,
                "Win Rate": f"{result.win_rate * 100:.2f}%",
                "Avg Profit": result.average_profit,
                "Max Drawdown": result.max_drawdown,
                "Profit Factor": result.profit_factor,
                "Exploratory Trades": result.exploratory_trades,
            }
            for k, v in summary_rows.items():
                table.add_row(k, str(v))
            console.print(table)
            if result.meta.get("report_path"):
                console.print(Panel(f"Report saved: {result.meta['report_path']}", style="green"))
            sys.exit(0)
        except Exception as e:
            console.print(Panel(f"Live backtest failed: {e}", style="bold red"))
            sys.exit(1)

    if args.backtest:
        rows = max(100, args.rows)
        df = _synthetic_df(rows)
        # Force exploration epsilon if specified for signal diversity
        restore_eps = None
        if args.force_epsilon is not None:
            restore_eps = os.getenv("NEXUS_FORCE_EPSILON")
            os.environ["NEXUS_FORCE_EPSILON"] = str(max(0.0, min(1.0, args.force_epsilon)))
        else:
            # Default mild exploration if no models provided
            if "NEXUS_FORCE_EPSILON" not in os.environ:
                os.environ["NEXUS_FORCE_EPSILON"] = "0.35"
        meta = MetaStrategy()
        bt = Backtester(window=args.window, expiration=60)
        result = await bt.run(meta, engine, df, asset=args.assets[0], timeframe=args.timeframe)
        if restore_eps is not None:
            if restore_eps is None:
                os.environ.pop("NEXUS_FORCE_EPSILON", None)
            else:
                os.environ["NEXUS_FORCE_EPSILON"] = restore_eps
        table = Table(title="Backtest Summary", box=box.SIMPLE_HEAVY)
        table.add_column("Metric")
        table.add_column("Value")
        summary_rows = {
            "Trades": result.total_trades,
            "Profit": result.total_profit,
            "Win Rate": f"{result.win_rate * 100:.2f}%",
            "Avg Profit": result.average_profit,
            "Max Drawdown": result.max_drawdown,
            "Profit Factor": result.profit_factor,
            "Exploratory Trades": result.exploratory_trades,
        }
        for k, v in summary_rows.items():
            table.add_row(k, str(v))
        console.print(table)
        if result.meta.get("report_path"):
            console.print(Panel(f"Report saved: {result.meta['report_path']}", style="green"))
        sys.exit(0)

    if args.evolve:
        evo_cfg = EvolutionConfig(
            generations=max(1, args.generations),
            population_size=max(2, args.population),
        )
        runner = EvolutionRunner(engine=engine, config=evo_cfg, settings=settings)
        generations = await runner.run()
        best = generations[-1].best_weights if generations else {}
        table = Table(title="Evolution Summary", box=box.SIMPLE)
        table.add_column("Gen", justify="right")
        table.add_column("Best Fitness")
        table.add_column("Weights Snapshot")
        for gr in generations:
            w_preview = ", ".join(f"{k}:{v:.2f}" for k, v in list(gr.best_weights.items())[:3])
            table.add_row(
                str(gr.generation),
                f"{gr.best_fitness:.4f}",
                w_preview + (" ..." if len(gr.best_weights) > 3 else ""),
            )
        console.print(table)
        console.print(Panel(f"Final Best Weights: {best}", style="green"))
        sys.exit(0)

    if args.start:
        console.print(
            Panel("Trading loop not implemented in lightweight prototype.", style="yellow")
        )
        sys.exit(0)

    if args.gui:
        from nexus.gui.launch_gui import launch_nexus_gui

        launch_nexus_gui(engine)  # type: ignore[arg-type]
        sys.exit(0)

    parser.print_help()


def run():  # pragma: no cover - thin wrapper
    """Synchronous entry point for console script 'nexus-cli'."""
    asyncio.run(main())


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())

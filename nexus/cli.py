"""
NEXUS CLI Interface

Provides command-line management, monitoring, and control for the NEXUS AI trading system.
Supports live trading, paper trading, strategy switching, analytics, and registry management.
"""

import argparse
import asyncio
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from nexus.core.engine import NexusEngine
from nexus.registry import registry

console = Console()

async def main():
    parser = argparse.ArgumentParser(description="NEXUS Autonomous AI Trader CLI")
    parser.add_argument("--email", type=str, help="Quotex account email")
    parser.add_argument("--password", type=str, help="Quotex account password")
    parser.add_argument("--demo", action="store_true", help="Use demo account")
    parser.add_argument("--assets", type=str, nargs='+', default=["EURUSD"], help="Assets to trade")
    parser.add_argument("--timeframe", type=int, default=5, help="Timeframe in minutes")
    parser.add_argument("--mode", type=str, choices=["live", "paper"], default="paper", help="Trading mode")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--switch-strategy", type=str, help="Switch to a different strategy")
    parser.add_argument("--train", action="store_true", help="Trigger model retraining")
    parser.add_argument("--stats", action="store_true", help="Show performance stats")
    parser.add_argument("--start", action="store_true", help="Start trading loop")
    parser.add_argument("--gui", action="store_true", help="Launch NEXUS GUI dashboard")
    args = parser.parse_args()

    engine = NexusEngine(email=args.email, password=args.password, demo_mode=args.demo)
    await engine.initialize_components()

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
            console.print(Panel(f"Switched to strategy: {args.switch_strategy}", style="bold green"))
        else:
            console.print(Panel(f"Strategy not found: {args.switch_strategy}", style="bold red"))
        sys.exit(0)

    if args.train:
        await engine.train_models(args.assets, [args.timeframe])
        console.print(Panel("Model retraining triggered.", style="bold green"))
        sys.exit(0)

    if args.stats:
        stats = engine.get_performance_stats()
        table = Table(title="Performance Stats", box=box.SIMPLE)
        for k, v in stats.items():
            table.add_row(str(k), str(v))
        console.print(table)
        sys.exit(0)

    if args.start:
        await engine.start_trading_loop(args.assets, timeframe=args.timeframe)
        sys.exit(0)

    if args.gui:
        from nexus.gui.launch_gui import launch_nexus_gui
        launch_nexus_gui(engine)
        sys.exit(0)

    parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())

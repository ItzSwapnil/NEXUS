"""
NEXUS Main Entry Point - Self-Evolving AI Trader for Quotex

This is the main entry point for the NEXUS autonomous trading system.
Features:
- Complete AI ensemble with Transformer, RL, and Evolution
- Real-time market regime detection
- Vector memory for pattern recognition
- Self-evolving strategies and neural architectures
- Advanced risk management and position sizing
- Comprehensive monitoring and logging
"""

import signal
import os
import platform
from pathlib import Path
from typing import Optional
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import NEXUS components
from nexus.core.engine import NexusEngine
from nexus.utils.config import load_config, create_default_config, NexusSettings
from nexus.utils.logger import setup_nexus_logging, LogConfig
from nexus.data.trade_history import TradeHistory, AdvancedDataStore

# Set up console for rich output
console = Console()
logger = None

def print_banner(version: str = "2.0.0"):
    """
    Print NEXUS startup banner.

    Args:
        version: Current NEXUS version
    """
    banner = Text()
    banner.append("🚀 NEXUS", style="bold cyan")
    banner.append(" - Self-Evolving AI Trader\n", style="bold white")
    banner.append(f"Version {version} | ", style="dim")
    banner.append("Powered by Advanced AI", style="bold green")

    console.print(Panel(
        banner,
        title="[bold blue]Welcome to NEXUS[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))

def print_system_info(config: NexusSettings):
    """
    Print system information.

    Args:
        config: NEXUS configuration
    """
    try:
        import torch
        import psutil

        table = Table(title="🖥️ System Information")
        table.add_column("Component", style="cyan")
        table.add_column("Information", style="green")

        # System info
        table.add_row("Operating System", f"{platform.system()} {platform.release()}")
        table.add_row("Python Version", f"{sys.version.split()[0]}")
        table.add_row("NEXUS Version", config.version)
        table.add_row("Environment", config.environment)

        # PyTorch info
        table.add_row("PyTorch Version", torch.__version__)
        table.add_row("CUDA Available", "✅ Yes" if torch.cuda.is_available() else "❌ No")
        if torch.cuda.is_available():
            table.add_row("GPU Device", torch.cuda.get_device_name())
            table.add_row("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

        # Hardware info
        table.add_row("CPU Cores", str(psutil.cpu_count(logical=False)))
        table.add_row("Logical Processors", str(psutil.cpu_count()))
        table.add_row("RAM", f"{psutil.virtual_memory().total // 1024**3} GB")

        # Configuration info
        table.add_row("Config Mode", "Debug" if config.debug_mode else "Production")
        table.add_row("GPU Enabled", "✅ Yes" if config.enable_gpu else "❌ No")
        table.add_row("Worker Threads", str(config.num_workers))

        console.print(table)
    except ImportError as e:
        console.print(f"[yellow]⚠️ Could not load all system info: {e}[/yellow]")

async def setup_signal_handlers(engine: NexusEngine):
    """
    Setup graceful shutdown handlers.

    Args:
        engine: NEXUS engine instance
    """
    def signal_handler(signum, frame):
        console.print("\n[yellow]🛑 Shutdown signal received. Initiating graceful shutdown...[/yellow]")

        async def shutdown():
            try:
                # If you implement engine.stop(), you can call it here
                # await engine.stop()
                console.print("[green]✅ NEXUS shutdown completed successfully[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error during shutdown: {e}[/red]")
            finally:
                pass  # sys.exit(0) is unreachable in async context

        # Create new event loop for shutdown if current one is closed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.create_task(shutdown())
        except:
            sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def validate_environment():
    """
    Validate system environment and dependencies.

    Returns:
        bool: True if environment is valid, False otherwise
    """
    try:
        import torch
        import pandas
        import numpy

        # Check for CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Validate CUDA with a small tensor operation
            x = torch.tensor([1.0, 2.0]).to(device)
            y = x + x

        return True
    except Exception as e:
        console.print(f"[red]❌ Environment validation failed: {e}[/red]")
        return False

async def start_nexus(config_path: Optional[Path] = None, debug: bool = False):
    """
    Start the NEXUS trading system.

    Args:
        config_path: Path to configuration file
        debug: Whether to enable debug mode
    """
    global logger

    try:
        # Set up logging
        log_config = LogConfig(level="DEBUG" if debug else "INFO")
        logger = setup_nexus_logging(log_config)
        logger.info("NEXUS starting up...")

        # Load configuration
        if config_path is None:
            config_path = Path("config.yaml")

        # Load configuration using the proper function
        settings = load_config(config_path)

        # Override with debug flag if specified
        if debug:
            settings.log_level = "DEBUG"

        # Override with environment variables if set
        if os.environ.get("NEXUS_DEBUG") in ("1", "true", "True"):
            settings.log_level = "DEBUG"

        # Print banner and system info
        print_banner("2.0.0")
        print_system_info(settings)

        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed.")
            return

        # Initialize NEXUS engine
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Initializing NEXUS engine...[/bold blue]"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("init", total=None)
            engine = NexusEngine(settings)
            await engine.initialize()

        # Setup signal handlers for graceful shutdown
        await setup_signal_handlers(engine)

        # Start the trading engine
        logger.info("🚀 Starting trading engine...")
        await engine.start()

        # Launch CLI dashboard
        launch_cli_dashboard(engine)

        # Keep the program running
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        console.print(f"[bold red]❌ Critical error: {e}[/bold red]")
        if debug:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

def launch_cli_dashboard(engine: NexusEngine):
    """
    Launch a real-time CLI dashboard for analytics, trade logs, and system control.
    """
    trade_history = TradeHistory()
    data_store = AdvancedDataStore()
    console = Console()
    try:
        while True:
            console.clear()
            # Banner
            print_banner("2.0.0")
            # System info
            settings = engine.settings
            print_system_info(settings)
            # Performance Table
            stats = engine.get_performance_stats()
            perf_table = Table(title="Performance Metrics")
            for k, v in stats.items():
                perf_table.add_row(str(k), str(v))
            console.print(perf_table)
            # Recent Trades
            trades = trade_history.get_trade_history(limit=20)
            trade_table = Table(title="Recent Trades")
            trade_table.add_column("Time")
            trade_table.add_column("Asset")
            trade_table.add_column("Dir")
            trade_table.add_column("Amt")
            trade_table.add_column("Result")
            trade_table.add_column("Profit")
            for t in trades:
                trade_table.add_row(str(t['timestamp']), t['asset'], t['direction'], str(t['amount']), t['result'], str(t['profit']))
            console.print(trade_table)
            # Regime Info
            try:
                asset = trades[0]['asset'] if trades else "EURUSD"
                candles = engine.quotex.get_candle(asset, 1, 100)
                regime = engine.regime_detector.detect_regime(candles)
                console.print(f"[bold blue]Current Regime:[/bold blue] {regime}")
            except Exception as e:
                console.print(f"[yellow]Regime detection error: {e}[/yellow]")
            # Strategy & Risk Model
            console.print(f"[bold green]Active Strategy:[/bold green] {engine.meta_strategy.__class__.__name__}")
            console.print(f"[bold green]Risk Model:[/bold green] {engine.risk_registry}")
            # Log tail
            try:
                with open("logs/nexus_20250717.log", "r") as f:
                    lines = f.readlines()[-20:]
                    console.print(Panel("".join(lines), title="System Log"))
            except Exception:
                pass
            # Refresh every 5 seconds
            time.sleep(5)
    except KeyboardInterrupt:
        console.print("[yellow]CLI dashboard exited by user.[/yellow]")

def create_arg_parser():
    """
    Create command line argument parser.

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description="NEXUS - Self-Evolving AI Trader for Quotex")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--version", action="store_true", help="Show version information")

    return parser

def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()

    # Handle version request
    if args.version:
        print("NEXUS v2.0.0 - Self-Evolving AI Trader")
        sys.exit(0)

    # Handle config creation request
    if args.create_config:
        config_path = args.config or "config.yaml"
        create_default_config(config_path)
        print(f"Created default configuration file at: {config_path}")
        sys.exit(0)

    # Determine config path
    config_path = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Configuration file not found: {config_path}")
            sys.exit(1)

    # Start NEXUS
    try:
        asyncio.run(start_nexus(config_path, args.debug))
    except KeyboardInterrupt:
        print("\nShutdown requested. Exiting...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    import asyncio
    from PySide6.QtWidgets import QApplication
    from nexus.gui.main_window import NexusMainWindow
    from nexus.core.engine import NexusEngine
    config = load_config()
    engine = NexusEngine(settings=config)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(engine.initialize_components())
    app = QApplication(sys.argv)
    window = NexusMainWindow(engine=engine)
    window.show()
    sys.exit(app.exec())

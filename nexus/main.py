"""
NEXUS Main Entry Point
Author: Swapnil De Sarkar
Created: 2025

Provides the primary interface for launching the NEXUS autonomous trading system.
Supports both GUI and CLI modes with comprehensive configuration.
"""

import asyncio
import sys
import os
from pathlib import Path
import argparse

# Add the project root to Python path if running as script
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

from nexus.utils.config import load_runtime_settings, NexusSettings
from nexus.utils.logger import get_nexus_logger, setup_logging
from nexus.core.engine import NexusEngine

logger = get_nexus_logger("nexus.main")

def setup_environment() -> None:
    """Set up the NEXUS runtime environment."""
    # Ensure required directories exist
    required_dirs = [
        "data",
        "logs", 
        "models",
        "models/transformer",
        "reports",
        "settings"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    setup_logging()
    
    # Set environment defaults if not specified
    if "NEXUS_ENGINE_RNG_SEED" not in os.environ:
        os.environ["NEXUS_ENGINE_RNG_SEED"] = "42"  # Reproducible for development
        
    logger.info("NEXUS environment initialized")

async def launch_gui_mode(engine: NexusEngine) -> None:
    """Launch the NEXUS GUI interface."""
    try:
        from nexus.gui.launch_gui import launch_nexus_gui
        logger.info("Launching NEXUS GUI...")
        launch_nexus_gui(engine)
    except ImportError as e:
        logger.error("GUI dependencies not available: %s", e)
        logger.error("Install GUI extras with: uv pip install .[gui]")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to launch GUI: %s", e)
        sys.exit(1)

async def launch_cli_mode(args: argparse.Namespace) -> None:
    """Launch the NEXUS CLI interface."""
    from nexus.cli import main as cli_main
    
    # Override sys.argv to pass arguments to CLI
    original_argv = sys.argv
    try:
        # Build CLI arguments
        cli_args = ["nexus"]
        if args.demo:
            cli_args.append("--demo")
        if args.backtest:
            cli_args.append("--backtest")
        if getattr(args, "live_backtest", False):
            cli_args.append("--live-backtest")
        # New pass-throughs
        if getattr(args, "place_trade", False):
            cli_args.append("--place-trade")
            if getattr(args, "direction", None):
                cli_args.extend(["--direction", args.direction])
            if getattr(args, "amount", None) is not None:
                cli_args.extend(["--amount", str(args.amount)])
            if getattr(args, "duration", None) is not None:
                cli_args.extend(["--duration", str(args.duration)])
        if getattr(args, "live_trade", False):
            cli_args.append("--live-trade")
            if getattr(args, "max_trades", None) is not None:
                cli_args.extend(["--max-trades", str(args.max_trades)])
            if getattr(args, "interval", None) is not None:
                cli_args.extend(["--interval", str(args.interval)])
        if args.stats:
            cli_args.append("--stats")
        if args.assets:
            cli_args.extend(["--assets"] + args.assets)
        
        sys.argv = cli_args
        await cli_main()
    finally:
        sys.argv = original_argv

async def run_autonomous_mode(engine: NexusEngine, settings: NexusSettings) -> None:
    """Run NEXUS in autonomous trading mode."""
    logger.info("Starting NEXUS autonomous trading mode...")
    logger.warning("Autonomous trading mode is experimental - use with caution!")
    
    from nexus.strategies.meta_strategy import MetaStrategy
    import pandas as pd
    
    # Initialize meta strategy
    meta_strategy = MetaStrategy()
    engine.meta_strategy = meta_strategy
    
    # Main trading loop
    trade_count = 0
    max_trades = settings.trading.max_daily_trades
    
    try:
        while trade_count < max_trades:
            logger.info(f"Trade cycle {trade_count + 1}/{max_trades}")
            
            # Generate synthetic market data for demonstration
            # In production, this would fetch real market data
            synthetic_data = pd.DataFrame({
                'open': [100.0, 100.1, 100.2, 100.3, 100.4],
                'high': [100.2, 100.3, 100.4, 100.5, 100.6], 
                'low': [99.8, 99.9, 100.0, 100.1, 100.2],
                'close': [100.1, 100.2, 100.3, 100.4, 100.5],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            # Get trading signal from meta strategy
            signal_result = await meta_strategy.generate_signal(
                synthetic_data,
                asset=settings.trading.default_asset,
                timeframe=settings.trading.prediction_interval
            )
            
            if signal_result:
                signal_type, position_size = signal_result
                amount = settings.trading.base_trade_amount * position_size
                
                logger.info(f"Generated signal: {signal_type.value} with size {amount:.2f}")
                
                # Execute trade
                trade_result = await engine.execute_trade(
                    settings.trading.default_asset,
                    signal_type.value,
                    amount,
                    settings.trading.default_expiration
                )
                
                # Update strategy performance
                await meta_strategy.update_performance(
                    meta_strategy.signal_history[-1] if meta_strategy.signal_history else None,
                    trade_result.get("success", False),
                    trade_result.get("profit", 0.0)
                )
                
                logger.info(f"Trade result: {trade_result}")
                trade_count += 1
            
            # Wait before next trade
            await asyncio.sleep(settings.trading.auto_trade_interval_seconds)
            
            # Check circuit breaker
            if engine.circuit_breaker_active:
                logger.error("Circuit breaker activated - stopping autonomous trading")
                break
                
    except KeyboardInterrupt:
        logger.info("Autonomous trading stopped by user")
    except Exception as e:
        logger.error(f"Error in autonomous trading: {e}")
        raise

async def main() -> None:
    """Main entry point for NEXUS."""
    parser = argparse.ArgumentParser(
        description="NEXUS - Autonomous, Self-Evolving AI Trader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nexus                          # Launch GUI (default)
  nexus --cli                    # Launch CLI mode
  nexus --cli --backtest         # Run backtest via CLI
  nexus --autonomous             # Run autonomous trading (experimental)
  nexus --demo --stats           # Show stats in demo mode
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--gui", action="store_true", default=True,
        help="Launch GUI interface (default)"
    )
    mode_group.add_argument(
        "--cli", action="store_true",
        help="Launch CLI interface"
    )
    mode_group.add_argument(
        "--autonomous", action="store_true",
        help="Run in autonomous trading mode (experimental)"
    )
    
    # Configuration options
    parser.add_argument(
        "--demo", action="store_true", default=True,
        help="Run in demo mode (default, safer)"
    )
    parser.add_argument(
        "--config", type=str, metavar="PATH",
        help="Path to custom configuration file"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Set logging level"
    )
    
    # CLI pass-through options
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run backtest (CLI mode)"
    )
    parser.add_argument(
        "--live-backtest", action="store_true",
        help="Run backtest on live OHLC (CLI mode)"
    )
    parser.add_argument(
        "--stats", action="store_true", 
        help="Show performance stats (CLI mode)"
    )
    parser.add_argument(
        "--assets", nargs="+", default=["EURUSD"],
        help="Trading assets (CLI mode)"
    )
    # New CLI pass-through arguments
    parser.add_argument(
        "--place-trade", action="store_true",
        help="Place a trade with specified parameters (CLI mode)"
    )
    parser.add_argument(
        "--direction", type=str, choices=["call", "put"],
        help="Trade direction for --place-trade (CLI mode)"
    )
    parser.add_argument(
        "--amount", type=float,
        help="Trade amount for --place-trade (CLI mode)"
    )
    parser.add_argument(
        "--duration", type=int,
        help="Trade duration in seconds for --place-trade (CLI mode)"
    )
    parser.add_argument(
        "--live-trade", action="store_true",
        help="Run a live trade with specified parameters (CLI mode)"
    )
    parser.add_argument(
        "--max-trades", type=int,
        help="Maximum number of trades for live trading (CLI mode)"
    )
    parser.add_argument(
        "--interval", type=int,
        help="Interval in seconds between trades for live trading (CLI mode)"
    )

    args = parser.parse_args()
    
    # Override GUI default if CLI or autonomous specified
    if args.cli or args.autonomous:
        args.gui = False
    
    # Set log level
    os.environ["NEXUS_LOG_LEVEL"] = args.log_level
    
    # Set up environment
    setup_environment()
    
    # Load configuration
    try:
        settings = load_runtime_settings(args.config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize engine
    try:
        engine = NexusEngine(settings=settings, demo_mode=args.demo)
        await engine.initialize_components()
        logger.info("NEXUS Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        sys.exit(1)
    
    # Launch appropriate mode
    try:
        if args.gui:
            await launch_gui_mode(engine)
        elif args.cli:
            await launch_cli_mode(args)
        elif args.autonomous:
            if not args.demo:
                logger.warning("Autonomous mode requires demo=True for safety")
                engine.demo_mode = True
            await run_autonomous_mode(engine, settings)
            
    except KeyboardInterrupt:
        logger.info("NEXUS shutdown requested")
    except Exception as e:
        logger.error(f"NEXUS execution error: {e}")
        raise
    finally:
        logger.info("NEXUS shutdown complete")

def run() -> None:
    """Synchronous entry point for console scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNEXUS shutdown by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nNEXUS error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run()

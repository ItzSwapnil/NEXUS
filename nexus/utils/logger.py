"""
Advanced Logging System for NEXUS

This module provides sophisticated logging capabilities with:
- Structured logging with JSON output
- Multi-level logging (console, file, remote)
- Performance metrics and timing
- Trade logging and audit trails
- Error tracking and alerts
- Log rotation and compression
- Real-time log streaming
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import time
import traceback
import codecs
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from loguru import logger as loguru_logger

# Initialize Rich console
console = Console()

@dataclass
class LogConfig:
    """Logging configuration."""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    structured_output: bool = True
    log_dir: Path = Path("logs")
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 10
    enable_performance_logging: bool = True
    enable_trade_logging: bool = True
    enable_error_tracking: bool = True

def setup_nexus_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """
    Set up NEXUS logging system.

    Args:
        config: Logging configuration

    Returns:
        Root logger
    """
    # Use default config if none provided
    if config is None:
        config = LogConfig()

    # Create log directory if it doesn't exist
    os.makedirs(config.log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Set up formatters
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        "%H:%M:%S"
    )

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # Console handler with Rich formatting
    if config.console_output:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=False,
            show_path=False
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, config.level))
        root_logger.addHandler(console_handler)

    # File handler with UTF-8 encoding for emoji support
    if config.file_output:
        log_file = config.log_dir / f"nexus_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, config.level))
        root_logger.addHandler(file_handler)

    # Set up loguru for performance logging
    loguru_logger.configure(
        handlers=[
            {
                "sink": config.log_dir / "nexus_performance.log",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                "rotation": "1 day",
                "retention": "30 days",
                "compression": "gz",
                "encoding": "utf-8"
            }
        ]
    )

    return root_logger

def get_nexus_logger(name: str) -> logging.Logger:
    """
    Get a NEXUS logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class PerformanceLogger:
    """
    Performance logging and metrics tracking.
    """

    def __init__(self, component: str):
        """
        Initialize performance logger.

        Args:
            component: Component name for logging
        """
        self.component = component
        self.start_time = None

    def start_operation(self, operation: str):
        """Start timing an operation."""
        self.operation = operation
        self.start_time = time.time()
        loguru_logger.info(f"{self.component}.{operation} - Started")

    def end_operation(self, success: bool = True, details: Optional[Dict] = None):
        """End timing an operation."""
        if self.start_time:
            duration = time.time() - self.start_time
            status = "Success" if success else "Failed"
            msg = f"{self.component}.{self.operation} - {status} ({duration:.3f}s)"
            if details:
                msg += f" - {details}"
            loguru_logger.info(msg)
            self.start_time = None

    @contextmanager
    def measure(self, operation: str):
        """Performance measurement context manager."""
        self.start_operation(operation)
        try:
            yield
            self.end_operation(success=True)
        except Exception as e:
            self.end_operation(success=False, details={"error": str(e)})
            raise

class TradeLogger:
    """
    Specialized logger for trade operations.
    """

    def __init__(self):
        self.trades_file = Path("logs") / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log trade execution details.

        Args:
            trade_data: Trade information
        """
        timestamp = datetime.now().isoformat()
        trade_record = {
            "timestamp": timestamp,
            "trade_id": trade_data.get("trade_id"),
            "asset": trade_data.get("asset"),
            "direction": trade_data.get("direction"),
            "amount": trade_data.get("amount"),
            "expiry": trade_data.get("expiry"),
            "confidence": trade_data.get("confidence"),
            "entry_price": trade_data.get("entry_price"),
            "result": trade_data.get("result"),
            "profit_loss": trade_data.get("profit_loss")
        }

        with open(self.trades_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trade_record) + '\n')

    def log_signal(self, signal_data: Dict[str, Any]):
        """
        Log trading signal generation.

        Args:
            signal_data: Signal information
        """
        timestamp = datetime.now().isoformat()
        signal_record = {
            "timestamp": timestamp,
            "type": "signal",
            "asset": signal_data.get("asset"),
            "direction": signal_data.get("direction"),
            "confidence": signal_data.get("confidence"),
            "features": signal_data.get("features"),
            "model_outputs": signal_data.get("model_outputs")
        }

        with open(self.trades_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(signal_record) + '\n')

class MetricsCollector:
    """
    System metrics collection and monitoring.
    """

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    def record_metric(self, name: str, value: Union[int, float], tags: Optional[Dict] = None):
        """Record a metric value."""
        timestamp = time.time()
        self.metrics[name] = {
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {}
        }

    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return time.time() - self.start_time

    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics."""
        return {
            "uptime": self.get_uptime(),
            "metrics_count": len(self.metrics),
            "latest_metrics": dict(list(self.metrics.items())[-10:])  # Last 10 metrics
        }

"""NEXUS logging utilities."""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import time
from contextlib import contextmanager

# Optional rich import
try:  # pragma: no cover - environment dependent
    from rich.console import Console  # type: ignore
    from rich.logging import RichHandler  # type: ignore
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    Console = None  # type: ignore
    RichHandler = None  # type: ignore
    _HAS_RICH = False

# Optional loguru import
try:  # pragma: no cover - environment dependent
    from loguru import logger as loguru_logger
    _HAS_LOGURU = True
except ImportError:  # pragma: no cover
    loguru_logger = None  # type: ignore
    _HAS_LOGURU = False

console = Console() if _HAS_RICH else None

@dataclass
class LogConfig:
    """Logging config."""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    structured_output: bool = True
    log_dir: Path = Path("logs")
    max_file_size: int = 100 * 1024 * 1024
    backup_count: int = 10
    enable_performance_logging: bool = True
    enable_trade_logging: bool = True
    enable_error_tracking: bool = True

def setup_nexus_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """Set up logging and return the root logger."""
    if config is None:
        config = LogConfig()
    os.makedirs(config.log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))
    root_logger.handlers.clear()

    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%H:%M:%S")
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S")

    if config.console_output:
        if _HAS_RICH and RichHandler:
            console_handler: logging.Handler = RichHandler(rich_tracebacks=True, markup=True, show_time=False, show_path=False)  # type: ignore
        else:
            console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, config.level, logging.INFO))
        root_logger.addHandler(console_handler)

    if config.file_output:
        log_file = config.log_dir / f"nexus_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, config.level, logging.INFO))
        root_logger.addHandler(file_handler)

    if _HAS_LOGURU and loguru_logger:
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
    """Return a logger by name."""
    return logging.getLogger(name)

class PerformanceLogger:
    """Performance logging and timing."""

    def __init__(self, component: str):
        """Create a performance logger for a component."""
        self.component = component
        self.start_time = None

    def start_operation(self, operation: str):
        """Start timing an operation."""
        self.operation = operation
        self.start_time = time.time()

    def end_operation(self, success: bool = True, details: Optional[Dict] = None):
        """End timing an operation and log it."""
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
        """Context manager to time an operation."""
        self.start_operation(operation)
        try:
            yield
            self.end_operation(success=True)
        except Exception as e:
            self.end_operation(success=False, details={"error": str(e)})
            raise
            self.start_time = None

class TradeLogger:
    """Logger for trades and signals."""

    def __init__(self):
        self.trades_file = Path("logs") / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Append a trade record to the log file."""
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
        """Append a signal record to the log file."""
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
    """In-memory metrics collector."""

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
        """Return process uptime in seconds."""
        return time.time() - self.start_time

    def get_metrics_summary(self) -> Dict:
        """Return a summary of tracked metrics."""
        return {
            "uptime": self.get_uptime(),
            "metrics_count": len(self.metrics),
            "latest_metrics": dict(list(self.metrics.items())[-10:])
        }

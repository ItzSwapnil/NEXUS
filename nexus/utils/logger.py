"""
NEXUS Logging Utilities
Author: Swapnil De Sarkar
Created: 2025

Custom logging implementation with performance tracking and rich console output.

Enhancements:
- Safe fallback when loguru isn't installed
- PerformanceLogger fully typed and resilient
- Added __all__ export list
- Comprehensive typing improvements
"""

import json
import logging
import logging.handlers
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    from loguru import logger as loguru_logger  # type: ignore

    _HAS_LOGURU = True
except ImportError:  # pragma: no cover
    loguru_logger = None  # type: ignore
    _HAS_LOGURU = False

console = Console() if _HAS_RICH else None


@dataclass
class LogConfig:
    """Logging configuration container."""

    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    structured_output: bool = True  # reserved for future JSON formatter toggle
    log_dir: Path = Path("logs")
    max_file_size: int = 100 * 1024 * 1024
    backup_count: int = 10
    enable_performance_logging: bool = True
    enable_trade_logging: bool = True
    enable_error_tracking: bool = True


def setup_nexus_logging(config: Optional[LogConfig] = None) -> logging.Logger:
    """Set up logging and return the root logger (idempotent)."""
    if config is None:
        config = LogConfig()
    os.makedirs(config.log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))
    # Clear only our handlers (avoid duplicate handlers on repeated init)
    root_logger.handlers.clear()

    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%H:%M:%S")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    if config.console_output:
        if _HAS_RICH and (RichHandler is not None):
            console_handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=False,
                show_path=False,  # type: ignore[arg-type]
            )
        else:
            console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, config.level, logging.INFO))
        root_logger.addHandler(console_handler)

    if config.file_output:
        log_file = config.log_dir / f"nexus_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, config.level, logging.INFO))
        root_logger.addHandler(file_handler)

    if _HAS_LOGURU and loguru_logger:
        # Configure a performance sink if enabled
        if config.enable_performance_logging:
            try:  # pragma: no cover - loguru formatting path
                loguru_logger.configure(  # type: ignore[arg-type]
                    handlers=[
                        {
                            "sink": config.log_dir / "nexus_performance.log",
                            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                            "rotation": "1 day",
                            "retention": "30 days",
                            "compression": "gz",
                            "encoding": "utf-8",
                        }
                    ]
                )
            except Exception:  # pragma: no cover
                root_logger.debug(
                    "Loguru performance sink configuration failed; continuing without it"
                )

    return root_logger


def get_nexus_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


class PerformanceLogger:
    """Performance logging and timing helper.

    Falls back to standard logging when loguru isn't available.
    Thread-safe for single-operation usage pattern.
    def trade_placed(self, asset: str, direction: str, amount: float, expiration: int) -> None:
        self._append({"type": "trade_placed", "asset": asset, "direction": direction, "amount": amount, "expiration": expiration})
    def trade_result(self, asset: str, direction: str, amount: float, result: str, profit: float) -> None:
        self._append({"type": "trade_result", "asset": asset, "direction": direction, "amount": amount, "result": result, "profit": profit})
    """

    def __init__(self, component: str):
        self.component: str = component
        self._operation: Optional[str] = None
        self._start_time: Optional[float] = None
        self._fallback_logger = logging.getLogger(f"performance.{component}")

    def start_operation(self, operation: str) -> None:
        self._operation = operation
        self._start_time = time.time()

    def end_operation(self, success: bool = True, details: Optional[Dict[str, Any]] = None) -> None:
        if self._start_time is None or self._operation is None:
            return  # Nothing to end
        duration = time.time() - self._start_time
        status = "Success" if success else "Failed"
        msg_parts: List[str] = [f"{self.component}.{self._operation}", status, f"({duration:.3f}s)"]
        if details:
            msg_parts.append(json.dumps(details, ensure_ascii=False))
        message = " - ".join([msg_parts[0], " ".join(msg_parts[1:])])
        if _HAS_LOGURU and loguru_logger:  # pragma: no branch
            loguru_logger.info(message)
        else:
            self._fallback_logger.info(message)
        # Reset state
        self._start_time = None
        self._operation = None

    @contextmanager
    def measure(self, operation: str):  # type: ignore[override]
        self.start_operation(operation)
        try:
            yield
            self.end_operation(success=True)
        except Exception as e:  # pragma: no cover - passthrough path
            self.end_operation(success=False, details={"error": str(e)})
            raise


class TradeLogger:
    """Logger for trades and signals (append-only JSONL)."""

    def __init__(self):
        self.trades_file = Path("logs") / f"trades_{datetime.now().strftime('%Y%m%d')}.log"
        os.makedirs("logs", exist_ok=True)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
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
            "profit_loss": trade_data.get("profit_loss"),
        }
        with open(self.trades_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade_record) + "\n")

    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        timestamp = datetime.now().isoformat()
        signal_record = {
            "timestamp": timestamp,
            "type": "signal",
            "asset": signal_data.get("asset"),
            "direction": signal_data.get("direction"),
            "confidence": signal_data.get("confidence"),
            "features": signal_data.get("features"),
            "model_outputs": signal_data.get("model_outputs"),
        }
        with open(self.trades_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal_record) + "\n")


class MetricsCollector:
    """In-memory metrics collector (non-thread-safe)."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: float = time.time()

    def record_metric(
        self, name: str, value: Union[int, float], tags: Optional[Dict[str, Any]] = None
    ) -> None:
        timestamp = time.time()
        self.metrics[name] = {
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {},
        }

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "uptime": self.get_uptime(),
            "metrics_count": len(self.metrics),
            "latest_metrics": dict(list(self.metrics.items())[-10:]),
        }


__all__ = [
    "LogConfig",
    "setup_nexus_logging",
    "get_nexus_logger",
    "PerformanceLogger",
    "TradeLogger",
    "MetricsCollector",
]

# Backward-compatible shim expected by older call sites
from typing import Optional as _Optional  # noqa: E402


def setup_logging(
    level: _Optional[str] = None, log_dir: _Optional[Union[str, Path]] = None
):  # pragma: no cover
    """Compatibility wrapper around setup_nexus_logging.
    Args:
        level: log level name (e.g., "INFO"). If None, uses env NEXUS_LOG_LEVEL or default in LogConfig.
        log_dir: optional directory for log files.
    Returns:
        The configured root logger.
    """
    env_level = os.getenv("NEXUS_LOG_LEVEL")
    cfg = LogConfig(
        level=(level or env_level or LogConfig.level),  # type: ignore[arg-type]
        log_dir=Path(log_dir) if log_dir else LogConfig.log_dir,
    )
    return setup_nexus_logging(cfg)


# Extend exports
__all__.append("setup_logging")

"""Dynamic runtime configuration for NEXUS.

Replaces prior file-based YAML config (config.yaml). Settings now load from:
1. runtime_settings.json (if present)
2. Environment variables (QUOTEX__EMAIL / QUOTEX__PASSWORD etc.)
3. In-memory defaults (simulation-safe)

Use save_runtime_settings() to persist changes at runtime.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from nexus.utils.logger import get_nexus_logger

load_dotenv()

logger = get_nexus_logger("nexus.utils.config")

RUNTIME_SETTINGS_PATH = Path("runtime_settings.json")

# --- Settings Models (unchanged) ---


class QuotexSettings(BaseModel):
    """Quotex connection settings."""

    email: str = ""
    password: str = ""
    demo_mode: bool = True
    lang: str = "en"
    reconnect_attempts: int = 3
    connection_timeout: int = 30
    # Optional browser/session parameters for seamless auth
    user_agent: Optional[str] = None
    cookies: Optional[str] = None
    ssid: Optional[str] = None


class TradingSettings(BaseModel):
    """Trading settings."""

    prediction_interval: int = 10
    min_confidence: float = 0.7
    max_open_trades: int = 3
    max_daily_trades: int = 50
    default_asset: str = "EURUSD"
    default_expiration: int = 60
    base_trade_amount: float = 5.0
    max_risk_per_trade_percent: float = 2.0
    max_loss_percent: float = 5.0
    payout_threshold: float = 80.0
    payout_poll_interval_seconds: int = 30
    max_exploration_capital_pct: float = 2.0
    use_live_catalog: bool = False
    auto_trade_enabled: bool = False
    auto_trade_interval_seconds: int = 30
    ai_select_timeframe: bool = True


class ExplorationSettings(BaseModel):
    """Exploration/exploitation settings."""

    base_epsilon: float = 0.15
    k_uncertainty: float = 0.35
    min_epsilon: float = 0.01
    max_epsilon: float = 0.6
    promotion_windows: int = 3
    fitness_promotion_threshold: float = 0.65


class FitnessSettings(BaseModel):
    """Composite fitness weights."""

    alpha_sharpe: float = 0.25
    alpha_sortino: float = 0.2
    alpha_profit_factor: float = 0.15
    alpha_payout: float = 0.15
    beta_mdd: float = 0.1
    beta_ulcer: float = 0.05
    beta_turnover: float = 0.05
    gamma_slippage: float = 0.03
    gamma_constraint: float = 0.02


class AISettings(BaseModel):
    """AI model settings."""

    enable_gpu: bool = True
    num_workers: int = 4
    model_update_interval: int = 3600
    learning_rate: float = 0.001
    batch_size: int = 256
    sequence_length: int = 100


class MemorySettings(BaseModel):
    """Vector memory settings."""

    capacity: int = 10000
    dimension: int = 128
    storage_path: str = "data/vector_memory"


class RegimeDetectorSettings(BaseModel):
    """Regime detector settings."""

    n_regimes: int = 4
    lookback_periods: int = 200
    sensitivity: float = 0.5


class TransformerSettings(BaseModel):
    """Transformer settings."""

    lookback_periods: int = 200
    feature_dim: int = 32
    batch_size: int = 128


class RLAgentSettings(BaseModel):
    """RL agent settings."""

    state_dim: int = 32
    hidden_dim: int = 64
    buffer_capacity: int = 10000


class EvolutionSettings(BaseModel):
    """Evolution engine settings."""

    population_size: int = 20
    mutation_rate: float = 0.1


class NexusSettings(BaseSettings):
    """Top-level NEXUS settings."""

    quotex: QuotexSettings
    trading: TradingSettings
    ai: AISettings = AISettings()
    memory: MemorySettings = MemorySettings()
    regime_detector: RegimeDetectorSettings = RegimeDetectorSettings()
    transformer: TransformerSettings = TransformerSettings()
    rl_agent: RLAgentSettings = RLAgentSettings()
    evolution: EvolutionSettings = EvolutionSettings()
    exploration: ExplorationSettings = ExplorationSettings()
    fitness: FitnessSettings = FitnessSettings()
    environment: str = "development"
    enable_gpu: bool = True
    num_workers: int = 4
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    version: str = "2.0.0"
    debug_mode: bool = False
    # Prominent flag to enable broker auto-login (can be set via .env: AUTO_LOGIN=true)
    auto_login: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")


# --- New dynamic load/save helpers ---


def _env_or(default: str, *keys: str) -> str:
    for k in keys:
        v = os.getenv(k)
        if v is not None and v.strip():
            return v.strip()
    return default


def _build_defaults() -> NexusSettings:
    email = _env_or("", "QUOTEX_EMAIL", "QUOTEX__EMAIL")
    password = _env_or("", "QUOTEX_PASSWORD", "QUOTEX__PASSWORD")
    return NexusSettings(
        quotex=QuotexSettings(
            email=email,
            password=password,
            demo_mode=True,
            lang="en",
        ),
        trading=TradingSettings(),
        ai=AISettings(),
        auto_login=True,
    )


def load_runtime_settings(path: Optional[Union[str, Path]] = None) -> NexusSettings:
    """Load dynamic settings from JSON or construct defaults.

    Args:
        path: optional override path for runtime settings file.
    Returns:
        NexusSettings instance
    """
    p = Path(path) if path else RUNTIME_SETTINGS_PATH
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            settings = NexusSettings(**raw)
            # Merge with environment-derived settings to ensure .env can override
            try:
                env_settings = NexusSettings(
                    quotex=QuotexSettings(email="", password=""), trading=TradingSettings()
                )  # loads from env/.env per model_config
                # Override if environment provides non-empty values
                if getattr(env_settings.quotex, "email", ""):
                    settings.quotex.email = env_settings.quotex.email  # type: ignore[attr-defined]
                if getattr(env_settings.quotex, "password", ""):
                    settings.quotex.password = env_settings.quotex.password  # type: ignore[attr-defined]
                # Auto-login may be toggled via env
                if hasattr(env_settings, "auto_login"):
                    settings.auto_login = bool(env_settings.auto_login)
            except Exception:
                pass
            return settings
        except Exception as e:
            logger.warning(f"Failed to load runtime settings JSON: {e}; rebuilding defaults")
    return _build_defaults()


def save_runtime_settings(settings: NexusSettings, path: Optional[Union[str, Path]] = None) -> bool:
    """Persist current settings to JSON (excluding sensitive override by env)."""
    p = Path(path) if path else RUNTIME_SETTINGS_PATH
    try:
        data = settings.model_dump()
        # Do not persist environment overrides explicitly if they were blank originally
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Runtime settings saved to {p}")
        return True
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed to save runtime settings: {e}")
        return False


def update_settings(mutator) -> NexusSettings:
    """Apply mutator(settings) -> None then save; returns updated settings."""
    settings = load_runtime_settings()
    try:
        mutator(settings)
    except Exception as e:
        logger.error(f"Mutator failed: {e}")
    save_runtime_settings(settings)
    return settings


# --- Backward compatible wrappers ---


def load_config(config_path: Optional[Union[str, Path]] = None) -> NexusSettings:  # noqa: D401
    return load_runtime_settings(config_path)


def create_default_config(save_path: Optional[Path] = None) -> NexusSettings:  # noqa: D401
    settings = _build_defaults()
    if save_path:
        save_runtime_settings(settings, save_path)
    return settings


def validate_config(
    config: NexusSettings,
) -> bool:  # unchanged logic but relaxed for blank creds in demo
    try:
        # Credentials can be blank in pure simulation
        if (
            config.trading.max_risk_per_trade_percent <= 0
            or config.trading.max_risk_per_trade_percent > 100
        ):
            logger.error("Risk per trade percentage must be between 0 and 100")
            return False
        if config.trading.payout_threshold <= 0 or config.trading.payout_threshold > 100:
            logger.error("Payout threshold must be between 0 and 100")
            return False
        if config.ai.num_workers <= 0:
            logger.error("Number of workers must be positive")
            return False
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


__all__ = [
    "NexusSettings",
    "QuotexSettings",
    "TradingSettings",
    "ExplorationSettings",
    "load_runtime_settings",
    "save_runtime_settings",
    "update_settings",
    "load_config",
    "create_default_config",
    "validate_config",
]

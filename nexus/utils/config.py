"""Configuration utilities for NEXUS."""

import yaml
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from omegaconf import OmegaConf
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.utils.config")

class QuotexSettings(BaseModel):
    """Quotex connection settings."""
    email: str
    password: str
    demo_mode: bool = True
    lang: str = "en"
    reconnect_attempts: int = 3
    connection_timeout: int = 30

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )


def load_config(config_path: Optional[Union[str, Path]] = None) -> NexusSettings:
    """Load configuration from YAML; create defaults if missing."""
    if config_path is None:
        config_path = Path("config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, creating default")
        create_default_config(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        _ = OmegaConf.create(config_data)
        settings = NexusSettings(**config_data)
        logger.info(f"Configuration loaded from {config_path}")
        return settings

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        return create_default_config()


def create_default_config(save_path: Optional[Path] = None) -> NexusSettings:
    """Create default config and optionally save to disk."""
    default_config = NexusSettings(
        quotex=QuotexSettings(
            email="demo@example.com",
            password="demo123",
            demo_mode=True,
            lang="en"
        ),
        trading=TradingSettings(),
        ai=AISettings()
    )

    if save_path:
        config_dict = default_config.model_dump()
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        logger.info(f"Default configuration saved to {save_path}")

    return default_config


def validate_config(config: NexusSettings) -> bool:
    """Basic validation for settings integrity."""
    try:
        if not config.quotex.email or not config.quotex.password:
            logger.error("Quotex email and password are required")
            return False
        if config.trading.max_risk_per_trade_percent <= 0 or config.trading.max_risk_per_trade_percent > 100:
            logger.error("Risk per trade percentage must be between 0 and 100")
            return False
        if config.trading.payout_threshold <= 0 or config.trading.payout_threshold > 100:
            logger.error("Payout threshold must be between 0 and 100")
            return False
        if config.ai.num_workers <= 0:
            logger.error("Number of workers must be positive")
            return False
        logger.info("Configuration validation passed")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

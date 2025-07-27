"""
Configuration utilities for NEXUS.

This module provides enhanced configuration management with:
- Pydantic-based settings validation
- Environment variable integration
- Dynamic configuration reloading
- Configuration encryption for sensitive data
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from omegaconf import OmegaConf

logger = logging.getLogger("nexus.utils.config")

class QuotexSettings(BaseModel):
    """Quotex connection settings."""
    email: str
    password: str
    demo_mode: bool = True
    lang: str = "en"
    reconnect_attempts: int = 3
    connection_timeout: int = 30

class TradingSettings(BaseModel):
    """Trading configuration settings."""
    prediction_interval: int = 10
    min_confidence: float = 0.7
    max_open_trades: int = 3
    max_daily_trades: int = 50
    default_asset: str = "EURUSD"
    default_expiration: int = 60
    base_trade_amount: float = 5.0
    max_risk_per_trade_percent: float = 2.0
    max_loss_percent: float = 5.0

class AISettings(BaseModel):
    """AI model configuration settings."""
    enable_gpu: bool = True
    num_workers: int = 4
    model_update_interval: int = 3600  # 1 hour
    learning_rate: float = 0.001
    batch_size: int = 256
    sequence_length: int = 100

class MemorySettings(BaseModel):
    """Vector memory configuration settings."""
    capacity: int = 10000
    dimension: int = 128
    storage_path: str = "data/vector_memory"

class RegimeDetectorSettings(BaseModel):
    """Regime detector configuration settings."""
    n_regimes: int = 4
    lookback_periods: int = 200
    sensitivity: float = 0.5

class TransformerSettings(BaseModel):
    """Transformer model configuration settings."""
    lookback_periods: int = 200
    feature_dim: int = 32
    batch_size: int = 128

class RLAgentSettings(BaseModel):
    """RL agent configuration settings."""
    state_dim: int = 32
    hidden_dim: int = 64
    buffer_capacity: int = 10000

class EvolutionSettings(BaseModel):
    """Evolution engine configuration settings."""
    population_size: int = 20
    mutation_rate: float = 0.1

class NexusSettings(BaseSettings):
    """Main NEXUS configuration settings."""
    quotex: QuotexSettings
    trading: TradingSettings
    ai: AISettings = AISettings()
    memory: MemorySettings = MemorySettings()
    regime_detector: RegimeDetectorSettings = RegimeDetectorSettings()
    transformer: TransformerSettings = TransformerSettings()
    rl_agent: RLAgentSettings = RLAgentSettings()
    evolution: EvolutionSettings = EvolutionSettings()
    environment: str = "development"
    enable_gpu: bool = True
    num_workers: int = 4
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    version: str = "2.0.0"
    debug_mode: bool = False

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

def load_config(config_path: Optional[Union[str, Path]] = None) -> NexusSettings:
    """
    Load NEXUS configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        NexusSettings: Loaded configuration
    """
    if config_path is None:
        config_path = Path("config.yaml")

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, creating default")
        create_default_config(config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Convert to OmegaConf for advanced features
        omega_config = OmegaConf.create(config_data)

        # Create Pydantic settings from loaded data
        settings = NexusSettings(**config_data)

        logger.info(f"Configuration loaded from {config_path}")
        return settings

    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.info("Using default configuration")
        return create_default_config()

def create_default_config(save_path: Optional[Path] = None) -> NexusSettings:
    """
    Create default NEXUS configuration.

    Args:
        save_path: Optional path to save the config file

    Returns:
        NexusSettings: Default configuration
    """
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
    """
    Validate configuration settings.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Validate quotex settings
        if not config.quotex.email or not config.quotex.password:
            logger.error("Quotex email and password are required")
            return False

        # Validate trading settings
        if config.trading.max_risk_per_trade_percent <= 0 or config.trading.max_risk_per_trade_percent > 100:
            logger.error("Risk per trade percentage must be between 0 and 100")
            return False

        # Validate AI settings
        if config.ai.num_workers <= 0:
            logger.error("Number of workers must be positive")
            return False

        logger.info("Configuration validation passed")
        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

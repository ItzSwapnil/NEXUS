"""
Configuration module for NEXUS.

This module defines the configuration structure using Pydantic models and provides
functions to load configuration from YAML files.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger("nexus.config")

class QuotexConfig(BaseModel):
    """Configuration for Quotex connection."""
    email: str
    password: str
    demo_mode: bool = True
    lang: str = "en"
    reconnect_attempts: int = 5

    class Config:
        frozen = True

class AssetConfig(BaseModel):
    """Configuration for a trading asset."""
    symbol: str
    timeframes: List[int] = Field(default_factory=lambda: [60, 300, 900])  # Default: 1m, 5m, 15m
    enabled: bool = True
    min_expiry: int = 60  # Minimum expiry time in seconds
    max_expiry: int = 300  # Maximum expiry time in seconds

    class Config:
        frozen = True

class RiskConfig(BaseModel):
    """Configuration for risk management."""
    max_trade_amount: float
    max_daily_loss: float
    max_consecutive_losses: int = 3
    position_sizing_method: str = "kelly"  # Options: fixed, kelly, percent
    position_size: float = 0.02  # 2% of balance for percent method
    kelly_fraction: float = 0.25  # Conservative Kelly fraction
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.06  # 6% take profit (3:1 risk reward)
    max_drawdown: float = 0.15  # 15% maximum drawdown

    @field_validator('position_sizing_method')
    def validate_position_sizing_method(cls, v):
        allowed_methods = ["fixed", "kelly", "percent"]
        if v not in allowed_methods:
            raise ValueError(f"Position sizing method must be one of {allowed_methods}")
        return v

class AIConfig(BaseModel):
    """Configuration for AI models and training."""
    model_type: str = "transformer_rl"  # transformer_rl, lstm_rl, hybrid
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 64
    sequence_length: int = 100
    prediction_horizon: int = 5

    # Reinforcement Learning
    rl_algorithm: str = "ppo"  # ppo, sac, td3, a2c
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    target_kl: float = 0.01

    # Evolution parameters
    evolution_enabled: bool = True
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 2.0

    @field_validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ["transformer_rl", "lstm_rl", "hybrid", "neuroevolution"]
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v

class MemoryConfig(BaseModel):
    """Configuration for memory and experience replay."""
    buffer_size: int = 100000
    min_buffer_size: int = 10000
    memory_type: str = "prioritized"  # simple, prioritized, episodic
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4  # Importance sampling
    epsilon: float = 1e-6  # Small constant for numerical stability

    # Vector memory for semantic storage
    vector_dim: int = 768
    max_vectors: int = 50000
    similarity_threshold: float = 0.85

class ModelConfig(BaseModel):
    """Configuration for model training and inference."""
    model_type: str = "lstm"  # Options: lstm, gru, transformer, randomforest, xgboost, etc.
    input_dim: int = 20  # Number of features
    hidden_dim: int = 128
    output_dim: int = 2  # Binary classification (up/down)
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    sequence_length: int = 60  # Lookback window
    prediction_horizon: int = 5  # How many steps ahead to predict
    use_gpu: bool = True
    save_path: str = "models"
    load_best: bool = True

    class Config:
        frozen = True

class DataConfig(BaseModel):
    """Configuration for data management."""
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    max_candles: int = 10000
    data_sources: List[str] = Field(default_factory=lambda: ["quotex", "yahoo"])
    enable_alternative_data: bool = True
    news_sources: List[str] = Field(default_factory=lambda: ["reuters", "bloomberg"])

class BacktestConfig(BaseModel):
    """Configuration for backtesting."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_balance: float = 10000.0
    commission: float = 0.0  # Quotex binary options typically no commission
    spread: float = 0.0001  # Small spread simulation
    slippage: float = 0.0001

class OptimizationConfig(BaseModel):
    """Configuration for hyperparameter optimization."""
    method: str = "optuna"  # optuna, hyperopt, bayesian
    n_trials: int = 100
    timeout: Optional[int] = 3600  # 1 hour timeout
    pruning: bool = True
    parallel_jobs: int = 4

class WebConfig(BaseModel):
    """Configuration for web interface."""
    host: str = "localhost"
    port: int = 8000
    enable_dashboard: bool = True
    enable_api: bool = True
    auth_required: bool = False

class StrategyConfig(BaseModel):
    """Configuration for trading strategies."""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeframes: List[int] = Field(default_factory=lambda: [60, 300])

    class Config:
        frozen = True

class Config(BaseModel):
    """Main configuration class for NEXUS."""
    quotex: QuotexConfig
    assets: List[AssetConfig]
    risk: RiskConfig
    ai: AIConfig = Field(default_factory=AIConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    strategies: List[StrategyConfig]

    class Config:
        frozen = True

def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    return Config.model_validate(config_data)

def save_config(config: Config, config_path: Union[str, Path] = "config.yaml"):
    """Save configuration to YAML file."""
    config_path = Path(config_path)

    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            # Convert to dict and then to YAML
            config_dict = config.model_dump()
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        raise

def create_default_config() -> Config:
    """Create a default configuration with sensible defaults."""

    return Config(
        quotex=QuotexConfig(
            email=os.getenv("QUOTEX_EMAIL", ""),
            password=os.getenv("QUOTEX_PASSWORD", ""),
            demo_mode=True,
            lang="en"
        ),
        assets=[],  # Empty list - will be populated dynamically from Quotex API
        risk=RiskConfig(
            max_trade_amount=100.0,
            max_daily_loss=500.0,
            max_consecutive_losses=3,
            position_sizing_method="kelly",
            kelly_fraction=0.25
        ),
        ai=AIConfig(),
        memory=MemoryConfig(),
        strategies=[]  # Will be configured separately
    )

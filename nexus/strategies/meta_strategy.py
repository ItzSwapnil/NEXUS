"""
Meta Strategy Coordinator for NEXUS Trading System

This module orchestrates multiple AI models and strategies to create
an intelligent ensemble trading system that adapts to market conditions.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from nexus.utils.logger import get_nexus_logger, PerformanceLogger

logger = get_nexus_logger("nexus.strategies.meta_strategy")
perf_logger = PerformanceLogger("meta_strategy")

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "call"
    SELL = "put"
    HOLD = "hold"

@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float
    asset: str
    timeframe: int
    reasoning: str
    source_model: str
    timestamp: datetime
    features: Optional[Dict] = None

@dataclass
class StrategyConfig:
    """Configuration for meta strategy."""
    confidence_threshold: float = 0.7
    ensemble_weights: Dict[str, float] = None
    regime_adaptation: bool = True
    risk_adjustment: bool = True
    signal_filtering: bool = True

    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'transformer': 0.4,
                'rl_agent': 0.3,
                'technical': 0.2,
                'regime': 0.1
            }

class MetaStrategy:
    """
    Advanced Meta Strategy that coordinates multiple AI models and strategies.

    Features:
    - Multi-model ensemble predictions
    - Regime-aware strategy switching
    - Dynamic weight adjustment
    - Signal filtering and validation
    - Risk-adjusted position sizing
    """

    def __init__(
        self,
        transformer=None,
        rl_agent=None,
        regime_detector=None,
        config: Optional[StrategyConfig] = None
    ):
        """
        Initialize the Meta Strategy.

        Args:
            transformer: Market transformer model
            rl_agent: Reinforcement learning agent
            regime_detector: Market regime detector
            config: Strategy configuration
        """
        self.transformer = transformer
        self.rl_agent = rl_agent
        self.regime_detector = regime_detector
        self.config = config or StrategyConfig()

        # Strategy state
        self.current_weights = self.config.ensemble_weights.copy()
        self.performance_history = []
        self.signal_history = []

        # Adaptive parameters
        self.adaptation_rate = 0.1
        self.min_weight = 0.05
        self.max_weight = 0.7

        # Performance tracking
        self.win_rate = 0.0
        self.trade_count = 0
        self.successful_trades = 0
        self.current_regime = None

        # Memory of past market conditions
        self.market_memory = {}

        logger.info("MetaStrategy initialized with weights: %s", self.current_weights)

    async def get_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime using the regime detector.

        Args:
            data: Market data as pandas DataFrame

        Returns:
            str: Detected market regime
        """
        if self.regime_detector and self.config.regime_adaptation:
            with perf_logger.measure("regime_detection"):
                regime = await self.regime_detector.detect_regime(data)
                logger.info(f"Current market regime: {regime}")
                self.current_regime = regime
                return regime
        return "unknown"

    async def collect_signals(self, data: pd.DataFrame, asset: str, timeframe: int) -> List[TradingSignal]:
        """
        Collect trading signals from all available models.

        Args:
            data: Market data as pandas DataFrame
            asset: Trading asset symbol
            timeframe: Analysis timeframe in minutes

        Returns:
            List[TradingSignal]: Collection of signals from all models
        """
        signals = []
        now = datetime.now()

        # Get market regime to inform models
        regime = await self.get_market_regime(data)

        # Collect signals from transformer model
        if self.transformer:
            with perf_logger.measure("transformer_prediction"):
                transformer_signal = await self.transformer.predict(
                    data,
                    asset=asset,
                    timeframe=timeframe,
                    regime=regime
                )
                signals.append(TradingSignal(
                    signal_type=SignalType(transformer_signal["signal"]),
                    confidence=transformer_signal["confidence"],
                    asset=asset,
                    timeframe=timeframe,
                    reasoning=transformer_signal.get("reasoning", "Transformer model prediction"),
                    source_model="transformer",
                    timestamp=now,
                    features=transformer_signal.get("features")
                ))
                logger.debug(f"Transformer signal: {transformer_signal['signal']} with {transformer_signal['confidence']:.2f} confidence")

        # Collect signals from RL agent
        if self.rl_agent:
            with perf_logger.measure("rl_agent_prediction"):
                rl_signal = await self.rl_agent.predict(
                    data,
                    asset=asset,
                    timeframe=timeframe,
                    regime=regime
                )
                signals.append(TradingSignal(
                    signal_type=SignalType(rl_signal["signal"]),
                    confidence=rl_signal["confidence"],
                    asset=asset,
                    timeframe=timeframe,
                    reasoning=rl_signal.get("reasoning", "RL agent decision"),
                    source_model="rl_agent",
                    timestamp=now,
                    features=rl_signal.get("features")
                ))
                logger.debug(f"RL agent signal: {rl_signal['signal']} with {rl_signal['confidence']:.2f} confidence")

        # Add additional signals from technical analysis or other sources here

        return signals

    def filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Filter signals based on confidence threshold and other criteria.

        Args:
            signals: List of collected trading signals

        Returns:
            List[TradingSignal]: Filtered signals
        """
        if not self.config.signal_filtering:
            return signals

        filtered_signals = [
            signal for signal in signals
            if signal.confidence >= self.config.confidence_threshold
        ]

        logger.info(f"Filtered {len(signals) - len(filtered_signals)} low-confidence signals")
        return filtered_signals

    async def ensemble_decision(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """
        Generate ensemble decision from multiple signals using weighted voting.

        Args:
            signals: List of collected and filtered trading signals

        Returns:
            Optional[TradingSignal]: Final ensemble decision or None if no decision
        """
        if not signals:
            logger.warning("No signals available for ensemble decision")
            return None

        # Score each signal type based on weighted votes
        votes = {
            SignalType.BUY: 0.0,
            SignalType.SELL: 0.0,
            SignalType.HOLD: 0.0
        }

        # Calculate weighted votes
        for signal in signals:
            source_weight = self.current_weights.get(signal.source_model, 0.1)
            weighted_confidence = signal.confidence * source_weight
            votes[signal.signal_type] += weighted_confidence

        # Find signal type with highest weighted vote
        max_vote = max(votes.values())
        if max_vote == 0:
            return None

        winning_signal_types = [st for st, vote in votes.items() if vote == max_vote]
        if len(winning_signal_types) > 1:
            logger.info(f"Tie between {[s.value for s in winning_signal_types]} signals. Defaulting to HOLD.")
            winning_signal_type = SignalType.HOLD
        else:
            winning_signal_type = winning_signal_types[0]

        # Calculate combined confidence and reasoning
        filtered_signals = [s for s in signals if s.signal_type == winning_signal_type]
        if not filtered_signals:
            return None

        # Use most confident signal's features and metadata
        most_confident = max(filtered_signals, key=lambda s: s.confidence)

        # Combine reasoning from top signals
        top_reasons = [
            f"{s.source_model}: {s.reasoning}"
            for s in sorted(filtered_signals, key=lambda x: x.confidence, reverse=True)[:2]
        ]
        combined_reason = " | ".join(top_reasons)

        # Calculate ensemble confidence
        ensemble_confidence = max_vote / sum(votes.values()) if sum(votes.values()) > 0 else 0

        logger.info(f"Ensemble decision: {winning_signal_type.value} with {ensemble_confidence:.2f} confidence")

        return TradingSignal(
            signal_type=winning_signal_type,
            confidence=ensemble_confidence,
            asset=most_confident.asset,
            timeframe=most_confident.timeframe,
            reasoning=combined_reason,
            source_model="ensemble",
            timestamp=datetime.now(),
            features=most_confident.features
        )

    async def adapt_weights(self, signal: TradingSignal, success: bool):
        """
        Adapt model weights based on trading success.

        Args:
            signal: The trading signal that was acted upon
            success: Whether the trade was successful
        """
        if signal.source_model == "ensemble":
            # Find which models contributed to this decision
            contributing_models = [
                s.source_model for s in self.signal_history[-10:]
                if s.signal_type == signal.signal_type and s.asset == signal.asset
            ]

            if not contributing_models:
                return

            # Reward or penalize contributing models
            adjustment = self.adaptation_rate * (1 if success else -1)

            for model in contributing_models:
                if model in self.current_weights:
                    # Adjust weight
                    self.current_weights[model] = max(
                        self.min_weight,
                        min(
                            self.max_weight,
                            self.current_weights[model] + adjustment
                        )
                    )

            # Normalize weights to sum to 1
            weight_sum = sum(self.current_weights.values())
            if weight_sum > 0:
                self.current_weights = {
                    model: weight / weight_sum
                    for model, weight in self.current_weights.items()
                }

            logger.info(f"Adapted model weights: {self.current_weights}")

    async def risk_position_size(self, signal: TradingSignal, balance: float) -> float:
        """
        Calculate position size based on signal confidence and risk management.

        Args:
            signal: Trading signal
            balance: Current account balance

        Returns:
            float: Position size as percentage of balance (0.0-1.0)
        """
        if not self.config.risk_adjustment:
            return 0.02  # Default 2% risk

        # Base position on signal confidence
        base_position = min(0.05, signal.confidence * 0.05)

        # Adjust based on win rate trend
        if self.win_rate > 0.6:
            risk_multiplier = 1.2
        elif self.win_rate < 0.4:
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0

        # Adjust based on market regime
        regime_multipliers = {
            "trending": 1.2,
            "ranging": 1.0,
            "volatile": 0.7,
            "unknown": 0.8
        }

        regime_mult = regime_multipliers.get(self.current_regime, 0.8)

        # Calculate final position size
        position_size = base_position * risk_multiplier * regime_mult

        # Cap at max 5% of balance
        position_size = min(position_size, 0.05)

        logger.info(f"Position sizing: {position_size:.2%} of balance")
        return position_size

    async def update_performance(self, signal: TradingSignal, success: bool, profit: float):
        """
        Update strategy performance metrics.

        Args:
            signal: The executed trading signal
            success: Whether the trade was successful
            profit: Profit/loss amount
        """
        self.trade_count += 1
        if success:
            self.successful_trades += 1

        self.win_rate = self.successful_trades / self.trade_count if self.trade_count > 0 else 0

        # Record performance for this condition
        if signal.features:
            # Create a simple key from the most important features
            feature_key = tuple(sorted([
                (k, round(v, 2)) for k, v in signal.features.items()
                if k in ['trend_strength', 'volatility', 'momentum']
            ]))

            if feature_key not in self.market_memory:
                self.market_memory[feature_key] = {
                    'wins': 0,
                    'losses': 0,
                    'profit': 0.0
                }

            self.market_memory[feature_key]['wins' if success else 'losses'] += 1
            self.market_memory[feature_key]['profit'] += profit

        # Adapt weights based on performance
        await self.adapt_weights(signal, success)

        logger.info(f"Updated performance: {self.win_rate:.2%} win rate after {self.trade_count} trades")

    async def generate_signal(self, data: pd.DataFrame, asset: str, timeframe: int) -> Optional[Tuple[SignalType, float]]:
        """
        Generate a trading signal for the specified asset and timeframe.

        Args:
            data: Market data as pandas DataFrame
            asset: Trading asset symbol
            timeframe: Analysis timeframe in minutes

        Returns:
            Optional[Tuple[SignalType, float]]: Signal type and position size or None if no action
        """
        with perf_logger.measure("generate_signal"):
            # Collect signals from all models
            signals = await self.collect_signals(data, asset, timeframe)

            # Store signals in history
            self.signal_history.extend(signals)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]

            # Filter signals
            filtered_signals = self.filter_signals(signals)

            # Generate ensemble decision
            final_signal = await self.ensemble_decision(filtered_signals)

            if not final_signal or final_signal.signal_type == SignalType.HOLD:
                logger.info(f"No actionable signal generated for {asset}")
                return None

            # Get account balance (placeholder - should be provided by caller)
            balance = 1000.0  # Example balance

            # Calculate position size
            position_size = await self.risk_position_size(final_signal, balance)

            return (final_signal.signal_type, position_size)

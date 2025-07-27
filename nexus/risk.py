"""
Risk management module for NEXUS.

This module handles risk management for the NEXUS trading system, including
position sizing, risk assessment, and trade filtering.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import pandas as pd

from nexus.config import RiskConfig

logger = logging.getLogger("nexus.risk")

class RiskManager:
    """
    Risk manager for the NEXUS trading system.
    
    This class handles risk management for the NEXUS trading system, including
    position sizing, risk assessment, and trade filtering.
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize the risk manager.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        self.current_balance = 0.0
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trade_history: List[Dict[str, Any]] = []
        self.asset_performance: Dict[str, Dict[str, Any]] = {}
        self.last_reset = datetime.now()
        self.initialized = False
    
    def initialize(self, balance: float) -> None:
        """
        Initialize the risk manager.
        
        Args:
            balance: Current account balance
        """
        logger.info("Initializing risk manager")
        
        self.current_balance = balance
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trade_history = []
        self.asset_performance = {}
        self.last_reset = datetime.now()
        self.initialized = True
        
        logger.info(f"Risk manager initialized with balance: {balance}")
    
    def update_balance(self, balance: float) -> None:
        """
        Update the current balance.
        
        Args:
            balance: New account balance
        """
        if not self.initialized:
            raise RuntimeError("Risk manager not initialized")
        
        self.current_balance = balance
        
        # Check if we need to reset daily metrics
        now = datetime.now()
        if now.date() > self.last_reset.date():
            self._reset_daily_metrics()
    
    def _reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        logger.info("Resetting daily risk metrics")
        
        self.daily_loss = 0.0
        self.daily_profit = 0.0
        self.last_reset = datetime.now()
    
    def calculate_position_size(self, asset: str, confidence: float) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            asset: Asset symbol
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            float: Position size
        """
        if not self.initialized:
            raise RuntimeError("Risk manager not initialized")
        
        # Check if we've reached the maximum daily loss
        if self.daily_loss >= self.config.max_daily_loss:
            logger.warning(f"Maximum daily loss reached: {self.daily_loss} >= {self.config.max_daily_loss}")
            return 0.0
        
        # Check if we've had too many consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"Maximum consecutive losses reached: {self.consecutive_losses} >= {self.config.max_consecutive_losses}")
            return 0.0
        
        # Calculate position size based on method
        if self.config.position_sizing_method == "fixed":
            # Fixed position size
            position_size = min(self.config.max_trade_amount, self.current_balance * 0.1)
            
        elif self.config.position_sizing_method == "percent":
            # Percentage of balance
            position_size = self.current_balance * self.config.position_size
            position_size = min(position_size, self.config.max_trade_amount)
            
        elif self.config.position_sizing_method == "kelly":
            # Kelly criterion
            # Get win rate and average win/loss ratio for this asset
            win_rate, win_loss_ratio = self._get_asset_performance(asset)
            
            # Apply Kelly formula
            if win_loss_ratio > 0:
                kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                kelly_fraction = max(0, kelly_fraction)  # Ensure non-negative
                kelly_fraction = min(kelly_fraction, 0.2)  # Cap at 20%
                
                position_size = self.current_balance * kelly_fraction
                position_size = min(position_size, self.config.max_trade_amount)
            else:
                position_size = self.current_balance * 0.01  # Default to 1%
        else:
            # Default to fixed
            position_size = min(self.config.max_trade_amount, self.current_balance * 0.1)
        
        # Adjust based on confidence
        position_size = position_size * confidence
        
        # Ensure position size is reasonable
        position_size = max(1.0, position_size)  # Minimum position size
        position_size = min(position_size, self.config.max_trade_amount)  # Maximum position size
        position_size = min(position_size, self.current_balance * 0.5)  # Never risk more than 50% of balance
        
        # Round to 2 decimal places
        position_size = round(position_size, 2)
        
        logger.debug(f"Calculated position size for {asset}: {position_size} (confidence: {confidence:.2f})")
        
        return position_size
    
    def _get_asset_performance(self, asset: str) -> Tuple[float, float]:
        """
        Get performance metrics for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Tuple[float, float]: Win rate and win/loss ratio
        """
        # If we don't have data for this asset, use default values
        if asset not in self.asset_performance:
            return 0.5, 1.0
        
        performance = self.asset_performance[asset]
        
        # Calculate win rate
        total_trades = performance.get("wins", 0) + performance.get("losses", 0)
        win_rate = performance.get("wins", 0) / total_trades if total_trades > 0 else 0.5
        
        # Calculate win/loss ratio
        avg_win = performance.get("avg_win", 0)
        avg_loss = performance.get("avg_loss", 0)
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 1.0
        
        return win_rate, win_loss_ratio
    
    def update_trade_result(self, asset: str, direction: str, amount: float, win: bool, profit: float) -> None:
        """
        Update trade result.
        
        Args:
            asset: Asset symbol
            direction: Trade direction
            amount: Trade amount
            win: Whether the trade was a win
            profit: Profit amount (positive for win, negative for loss)
        """
        if not self.initialized:
            raise RuntimeError("Risk manager not initialized")
        
        # Create trade record
        trade = {
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "win": win,
            "profit": profit,
            "timestamp": datetime.now()
        }
        
        # Add to trade history
        self.trade_history.append(trade)
        
        # Update consecutive wins/losses
        if win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.daily_profit += profit
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.daily_loss += amount
        
        # Update asset performance
        if asset not in self.asset_performance:
            self.asset_performance[asset] = {
                "wins": 0,
                "losses": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0
            }
        
        performance = self.asset_performance[asset]
        
        if win:
            performance["wins"] += 1
            performance["total_profit"] += profit
            performance["avg_win"] = performance["total_profit"] / performance["wins"]
        else:
            performance["losses"] += 1
            performance["total_loss"] += amount
            performance["avg_loss"] = performance["total_loss"] / performance["losses"]
        
        logger.debug(
            f"Updated trade result for {asset}: win={win}, profit={profit}, "
            f"consecutive_wins={self.consecutive_wins}, consecutive_losses={self.consecutive_losses}"
        )
    
    def filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and prioritize trading signals based on risk parameters.
        
        Args:
            signals: List of trading signals
            
        Returns:
            List[Dict[str, Any]]: Filtered and prioritized signals
        """
        if not self.initialized:
            raise RuntimeError("Risk manager not initialized")
        
        # Check if we've reached the maximum daily loss
        if self.daily_loss >= self.config.max_daily_loss:
            logger.warning(f"Maximum daily loss reached: {self.daily_loss} >= {self.config.max_daily_loss}")
            return []
        
        # Check if we've had too many consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            logger.warning(f"Maximum consecutive losses reached: {self.consecutive_losses} >= {self.config.max_consecutive_losses}")
            return []
        
        # Filter out signals with low confidence
        min_confidence = 0.6
        filtered_signals = [s for s in signals if s.get("confidence", 0) >= min_confidence]
        
        # Sort signals by confidence (highest first)
        filtered_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        
        # Limit the number of signals
        max_signals = 3
        filtered_signals = filtered_signals[:max_signals]
        
        logger.debug(f"Filtered {len(signals)} signals to {len(filtered_signals)} based on risk parameters")
        
        return filtered_signals
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get risk metrics.
        
        Returns:
            Dict[str, Any]: Risk metrics
        """
        if not self.initialized:
            raise RuntimeError("Risk manager not initialized")
        
        # Calculate overall performance
        total_trades = len(self.trade_history)
        wins = sum(1 for trade in self.trade_history if trade["win"])
        losses = total_trades - wins
        
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(trade["profit"] for trade in self.trade_history if trade["win"])
        total_loss = sum(trade["amount"] for trade in self.trade_history if not trade["win"])
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Calculate drawdown
        balance_history = []
        current = self.current_balance
        
        for trade in reversed(self.trade_history):
            if trade["win"]:
                current -= trade["profit"]
            else:
                current += trade["amount"]
            balance_history.append(current)
        
        balance_history.reverse()
        
        max_balance = self.current_balance
        max_drawdown = 0.0
        current_drawdown = 0.0
        
        for balance in balance_history:
            if balance > max_balance:
                max_balance = balance
                current_drawdown = 0.0
            else:
                current_drawdown = (max_balance - balance) / max_balance
                max_drawdown = max(max_drawdown, current_drawdown)
        
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "daily_profit": self.daily_profit,
            "daily_loss": self.daily_loss,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses
        }

class RiskRegistry:
    """
    Registry for dynamic risk model loading and hot-swapping.
    """
    _risk_models: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, risk_model: Any):
        cls._risk_models[name] = risk_model
        logger.info(f"Registered risk model: {name}")

    @classmethod
    def unregister(cls, name: str):
        if name in cls._risk_models:
            del cls._risk_models[name]
            logger.info(f"Unregistered risk model: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        return cls._risk_models.get(name)

    @classmethod
    def all(cls) -> List[str]:
        return list(cls._risk_models.keys())

class KellyRiskModel:
    """
    Kelly criterion-based position sizing.
    """
    def __init__(self, win_rate: float = 0.5, win_loss_ratio: float = 1.0):
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio

    def position_size(self, balance: float) -> float:
        kelly_fraction = (self.win_rate - (1 - self.win_rate) / self.win_loss_ratio)
        kelly_fraction = max(0.01, min(kelly_fraction, 0.2))  # Clamp for safety
        return balance * kelly_fraction

class VaRRiskModel:
    """
    Value-at-Risk (VaR) based position sizing.
    """
    def __init__(self, confidence: float = 0.95, lookback: int = 100):
        self.confidence = confidence
        self.lookback = lookback
        self.returns = []

    def update(self, pnl: float):
        self.returns.append(pnl)
        if len(self.returns) > self.lookback:
            self.returns.pop(0)

    def position_size(self, balance: float) -> float:
        if len(self.returns) < self.lookback:
            return balance * 0.02
        var = np.percentile(self.returns, (1 - self.confidence) * 100)
        risk = min(abs(var), 0.05)
        return balance * risk

class EmotionalRiskModel:
    """
    Emotion-driven risk adjustment (FOMO, greed, fear).
    """
    def __init__(self, base_model: Any, emotion_state: Dict[str, float]):
        self.base_model = base_model
        self.emotion_state = emotion_state

    def position_size(self, balance: float) -> float:
        base = self.base_model.position_size(balance)
        greed_boost = 1.0 + self.emotion_state.get('greed', 0.0) * 0.2
        fear_cut = 1.0 - self.emotion_state.get('fear', 0.0) * 0.5
        return max(1.0, base * greed_boost * fear_cut)

# Register risk models
RiskRegistry.register('kelly', KellyRiskModel())
RiskRegistry.register('var', VaRRiskModel())

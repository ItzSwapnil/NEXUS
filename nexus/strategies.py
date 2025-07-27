"""
Trading strategies module for NEXUS.

This module implements trading strategies for the NEXUS trading system.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Type

import numpy as np
import pandas as pd
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

from nexus.config import StrategyConfig
from nexus.models import ModelManager

logger = logging.getLogger("nexus.strategies")

class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface for all trading strategies.
    """
    
    def __init__(self, name: str, config: StrategyConfig, model_manager: ModelManager):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            model_manager: Model manager for predictions
        """
        self.name = name
        self.config = config
        self.model_manager = model_manager
        self.parameters = config.parameters
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        pass

class TrendFollowingStrategy(Strategy):
    """
    Trend following strategy using technical indicators.
    
    This strategy uses moving averages, MACD, and RSI to identify trends.
    """
    
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        signals = []
        
        # Get parameters
        sma_periods = self.parameters.get("sma_periods", [10, 20, 50])
        ema_periods = self.parameters.get("ema_periods", [5, 10, 20])
        rsi_period = self.parameters.get("rsi_period", 14)
        rsi_overbought = self.parameters.get("rsi_overbought", 70)
        rsi_oversold = self.parameters.get("rsi_oversold", 30)
        
        # Process each asset and timeframe
        for asset in self.config.assets:
            if asset not in data:
                continue
                
            for timeframe in self.config.timeframes:
                if timeframe not in data[asset]:
                    continue
                
                df = data[asset][timeframe]
                
                if len(df) < 50:  # Need enough data for indicators
                    continue
                
                # Get the latest candle
                latest = df.iloc[-1]
                
                # Calculate indicators if not already in the DataFrame
                if f"sma_{sma_periods[0]}" not in df.columns:
                    for period in sma_periods:
                        sma = SMAIndicator(df["close"], window=period)
                        df[f"sma_{period}"] = sma.sma_indicator()
                
                if f"ema_{ema_periods[0]}" not in df.columns:
                    for period in ema_periods:
                        ema = EMAIndicator(df["close"], window=period)
                        df[f"ema_{period}"] = ema.ema_indicator()
                
                if "macd" not in df.columns:
                    macd_ind = MACD(df["close"])
                    df["macd"] = macd_ind.macd()
                    df["macd_signal"] = macd_ind.macd_signal()
                    df["macd_diff"] = macd_ind.macd_diff()
                
                if "rsi" not in df.columns:
                    rsi_ind = RSIIndicator(df["close"], window=rsi_period)
                    df["rsi"] = rsi_ind.rsi()
                
                # Get the latest values
                latest = df.iloc[-1]
                
                # Initialize signal components
                trend_signals = []
                trend_strengths = []
                
                # Check SMA trend
                for i in range(len(sma_periods) - 1):
                    fast_sma = latest[f"sma_{sma_periods[i]}"]
                    slow_sma = latest[f"sma_{sma_periods[i+1]}"]
                    
                    if fast_sma > slow_sma:
                        trend_signals.append(1)  # Bullish
                        trend_strengths.append((fast_sma - slow_sma) / slow_sma)
                    elif fast_sma < slow_sma:
                        trend_signals.append(-1)  # Bearish
                        trend_strengths.append((slow_sma - fast_sma) / slow_sma)
                    else:
                        trend_signals.append(0)  # Neutral
                        trend_strengths.append(0)
                
                # Check EMA trend
                for i in range(len(ema_periods) - 1):
                    fast_ema = latest[f"ema_{ema_periods[i]}"]
                    slow_ema = latest[f"ema_{ema_periods[i+1]}"]
                    
                    if fast_ema > slow_ema:
                        trend_signals.append(1)  # Bullish
                        trend_strengths.append((fast_ema - slow_ema) / slow_ema)
                    elif fast_ema < slow_ema:
                        trend_signals.append(-1)  # Bearish
                        trend_strengths.append((slow_ema - fast_ema) / slow_ema)
                    else:
                        trend_signals.append(0)  # Neutral
                        trend_strengths.append(0)
                
                # Check MACD
                if latest["macd"] > latest["macd_signal"]:
                    trend_signals.append(1)  # Bullish
                    trend_strengths.append(abs(latest["macd_diff"]) / latest["macd_signal"] if latest["macd_signal"] != 0 else 0)
                elif latest["macd"] < latest["macd_signal"]:
                    trend_signals.append(-1)  # Bearish
                    trend_strengths.append(abs(latest["macd_diff"]) / latest["macd_signal"] if latest["macd_signal"] != 0 else 0)
                else:
                    trend_signals.append(0)  # Neutral
                    trend_strengths.append(0)
                
                # Check RSI
                if latest["rsi"] > rsi_overbought:
                    trend_signals.append(-1)  # Overbought, bearish
                    trend_strengths.append((latest["rsi"] - rsi_overbought) / (100 - rsi_overbought))
                elif latest["rsi"] < rsi_oversold:
                    trend_signals.append(1)  # Oversold, bullish
                    trend_strengths.append((rsi_oversold - latest["rsi"]) / rsi_oversold)
                else:
                    # RSI in middle range, use slope
                    rsi_slope = df["rsi"].diff().iloc[-1]
                    if rsi_slope > 0:
                        trend_signals.append(1)  # Rising RSI, bullish
                        trend_strengths.append(min(rsi_slope / 5, 1.0))  # Normalize
                    elif rsi_slope < 0:
                        trend_signals.append(-1)  # Falling RSI, bearish
                        trend_strengths.append(min(abs(rsi_slope) / 5, 1.0))  # Normalize
                    else:
                        trend_signals.append(0)  # Flat RSI, neutral
                        trend_strengths.append(0)
                
                # Combine signals
                if len(trend_signals) == 0:
                    continue
                
                # Weight the signals by their strength
                weighted_signals = [s * w for s, w in zip(trend_signals, trend_strengths)]
                overall_signal = sum(weighted_signals) / sum(trend_strengths) if sum(trend_strengths) > 0 else 0
                
                # Determine direction and confidence
                if overall_signal > 0.2:
                    direction = "buy"
                    confidence = min(abs(overall_signal), 1.0)
                elif overall_signal < -0.2:
                    direction = "sell"
                    confidence = min(abs(overall_signal), 1.0)
                else:
                    direction = "neutral"
                    confidence = 0.5
                
                # Create signal
                signal = {
                    "strategy": self.name,
                    "asset": asset,
                    "timeframe": timeframe,
                    "direction": direction,
                    "confidence": confidence,
                    "timestamp": datetime.now()
                }
                
                signals.append(signal)
        
        return signals

class MachineLearningStrategy(Strategy):
    """
    Machine learning strategy using trained models.
    
    This strategy uses machine learning models to predict price movements.
    """
    
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        signals = []
        
        # Get parameters
        model_names = self.config.models
        
        # Process each asset and timeframe
        for asset in self.config.assets:
            if asset not in data:
                continue
                
            for timeframe in self.config.timeframes:
                if timeframe not in data[asset]:
                    continue
                
                df = data[asset][timeframe]
                
                if len(df) < 50:  # Need enough data
                    continue
                
                # Make predictions with each model
                predictions = []
                confidences = []
                
                for model_name in model_names:
                    try:
                        # Get prediction and confidence
                        pred, conf = self.model_manager.predict(model_name, df)
                        
                        predictions.append(pred)
                        confidences.append(conf)
                        
                    except Exception as e:
                        logger.exception(f"Error making prediction with model {model_name}: {e}")
                
                if not predictions:
                    continue
                
                # Combine predictions (weighted by confidence)
                weighted_preds = [p * c for p, c in zip(predictions, confidences)]
                overall_pred = sum(weighted_preds) / sum(confidences) if sum(confidences) > 0 else 0.5
                
                # Determine direction and confidence
                if overall_pred > 0.6:
                    direction = "buy"
                    confidence = overall_pred
                elif overall_pred < 0.4:
                    direction = "sell"
                    confidence = 1 - overall_pred
                else:
                    direction = "neutral"
                    confidence = 0.5
                
                # Create signal
                signal = {
                    "strategy": self.name,
                    "asset": asset,
                    "timeframe": timeframe,
                    "direction": direction,
                    "confidence": confidence,
                    "timestamp": datetime.now()
                }
                
                signals.append(signal)
        
        return signals

class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.
    
    This strategy looks for price movements outside of Bollinger Bands
    and expects a reversion to the mean.
    """
    
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        signals = []
        
        # Get parameters
        bb_window = self.parameters.get("bb_window", 20)
        bb_std = self.parameters.get("bb_std", 2)
        
        # Process each asset and timeframe
        for asset in self.config.assets:
            if asset not in data:
                continue
                
            for timeframe in self.config.timeframes:
                if timeframe not in data[asset]:
                    continue
                
                df = data[asset][timeframe]
                
                if len(df) < bb_window:  # Need enough data
                    continue
                
                # Calculate Bollinger Bands if not already in the DataFrame
                if "bb_high" not in df.columns:
                    bb = BollingerBands(df["close"], window=bb_window, window_dev=bb_std)
                    df["bb_high"] = bb.bollinger_hband()
                    df["bb_mid"] = bb.bollinger_mavg()
                    df["bb_low"] = bb.bollinger_lband()
                    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
                    df["bb_pct"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])
                
                # Get the latest values
                latest = df.iloc[-1]
                
                # Check if price is outside Bollinger Bands
                if latest["close"] > latest["bb_high"]:
                    # Price above upper band, expect reversion down
                    direction = "sell"
                    # Calculate confidence based on distance from band
                    distance = (latest["close"] - latest["bb_high"]) / latest["bb_high"]
                    confidence = min(0.5 + distance * 5, 0.95)  # Cap at 0.95
                    
                elif latest["close"] < latest["bb_low"]:
                    # Price below lower band, expect reversion up
                    direction = "buy"
                    # Calculate confidence based on distance from band
                    distance = (latest["bb_low"] - latest["close"]) / latest["bb_low"]
                    confidence = min(0.5 + distance * 5, 0.95)  # Cap at 0.95
                    
                else:
                    # Price within bands, check position relative to middle band
                    if latest["close"] > latest["bb_mid"]:
                        # Price above middle band but below upper band
                        direction = "neutral"
                        confidence = 0.5
                    elif latest["close"] < latest["bb_mid"]:
                        # Price below middle band but above lower band
                        direction = "neutral"
                        confidence = 0.5
                    else:
                        # Price at middle band
                        direction = "neutral"
                        confidence = 0.5
                
                # Create signal
                signal = {
                    "strategy": self.name,
                    "asset": asset,
                    "timeframe": timeframe,
                    "direction": direction,
                    "confidence": confidence,
                    "timestamp": datetime.now()
                }
                
                signals.append(signal)
        
        return signals

class StrategyManager:
    """
    Strategy manager for the NEXUS trading system.
    
    This class manages trading strategies and coordinates signal generation.
    """
    
    def __init__(self, strategy_configs: List[StrategyConfig]):
        """
        Initialize the strategy manager.
        
        Args:
            strategy_configs: List of strategy configurations
        """
        self.strategy_configs = strategy_configs
        self.strategies: Dict[str, Strategy] = {}
        self.model_manager: Optional[ModelManager] = None
        self.initialized = False
    
    def initialize(self, model_manager: ModelManager) -> None:
        """
        Initialize the strategy manager.
        
        Args:
            model_manager: Model manager for predictions
        """
        logger.info("Initializing strategy manager")
        
        self.model_manager = model_manager
        
        # Create strategies
        for config in self.strategy_configs:
            if not config.enabled:
                continue
            
            try:
                # Create strategy based on name
                if config.name == "trend_following":
                    strategy = TrendFollowingStrategy(config.name, config, model_manager)
                elif config.name == "machine_learning":
                    strategy = MachineLearningStrategy(config.name, config, model_manager)
                elif config.name == "mean_reversion":
                    strategy = MeanReversionStrategy(config.name, config, model_manager)
                elif config.name == "deep_learning":
                    strategy = MachineLearningStrategy(config.name, config, model_manager)
                elif config.name == "reinforcement_learning":
                    strategy = MachineLearningStrategy(config.name, config, model_manager)
                else:
                    logger.warning(f"Unknown strategy: {config.name}")
                    continue
                
                # Add strategy to dictionary
                self.strategies[config.name] = strategy
                
                logger.info(f"Strategy {config.name} initialized")
                
            except Exception as e:
                logger.exception(f"Error initializing strategy {config.name}: {e}")
        
        self.initialized = True
        logger.info(f"Strategy manager initialized with {len(self.strategies)} strategies")
    
    def generate_signals(self, strategy_name: str, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        if not self.initialized:
            raise RuntimeError("Strategy manager not initialized")
        
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy = self.strategies[strategy_name]
        
        try:
            signals = strategy.generate_signals(data)
            logger.debug(f"Strategy {strategy_name} generated {len(signals)} signals")
            return signals
        except Exception as e:
            logger.exception(f"Error generating signals for strategy {strategy_name}: {e}")
            return []
    
    def get_all_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """
        Generate trading signals for all strategies.
        
        Args:
            data: Dictionary of feature DataFrames
            
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        if not self.initialized:
            raise RuntimeError("Strategy manager not initialized")
        
        all_signals = []
        
        for name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)
                all_signals.extend(signals)
            except Exception as e:
                logger.exception(f"Error generating signals for strategy {name}: {e}")
        
        logger.info(f"Generated {len(all_signals)} signals from all strategies")
        
        return all_signals

class StrategyRegistry:
    """
    Registry for dynamic strategy loading, hot-swapping, and experimentation.
    """
    _strategies: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Strategy]):
        cls._strategies[name] = strategy_cls
        logger.info(f"Registered strategy: {name}")

    @classmethod
    def unregister(cls, name: str):
        if name in cls._strategies:
            del cls._strategies[name]
            logger.info(f"Unregistered strategy: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[Strategy]]:
        return cls._strategies.get(name)

    @classmethod
    def all(cls) -> List[str]:
        return list(cls._strategies.keys())

# Example advanced strategies
class AIBasedStrategy(Strategy):
    """
    AI-based strategy using model manager predictions and regime context.
    """
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        signals = []
        for asset, tf_data in data.items():
            for tf, df in tf_data.items():
                if df.empty:
                    continue
                # Use model manager for prediction
                prediction = self.model_manager.predict(asset, tf, df)
                if prediction['signal'] in ['call', 'put'] and prediction['confidence'] > 0.6:
                    signals.append({
                        'asset': asset,
                        'timeframe': tf,
                        'signal': prediction['signal'],
                        'confidence': prediction['confidence'],
                        'reasoning': prediction.get('reasoning', 'AI model'),
                        'timestamp': datetime.now()
                    })
        return signals

class RegimeAwareStrategy(Strategy):
    """
    Regime-aware strategy that adapts logic based on detected market regime.
    """
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        signals = []
        for asset, tf_data in data.items():
            for tf, df in tf_data.items():
                if df.empty:
                    continue
                regime = self.model_manager.get_regime(asset, tf, df)
                if regime == 'trending':
                    # Trend-following logic
                    macd = MACD(df['close']).macd_diff().iloc[-1]
                    if macd > 0:
                        signals.append({'asset': asset, 'timeframe': tf, 'signal': 'call', 'confidence': 0.7, 'reasoning': 'MACD trend', 'timestamp': datetime.now()})
                    elif macd < 0:
                        signals.append({'asset': asset, 'timeframe': tf, 'signal': 'put', 'confidence': 0.7, 'reasoning': 'MACD trend', 'timestamp': datetime.now()})
                elif regime == 'ranging':
                    # Mean-reversion logic
                    rsi = RSIIndicator(df['close']).rsi().iloc[-1]
                    if rsi < 30:
                        signals.append({'asset': asset, 'timeframe': tf, 'signal': 'call', 'confidence': 0.6, 'reasoning': 'RSI oversold', 'timestamp': datetime.now()})
                    elif rsi > 70:
                        signals.append({'asset': asset, 'timeframe': tf, 'signal': 'put', 'confidence': 0.6, 'reasoning': 'RSI overbought', 'timestamp': datetime.now()})
        return signals

class FallbackStrategy(Strategy):
    """
    Fallback strategy for unstable or unknown market regimes.
    """
    def generate_signals(self, data: Dict[str, Dict[int, pd.DataFrame]]) -> List[Dict[str, Any]]:
        signals = []
        for asset, tf_data in data.items():
            for tf, df in tf_data.items():
                if df.empty:
                    continue
                # Simple volatility breakout
                bb = BollingerBands(df['close'])
                close = df['close'].iloc[-1]
                if close > bb.bollinger_hband().iloc[-1]:
                    signals.append({'asset': asset, 'timeframe': tf, 'signal': 'put', 'confidence': 0.5, 'reasoning': 'Volatility breakout', 'timestamp': datetime.now()})
                elif close < bb.bollinger_lband().iloc[-1]:
                    signals.append({'asset': asset, 'timeframe': tf, 'signal': 'call', 'confidence': 0.5, 'reasoning': 'Volatility breakout', 'timestamp': datetime.now()})
        return signals

# Register strategies
StrategyRegistry.register('ai_based', AIBasedStrategy)
StrategyRegistry.register('regime_aware', RegimeAwareStrategy)
StrategyRegistry.register('fallback', FallbackStrategy)

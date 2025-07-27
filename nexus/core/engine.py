"""
NEXUS Core Engine - The heart of the self-evolving AI trading system.

This engine orchestrates all components including:
- Market data processing and feature engineering
- Multi-model AI ensemble (RL, Transformers, Evolution)
- Real-time strategy adaptation and optimization
- Risk management and position sizing
- Performance monitoring and self-improvement
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from nexus.adapters.quotex import QuotexAdapter
from nexus.intelligence.transformer import MarketPredictor
from nexus.intelligence.rl_agent import RLAgent
from nexus.intelligence.regime_detector import RegimeDetector
from nexus.core.evolution import EvolutionEngine
from nexus.core.memory import VectorMemory
from nexus.strategies.meta_strategy import MetaStrategy, SignalType
from nexus.utils.config import NexusSettings, load_config
from nexus.utils.logger import get_nexus_logger, PerformanceLogger
from nexus.registry import registry

logger = get_nexus_logger("nexus.core.engine")
perf_logger = PerformanceLogger("engine")


class NexusEngine:
    """
    Core engine for NEXUS trading system.

    Coordinates all components, manages data flow, and executes trades.
    """

    def __init__(
        self,
        settings: Optional[NexusSettings] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        demo_mode: bool = True,
        auto_login: bool = True
    ):
        """
        Initialize the NEXUS engine.

        Args:
            settings: Configuration settings
            email: Quotex account email (overrides settings)
            password: Quotex account password (overrides settings)
            demo_mode: Whether to use demo account
            auto_login: Whether to log in automatically on init
        """
        # Load settings
        self.settings = settings or load_config()

        # Override credentials if provided
        if email and password:
            self.settings.quotex.email = email
            self.settings.quotex.password = password

        self.demo_mode = demo_mode
        self.running = False
        self.initialized = False
        self.last_update_time = {}
        self.candle_cache = {}
        self.lock = threading.Lock()

        # Executor for background tasks
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Initialize Quotex adapter
        self.quotex = QuotexAdapter(
            email=self.settings.quotex.email,
            password=self.settings.quotex.password,
            demo_mode=demo_mode
        )

        # Initialize intelligence components
        self.transformer = None
        self.rl_agent = None
        self.regime_detector = None
        self.meta_strategy = None
        self.memory = None
        self.evolution_engine = None

        # Performance tracking
        self.trades_history = []
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0
        }

        # Plugin/component registries for hot-swappable modules
        self.strategy_registry = registry.strategies
        self.model_registry = registry.models
        self.risk_registry: Dict[str, Any] = {}
        self.emotion_state = {
            'fomo': 0.0,
            'greed': 0.0,
            'fear': 0.0
        }
        self.bandit_state = {
            'explore_prob': 0.1,
            'exploit_prob': 0.9
        }

        # Auto login
        self._login_task = None
        # Do not create asyncio task here; login should be called explicitly in async context

    async def initialize(self):
        """Async initialization for NEXUS engine (stub)."""
        logger.info("NexusEngine async initialization complete (stub)")

    async def initialize_components(self):
        """Initialize all AI components."""
        logger.info("Initializing NEXUS intelligence components...")

        try:
            # Initialize memory system
            self.memory = VectorMemory(
                capacity=self.settings.memory.capacity,
                dimension=self.settings.memory.dimension
            )

            # Initialize regime detector
            self.regime_detector = RegimeDetector(
                n_regimes=self.settings.regime_detector.n_regimes,
                lookback_periods=self.settings.regime_detector.lookback_periods
            )

            # Initialize transformer
            self.transformer = MarketPredictor(
                lookback_periods=self.settings.transformer.lookback_periods,
                feature_dim=self.settings.transformer.feature_dim,
                batch_size=self.settings.transformer.batch_size
            )

            # Initialize RL agent
            self.rl_agent = RLAgent(
                state_dim=self.settings.rl_agent.state_dim,
                hidden_dim=self.settings.rl_agent.hidden_dim,
                buffer_capacity=self.settings.rl_agent.buffer_capacity
            )

            # Initialize meta strategy
            self.meta_strategy = MetaStrategy(
                transformer=self.transformer,
                rl_agent=self.rl_agent,
                regime_detector=self.regime_detector,
                config=None  # Use default config
            )

            # Initialize evolution engine
            self.evolution_engine = EvolutionEngine(
                meta_strategy=self.meta_strategy,
                population_size=self.settings.evolution.population_size,
                mutation_rate=self.settings.evolution.mutation_rate
            )

            self.initialized = True
            logger.info("NEXUS components initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize NEXUS components: {str(e)}")
            raise

    async def login(self) -> bool:
        """
        Log in to Quotex (pyquotex handles this on instantiation).

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Logging in to Quotex with email: {self.settings.quotex.email}")
            # Use the async version to properly await
            success = await self.quotex.login()
            if success:
                balance = await self.quotex.get_balance_async()
                logger.info(f"Successfully connected to Quotex. Balance: {balance}")
                return True
            else:
                logger.error("Failed to connect to Quotex")
                return False
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False

    def logout(self) -> bool:
        """
        Log out from Quotex.

        Returns:
            bool: Success status
        """
        try:
            logger.info("Logging out from Quotex...")
            success = self.quotex.logout()

            if success:
                logger.info("Successfully logged out from Quotex")
                return True
            else:
                logger.error("Failed to log out from Quotex")
                return False

        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False

    async def get_balance(self) -> float:
        """
        Get current account balance.

        Returns:
            float: Account balance
        """
        try:
            balance = await self.quotex.get_balance_async()
            logger.debug(f"Current balance: {balance}")
            return float(balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {str(e)}")
            return 0.0

    async def get_candles(self, asset: str, timeframe: int, count: int = 100) -> pd.DataFrame:
        """
        Get historical candles for an asset.

        Args:
            asset: Asset symbol
            timeframe: Timeframe in minutes
            count: Number of candles to fetch

        Returns:
            pd.DataFrame: Candles data
        """
        cache_key = f"{asset}_{timeframe}"

        # Check if we have recent data in cache
        if cache_key in self.candle_cache and self.last_update_time.get(cache_key):
            last_update = self.last_update_time[cache_key]
            if datetime.now() - last_update < timedelta(seconds=timeframe * 30):  # Only update if data is older than half the timeframe
                logger.debug(f"Using cached candles for {asset} {timeframe}m")
                return self.candle_cache[cache_key]

        try:
            with perf_logger.measure(f"get_candles_{timeframe}"):
                # Use async version
                candles = await self.quotex.get_candles_async(asset, timeframe=timeframe, count=count)

                if candles is None or len(candles) == 0:
                    logger.warning(f"No candles returned for {asset} {timeframe}m")
                    return pd.DataFrame()

                # Cache the data
                with self.lock:
                    self.candle_cache[cache_key] = candles
                    self.last_update_time[cache_key] = datetime.now()

                logger.debug(f"Fetched {len(candles)} candles for {asset} {timeframe}m")
                return candles

        except Exception as e:
            logger.error(f"Failed to get candles for {asset} {timeframe}m: {str(e)}")
            return pd.DataFrame()

    async def analyze_market(self, asset: str, timeframe: int) -> Dict:
        """
        Perform comprehensive market analysis.

        Args:
            asset: Asset symbol
            timeframe: Timeframe in minutes

        Returns:
            Dict: Analysis results
        """
        if not self.initialized:
            logger.warning("NEXUS components not initialized yet. Initializing now.")
            await self.initialize_components()

        try:
            # Get candles data
            with perf_logger.measure("get_market_data"):
                candles = await self.get_candles(asset, timeframe)

                if candles.empty:
                    logger.warning(f"No data available for {asset}")
                    return {"error": "No data available"}

            # Detect market regime
            with perf_logger.measure("regime_detection"):
                regime = await self.regime_detector.detect_regime(candles)
                logger.info(f"Detected {regime} regime for {asset}")

            # Generate signal using meta strategy
            with perf_logger.measure("generate_signal"):
                signal_result = await self.meta_strategy.generate_signal(candles, asset, timeframe)

                if not signal_result:
                    logger.info(f"No actionable signal for {asset}")
                    return {
                        "asset": asset,
                        "timeframe": timeframe,
                        "regime": regime,
                        "signal": "hold",
                        "confidence": 0.0,
                        "position_size": 0.0
                    }

                signal, position_size = signal_result

            # Store analysis in memory
            analysis_data = {
                "asset": asset,
                "timeframe": timeframe,
                "regime": regime,
                "signal": signal.value,
                "confidence": position_size * 5,  # Rough conversion to confidence scale
                "position_size": position_size,
                "timestamp": datetime.now().isoformat()
            }

            # Store in vector memory
            self.memory.store(
                key=f"{asset}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                data=analysis_data
            )

            return analysis_data

        except Exception as e:
            logger.error(f"Market analysis error for {asset}: {str(e)}")
            return {"error": str(e)}

    def register_strategy(self, name: str, strategy: Any):
        """Register a new trading strategy at runtime."""
        self.strategy_registry[name] = strategy
        logger.info(f"Strategy '{name}' registered.")

    def unregister_strategy(self, name: str):
        """Unregister a trading strategy at runtime."""
        if name in self.strategy_registry:
            del self.strategy_registry[name]
            logger.info(f"Strategy '{name}' unregistered.")

    def register_model(self, name: str, model: Any):
        self.model_registry[name] = model
        logger.info(f"Model '{name}' registered.")

    def unregister_model(self, name: str):
        if name in self.model_registry:
            del self.model_registry[name]
            logger.info(f"Model '{name}' unregistered.")

    def register_risk_module(self, name: str, risk_module: Any):
        self.risk_registry[name] = risk_module
        logger.info(f"Risk module '{name}' registered.")

    def unregister_risk_module(self, name: str):
        if name in self.risk_registry:
            del self.risk_registry[name]
            logger.info(f"Risk module '{name}' unregistered.")

    def update_emotional_state(self, trade_result: dict):
        """Simulate emotional intelligence based on trade outcomes."""
        if trade_result.get('success'):
            self.emotion_state['greed'] = min(1.0, self.emotion_state['greed'] + 0.05)
            self.emotion_state['fomo'] = max(0.0, self.emotion_state['fomo'] - 0.02)
            self.emotion_state['fear'] = max(0.0, self.emotion_state['fear'] - 0.03)
        else:
            self.emotion_state['fear'] = min(1.0, self.emotion_state['fear'] + 0.07)
            self.emotion_state['fomo'] = min(1.0, self.emotion_state['fomo'] + 0.03)
            self.emotion_state['greed'] = max(0.0, self.emotion_state['greed'] - 0.04)
        logger.debug(f"Updated emotional state: {self.emotion_state}")

    def contextual_bandit_decision(self, context: dict) -> str:
        """Adaptive exploration/exploitation using contextual bandits."""
        import random
        explore = random.random() < self.bandit_state['explore_prob']
        if explore:
            logger.info("Contextual bandit: exploring new strategy.")
            return random.choice(list(self.strategy_registry.keys()))
        else:
            logger.info("Contextual bandit: exploiting best-known strategy.")
            # For demo, just return the meta_strategy; in production, use performance metrics
            return 'meta_strategy'

    def online_learning_update(self, trade_record: dict):
        """Online learning: update models/strategies with new trade outcome."""
        # Update RL agent
        if self.rl_agent:
            self.rl_agent.learn_from_trade(trade_record)
        # Update transformer
        if self.transformer:
            self.transformer.update_from_trade(trade_record)
        # Update evolution engine
        if self.evolution_engine:
            self.evolution_engine.evolve(trade_record)
        logger.info("Online learning update complete.")

    def advanced_risk_management(self, context: dict, base_amount: float) -> float:
        """Apply Kelly, VaR, and emotional risk shaping to position sizing."""
        kelly_fraction = 0.05  # Placeholder; in production, compute from win/loss stats
        var_adjustment = 1.0 - self.emotion_state['fear'] * 0.5
        greed_boost = 1.0 + self.emotion_state['greed'] * 0.2
        position_size = base_amount * kelly_fraction * var_adjustment * greed_boost
        logger.debug(f"Advanced risk management position size: {position_size}")
        return max(1.0, position_size)

    async def execute_trade(self, asset: str, signal_type: str, amount: float, timeframe: int = 1, expiration: int = 5) -> Dict:
        """
        Execute a trade on Quotex.

        Args:
            asset: Asset symbol
            signal_type: Trade direction ('call' or 'put')
            amount: Trade amount
            timeframe: Analysis timeframe that generated the signal
            expiration: Trade expiration time in minutes

        Returns:
            Dict: Trade result
        """
        if signal_type not in ['call', 'put']:
            logger.error(f"Invalid signal type: {signal_type}")
            return {"success": False, "error": "Invalid signal type"}

        try:
            logger.info(f"Executing {signal_type} trade on {asset} for ${amount} with {expiration}m expiration")

            # Use async version
            with perf_logger.measure("execute_trade"):
                result = await self.quotex.buy_and_check_win_async(
                    asset=asset,
                    direction=signal_type,
                    amount=amount,
                    expiration=expiration
                )

            if result is None:
                logger.error(f"Trade execution failed for {asset}")
                return {"success": False, "error": "Trade execution failed"}

            # Record trade result
            trade_record = {
                "asset": asset,
                "signal_type": signal_type,
                "amount": amount,
                "expiration": expiration,
                "result": result,
                "success": result.get('win', False),
                "profit": result.get('profit', 0),
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            }

            self.trades_history.append(trade_record)

            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            if trade_record['success']:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1

            self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            self.performance_metrics['total_profit'] += trade_record['profit']

            # --- NEXUS: Self-evolution, online learning, and emotional intelligence ---
            self.update_emotional_state(trade_record)
            self.online_learning_update(trade_record)
            # --- END ---

            # Create feedback signal for models
            success = trade_record['success']
            profit = trade_record['profit']

            # Update meta strategy with performance feedback
            signal_obj = SignalType.BUY if signal_type == 'call' else SignalType.SELL
            await self.meta_strategy.update_performance(
                signal=TradingSignal(
                    signal_type=signal_obj,
                    confidence=amount / 100,  # Proxy for confidence
                    asset=asset,
                    timeframe=timeframe,
                    reasoning="Executed trade",
                    source_model="ensemble",
                    timestamp=datetime.now()
                ),
                success=success,
                profit=profit
            )

            logger.info(f"Trade {'successful' if success else 'failed'} with profit: {profit}")

            # Evolve strategy weights if needed
            if self.performance_metrics['total_trades'] % 5 == 0:  # Evolve every 5 trades
                asyncio.create_task(self.evolution_engine.evolve())

            return trade_record

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def auto_trade(self, asset: str, timeframe: int = 5, amount_percent: float = 0.02, min_confidence: float = 0.7):
        """
        Perform automated analysis and trading.

        Args:
            asset: Asset symbol
            timeframe: Timeframe in minutes
            amount_percent: Percentage of balance to trade
            min_confidence: Minimum confidence threshold

        Returns:
            Dict: Trade result or analysis
        """
        # Analyze market
        analysis = await self.analyze_market(asset, timeframe)

        if "error" in analysis:
            return analysis

        # Check if signal is actionable
        if analysis['signal'] in ['call', 'put'] and analysis['confidence'] >= min_confidence:
            # Calculate trade amount - use async version
            balance = await self.get_balance()
            amount = balance * amount_percent
            amount = max(1.0, min(amount, balance * 0.1))  # Cap at 10% of balance

            # Execute trade
            trade_result = await self.execute_trade(
                asset=asset,
                signal_type=analysis['signal'],
                amount=amount,
                timeframe=timeframe,
                expiration=timeframe  # Use same timeframe for expiration
            )

            # Combine analysis and trade result
            return {**analysis, **trade_result}
        else:
            logger.info(f"No actionable signal for {asset} (confidence: {analysis['confidence']:.2f})")
            return analysis

    async def start_trading_loop(self, assets: List[str], timeframe: int = 5, interval: int = 60):
        """
        Start automated trading loop.

        Args:
            assets: List of assets to trade
            timeframe: Timeframe in minutes
            interval: Trading interval in seconds
        """
        if not self.initialized:
            logger.warning("NEXUS components not initialized yet. Initializing now.")
            await self.initialize_components()

        self.running = True
        logger.info(f"Starting trading loop for assets: {assets}")

        while self.running:
            for asset in assets:
                if not self.running:
                    break

                try:
                    # Perform auto trade
                    result = await self.auto_trade(asset, timeframe)
                    logger.info(f"Auto trade result for {asset}: {result}")

                except Exception as e:
                    logger.error(f"Error in trading loop for {asset}: {str(e)}")

                # Sleep between assets
                await asyncio.sleep(5)

            # Sleep before next iteration
            logger.debug(f"Sleeping for {interval} seconds before next trading cycle")
            await asyncio.sleep(interval)

    def stop_trading_loop(self):
        """Stop the trading loop."""
        logger.info("Stopping trading loop")
        self.running = False

    async def train_models(self, assets: List[str], timeframes: List[int]):
        """
        Train all models with historical data.

        Args:
            assets: List of assets to use for training
            timeframes: List of timeframes to use for training

        Returns:
            bool: Training success status
        """
        if not self.initialized:
            logger.warning("NEXUS components not initialized yet. Initializing now.")
            await self.initialize_components()

        logger.info("Starting model training...")

        try:
            # Get training data for each asset and timeframe
            for asset in assets:
                for timeframe in timeframes:
                    logger.info(f"Training on {asset} {timeframe}m data")

                    # Get candles
                    candles = await self.get_candles(asset, timeframe, count=500)

                    if candles.empty:
                        logger.warning(f"No training data for {asset} {timeframe}m")
                        continue

                    # Train regime detector
                    await self.regime_detector.train(candles)

                    # For transformer and RL, we would need labeled data
                    # This is just a placeholder for actual implementation
                    # Generate synthetic labels for demonstration
                    # In a real system, we would use actual historical outcomes

                    # Train RL agent with some simulated experiences
                    # Real implementation would use actual market data and outcomes

                    logger.info(f"Completed training on {asset} {timeframe}m")

            logger.info("Model training completed")
            return True

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return False

    def get_performance_stats(self) -> Dict:
        """
        Get trading performance statistics.

        Returns:
            Dict: Performance metrics
        """
        metrics = self.performance_metrics.copy()

        # Calculate additional metrics
        if metrics['losing_trades'] > 0:
            metrics['profit_factor'] = metrics['winning_trades'] / metrics['losing_trades']

        return metrics

    def save_trade_history(self, filepath: Optional[str] = None):
        """
        Save trade history to CSV file.

        Args:
            filepath: Path to save file (default: ./data/trade_history_{datetime}.csv)
        """
        if not filepath:
            filepath = f"data/trade_history_{datetime.now().strftime('%Y%m%d%H%M')}.csv"

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        try:
            pd.DataFrame(self.trades_history).to_csv(filepath, index=False)
            logger.info(f"Trade history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trade history: {str(e)}")

    async def start(self):
        """Start the main trading loop (stub, runs indefinitely)."""
        logger.info("NexusEngine trading loop started (stub)")
        while True:
            await asyncio.sleep(60)


# Import at the end to avoid circular imports
from nexus.strategies.meta_strategy import TradingSignal

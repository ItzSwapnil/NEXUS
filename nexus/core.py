"""
Core module for NEXUS.

This module contains the main engine for the NEXUS trading system, coordinating
data collection, model training, strategy execution, and trade placement.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any

from nexus.client import QuotexClient
from nexus.config import Config
from nexus.data import DataManager
from nexus.models import ModelManager
from nexus.risk import RiskManager
from nexus.strategies import StrategyManager
from nexus.utils.logging import TradeLogger

logger = logging.getLogger("nexus.core")
trade_logger = TradeLogger()

class NexusCore:
    """
    Core engine for the NEXUS trading system.
    
    This class coordinates the different components of the system, including
    data collection, model training, strategy execution, and trade placement.
    """
    
    def __init__(self, client: QuotexClient, config: Config):
        """
        Initialize the NEXUS core engine.
        
        Args:
            client: Quotex client for API interactions
            config: System configuration
        """
        self.client = client
        self.config = config
        
        # Initialize components
        self.data_manager = DataManager(client, config.data)
        self.model_manager = ModelManager(config.models)
        self.strategy_manager = StrategyManager(config.strategies)
        self.risk_manager = RiskManager(config.risk)
        
        # State variables
        self.running = False
        self.available_assets: List[str] = []
        self.active_timeframes: Set[int] = set()
        self.last_data_update: Dict[str, Dict[int, datetime]] = {}
        self.last_model_update = datetime.now() - timedelta(days=1)  # Force initial update
        
        # Performance tracking
        self.performance = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "start_balance": 0.0,
            "current_balance": 0.0,
            "start_time": datetime.now(),
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the NEXUS system.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        logger.info("Initializing NEXUS system")
        
        try:
            # Get account information
            account_info = await self.client.update_account_info()
            self.performance["start_balance"] = account_info["balance"]
            self.performance["current_balance"] = account_info["balance"]
            
            # Get available assets
            self.available_assets = await self.client.get_available_assets()
            logger.info(f"Found {len(self.available_assets)} available assets")
            
            # Filter enabled assets that are available
            enabled_assets = [
                asset.symbol for asset in self.config.assets 
                if asset.enabled and asset.symbol in self.available_assets
            ]
            
            if not enabled_assets:
                logger.warning("No enabled assets are available for trading")
                return False
            
            logger.info(f"Enabled assets for trading: {', '.join(enabled_assets)}")
            
            # Collect timeframes from strategies and assets
            for strategy in self.config.strategies:
                if strategy.enabled:
                    self.active_timeframes.update(strategy.timeframes)
            
            for asset in self.config.assets:
                if asset.enabled and asset.symbol in self.available_assets:
                    self.active_timeframes.update(asset.timeframes)
            
            logger.info(f"Active timeframes: {', '.join(str(tf) for tf in sorted(self.active_timeframes))}")
            
            # Initialize data structures for tracking last updates
            for asset_symbol in enabled_assets:
                self.last_data_update[asset_symbol] = {
                    tf: datetime.now() - timedelta(days=1) for tf in self.active_timeframes
                }
            
            # Initialize data manager
            await self.data_manager.initialize(enabled_assets, list(self.active_timeframes))
            
            # Initialize model manager
            self.model_manager.initialize()
            
            # Initialize strategy manager
            self.strategy_manager.initialize(self.model_manager)
            
            # Initialize risk manager
            self.risk_manager.initialize(self.performance["current_balance"])
            
            logger.info("NEXUS system initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error initializing NEXUS system: {e}")
            return False
    
    async def run(self) -> None:
        """
        Run the NEXUS trading system.
        
        This method starts the main loop that coordinates data collection,
        model training, strategy execution, and trade placement.
        """
        logger.info("Starting NEXUS trading system")
        
        # Initialize the system
        if not await self.initialize():
            logger.error("Failed to initialize NEXUS system")
            return
        
        self.running = True
        
        try:
            # Main loop
            while self.running:
                # Update account information
                await self.update_account_info()
                
                # Update data for all assets and timeframes
                await self.update_data()
                
                # Update models if needed
                await self.update_models()
                
                # Generate trading signals
                signals = await self.generate_signals()
                
                # Execute trades based on signals
                if signals:
                    await self.execute_trades(signals)
                
                # Sleep to avoid excessive API calls
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("NEXUS system shutdown requested")
        except Exception as e:
            logger.exception(f"Error in NEXUS main loop: {e}")
        finally:
            self.running = False
            logger.info("NEXUS trading system stopped")
    
    async def update_account_info(self) -> None:
        """Update account information."""
        try:
            account_info = await self.client.update_account_info()
            current_balance = account_info["balance"]
            
            # Update performance tracking
            if current_balance != self.performance["current_balance"]:
                self.performance["current_balance"] = current_balance
                trade_logger.balance_update(current_balance, account_info["currency"])
                
                # Update risk manager with new balance
                self.risk_manager.update_balance(current_balance)
                
        except Exception as e:
            logger.error(f"Error updating account info: {e}")
    
    async def update_data(self) -> None:
        """Update data for all assets and timeframes."""
        now = datetime.now()
        
        for asset_symbol, timeframes in self.last_data_update.items():
            for timeframe, last_update in timeframes.items():
                # Check if update is needed (update at least every timeframe seconds)
                if (now - last_update).total_seconds() >= timeframe:
                    try:
                        # Update data for this asset and timeframe
                        await self.data_manager.update_data(asset_symbol, timeframe)
                        
                        # Update last update time
                        self.last_data_update[asset_symbol][timeframe] = now
                        
                    except Exception as e:
                        logger.error(f"Error updating data for {asset_symbol} at {timeframe}s: {e}")
    
    async def update_models(self) -> None:
        """Update prediction models if needed."""
        now = datetime.now()
        
        # Update models every hour by default
        if (now - self.last_model_update).total_seconds() >= 3600:
            try:
                # Get data for training
                training_data = await self.data_manager.get_training_data()
                
                # Update models
                await asyncio.to_thread(self.model_manager.update_models, training_data)
                
                # Update last update time
                self.last_model_update = now
                
                logger.info("Models updated successfully")
                
            except Exception as e:
                logger.error(f"Error updating models: {e}")
    
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals from all strategies.
        
        Returns:
            List[Dict[str, Any]]: List of trading signals
        """
        signals = []
        
        try:
            # Get latest data for all assets and timeframes
            latest_data = await self.data_manager.get_latest_data()
            
            # Generate signals from strategies
            for strategy_config in self.config.strategies:
                if not strategy_config.enabled:
                    continue
                
                strategy_signals = await asyncio.to_thread(
                    self.strategy_manager.generate_signals,
                    strategy_config.name,
                    latest_data
                )
                
                # Add signals to the list
                signals.extend(strategy_signals)
                
                # Log signals
                for signal in strategy_signals:
                    trade_logger.strategy_signal(
                        strategy=signal["strategy"],
                        asset=signal["asset"],
                        timeframe=signal["timeframe"],
                        signal=signal["direction"],
                        confidence=signal["confidence"]
                    )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def execute_trades(self, signals: List[Dict[str, Any]]) -> None:
        """
        Execute trades based on signals.
        
        Args:
            signals: List of trading signals
        """
        if not signals:
            return
        
        try:
            # Filter and prioritize signals
            filtered_signals = self.risk_manager.filter_signals(signals)
            
            for signal in filtered_signals:
                # Calculate trade parameters
                asset = signal["asset"]
                direction = signal["direction"]
                confidence = signal["confidence"]
                timeframe = signal["timeframe"]
                
                # Skip neutral signals
                if direction.lower() == "neutral":
                    continue
                
                # Map direction to Quotex format
                quotex_direction = "call" if direction.lower() == "buy" else "put"
                
                # Calculate amount based on risk management
                amount = self.risk_manager.calculate_position_size(
                    asset=asset,
                    confidence=confidence
                )
                
                # Skip if amount is too small
                if amount <= 0:
                    logger.warning(f"Skipping trade for {asset}: amount too small")
                    continue
                
                # Calculate expiration (use timeframe as expiration by default)
                expiration = timeframe
                
                # Log trade placement
                trade_logger.trade_placed(
                    asset=asset,
                    direction=quotex_direction,
                    amount=amount,
                    expiration=expiration
                )
                
                # Place trade and wait for result
                trade_result = await self.client.place_trade(
                    asset=asset,
                    amount=amount,
                    direction=quotex_direction,
                    expiration=expiration,
                    wait_for_result=True
                )
                
                # Process trade result
                result = trade_result.get("result", {})
                win = result.get("win", False)
                profit = result.get("profit", 0.0)
                
                # Update performance tracking
                self.performance["trades"] += 1
                if win:
                    self.performance["wins"] += 1
                    self.performance["profit"] += profit
                else:
                    self.performance["losses"] += 1
                    self.performance["profit"] -= amount
                
                # Log trade result
                trade_logger.trade_result(
                    asset=asset,
                    direction=quotex_direction,
                    amount=amount,
                    result="win" if win else "loss",
                    profit=profit if win else -amount
                )
                
                # Update risk manager with trade result
                self.risk_manager.update_trade_result(
                    asset=asset,
                    direction=quotex_direction,
                    amount=amount,
                    win=win,
                    profit=profit if win else -amount
                )
                
                # Wait a bit between trades to avoid rate limiting
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.exception(f"Error executing trades: {e}")
    
    def stop(self) -> None:
        """Stop the NEXUS trading system."""
        logger.info("Stopping NEXUS trading system")
        self.running = False
    
    def get_performance(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        # Calculate additional statistics
        win_rate = self.performance["wins"] / self.performance["trades"] if self.performance["trades"] > 0 else 0
        roi = self.performance["profit"] / self.performance["start_balance"] if self.performance["start_balance"] > 0 else 0
        runtime = (datetime.now() - self.performance["start_time"]).total_seconds() / 3600  # hours
        
        return {
            **self.performance,
            "win_rate": win_rate,
            "roi": roi,
            "runtime_hours": runtime
        }
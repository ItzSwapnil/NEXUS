"""
Exploration controller for balancing exploitation vs. exploration in trading decisions.

This module implements sophisticated epsilon-greedy exploration strategies
that adapt based on model confidence, market uncertainty, and performance metrics.
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from nexus.utils.config import NexusSettings
from nexus.utils.logger import get_nexus_logger

logger = get_nexus_logger("nexus.intelligence.exploration")

@dataclass
class ExplorationMetrics:
    """Metrics used to compute exploration rate."""
    confidence: float = 0.5
    uncertainty: float = 0.5
    win_rate: float = 0.5
    volatility: float = 0.5
    regime_stability: float = 0.5

class ExplorationController:
    """
    Advanced exploration controller that dynamically adjusts exploration rates
    based on market conditions, model confidence, and performance metrics.
    
    The controller uses a sophisticated epsilon calculation that considers:
    - Model confidence metrics (Sharpe ratio, stability, win rate)  
    - Market uncertainty indicators (ATR, disagreement, spread, OTC status)
    - Payout incentives (higher payouts reduce exploration need)
    - Performance feedback loops
    """
    
    def __init__(self, settings: NexusSettings):
        """
        Initialize exploration controller with configuration.
        
        Args:
            settings: NEXUS configuration settings
        """
        self.cfg = settings.exploration
        self.settings = settings
        
        # Adaptation parameters
        self.adaptation_rate = 0.05
        self.performance_window = 20
        self.recent_performance: list[float] = []
        
        # Regime-specific exploration multipliers
        self.regime_multipliers = {
            "trending": 0.8,      # Less exploration in trending markets
            "ranging": 1.2,       # More exploration in ranging markets  
            "volatile": 1.5,      # Much more exploration in volatile markets
            "reversal": 1.0,      # Normal exploration during reversals
            "unknown": 1.1        # Slightly more exploration when uncertain
        }
        
        logger.info(f"Exploration controller initialized with base epsilon: {self.cfg.base_epsilon}")
    
    def compute_epsilon(
        self,
        confidence_metrics: Dict[str, float],
        uncertainty_metrics: Dict[str, float],
        payout: float = 80.0,
        regime: Optional[str] = None
    ) -> float:
        """
        Compute dynamic exploration rate (epsilon) based on current conditions.
        
        Args:
            confidence_metrics: Model confidence indicators
                - sharpe: Sharpe ratio of recent trades
                - stability: Strategy performance stability  
                - win_rate: Recent win rate percentage
            uncertainty_metrics: Market uncertainty indicators
                - atr: Average True Range (volatility)
                - disagreement: Model prediction disagreement
                - spread: Bid-ask spread
                - otc: Whether asset is OTC (boolean converted to float)
            payout: Expected payout percentage for the trade
            regime: Current market regime (if detected)
            
        Returns:
            float: Computed epsilon value between min_epsilon and max_epsilon
        """
        # Normalize confidence metrics to [0, 1]
        sharpe = self._normalize_sharpe(confidence_metrics.get("sharpe", 0.0))
        stability = self._clamp(confidence_metrics.get("stability", 0.5))
        win_rate = self._clamp(confidence_metrics.get("win_rate", 0.5))
        
        # Composite confidence score
        confidence = (sharpe + stability + win_rate) / 3.0
        
        # Normalize uncertainty metrics  
        atr = self._clamp(uncertainty_metrics.get("atr", 0.5))
        disagreement = self._clamp(uncertainty_metrics.get("disagreement", 0.0))
        spread = self._clamp(uncertainty_metrics.get("spread", 0.0) * 10000)  # Scale spread 
        otc = float(uncertainty_metrics.get("otc", False))
        
        # Composite uncertainty score with weights
        uncertainty = (
            0.4 * atr +
            0.3 * disagreement + 
            0.2 * spread +
            0.1 * otc
        )
        
        # Base exploration rate
        base_eps = self.cfg.base_epsilon
        
        # Confidence adjustment: lower confidence -> higher exploration
        confidence_factor = 1.0 - (confidence * 0.6)  # Range: [0.4, 1.0]
        
        # Uncertainty adjustment: higher uncertainty -> higher exploration  
        uncertainty_factor = 1.0 + (uncertainty * self.cfg.k_uncertainty)
        
        # Payout adjustment: higher payout -> lower exploration (more exploitation)
        payout_factor = 1.0 - ((payout - 70.0) / 100.0) * 0.2  # Normalize around 70% baseline
        payout_factor = self._clamp(payout_factor, 0.8, 1.2)
        
        # Regime adjustment
        regime_factor = self.regime_multipliers.get(regime, 1.0) if regime else 1.0
        
        # Performance feedback adjustment
        perf_factor = self._compute_performance_factor()
        
        # Combine all factors
        epsilon = base_eps * confidence_factor * uncertainty_factor * payout_factor * regime_factor * perf_factor
        
        # Ensure epsilon stays within configured bounds
        epsilon = self._clamp(epsilon, self.cfg.min_epsilon, self.cfg.max_epsilon)
        
        logger.debug(
            f"Epsilon computation: base={base_eps:.3f}, conf={confidence_factor:.3f}, "
            f"unc={uncertainty_factor:.3f}, payout={payout_factor:.3f}, "
            f"regime={regime_factor:.3f}, perf={perf_factor:.3f} -> eps={epsilon:.3f}"
        )
        
        return round(epsilon, 4)
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update exploration controller with trade performance feedback.
        
        Args:
            trade_result: Dictionary containing trade outcome
                - success: Whether trade was successful
                - profit: Profit/loss amount
                - exploratory: Whether trade was exploratory
        """
        success = trade_result.get("success", False)
        profit = float(trade_result.get("profit", 0.0))
        exploratory = trade_result.get("exploratory", False)
        
        # Track performance for adaptation
        performance_score = 1.0 if success else 0.0
        if profit != 0:
            # Weight by profit magnitude (normalized)
            profit_weight = min(2.0, abs(profit) / 10.0)  # Cap at 2x weight
            performance_score *= profit_weight if profit > 0 else (1.0 / profit_weight)
        
        self.recent_performance.append(performance_score)
        
        # Keep only recent performance window
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance.pop(0)
            
        # Log exploratory trade outcomes for monitoring
        if exploratory:
            outcome = "WIN" if success else "LOSS"
            logger.info(f"Exploratory trade {outcome}: profit={profit:.2f}")
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """
        Get current exploration statistics.
        
        Returns:
            Dict containing exploration metrics and performance
        """
        avg_performance = sum(self.recent_performance) / len(self.recent_performance) if self.recent_performance else 0.5
        
        return {
            "base_epsilon": self.cfg.base_epsilon,
            "min_epsilon": self.cfg.min_epsilon,
            "max_epsilon": self.cfg.max_epsilon,
            "recent_performance": avg_performance,
            "performance_samples": len(self.recent_performance),
            "adaptation_rate": self.adaptation_rate
        }
    
    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to [0, 1] range.""" 
        # Typical Sharpe ratios range from -2 to +3
        # Map to sigmoid-like curve
        return 1.0 / (1.0 + math.exp(-sharpe))
    
    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to specified range."""
        return max(min_val, min(max_val, value))
    
    def _compute_performance_factor(self) -> float:
        """
        Compute performance-based adjustment factor.
        
        Returns:
            float: Adjustment factor based on recent performance
        """
        if not self.recent_performance:
            return 1.0
            
        avg_perf = sum(self.recent_performance) / len(self.recent_performance)
        
        # If performing well, reduce exploration slightly
        # If performing poorly, increase exploration
        if avg_perf > 0.6:
            return 0.9  # Reduce exploration when doing well
        elif avg_perf < 0.4:
            return 1.2  # Increase exploration when struggling
        else:
            return 1.0  # Neutral adjustment
    
    def reset_performance(self) -> None:
        """Reset performance tracking (useful for strategy changes)."""
        self.recent_performance.clear()
        logger.info("Exploration performance metrics reset")

__all__ = ["ExplorationController", "ExplorationMetrics"]

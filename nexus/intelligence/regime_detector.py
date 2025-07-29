"""
Advanced Market Regime Detection for NEXUS Trading System

This module implements sophisticated market regime detection using:
- Hidden Markov Models for regime transitions
- Technical indicators analysis
- Volatility clustering detection
- Machine learning classification
- Real-time regime switching detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import asyncio
from pathlib import Path
import joblib

from nexus.utils.logger import get_nexus_logger
from nexus.utils.technical import calculate_features

logger = get_nexus_logger("nexus.intelligence.regime_detector")

class RegimeDetector:
    """
    Market regime detector using Hidden Markov Models and technical analysis
    to identify market conditions and adapt trading strategy accordingly.
    """

    REGIMES = ["trending", "ranging", "volatile", "reversal", "unknown"]

    def __init__(self, n_regimes: int = 4, lookback_periods: int = 200, sensitivity: float = 0.5):
        """
        Initialize the regime detector.

        Args:
            n_regimes: Number of market regimes to identify
            lookback_periods: Number of candles to analyze for regime detection
            sensitivity: Sensitivity factor for regime detection
        """
        self.n_regimes = n_regimes
        self.lookback_periods = lookback_periods
        self.sensitivity = sensitivity

        # HMM model for regime detection
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        # KMeans for feature clustering
        self.kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=42
        )

        # Scaler for normalizing features
        self.scaler = StandardScaler()

        # Regime classification mapping
        self.regime_mapping = {}

        # Last known regime and history
        self.current_regime = None
        self.regime_history = []
        self.regime_start_time = None

        # Model persistence
        self.model_path = Path("models/regime_detector/")
        self.model_path.mkdir(exist_ok=True, parents=True)

        # Load existing model if available
        self._load_model()

    def _load_model(self):
        """Load trained regime detector model if it exists."""
        try:
            hmm_path = self.model_path / "hmm_model.pkl"
            kmeans_path = self.model_path / "kmeans_model.pkl"
            scaler_path = self.model_path / "scaler.pkl"
            mapping_path = self.model_path / "regime_mapping.pkl"

            if hmm_path.exists() and kmeans_path.exists() and mapping_path.exists():
                self.hmm_model = joblib.load(hmm_path)
                self.kmeans = joblib.load(kmeans_path)
                self.scaler = joblib.load(scaler_path)
                self.regime_mapping = joblib.load(mapping_path)
                logger.info("Loaded existing regime detector models")
                return True
        except Exception as e:
            logger.error(f"Failed to load regime models: {e}")

        return False

    def _save_model(self):
        """Save trained regime detector model."""
        try:
            joblib.dump(self.hmm_model, self.model_path / "hmm_model.pkl")
            joblib.dump(self.kmeans, self.model_path / "kmeans_model.pkl")
            joblib.dump(self.scaler, self.model_path / "scaler.pkl")
            joblib.dump(self.regime_mapping, self.model_path / "regime_mapping.pkl")
            logger.info("Saved regime detector models")
        except Exception as e:
            logger.error(f"Failed to save regime models: {e}")

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract relevant features for regime detection.

        Args:
            data: OHLCV market data

        Returns:
            np.ndarray: Feature matrix for regime detection
        """
        # Apply technical indicators
        features = calculate_features(data)

        # Select relevant features for regime detection
        selected_features = [
            'trend_strength',
            'volatility',
            'momentum',
            'mean_reversion',
            'volume_trend',
            'price_pattern'
        ]

        # Ensure all features exist
        for feature in selected_features:
            if feature not in features.columns:
                features[feature] = 0.0

        # Get selected features and handle NaN values
        X = features[selected_features].fillna(0).values

        return X

    def _map_regimes_to_labels(self, regime_states: np.ndarray, X: np.ndarray):
        """
        Map numerical regime states to meaningful labels based on feature values.

        Args:
            regime_states: Numerical regime states
            X: Feature matrix used for regime detection
        """
        if len(regime_states) == 0:
            return

        # Calculate cluster centers for each regime
        regime_centers = {}
        for i in range(self.n_regimes):
            mask = (regime_states == i)
            if np.sum(mask) > 0:
                regime_centers[i] = np.mean(X[mask], axis=0)

        # Define regime characteristics based on feature importance
        trend_idx = 0  # Index of trend_strength in features
        vol_idx = 1    # Index of volatility in features
        mom_idx = 2    # Index of momentum in features

        # Map regimes to labels based on their characteristics
        mapping = {}
        for regime, center in regime_centers.items():
            trend = center[trend_idx]
            vol = center[vol_idx]
            mom = center[mom_idx]

            if vol > 0.7:  # High volatility
                mapping[regime] = "volatile"
            elif abs(trend) > 0.6:  # Strong trend
                mapping[regime] = "trending"
            elif abs(mom) > 0.7 and abs(trend) < 0.3:  # High momentum but weak trend
                mapping[regime] = "reversal"
            else:  # Low volatility, low trend
                mapping[regime] = "ranging"

        self.regime_mapping = mapping
        logger.info(f"Updated regime mapping: {mapping}")

    async def train(self, data: pd.DataFrame):
        """
        Train the regime detector on historical market data.

        Args:
            data: OHLCV market data
        """
        if len(data) < self.lookback_periods:
            logger.warning(f"Not enough data for training regime detector. Need {self.lookback_periods} periods.")
            return False

        # Extract features
        X = self._extract_features(data)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train HMM model
        try:
            self.hmm_model.fit(X_scaled)

            # Get regime states
            regime_states = self.hmm_model.predict(X_scaled)

            # Train KMeans as backup method
            self.kmeans.fit(X_scaled)

            # Map numerical states to meaningful labels
            self._map_regimes_to_labels(regime_states, X)

            # Save the trained model
            self._save_model()

            logger.info("Successfully trained regime detector")
            return True

        except Exception as e:
            logger.error(f"Failed to train regime detector: {e}")
            return False

    async def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime based on recent price action.

        Args:
            data: OHLCV market data

        Returns:
            str: Detected market regime
        """
        if len(data) < 30:  # Need at least 30 candles for meaningful detection
            logger.warning("Not enough data for regime detection")
            return "unknown"

        # Use recent data for detection
        recent_data = data.tail(min(self.lookback_periods, len(data)))

        # Extract features
        X = self._extract_features(recent_data)

        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except:
            # If scaler not fitted, use StandardScaler directly
            X_scaled = StandardScaler().fit_transform(X)

        try:
            # Primary method: HMM prediction
            regime_state = self.hmm_model.predict(X_scaled)[-1]  # Get the last state

            # Map numerical state to label
            if regime_state in self.regime_mapping:
                regime = self.regime_mapping[regime_state]
            else:
                # Fallback to KMeans
                regime_state = self.kmeans.predict(X_scaled[-1:].reshape(1, -1))[0]
                if regime_state in self.regime_mapping:
                    regime = self.regime_mapping[regime_state]
                else:
                    # Default to most common regime if mapping fails
                    regime = "ranging"

            # Track regime changes
            if self.current_regime != regime:
                logger.info(f"Regime change detected: {self.current_regime} -> {regime}")
                self.regime_start_time = datetime.now()

            # Update current regime and history
            self.current_regime = regime
            self.regime_history.append((datetime.now(), regime))

            # Keep history manageable
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            return regime

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return "unknown"

    def get_regime_duration(self) -> Optional[timedelta]:
        """
        Get the duration of the current regime.

        Returns:
            Optional[timedelta]: Duration of current regime or None if unknown
        """
        if self.regime_start_time is None:
            return None

        return datetime.now() - self.regime_start_time

    def get_regime_history(self, lookback_hours: int = 24) -> Dict[str, int]:
        """
        Get regime distribution over recent history.

        Args:
            lookback_hours: Number of hours to look back

        Returns:
            Dict[str, int]: Count of each regime in the lookback period
        """
        if not self.regime_history:
            return {regime: 0 for regime in self.REGIMES}

        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_regimes = [r for t, r in self.regime_history if t >= cutoff_time]

        # Count occurrences of each regime
        regime_counts = {regime: 0 for regime in self.REGIMES}
        for regime in recent_regimes:
            if regime in regime_counts:
                regime_counts[regime] += 1

        return regime_counts

    def detect(self, candles: List[Dict]) -> str:
        """Detect market regime based on candle data."""
        returns = [candle['close'] / candle['open'] - 1 for candle in candles]
        volatility = np.std(returns)
        trend = np.mean(returns)

        if trend > self.sensitivity and volatility < self.sensitivity:
            return "Trending"
        elif volatility > self.sensitivity:
            return "Volatile"
        else:
            return "Sideways"

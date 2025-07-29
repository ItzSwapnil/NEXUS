import pytest
import numpy as np
import pandas as pd
import asyncio
from nexus.intelligence.transformer import MarketPredictor
from nexus.intelligence.rl_agent import RLAgent
from nexus.intelligence.regime_detector import RegimeDetector

def test_market_predictor_forward():
    predictor = MarketPredictor(lookback_periods=10, feature_dim=5, batch_size=2)
    # Generate dummy data
    data = pd.DataFrame(np.random.randn(10, 5), columns=[f'f{i}' for i in range(5)])
    tensor = predictor.preprocess(data)
    logits, confidence = predictor.model(tensor)
    assert logits.shape[-1] == 3
    assert confidence.shape[-1] == 1

def test_rl_agent_learn():
    agent = RLAgent(state_dim=4, hidden_dim=8, buffer_capacity=100)
    # Simulate experience
    state = np.random.randn(4)
    next_state = np.random.randn(4)
    agent.store_transition(state, 1, 1.0, next_state, False)
    agent.learn_from_trade({'state': state, 'action': 1, 'reward': 1.0, 'next_state': next_state, 'done': False})
    assert len(agent.memory) > 0

@pytest.mark.asyncio
async def test_regime_detector_detect():
    detector = RegimeDetector(n_regimes=3, lookback_periods=50)
    # Generate dummy OHLCV data
    data = pd.DataFrame({
        'open': np.random.rand(50),
        'high': np.random.rand(50),
        'low': np.random.rand(50),
        'close': np.random.rand(50),
        'volume': np.random.rand(50)
    })
    regime = await detector.detect_regime(data)
    assert regime in detector.REGIMES

    # Test with different regime configurations
    detector = RegimeDetector(n_regimes=2, lookback_periods=100)
    data = pd.DataFrame({
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'volume': np.random.rand(100)
    })
    regime = await detector.detect_regime(data)
    assert regime in detector.REGIMES

    detector = RegimeDetector(n_regimes=4, lookback_periods=75)
    data = pd.DataFrame({
        'open': np.random.rand(75),
        'high': np.random.rand(75),
        'low': np.random.rand(75),
        'close': np.random.rand(75),
        'volume': np.random.rand(75)
    })
    regime = await detector.detect_regime(data)
    assert regime in detector.REGIMES

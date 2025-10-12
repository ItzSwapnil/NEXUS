"""
AI Module - Real Machine Learning Models for Trading.

This module contains actual AI/ML implementations:
- LSTM with Attention for sequence prediction
- Deep Q-Network (DQN) for reinforcement learning
- Ensemble Manager for combining multiple models
"""

try:
    from nexus.ai.lstm_predictor import LSTMPredictor, MarketPredictor
    from nexus.ai.deep_rl_agent import DeepRLAgent, DuelingDQN
    from nexus.ai.ensemble_manager import AIEnsembleManager

    __all__ = [
        'LSTMPredictor',
        'MarketPredictor',
        'DeepRLAgent',
        'DuelingDQN',
        'AIEnsembleManager',
    ]
except ImportError as e:
    # PyTorch not installed - AI features disabled
    import warnings
    warnings.warn(f"AI features unavailable: {e}. Install PyTorch to enable AI models.")
    __all__ = []


# Models Directory

This directory stores trained AI models and their configurations for NEXUS. Model files are generated locally and are intentionally excluded from Git because they can be large, frequently changing, and may contain proprietary training results.

## Structure

```
models/
├── meta_strategy_champion.json    # Best performing strategy configuration
├── lstm_predictor.pth             # Trained LSTM model weights
├── dqn_agent.pth                  # Trained DQN agent weights
├── ensemble_weights.json          # Ensemble model weights
├── transformer/                   # Transformer model files
└── README.md                      # This file
```

## Model Files

### LSTM Predictor (`lstm_predictor.pth`)
- Deep learning model for price prediction
- Uses attention mechanisms
- Trained on historical market data
- Input: 60 timesteps of OHLCV + indicators
- Output: Direction, confidence, value predictions

### DQN Agent (`dqn_agent.pth`)
- Reinforcement learning agent
- Learns optimal trading strategies
- Dueling architecture with noisy layers
- Trained through market interaction

### Ensemble Weights (`ensemble_weights.json`)
- Meta-learning model combination
- Adaptive weighting based on performance
- Updated during online learning

### Meta Strategy Champion (`meta_strategy_champion.json`)
- Best performing strategy configuration
- Stores strategy parameters and performance metrics
- Used as baseline for new strategies

## Training Models

### Train All Models
```bash
python -m nexus.ai.train_models
```

### Train Individual Models
```bash
# LSTM Predictor
python -m nexus.ai.train_models --model lstm

# DQN Agent
python -m nexus.ai.train_models --model dqn

# Ensemble Manager
python -m nexus.ai.train_models --model ensemble
```

## Loading Models

```python
from nexus.ai.lstm_predictor import MarketPredictor
from nexus.ai.deep_rl_agent import DQNAgent

# Load LSTM
predictor = MarketPredictor.load("models/lstm_predictor.pth")

# Load DQN
agent = DQNAgent.load("models/dqn_agent.pth")
```

## Model Versioning

Models are versioned by:
- Timestamp in filename (e.g., `lstm_predictor_20251007.pth`)
- Version metadata stored in model file
- An external artifact store or backup, rather than Git

## Best Practices

1. **Backup models** before retraining
2. **Validate on test set** before deployment
3. **Track performance metrics** in logs
4. **Use demo mode** to test new models
5. **Keep champion models** for comparison

## Notes

- Model files (`.pt`, `.pth`, `.joblib`, `.pkl`, `.h5`, `.onnx`) and generated model metadata are excluded from Git
- Store production models securely
- Retrain periodically as markets evolve
- Monitor model performance in production

---

For training instructions, see [docs/OVERVIEW.md](../docs/OVERVIEW.md)

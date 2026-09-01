# Data Directory

This directory stores market data, cache, and historical information for NEXUS.

## Structure

```
data/
├── cache/              # Cached market data (gitignored)
│   ├── EURUSD_5m.csv
│   ├── GBPUSD_15m.csv
│   └── ...
├── historical/         # Historical data downloads
├── features/           # Pre-computed features
└── README.md          # This file
```

## Cache Directory

The `cache/` folder stores:
- Downloaded market data from brokers
- Pre-processed OHLCV data
- Technical indicator calculations
- Feature engineering outputs

**Note**: Cache files are automatically managed and excluded from Git.

## Usage

### Data Provider
```python
from nexus.data.provider import DataProvider

provider = DataProvider(cache_dir="data/cache")

# Get data (auto-cached)
df = await provider.get_ohlcv(
    symbol="EURUSD",
    timeframe=5,
    limit=500,
    use_cache=True,  # Uses cache if available
)
```

### Feature Engineering
```python
from nexus.data.provider import FeatureEngineer

engineer = FeatureEngineer()
df_with_features = engineer.add_features(df)

# Save for reuse
provider.save_to_csv(df_with_features, "EURUSD", 5)
```

## Data Sources

1. **Quotex Broker**: Live market data
2. **CSV Files**: Historical data files
3. **Synthetic**: Generated data for testing

## Cache Management

### Clear Cache
```powershell
# Via run script
.\run.ps1  # Select option 7

# Or manually
Remove-Item data/cache/* -Recurse
```

### Cache Statistics
```python
provider = DataProvider()
stats = provider.get_cache_stats()
print(f"Cached items: {stats['cached_items']}")
```

## Data Format

Standard OHLCV format:
```csv
timestamp,open,high,low,close,volume
2025-01-01 00:00:00,1.0950,1.0955,1.0945,1.0952,1000
2025-01-01 00:05:00,1.0952,1.0960,1.0950,1.0958,1200
...
```

## Best Practices

1. **Regular cleanup**: Remove old cache files
2. **Backup important data**: Don't rely solely on cache
3. **Verify data quality**: Check for gaps and anomalies
4. **Use appropriate timeframes**: Match your strategy needs
5. **Monitor cache size**: Avoid filling disk space

## Data Privacy

- No sensitive personal data stored here
- Market data is public information
- Cache can be safely deleted anytime

---

**Note**: This directory is for data storage only. Large files are gitignored.
# AI Models Directory

This directory stores trained AI models for NEXUS trading system.

## Model Files

### LSTM Predictor
- `lstm_predictor.pth` - Main LSTM model with attention
- `lstm_predictor_best.pth` - Best validation checkpoint

### Deep RL Agent
- `dqn_agent.pth` - Deep Q-Network agent
- `dqn_agent_final.pth` - Final trained agent
- `dqn_agent_ep*.pth` - Episode checkpoints

### Ensemble
- `ensemble_state.json` - Ensemble model weights and performance

### Transformer (Future)
- `transformer_model.pth` - Transformer-based predictor

## Model Training

Train models using:
```powershell
.\train.ps1
```

Or programmatically:
```python
from nexus.ai.train_models import ModelTrainingPipeline
import asyncio

pipeline = ModelTrainingPipeline()
asyncio.run(pipeline.train_lstm(epochs=50))
asyncio.run(pipeline.train_dqn(episodes=1000))
```

## Model Loading

```python
from nexus.ai import MarketPredictor, DeepRLAgent
from pathlib import Path

# Load LSTM
predictor = MarketPredictor(model_path=Path("models/lstm_predictor.pth"))

# Load DQN
agent = DeepRLAgent(state_dim=20)
agent.load(Path("models/dqn_agent.pth"))
```

## Storage

**Important**: Model files (`.pth`, `.pt`) are excluded from Git to save space.
Train your own models or download pre-trained models separately.

## Metadata

Each model checkpoint includes:
- Model state dict (learned parameters)
- Optimizer state (for continuing training)
- Training history (loss, accuracy, etc.)
- Hyperparameters used

## Best Practices

1. **Version your models**: Use descriptive names with dates
2. **Keep best checkpoints**: Save top-k models by validation performance
3. **Document experiments**: Track hyperparameters and results
4. **Regular backups**: Store important models securely
5. **Clean old models**: Remove obsolete checkpoints periodically

## Model Performance Tracking

Track model performance in `ensemble_state.json`:
```json
{
  "weights": {
    "lstm": 0.45,
    "dqn": 0.35,
    "other": 0.20
  },
  "performance": {
    "lstm": {
      "accuracy": 0.67,
      "total_profit": 150.25
    }
  }
}
```

---

**Note**: Never commit large model files to Git. Use Git LFS or separate storage.


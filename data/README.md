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

**Note**: This directory is for local data storage. The database, downloaded market data, caches, reports, and other generated files are gitignored. Back up any important data separately before cleaning the workspace.

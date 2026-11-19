# Local data

NEXUS uses `data/` for downloaded or generated market data and local state. Keep account-specific or licensed broker data private. Runtime data is ignored by Git; this README is kept to describe the directory.

## Common data

| Path | Purpose |
|---|---|
| `cache/` | Cached OHLCV data used by `DataProvider` |
| `historical/` | Optional historical data files for local research |
| `features/` | Optional saved or precomputed feature data |
| `experience.sqlite3` | Local experience memory used by adaptive research components |

Not every workflow creates every directory. Additional scripts may store artifacts under `reports/` or `models/`.

## Using the data provider

`DataProvider` caches requests under `data/cache` by default. It supports broker, CSV, and synthetic sources; broker access requires a configured adapter.

```python
import asyncio

from nexus.data.provider import DataProvider


async def load_data():
    provider = DataProvider(cache_dir="data/cache")
    return await provider.get_ohlcv(
        symbol="EURUSD",
        timeframe=5,
        limit=500,
        source="synthetic",  # use "csv" or "broker" when configured
    )


df = asyncio.run(load_data())
```

`get_ohlcv` returns a pandas DataFrame or `None` when the selected source cannot provide data. The exact columns depend on the provider’s normalized output; inspect the returned frame before relying on a schema.

## Cleanup and privacy

Cache files can be deleted and fetched again. Do not delete local research or experience data unless you are comfortable losing it. The launcher’s clean option removes `data/cache/*`; review the code before using cleanup tools.

Logs, reports, browser sessions, and databases may contain sensitive material. Inspect and sanitize them before sharing or backing them up to a public location. See [Security](../SECURITY.md).

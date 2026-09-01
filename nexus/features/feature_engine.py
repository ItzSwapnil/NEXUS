"""Provider-isolated feature registry for broad technical-analysis coverage.

Providers are optional. Each provider contributes namespaced numeric features;
one broken or unavailable package cannot interrupt the live broker pipeline.
"""

from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    installed: bool
    available_functions: int


def _module_functions(module: Any) -> list[str]:
    try:
        return sorted(name for name in dir(module) if not name.startswith("_"))
    except Exception:
        return []


def _configure_dotnet() -> bool:
    """Make a user-local .NET install visible to pythonnet/Stock Indicators."""
    dotnet = shutil.which("dotnet")
    if dotnet:
        root = str(Path(dotnet).resolve().parent)
    else:
        local_root = Path.home() / ".dotnet"
        dotnet_path = local_root / "dotnet"
        if not dotnet_path.is_file():
            return False
        root = str(local_root)
        os.environ["PATH"] = f"{root}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("DOTNET_ROOT", root)
    return True


def _stock_indicators_module() -> Any | None:
    # Importing this package without a CoreCLR runtime logs an error on every
    # import. Check first so live analysis stays quiet and fast on bare hosts.
    if not _configure_dotnet():
        return None
    try:
        return importlib.import_module("stock_indicators.indicators")
    except Exception:
        return None


def get_feature_provider_catalog() -> Dict[str, Any]:
    """Return installed providers and their discoverable function counts."""
    providers: list[ProviderInfo] = []
    try:
        talib = importlib.import_module("talib")
        providers.append(ProviderInfo("TA-Lib", True, len(talib.get_functions())))
    except Exception:
        providers.append(ProviderInfo("TA-Lib", False, 0))
    try:
        stock = _stock_indicators_module()
        providers.append(ProviderInfo("Stock Indicators", stock is not None, len(_module_functions(stock))))
    except Exception:
        providers.append(ProviderInfo("Stock Indicators", False, 0))
    try:
        pandas_ta = importlib.import_module("pandas_ta_classic")
        providers.append(ProviderInfo("pandas-ta-classic", True, len(_module_functions(pandas_ta))))
    except Exception:
        providers.append(ProviderInfo("pandas-ta-classic", False, 0))
    return {
        "providers": [provider.__dict__ for provider in providers],
        "installed_count": sum(provider.installed for provider in providers),
        "function_count": sum(provider.available_functions for provider in providers),
    }


def _append_numeric(features: dict[str, np.ndarray], length: int, prefix: str, values: Any) -> None:
    if isinstance(values, pd.DataFrame):
        for column in values.columns:
            _append_numeric(features, length, f"{prefix}_{column}", values[column])
        return
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return
    if array.ndim == 1 and len(array) == length:
        features[prefix] = array


def _add_talib(result: pd.DataFrame, features: dict[str, np.ndarray]) -> None:
    try:
        talib = importlib.import_module("talib")
        abstract = importlib.import_module("talib.abstract")
        for name in talib.get_functions():
            try:
                fn = abstract.Function(name)
                # The abstract API maps standard OHLCV dataframe columns to
                # each function's required inputs (price, real, etc.).
                # Passing a hand-built dict incorrectly skipped nearly every
                # TA-Lib function because its input names are not OHLCV keys.
                output = fn(result)
                if isinstance(output, dict):
                    for suffix, values in output.items():
                        _append_numeric(features, len(result), f"talib_{name.lower()}_{str(suffix).lower()}", values)
                else:
                    _append_numeric(features, len(result), f"talib_{name.lower()}", output)
            except Exception:
                continue
    except Exception:
        return


def _add_pandas_ta(result: pd.DataFrame, features: dict[str, np.ndarray]) -> None:
    try:
        importlib.import_module("pandas_ta_classic")
        # The accessor is registered on DataFrame by the package import.
        if hasattr(result, "ta"):
            # VWAP and several time-based indicators require a datetime index.
            # Use a deterministic synthetic timeline only when the broker
            # frame does not already provide one.  Run in-process: spawning a
            # pool per live candle request is both slow and unsafe in servers.
            if not isinstance(result.index, pd.DatetimeIndex):
                result.index = pd.date_range("2000-01-01", periods=len(result), freq="min")
            # `cores` is exposed as a read-only property in newer
            # pandas-ta-classic releases; assigning it silently has no effect.
            # Set the backing value explicitly so strategy() cannot spawn a
            # process pool inside a web request (or from a stdin-launched job).
            ta = result.ta
            ta._cores = 0
            original_columns = set(result.columns)
            # pandas-ta-classic appends results to the frame in place and
            # returns None; older releases returned a frame. Support both.
            generated = ta.strategy("all", timed=False)
            generated_frame = generated if isinstance(generated, pd.DataFrame) else result
            new_columns = [column for column in generated_frame.columns if column not in original_columns]
            for column in new_columns:
                values = generated_frame[column]
                _append_numeric(features, len(result), f"pandas_ta_{str(column).lower()}", values)
            # Keep the provider columns namespaced and remove the unprefixed
            # columns added by the in-place API to avoid duplicate model input.
            for column in new_columns:
                if column in result.columns:
                    result.drop(columns=[column], inplace=True)
    except Exception:
        return


def _add_stock_indicators(result: pd.DataFrame, features: dict[str, np.ndarray]) -> None:
    """Use the common Stock Indicators API when installed.

    Its results are object records rather than arrays, so this adapter keeps
    the standard OHLCV-compatible indicators and safely skips unavailable ones.
    """
    try:
        module = _stock_indicators_module()
        if module is None or len(result) < 130:
            return
        quotes_type = importlib.import_module("stock_indicators").Quote
        dates = result.index
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.date_range(datetime(2000, 1, 1), periods=len(result), freq="min")
        quotes = [
            quotes_type(
                date.to_pydatetime(), float(o), float(h), float(low_value), float(c), float(v)
            )
            for date, o, h, low_value, c, v in zip(
                dates, result.open, result.high, result.low, result.close, result.volume,
                strict=True,
            )
        ]
        calls: dict[str, tuple[str, tuple[Any, ...]]] = {
            "get_sma": ("sma", (14,)), "get_ema": ("ema", (14,)),
            "get_rsi": ("rsi", (14,)), "get_atr": ("atr", (14,)),
            "get_adx": ("adx", (14,)), "get_cci": ("cci", (20,)),
            "get_macd": ("macd", (12, 26, 9)),
        }
        for function_name, (prefix, args) in calls.items():
            function = getattr(module, function_name, None)
            if function is None:
                continue
            records = function(quotes, *args)
            fields = {
                field
                for field in dir(records[0])
                if not field.startswith("_") and field not in {"date"}
            } if records else set()
            for field in fields:
                _append_numeric(features, len(result), f"stock_{prefix}_{field}", [getattr(record, field, np.nan) for record in records])
    except Exception:
        return


def add_external_features(data: pd.DataFrame) -> pd.DataFrame:
    """Run all installed providers and return a merged feature frame."""
    result = data.copy()
    features: dict[str, np.ndarray] = {}
    _add_talib(result, features)
    _add_stock_indicators(result, features)
    _add_pandas_ta(result, features)
    if features:
        result = pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)
    return result


class FeatureRegistry:
    """Facade used by callers that need provider metadata and features."""

    @staticmethod
    def catalog() -> Dict[str, Any]:
        return get_feature_provider_catalog()

    @staticmethod
    def compute(data: pd.DataFrame) -> pd.DataFrame:
        return add_external_features(data)

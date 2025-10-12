"""Financial performance metrics compatible with Python 3.13."""

import numpy as np
import pandas as pd
from typing import Union


def _as_series(returns: Union[pd.Series, np.ndarray]) -> pd.Series:
    if isinstance(returns, pd.Series):
        return pd.to_numeric(returns.dropna(), errors="coerce").dropna()
    # numpy array or other sequence
    return pd.Series(returns, dtype=float).dropna()


def annual_return(returns: Union[pd.Series, np.ndarray], period: str = 'daily') -> float:
    """Calculate annualized return from a series of returns."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)
    total_return = float((1.0 + s).prod())
    years = len(s) / periods
    if years == 0:
        return 0.0
    annual_ret = float(total_return ** (1.0 / years) - 1.0)
    return float(annual_ret)


def sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, period: str = 'daily') -> float:
    """Calculate Sharpe ratio."""
    s = _as_series(returns)
    if len(s) == 0 or float(s.std()) == 0.0:
        return 0.0
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)
    rf_period = risk_free_rate / periods
    excess_returns = s - rf_period
    return float(excess_returns.mean() / s.std() * np.sqrt(periods))


def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate maximum drawdown."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    cumulative = (1.0 + s).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: Union[pd.Series, np.ndarray], period: str = 'daily') -> float:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    annual_ret = annual_return(returns, period)
    max_dd = abs(max_drawdown(returns))
    if max_dd == 0:
        return float(np.inf) if annual_ret > 0 else 0.0
    return float(annual_ret / max_dd)


def sortino_ratio(returns: Union[pd.Series, np.ndarray], target_return: float = 0.0, period: str = 'daily') -> float:
    """Calculate Sortino ratio (focuses on downside volatility)."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    downside_returns = s[s < target_return]
    if len(downside_returns) == 0:
        return float(np.inf) if float(s.mean()) > target_return else 0.0
    downside_std = float(downside_returns.std())
    if downside_std == 0:
        return float(np.inf) if float(s.mean()) > target_return else 0.0
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)
    excess_return = float(s.mean()) - target_return
    return float(excess_return / downside_std * np.sqrt(periods))


def value_at_risk(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    return float(np.percentile(s.to_numpy(dtype=float), confidence_level * 100.0))


def conditional_value_at_risk(returns: Union[pd.Series, np.ndarray], confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    var = value_at_risk(s, confidence_level)
    tail = s[s <= var]
    return float(tail.mean()) if len(tail) > 0 else 0.0


def omega_ratio(returns: Union[pd.Series, np.ndarray], target_return: float = 0.0) -> float:
    """Calculate Omega ratio."""
    s = _as_series(returns)
    if len(s) == 0:
        return 1.0
    gains = s[s > target_return] - target_return
    losses = target_return - s[s <= target_return]
    total_gains = float(gains.sum()) if len(gains) > 0 else 0.0
    total_losses = float(losses.sum()) if len(losses) > 0 else 0.0
    if total_losses == 0.0:
        return float(np.inf) if total_gains > 0.0 else 1.0
    return float(total_gains / total_losses)


def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate win rate (percentage of positive returns)."""
    s = _as_series(returns)
    if len(s) == 0:
        return 0.0
    return float((s > 0).sum() / len(s))


def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """Calculate profit factor (total gains / total losses)."""
    s = _as_series(returns)
    if len(s) == 0:
        return 1.0
    gains = float(s[s > 0].sum())
    losses = abs(float(s[s < 0].sum()))
    if losses == 0.0:
        return float(np.inf) if gains > 0.0 else 1.0
    return float(gains / losses)


def calculate_performance_metrics(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, period: str = 'daily') -> dict:
    """Calculate comprehensive performance metrics."""
    s = _as_series(returns)
    if len(s) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'omega_ratio': 1.0
        }

    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)

    return {
        'total_return': float((1.0 + s).prod() - 1.0),
        'annual_return': annual_return(s, period),
        'volatility': float(s.std() * np.sqrt(periods)),
        'sharpe_ratio': sharpe_ratio(s, risk_free_rate, period),
        'sortino_ratio': sortino_ratio(s, 0.0, period),
        'max_drawdown': max_drawdown(s),
        'calmar_ratio': calmar_ratio(s, period),
        'win_rate': win_rate(s),
        'profit_factor': profit_factor(s),
        'var_95': value_at_risk(s, 0.05),
        'cvar_95': conditional_value_at_risk(s, 0.05),
        'omega_ratio': omega_ratio(s, 0.0)
    }

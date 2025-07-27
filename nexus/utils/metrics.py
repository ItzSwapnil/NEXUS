"""
Financial Performance Metrics - Replacement for empyrical

This module provides financial performance calculations that are compatible
with Python 3.13, replacing the deprecated empyrical package functionality.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

def annual_return(returns: Union[pd.Series, np.ndarray], period: str = 'daily') -> float:
    """
    Calculate annualized return from a series of returns.

    Args:
        returns: Series of returns
        period: Frequency of returns ('daily', 'weekly', 'monthly')

    Returns:
        Annualized return as a decimal
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    # Periods per year
    periods_per_year = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12
    }

    periods = periods_per_year.get(period, 252)

    # Calculate compound annual growth rate
    total_return = (1 + returns).prod()
    years = len(returns) / periods

    if years == 0:
        return 0.0

    annual_ret = total_return ** (1 / years) - 1
    return annual_ret

def sharpe_ratio(returns: Union[pd.Series, np.ndarray],
                risk_free_rate: float = 0.0,
                period: str = 'daily') -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        period: Frequency of returns

    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Convert risk-free rate to same period
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)
    rf_period = risk_free_rate / periods

    excess_returns = returns - rf_period
    return excess_returns.mean() / returns.std() * np.sqrt(periods)

def max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Series of returns

    Returns:
        Maximum drawdown as a negative decimal
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cumulative.expanding().max()

    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()

def calmar_ratio(returns: Union[pd.Series, np.ndarray], period: str = 'daily') -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Series of returns
        period: Frequency of returns

    Returns:
        Calmar ratio
    """
    annual_ret = annual_return(returns, period)
    max_dd = abs(max_drawdown(returns))

    if max_dd == 0:
        return np.inf if annual_ret > 0 else 0.0

    return annual_ret / max_dd

def sortino_ratio(returns: Union[pd.Series, np.ndarray],
                 target_return: float = 0.0,
                 period: str = 'daily') -> float:
    """
    Calculate Sortino ratio (focuses on downside volatility).

    Args:
        returns: Series of returns
        target_return: Target return threshold
        period: Frequency of returns

    Returns:
        Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    # Calculate downside returns
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return np.inf if returns.mean() > target_return else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return np.inf if returns.mean() > target_return else 0.0

    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}
    periods = periods_per_year.get(period, 252)

    excess_return = returns.mean() - target_return
    return excess_return / downside_std * np.sqrt(periods)

def value_at_risk(returns: Union[pd.Series, np.ndarray],
                 confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)

    Returns:
        VaR as a negative value
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    return np.percentile(returns, confidence_level * 100)

def conditional_value_at_risk(returns: Union[pd.Series, np.ndarray],
                            confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

    Args:
        returns: Series of returns
        confidence_level: Confidence level

    Returns:
        CVaR as a negative value
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    var = value_at_risk(returns, confidence_level)
    return returns[returns <= var].mean()

def omega_ratio(returns: Union[pd.Series, np.ndarray],
               target_return: float = 0.0) -> float:
    """
    Calculate Omega ratio.

    Args:
        returns: Series of returns
        target_return: Target return threshold

    Returns:
        Omega ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 1.0

    gains = returns[returns > target_return] - target_return
    losses = target_return - returns[returns <= target_return]

    total_gains = gains.sum() if len(gains) > 0 else 0
    total_losses = losses.sum() if len(losses) > 0 else 0

    if total_losses == 0:
        return np.inf if total_gains > 0 else 1.0

    return total_gains / total_losses

def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Series of returns

    Returns:
        Win rate as a decimal (0-1)
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 0.0

    return (returns > 0).sum() / len(returns)

def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor (total gains / total losses).

    Args:
        returns: Series of returns

    Returns:
        Profit factor
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
        return 1.0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if gains > 0 else 1.0

    return gains / losses

def calculate_performance_metrics(returns: Union[pd.Series, np.ndarray],
                                risk_free_rate: float = 0.0,
                                period: str = 'daily') -> dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        period: Frequency of returns

    Returns:
        Dictionary of performance metrics
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    if len(returns) == 0:
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
        'total_return': (1 + returns).prod() - 1,
        'annual_return': annual_return(returns, period),
        'volatility': returns.std() * np.sqrt(periods),
        'sharpe_ratio': sharpe_ratio(returns, risk_free_rate, period),
        'sortino_ratio': sortino_ratio(returns, 0.0, period),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns, period),
        'win_rate': win_rate(returns),
        'profit_factor': profit_factor(returns),
        'var_95': value_at_risk(returns, 0.05),
        'cvar_95': conditional_value_at_risk(returns, 0.05),
        'omega_ratio': omega_ratio(returns, 0.0)
    }

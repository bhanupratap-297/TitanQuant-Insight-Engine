"""
Utility Functions
utils.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_currency(value, currency='$'):
    """Format number as currency"""
    if value >= 1e12:
        return f"{currency}{value/1e12:.2f}T"
    elif value >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif value >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    else:
        return f"{currency}{value:.2f}"


def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100


def get_sentiment_label(score):
    """Get sentiment label based on score"""
    if score > 0.5:
        return "🟢 STRONGLY BULLISH"
    elif score > 0.1:
        return "🟢 BULLISH"
    elif score > -0.1:
        return "🟡 NEUTRAL"
    elif score > -0.5:
        return "🔴 BEARISH"
    else:
        return "🔴 STRONGLY BEARISH"


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative_returns = (1 + prices.pct_change()).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min() * 100


def normalize_data(data, method='minmax'):
    """Normalize data"""
    if method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'zscore':
        return (data - data.mean()) / data.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

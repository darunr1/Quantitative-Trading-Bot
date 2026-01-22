"""Multi-asset portfolio management with risk parity position sizing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.trading_bot import StrategyConfig, compute_strategy_returns


@dataclass
class PortfolioConfig:
    """Configuration for multi-asset portfolio."""

    symbols: List[str]
    strategy_config: StrategyConfig
    target_portfolio_vol: float = 0.15
    rebalance_frequency: str = "D"  # 'D' for daily, 'W' for weekly, etc.
    risk_parity_method: str = "equal_risk"  # 'equal_risk' or 'inverse_vol'


def compute_risk_parity_weights(
    returns: pd.DataFrame,
    method: str = "equal_risk",
) -> pd.Series:
    """Compute risk parity weights for assets.

    Args:
        returns: DataFrame with asset returns (columns = symbols)
        method: 'equal_risk' or 'inverse_vol'

    Returns:
        Series with weights for each asset
    """
    if method == "inverse_vol":
        # Inverse volatility weighting
        volatilities = returns.std()
        inv_vol = 1.0 / volatilities.replace(0.0, np.inf)
        weights = inv_vol / inv_vol.sum()
    else:  # equal_risk
        # Equal risk contribution (simplified)
        # Each asset contributes equally to portfolio risk
        volatilities = returns.std()
        inv_vol = 1.0 / volatilities.replace(0.0, np.inf)
        # Normalize so sum of weights = 1
        weights = inv_vol / inv_vol.sum()

    return weights.fillna(0.0)


def compute_portfolio_returns(
    prices: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.Series:
    """Compute portfolio returns for multiple assets.

    Args:
        prices: DataFrame with prices (columns = symbols, index = dates)
        config: Portfolio configuration

    Returns:
        Series with portfolio returns
    """
    # Compute individual strategy returns for each asset
    asset_returns = {}
    for symbol in config.symbols:
        if symbol not in prices.columns:
            continue
        asset_prices = prices[symbol].dropna()
        if len(asset_prices) == 0:
            continue
        asset_returns[symbol] = compute_strategy_returns(
            asset_prices, config.strategy_config
        )

    if not asset_returns:
        return pd.Series(dtype=float)

    # Align all returns to common index
    returns_df = pd.DataFrame(asset_returns)
    returns_df = returns_df.fillna(0.0)

    # Compute risk parity weights
    # Use rolling window for dynamic weights
    lookback = config.strategy_config.vol_lookback
    portfolio_returns = pd.Series(index=returns_df.index, dtype=float)
    historical_portfolio_returns = []

    for i in range(len(returns_df)):
        if i < lookback:
            # Use equal weights initially
            weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
        else:
            # Compute weights based on recent volatility
            recent_returns = returns_df.iloc[max(0, i - lookback):i]
            weights = compute_risk_parity_weights(
                recent_returns, config.risk_parity_method
            )

        # Apply portfolio volatility targeting based on historical portfolio returns
        if i >= lookback and len(historical_portfolio_returns) >= lookback:
            portfolio_vol = np.std(historical_portfolio_returns[-lookback:]) * np.sqrt(252)
            if portfolio_vol > 0:
                vol_scale = config.target_portfolio_vol / portfolio_vol
                weights = weights * min(vol_scale, config.strategy_config.max_leverage)

        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Compute weighted portfolio return
        portfolio_return = (returns_df.iloc[i] * weights).sum()
        portfolio_returns.iloc[i] = portfolio_return
        historical_portfolio_returns.append(portfolio_return)

    return portfolio_returns


def run_portfolio_backtest(
    prices: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.Series:
    """Run backtest for multi-asset portfolio.

    Args:
        prices: DataFrame with prices (columns = symbols)
        config: Portfolio configuration

    Returns:
        Series with portfolio returns
    """
    return compute_portfolio_returns(prices, config)

"""Walk-forward validation framework for robust backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from src.trading_bot import BacktestResult, StrategyConfig, compute_strategy_returns, calculate_performance


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: Optional[BacktestResult] = None
    test_result: Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""

    periods: List[WalkForwardPeriod]
    summary_stats: pd.DataFrame

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = ["Walk-Forward Analysis Summary", "=" * 50]
        
        test_returns = [p.test_result.annual_return for p in self.periods if p.test_result]
        test_sharpes = [p.test_result.sharpe_ratio for p in self.periods if p.test_result]
        test_drawdowns = [p.test_result.max_drawdown for p in self.periods if p.test_result]

        if test_returns:
            lines.append(f"\nTest Period Statistics:")
            lines.append(f"  Average Annual Return: {sum(test_returns) / len(test_returns):.2%}")
            lines.append(f"  Average Sharpe Ratio: {sum(test_sharpes) / len(test_sharpes):.2f}")
            lines.append(f"  Average Max Drawdown: {sum(test_drawdowns) / len(test_drawdowns):.2%}")
            lines.append(f"  Number of Periods: {len(test_returns)}")
            lines.append(f"  Positive Periods: {sum(1 for r in test_returns if r > 0)}/{len(test_returns)}")

        return "\n".join(lines)


def run_walk_forward(
    prices: pd.Series,
    config: StrategyConfig,
    train_window_days: int = 252,  # 1 year
    test_window_days: int = 63,    # ~3 months
    step_days: int = 21,            # ~1 month step
    min_train_days: int = 126,      # Minimum training data
) -> WalkForwardResult:
    """Run walk-forward validation.

    Args:
        prices: Price series with datetime index
        config: Strategy configuration
        train_window_days: Training window size in days
        test_window_days: Testing window size in days
        step_days: Step size between periods
        min_train_days: Minimum days required for training

    Returns:
        WalkForwardResult with all periods and summary
    """
    periods = []
    start_date = prices.index[0]
    end_date = prices.index[-1]

    current_date = start_date + timedelta(days=min_train_days)

    while current_date + timedelta(days=test_window_days) <= end_date:
        # Define training period
        train_end = current_date
        train_start = train_end - timedelta(days=train_window_days)

        # Define test period
        test_start = current_date
        test_end = test_start + timedelta(days=test_window_days)

        # Ensure we have enough data
        if train_start < start_date:
            current_date += timedelta(days=step_days)
            continue

        period = WalkForwardPeriod(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        # Run training backtest (optional - can be used for parameter optimization)
        train_prices = prices[(prices.index >= train_start) & (prices.index < train_end)]
        if len(train_prices) > 0:
            train_returns = compute_strategy_returns(train_prices, config)
            period.train_result = calculate_performance(train_returns)

        # Run test backtest
        test_prices = prices[(prices.index >= test_start) & (prices.index < test_end)]
        if len(test_prices) > 0:
            test_returns = compute_strategy_returns(test_prices, config)
            period.test_result = calculate_performance(test_returns)

        periods.append(period)
        current_date += timedelta(days=step_days)

    # Create summary statistics
    summary_data = []
    for i, period in enumerate(periods):
        if period.test_result:
            summary_data.append({
                "period": i + 1,
                "test_start": period.test_start,
                "test_end": period.test_end,
                "annual_return": period.test_result.annual_return,
                "sharpe_ratio": period.test_result.sharpe_ratio,
                "max_drawdown": period.test_result.max_drawdown,
                "annual_volatility": period.test_result.annual_volatility,
            })
    
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()

    return WalkForwardResult(periods=periods, summary_stats=summary_df)

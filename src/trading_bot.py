"""Quantitative trading bot with volatility-targeted trend strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    fast_ema_span: int = 12
    slow_ema_span: int = 48
    vol_lookback: int = 20
    target_vol: float = 0.15
    max_leverage: float = 2.0
    transaction_cost_bps: float = 1.0


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float


def load_price_data(csv_path: str, price_column: str = "close") -> pd.DataFrame:
    """Load OHLCV data from CSV and return cleaned dataframe.

    Expected columns: date, open, high, low, close, volume (case-insensitive).
    """
    data = pd.read_csv(csv_path)
    data.columns = [col.lower() for col in data.columns]
    if "date" not in data.columns:
        raise ValueError("CSV must contain a 'date' column.")
    if price_column.lower() not in data.columns:
        raise ValueError(f"CSV must contain '{price_column}' column.")
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.sort_values("date").set_index("date")
    return data


def compute_strategy_returns(
    prices: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    """Compute daily strategy returns with volatility targeting and costs."""
    fast_ema = prices.ewm(span=config.fast_ema_span, adjust=False).mean()
    slow_ema = prices.ewm(span=config.slow_ema_span, adjust=False).mean()
    trend_signal = (fast_ema > slow_ema).astype(float)

    daily_returns = prices.pct_change().fillna(0.0)
    rolling_vol = daily_returns.rolling(config.vol_lookback).std().replace(0.0, pd.NA)
    vol_target = config.target_vol / (rolling_vol * (252**0.5))
    position_size = vol_target.clip(upper=config.max_leverage).fillna(0.0)

    raw_position = trend_signal * position_size
    position = raw_position.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)
    transaction_cost = turnover * (config.transaction_cost_bps / 10_000)

    strategy_returns = position * daily_returns - transaction_cost
    return strategy_returns


def calculate_performance(strategy_returns: pd.Series) -> BacktestResult:
    """Calculate performance metrics from strategy returns."""
    equity_curve = (1 + strategy_returns).cumprod()
    daily_vol = strategy_returns.std()
    sharpe_ratio = (
        (strategy_returns.mean() / daily_vol) * (252**0.5) if daily_vol else 0.0
    )
    annual_return = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    annual_volatility = daily_vol * (252**0.5)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=strategy_returns,
        sharpe_ratio=sharpe_ratio,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
    )


def run_backtest(
    csv_path: str,
    price_column: str = "close",
    config: Optional[StrategyConfig] = None,
) -> BacktestResult:
    """Run backtest using the configured strategy."""
    config = config or StrategyConfig()
    data = load_price_data(csv_path, price_column=price_column)
    prices = data[price_column.lower()].astype(float)
    strategy_returns = compute_strategy_returns(prices, config)
    return calculate_performance(strategy_returns)


def format_report(result: BacktestResult) -> str:
    """Create a human-readable performance summary."""
    return "\n".join(
        [
            "Performance Summary",
            "-------------------",
            f"Annual Return: {result.annual_return:.2%}",
            f"Annual Volatility: {result.annual_volatility:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
        ]
    )


def main() -> None:
    """CLI entrypoint."""
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Quantitative trading bot - backtest, walk-forward, live trading, and portfolios"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["backtest", "walk-forward", "live", "portfolio"],
        default="backtest",
        help="Trading mode (default: backtest)",
    )

    # Data source arguments
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="?",
        help="Path to CSV file with price data (for backtest/walk-forward)",
    )
    parser.add_argument(
        "--price-column",
        dest="price_column",
        default="close",
        help="Column for price data (default: close)",
    )
    parser.add_argument(
        "--data-source",
        choices=["csv", "alpaca", "yahoo"],
        default="csv",
        help="Data source type (default: csv)",
    )

    # Live trading arguments
    parser.add_argument(
        "--symbol",
        type=str,
        help="Trading symbol (required for live/portfolio modes)",
    )
    parser.add_argument(
        "--alpaca-key",
        type=str,
        help="Alpaca API key (for live trading with Alpaca)",
    )
    parser.add_argument(
        "--alpaca-secret",
        type=str,
        help="Alpaca API secret (for live trading with Alpaca)",
    )

    # Portfolio arguments
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="List of symbols for portfolio mode",
    )

    # Strategy parameters
    parser.add_argument("--fast-ema", type=int, default=12, help="Fast EMA span")
    parser.add_argument("--slow-ema", type=int, default=48, help="Slow EMA span")
    parser.add_argument("--vol-lookback", type=int, default=20, help="Volatility window")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target volatility")
    parser.add_argument("--max-leverage", type=float, default=2.0, help="Max leverage")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in basis points",
    )

    # Walk-forward arguments
    parser.add_argument(
        "--train-window",
        type=int,
        default=252,
        help="Training window in days (walk-forward mode)",
    )
    parser.add_argument(
        "--test-window",
        type=int,
        default=63,
        help="Test window in days (walk-forward mode)",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=21,
        help="Step size in days (walk-forward mode)",
    )

    args = parser.parse_args()

    config = StrategyConfig(
        fast_ema_span=args.fast_ema,
        slow_ema_span=args.slow_ema,
        vol_lookback=args.vol_lookback,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    # Route to appropriate mode
    if args.mode == "backtest":
        if not args.csv_path:
            parser.error("csv_path is required for backtest mode")
        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        else:
            csv_path = csv_path.resolve()
        if not csv_path.exists():
            parser.error(f"CSV file not found: {csv_path}")
        result = run_backtest(str(csv_path), args.price_column, config)
        print(format_report(result))

    elif args.mode == "walk-forward":
        if not args.csv_path:
            parser.error("csv_path is required for walk-forward mode")
        from src.walk_forward import run_walk_forward

        csv_path = Path(args.csv_path)
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()
        if not csv_path.exists():
            parser.error(f"CSV file not found: {csv_path}")

        data = load_price_data(str(csv_path), args.price_column)
        prices = data[args.price_column.lower()].astype(float)

        wf_result = run_walk_forward(
            prices,
            config,
            train_window_days=args.train_window,
            test_window_days=args.test_window,
            step_days=args.step_days,
        )
        print(wf_result.get_summary())

    elif args.mode == "live":
        if not args.symbol:
            parser.error("--symbol is required for live trading mode")

        from src.data_source import AlpacaDataSource, CSVDataSource, YahooFinanceDataSource
        from src.broker import AlpacaBroker, PaperTradingBroker
        from src.live_trading import LiveTrader

        # Setup data source
        if args.data_source == "alpaca":
            if not args.alpaca_key or not args.alpaca_secret:
                parser.error("--alpaca-key and --alpaca-secret required for Alpaca data source")
            data_source = AlpacaDataSource(args.alpaca_key, args.alpaca_secret)
            broker = AlpacaBroker(args.alpaca_key, args.alpaca_secret, paper=True)
        elif args.data_source == "yahoo":
            data_source = YahooFinanceDataSource()
            broker = PaperTradingBroker()
        else:
            parser.error("CSV data source not supported for live trading")

        trader = LiveTrader(data_source, broker, args.symbol, config)
        trader.run(check_interval_seconds=3600)

    elif args.mode == "portfolio":
        if not args.symbols:
            parser.error("--symbols is required for portfolio mode")

        from src.portfolio import PortfolioConfig, run_portfolio_backtest
        from src.data_source import CSVDataSource, YahooFinanceDataSource
        from datetime import datetime, timedelta

        # Load data for all symbols
        if args.data_source == "yahoo":
            data_source = YahooFinanceDataSource()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * 2)  # 2 years

            prices_dict = {}
            for symbol in args.symbols:
                try:
                    data = data_source.get_historical_data(
                        symbol, start_date, end_date, interval="1d"
                    )
                    prices_dict[symbol] = data.set_index("date")["close"]
                except Exception as e:
                    print(f"Warning: Could not load data for {symbol}: {e}")

            if not prices_dict:
                parser.error("No valid price data loaded")

            prices_df = pd.DataFrame(prices_dict)
            prices_df = prices_df.ffill().dropna()

        elif args.csv_path:
            # Multi-asset CSV (each column is a symbol)
            data = pd.read_csv(args.csv_path)
            data.columns = [col.lower() for col in data.columns]
            if "date" not in data.columns:
                parser.error("CSV must contain a 'date' column")
            data["date"] = pd.to_datetime(data["date"], utc=True)
            data = data.set_index("date")
            prices_df = data[args.symbols].astype(float)
        else:
            parser.error("Either --symbols with yahoo data source or csv_path required")

        portfolio_config = PortfolioConfig(
            symbols=args.symbols,
            strategy_config=config,
            target_portfolio_vol=args.target_vol,
        )

        portfolio_returns = run_portfolio_backtest(prices_df, portfolio_config)
        result = calculate_performance(portfolio_returns)
        print(format_report(result))


if __name__ == "__main__":
    main()

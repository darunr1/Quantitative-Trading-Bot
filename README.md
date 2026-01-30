# Quantitative Trading Research Framework

A comprehensive quantitative trading bot with backtesting, walk-forward validation, live trading, and multi-asset portfolio support. Implements a volatility-targeted trend strategy (EMA crossover with position sizing) with risk controls and performance metrics.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **⚠️ Disclaimer**: This project is for research and educational use only. Past performance does not guarantee future results. You are responsible for ensuring compliance with any applicable regulations and broker requirements. Trading involves risk of loss.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Strategy Parameters](#strategy-parameters)
- [Architecture](#architecture)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Strategy
- ✅ **EMA-based trend signal** - Fast/slow EMA crossover for trend detection
- ✅ **Volatility targeting** - Dynamic position sizing based on realized volatility
- ✅ **Leverage caps** - Configurable maximum leverage limits
- ✅ **Transaction cost modeling** - Realistic cost accounting
- ✅ **Performance metrics** - Annual return, Sharpe ratio, max drawdown

### Advanced Features
- 🚀 **Live Trading** - Real-time data feeds and order execution via broker APIs
- 📊 **Walk-Forward Validation** - Robust out-of-sample testing with rolling windows
- 📈 **Multi-Asset Portfolios** - Trade multiple assets simultaneously
- ⚖️ **Risk Parity Sizing** - Equal risk contribution position sizing across assets

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Quantitative-Trading-Bot.git
cd Quantitative-Trading-Bot
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

### Optional Dependencies

For **live trading with Alpaca**:
```bash
pip install alpaca-py
```

For **Yahoo Finance data source**:
```bash
pip install yfinance
```

Or install all optional dependencies:
```bash
pip install alpaca-py yfinance
```

## Quick Start

### 1. Prepare Your Data

Create a CSV file with at least these columns:
- `date` - Date in parseable format (e.g., YYYY-MM-DD)
- `close` - Closing price (or specify another column via `--price-column`)

Example CSV format:
```csv
date,open,high,low,close,volume
2020-01-02,100.0,101.5,99.2,100.8,1000000
2020-01-03,100.8,102.1,100.0,101.2,980000
```

### 2. Run Your First Backtest

```bash
python -m src.trading_bot path/to/data.csv
```

Expected output:
```
Performance Summary
-------------------
Annual Return: 5.47%
Annual Volatility: 1.54%
Sharpe Ratio: 3.46
Max Drawdown: 0.00%
```

## Usage Guide

### Mode 1: Backtest (Default)

Run a simple backtest on historical data:

```bash
python -m src.trading_bot path/to/data.csv
```

With custom parameters:
```bash
python -m src.trading_bot data.csv \
  --mode backtest \
  --fast-ema 10 \
  --slow-ema 50 \
  --vol-lookback 20 \
  --target-vol 0.12 \
  --max-leverage 1.5 \
  --transaction-cost-bps 2.0
```

### Mode 2: Walk-Forward Validation

Test strategy robustness with rolling train/test windows:

```bash
python -m src.trading_bot data.csv \
  --mode walk-forward \
  --train-window 252 \
  --test-window 63 \
  --step-days 21
```

**Parameters:**
- `--train-window`: Training window in days (default: 252 = 1 year)
- `--test-window`: Testing window in days (default: 63 = ~3 months)
- `--step-days`: Step size between periods (default: 21 = ~1 month)

**What it does:**
- Creates multiple train/test periods
- Trains on historical data, tests on future data
- Provides average performance across all test periods
- Helps identify overfitting

**Example output:**
```
Walk-Forward Analysis Summary
==================================================

Test Period Statistics:
  Average Annual Return: 8.23%
  Average Sharpe Ratio: 2.15
  Average Max Drawdown: -5.42%
  Number of Periods: 12
  Positive Periods: 10/12
```

### Mode 3: Live Trading

Execute trades in real-time (paper trading recommended first).

#### Option A: Yahoo Finance (Simulated - No API Key Required)

```bash
python -m src.trading_bot \
  --mode live \
  --symbol AAPL \
  --data-source yahoo
```

#### Option B: Alpaca (Paper Trading)

1. Get your API credentials from [Alpaca](https://alpaca.markets/)
2. Run with your credentials:

```bash
python -m src.trading_bot \
  --mode live \
  --symbol AAPL \
  --data-source alpaca \
  --alpaca-key YOUR_API_KEY \
  --alpaca-secret YOUR_API_SECRET
```

**What happens:**
- Bot fetches real-time price data every hour (configurable)
- Computes trading signals based on strategy
- Executes orders automatically when signals change
- Monitors positions and account status
- Press `Ctrl+C` to stop

### Mode 4: Multi-Asset Portfolio

Trade multiple assets with risk parity position sizing.

#### Option A: Yahoo Finance

```bash
python -m src.trading_bot \
  --mode portfolio \
  --symbols AAPL MSFT GOOGL \
  --data-source yahoo \
  --target-vol 0.15
```

#### Option B: CSV File

Create a CSV with columns for each symbol:
```csv
date,AAPL,MSFT,GOOGL
2020-01-02,100.0,200.0,150.0
2020-01-03,101.0,201.0,151.0
```

Then run:
```bash
python -m src.trading_bot portfolio_data.csv \
  --mode portfolio \
  --symbols AAPL MSFT GOOGL \
  --data-source csv
```

**Portfolio features:**
- Computes individual strategy signals for each asset
- Applies risk parity weighting (equal risk contribution)
- Targets portfolio-level volatility
- Rebalances dynamically based on volatility

## Strategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fast-ema` | 12 | Fast EMA span (shorter-term trend) |
| `--slow-ema` | 48 | Slow EMA span (longer-term trend) |
| `--vol-lookback` | 20 | Volatility calculation window (days) |
| `--target-vol` | 0.15 | Target annual volatility (15%) |
| `--max-leverage` | 2.0 | Maximum position leverage |
| `--transaction-cost-bps` | 1.0 | Transaction cost in basis points |

### Understanding the Strategy

1. **Trend Signal**: When fast EMA > slow EMA, the strategy goes long (trending up)
2. **Volatility Targeting**: Position size is adjusted to target a specific volatility level
3. **Risk Management**: Maximum leverage caps prevent excessive exposure
4. **Costs**: Transaction costs are deducted from returns

## Architecture

### Data Sources

The bot supports multiple data sources through an abstract interface:

- **CSVDataSource** - Load historical data from CSV files
- **AlpacaDataSource** - Real-time and historical data from Alpaca Markets
- **YahooFinanceDataSource** - Free historical data from Yahoo Finance

### Brokers

Order execution is handled through broker interfaces:

- **PaperTradingBroker** - Simulated trading (no real money, for testing)
- **AlpacaBroker** - Real order execution via Alpaca API

### Core Components

| File | Purpose |
|------|---------|
| `trading_bot.py` | Core strategy, backtesting engine, and CLI |
| `data_source.py` | Abstract data source interface and implementations |
| `broker.py` | Abstract broker interface and implementations |
| `walk_forward.py` | Walk-forward validation framework |
| `portfolio.py` | Multi-asset portfolio management with risk parity |
| `live_trading.py` | Live trading execution engine |

## Examples

### Example 1: Basic Backtest

```bash
# Simple backtest with default parameters
python -m src.trading_bot data.csv

# Custom strategy parameters
python -m src.trading_bot data.csv \
  --fast-ema 8 \
  --slow-ema 40 \
  --target-vol 0.20
```

### Example 2: Walk-Forward Analysis

```bash
# Standard walk-forward (1 year train, 3 month test)
python -m src.trading_bot data.csv --mode walk-forward

# Custom windows
python -m src.trading_bot data.csv \
  --mode walk-forward \
  --train-window 180 \
  --test-window 30 \
  --step-days 15
```

### Example 3: Portfolio Backtest

```bash
# 3-asset portfolio with Yahoo Finance
python -m src.trading_bot \
  --mode portfolio \
  --symbols AAPL MSFT GOOGL TSLA \
  --data-source yahoo \
  --target-vol 0.12 \
  --max-leverage 1.5
```

### Example 4: Live Trading Simulation

```bash
# Test live trading logic with Yahoo Finance (no API needed)
python -m src.trading_bot \
  --mode live \
  --symbol SPY \
  --data-source yahoo
```

## Project Structure

```
Quantitative-Trading-Bot/
├── src/
│   ├── __init__.py
│   ├── trading_bot.py      # Core strategy, backtest, CLI
│   ├── data_source.py      # Data source abstractions
│   ├── broker.py           # Broker API abstractions
│   ├── walk_forward.py     # Walk-forward validation
│   ├── portfolio.py        # Multi-asset portfolio management
│   ├── live_trading.py     # Live trading engine
├── path/
│   └── to/
│       └── data.csv        # Example data file
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── QUICK_START.md          # Quick reference guide
```

## Command-Line Reference

### Basic Syntax
```bash
python -m src.trading_bot [CSV_PATH] [OPTIONS]
```

### All Available Options

```bash
python -m src.trading_bot --help
```

**Mode Selection:**
- `--mode {backtest,walk-forward,live,portfolio}` - Trading mode (default: backtest)

**Data Source:**
- `--data-source {csv,alpaca,yahoo}` - Data source type (default: csv)
- `--price-column COLUMN` - Price column name (default: close)

**Live Trading:**
- `--symbol SYMBOL` - Trading symbol (required for live/portfolio)
- `--alpaca-key KEY` - Alpaca API key
- `--alpaca-secret SECRET` - Alpaca API secret

**Portfolio:**
- `--symbols SYM1 SYM2 ...` - List of symbols for portfolio mode

**Strategy Parameters:**
- `--fast-ema INT` - Fast EMA span
- `--slow-ema INT` - Slow EMA span
- `--vol-lookback INT` - Volatility window
- `--target-vol FLOAT` - Target volatility
- `--max-leverage FLOAT` - Max leverage
- `--transaction-cost-bps FLOAT` - Transaction cost (basis points)

**Walk-Forward:**
- `--train-window INT` - Training window (days)
- `--test-window INT` - Test window (days)
- `--step-days INT` - Step size (days)

## Troubleshooting

### Common Issues

**Issue: "CSV file not found"**
- Solution: Use absolute path or ensure file is in correct location
- Example: `python -m src.trading_bot C:\full\path\to\data.csv`

**Issue: "Module not found" errors**
- Solution: Install missing dependencies: `pip install -r requirements.txt`

**Issue: Alpaca API errors**
- Solution: Verify API keys are correct and account has paper trading enabled

**Issue: Yahoo Finance data not loading**
- Solution: Install yfinance: `pip install yfinance`
- Check internet connection
- Some symbols may not be available

**Issue: Portfolio mode requires symbols**
- Solution: Use `--symbols AAPL MSFT GOOGL` flag

## Best Practices

1. **Start Simple**: Begin with backtest mode to understand the strategy
2. **Validate Robustness**: Always run walk-forward validation before live trading
3. **Paper Trade First**: Test live trading logic with paper trading
4. **Monitor Performance**: Regularly check performance metrics
5. **Risk Management**: Never risk more than you can afford to lose
6. **Document Changes**: Keep track of parameter changes and results

## Recommended Workflow

1. **Develop Strategy** → Start with backtest mode to test ideas
2. **Validate Robustness** → Use walk-forward to ensure strategy works across periods
3. **Test Execution** → Use paper trading to validate execution logic
4. **Scale Up** → Use portfolio mode to trade multiple assets
5. **Go Live** → Switch to real broker (with extreme caution!)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Remember**: Trading involves substantial risk of loss. This software is provided for educational purposes only. Always test thoroughly and never risk more than you can afford to lose.

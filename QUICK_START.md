# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt

# Optional: For live trading
pip install alpaca-py
```

## Basic Usage Examples

### 1. Simple Backtest
```bash
python -m src.trading_bot path/to/data.csv
```

### 2. Walk-Forward Validation
```bash
python -m src.trading_bot path/to/data.csv --mode walk-forward
```

### 3. Live Trading (Paper)
```bash
# With Yahoo Finance (simulated)
python -m src.trading_bot --mode live --symbol AAPL --data-source yahoo

# With Alpaca (paper trading)
python -m src.trading_bot --mode live --symbol AAPL \
  --data-source alpaca \
  --alpaca-key YOUR_KEY \
  --alpaca-secret YOUR_SECRET
```

### 4. Multi-Asset Portfolio
```bash
# With Yahoo Finance
python -m src.trading_bot --mode portfolio \
  --symbols AAPL MSFT GOOGL \
  --data-source yahoo

# With CSV (columns = symbols)
python -m src.trading_bot portfolio_data.csv \
  --mode portfolio \
  --symbols AAPL MSFT GOOGL \
  --data-source csv
```

## Common Parameters

- `--fast-ema 12`: Fast EMA period
- `--slow-ema 48`: Slow EMA period  
- `--target-vol 0.15`: Target 15% annual volatility
- `--max-leverage 2.0`: Maximum 2x leverage
- `--transaction-cost-bps 1.0`: 1 basis point transaction cost

## Modes Summary

| Mode | Description | Data Source |
|------|-------------|-------------|
| `backtest` | Simple historical backtest | CSV |
| `walk-forward` | Rolling train/test validation | CSV |
| `live` | Real-time trading | Alpaca, Yahoo |
| `portfolio` | Multi-asset with risk parity | CSV, Yahoo |

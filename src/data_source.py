"""Abstract data source interface and implementations for live and historical data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class DataSource(ABC):
    """Abstract interface for price data sources."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USD')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval ('1d', '1h', '5m', etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price
        """
        pass

    @abstractmethod
    def get_current_data(self, symbol: str) -> pd.Series:
        """Get current OHLCV data point.

        Args:
            symbol: Trading symbol

        Returns:
            Series with open, high, low, close, volume
        """
        pass


class CSVDataSource(DataSource):
    """CSV file-based data source for backtesting."""

    def __init__(self, csv_path: str, price_column: str = "close"):
        """Initialize CSV data source.

        Args:
            csv_path: Path to CSV file
            price_column: Column name for price data
        """
        self.csv_path = csv_path
        self.price_column = price_column
        self._data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load and prepare CSV data."""
        data = pd.read_csv(self.csv_path)
        data.columns = [col.lower() for col in data.columns]
        if "date" not in data.columns:
            raise ValueError("CSV must contain a 'date' column.")
        if self.price_column.lower() not in data.columns:
            raise ValueError(f"CSV must contain '{self.price_column}' column.")
        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data.sort_values("date").set_index("date")
        self._data = data

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical data from CSV."""
        if self._data is None:
            self._load_data()
        mask = (self._data.index >= start_date) & (self._data.index <= end_date)
        return self._data[mask].reset_index()

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price from CSV."""
        if self._data is None:
            self._load_data()
        return float(self._data[self.price_column.lower()].iloc[-1])

    def get_current_data(self, symbol: str) -> pd.Series:
        """Get current data point from CSV."""
        if self._data is None:
            self._load_data()
        return self._data.iloc[-1]


class AlpacaDataSource(DataSource):
    """Alpaca Markets API data source for live trading."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
    ):
        """Initialize Alpaca data source.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Base URL (defaults to paper trading URL)
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required. Install with: pip install alpaca-py"
            )

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or "https://paper-api.alpaca.markets"

        self.historical_client = StockHistoricalDataClient(
            api_key, api_secret, base_url=base_url
        )
        self.trading_client = TradingClient(api_key, api_secret, paper=True)

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        # Map interval string to TimeFrame
        timeframe_map = {
            "1d": TimeFrame.Day,
            "1h": TimeFrame.Hour,
            "5m": TimeFrame.Minute,
        }
        timeframe = timeframe_map.get(interval, TimeFrame.Day)

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )
        bars = self.historical_client.get_stock_bars(request)
        df = bars.df.reset_index()
        df = df.rename(
            columns={
                "timestamp": "date",
                "trade_count": "volume",  # Alpaca uses trade_count
            }
        )
        return df[["date", "open", "high", "low", "close", "volume"]]

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price from Alpaca."""
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=[symbol])
        bars = self.historical_client.get_stock_latest_bar(request)
        return float(bars[symbol].close)

    def get_current_data(self, symbol: str) -> pd.Series:
        """Get current data from Alpaca."""
        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=[symbol])
        bars = self.historical_client.get_stock_latest_bar(request)
        bar = bars[symbol]
        return pd.Series(
            {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
        )


class YahooFinanceDataSource(DataSource):
    """Yahoo Finance data source (free, no API key required)."""

    def __init__(self):
        """Initialize Yahoo Finance data source."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required. Install with: pip install yfinance"
            )
        self.yf = yf

    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        df = df.reset_index()
        df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        return df[["date", "Open", "High", "Low", "Close", "Volume"]].rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price from Yahoo Finance."""
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return float(data["Close"].iloc[-1])

    def get_current_data(self, symbol: str) -> pd.Series:
        """Get current data from Yahoo Finance."""
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(period="1d")
        latest = data.iloc[-1]
        return pd.Series(
            {
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": float(latest["Volume"]),
            }
        )

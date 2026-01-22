"""Abstract broker interface and implementations for order execution."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Represents a current position."""

    symbol: str
    quantity: float
    avg_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.avg_price) * self.quantity


@dataclass
class Account:
    """Represents trading account information."""

    equity: float
    cash: float
    positions: dict[str, Position]
    buying_power: float

    @property
    def total_value(self) -> float:
        """Total account value (cash + positions)."""
        return self.cash + sum(pos.market_value for pos in self.positions.values())


class Broker(ABC):
    """Abstract interface for broker APIs."""

    @abstractmethod
    def get_account(self) -> Account:
        """Get current account information."""
        pass

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order and return order ID."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Get order status."""
        pass


class PaperTradingBroker(Broker):
    """Paper trading broker for simulation (no real money)."""

    def __init__(self, initial_cash: float = 100000.0):
        """Initialize paper trading broker.

        Args:
            initial_cash: Starting cash balance
        """
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.order_counter = 0
        self.price_data: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Set current price for a symbol (for simulation)."""
        self.price_data[symbol] = price

    def get_account(self) -> Account:
        """Get account information."""
        total_positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return Account(
            equity=self.cash + total_positions_value,
            cash=self.cash,
            positions=self.positions.copy(),
            buying_power=self.cash * 2.0,  # Assume 2x margin
        )

    def get_positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()

    def submit_order(self, order: Order) -> str:
        """Submit an order (simulated execution)."""
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        self.orders[order_id] = order

        # Simulate immediate execution at current price
        current_price = self.price_data.get(order.symbol, 0.0)
        if current_price == 0.0:
            raise ValueError(f"No price data for {order.symbol}")

        execution_price = (
            order.limit_price
            if order.order_type == OrderType.LIMIT and order.limit_price
            else current_price
        )

        cost = order.quantity * execution_price

        if order.side == OrderSide.BUY:
            if cost > self.cash:
                raise ValueError("Insufficient cash")
            self.cash -= cost
            if order.symbol in self.positions:
                # Update existing position
                pos = self.positions[order.symbol]
                total_cost = (pos.quantity * pos.avg_price) + cost
                total_quantity = pos.quantity + order.quantity
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=total_quantity,
                    avg_price=total_cost / total_quantity,
                    current_price=execution_price,
                )
            else:
                # New position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_price=execution_price,
                    current_price=execution_price,
                )
        else:  # SELL
            if order.symbol not in self.positions:
                raise ValueError(f"No position to sell for {order.symbol}")
            pos = self.positions[order.symbol]
            if order.quantity > pos.quantity:
                raise ValueError("Insufficient position quantity")
            self.cash += order.quantity * execution_price
            if order.quantity == pos.quantity:
                # Close position
                del self.positions[order.symbol]
            else:
                # Reduce position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=pos.quantity - order.quantity,
                    avg_price=pos.avg_price,
                    current_price=execution_price,
                )

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    def get_order_status(self, order_id: str) -> dict:
        """Get order status."""
        if order_id in self.orders:
            return {"status": "filled", "order": self.orders[order_id]}
        return {"status": "not_found"}


class AlpacaBroker(Broker):
    """Alpaca Markets broker implementation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True,
    ):
        """Initialize Alpaca broker.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Use paper trading (default: True)
        """
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        except ImportError:
            raise ImportError(
                "alpaca-py is required. Install with: pip install alpaca-py"
            )

        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.MarketOrderRequest = MarketOrderRequest
        self.LimitOrderRequest = LimitOrderRequest

    def get_account(self) -> Account:
        """Get account information from Alpaca."""
        account = self.client.get_account()
        positions = self.get_positions()

        return Account(
            equity=float(account.equity),
            cash=float(account.cash),
            positions=positions,
            buying_power=float(account.buying_power),
        )

    def get_positions(self) -> dict[str, Position]:
        """Get all positions from Alpaca."""
        alpaca_positions = self.client.get_all_positions()
        positions = {}

        for pos in alpaca_positions:
            # Get current price
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest

            # This is a simplified version - in production, you'd cache prices
            data_client = StockHistoricalDataClient(
                self.client._key_id, self.client._secret_key
            )
            request = StockLatestBarRequest(symbol_or_symbols=[pos.symbol])
            bars = data_client.get_stock_latest_bar(request)
            current_price = float(bars[pos.symbol].close)

            positions[pos.symbol] = Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_price=float(pos.avg_entry_price),
                current_price=current_price,
            )

        return positions

    def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        if order.order_type == OrderType.MARKET:
            request = self.MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
            )
        else:
            request = self.LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=order.side.value,
                limit_price=order.limit_price,
            )

        submitted_order = self.client.submit_order(request)
        return str(submitted_order.id)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order on Alpaca."""
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> dict:
        """Get order status from Alpaca."""
        order = self.client.get_order_by_id(order_id)
        return {
            "status": order.status,
            "filled_qty": float(order.filled_qty),
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        }

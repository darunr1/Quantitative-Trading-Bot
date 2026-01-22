"""Live trading execution engine."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import pandas as pd

from src.broker import Broker, Order, OrderSide, OrderType, PaperTradingBroker
from src.data_source import DataSource
from src.trading_bot import StrategyConfig, compute_strategy_returns


class LiveTrader:
    """Live trading engine that executes strategy in real-time."""

    def __init__(
        self,
        data_source: DataSource,
        broker: Broker,
        symbol: str,
        config: StrategyConfig,
        lookback_days: int = 100,
    ):
        """Initialize live trader.

        Args:
            data_source: Data source for price data
            broker: Broker for order execution
            symbol: Trading symbol
            config: Strategy configuration
            lookback_days: Days of historical data to use for signals
        """
        self.data_source = data_source
        self.broker = broker
        self.symbol = symbol
        self.config = config
        self.lookback_days = lookback_days
        self.current_position = 0.0
        self.last_signal = 0.0

    def get_current_signal(self) -> float:
        """Compute current trading signal based on recent data."""
        end_date = datetime.now()
        start_date = datetime.fromtimestamp(
            end_date.timestamp() - (self.lookback_days * 24 * 3600)
        )

        # Get historical data
        data = self.data_source.get_historical_data(
            self.symbol, start_date, end_date, interval="1d"
        )
        if len(data) == 0:
            return 0.0

        # Prepare prices
        prices = data.set_index("date")["close"]
        if len(prices) < self.config.slow_ema_span:
            return 0.0

        # Compute strategy returns to get position signal
        strategy_returns = compute_strategy_returns(prices, self.config)

        # Get the latest position (before shift)
        fast_ema = prices.ewm(span=self.config.fast_ema_span, adjust=False).mean()
        slow_ema = prices.ewm(span=self.config.slow_ema_span, adjust=False).mean()
        trend_signal = (fast_ema > slow_ema).astype(float)

        daily_returns = prices.pct_change().fillna(0.0)
        rolling_vol = daily_returns.rolling(self.config.vol_lookback).std().replace(
            0.0, pd.NA
        )
        vol_target = self.config.target_vol / (rolling_vol * (252**0.5))
        position_size = vol_target.clip(upper=self.config.max_leverage).fillna(0.0)

        raw_position = trend_signal * position_size
        target_position = float(raw_position.iloc[-1])

        return target_position

    def execute_trade(self, target_position: float) -> Optional[str]:
        """Execute trade to reach target position.

        Args:
            target_position: Target position size (can be negative for short)

        Returns:
            Order ID if order was placed, None otherwise
        """
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        current_position = positions.get(self.symbol, None)
        current_qty = current_position.quantity if current_position else 0.0

        # Get current price
        current_price = self.data_source.get_latest_price(self.symbol)

        # Calculate target quantity based on account equity
        account_value = account.equity
        target_value = abs(target_position) * account_value
        target_qty = target_value / current_price if current_price > 0 else 0.0

        # Determine if we need to trade
        position_diff = target_qty - current_qty
        min_trade_size = 0.01  # Minimum trade size

        if abs(position_diff) < min_trade_size:
            return None  # No trade needed

        # Determine order side
        if position_diff > 0:
            # Need to buy
            if current_qty < 0:
                # Close short position first
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=abs(current_qty),
                    order_type=OrderType.MARKET,
                )
                self.broker.submit_order(order)
                # Then open long
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=target_qty,
                    order_type=OrderType.MARKET,
                )
            else:
                # Increase long position
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=position_diff,
                    order_type=OrderType.MARKET,
                )
        else:
            # Need to sell
            if current_qty > 0:
                # Close long position first
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=current_qty,
                    order_type=OrderType.MARKET,
                )
                self.broker.submit_order(order)
                # Then open short if needed
                if target_position < 0:
                    order = Order(
                        symbol=self.symbol,
                        side=OrderSide.SELL,
                        quantity=abs(target_qty),
                        order_type=OrderType.MARKET,
                    )
            else:
                # Increase short position
                order = Order(
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=abs(position_diff),
                    order_type=OrderType.MARKET,
                )

        try:
            order_id = self.broker.submit_order(order)
            self.current_position = target_qty
            return order_id
        except Exception as e:
            print(f"Error executing trade: {e}")
            return None

    def run(self, check_interval_seconds: int = 3600) -> None:
        """Run live trading loop.

        Args:
            check_interval_seconds: Seconds between signal checks
        """
        print(f"Starting live trading for {self.symbol}")
        print(f"Check interval: {check_interval_seconds} seconds")
        print("Press Ctrl+C to stop")

        try:
            while True:
                signal = self.get_current_signal()
                if signal != self.last_signal:
                    print(f"\n[{datetime.now()}] Signal changed: {self.last_signal:.4f} -> {signal:.4f}")
                    order_id = self.execute_trade(signal)
                    if order_id:
                        print(f"Order placed: {order_id}")
                    self.last_signal = signal

                # Print account status
                account = self.broker.get_account()
                print(f"[{datetime.now()}] Equity: ${account.equity:,.2f}, Cash: ${account.cash:,.2f}")

                time.sleep(check_interval_seconds)
        except KeyboardInterrupt:
            print("\nStopping live trading...")

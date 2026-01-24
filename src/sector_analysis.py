"""Sector-based analysis: strategy metrics, rankings, and invest reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.sectors import Sector, SECTORS, get_sector
from src.trading_bot import (
    StrategyConfig,
    calculate_performance,
    compute_strategy_returns,
)


@dataclass
class TickerAnalysis:
    """Analysis result for a single ticker."""

    symbol: str
    sector_id: str
    is_etf: bool
    current_price: float
    fast_ema: float
    slow_ema: float
    trend_signal: str  # "bullish" | "bearish"
    position_size: float
    annual_return: float
    sharpe_ratio: float
    annual_volatility: float
    max_drawdown: float
    n_observations: int
    reasoning: str
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorRecommendation:
    """Sector-level recommendation with ticker-level breakdown."""

    sector: Sector
    etf_analysis: Optional[TickerAnalysis] = None
    stock_analyses: List[TickerAnalysis] = field(default_factory=list)
    sector_score: float = 0.0
    reasoning: str = ""


def _fetch_prices(symbol: str, lookback_days: int = 504) -> Optional[pd.Series]:
    """Fetch daily close prices from Yahoo Finance. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    if df is None or df.empty or len(df) < 60:
        return None
    return df["Close"].dropna()


def _analyze_ticker(
    symbol: str,
    sector_id: str,
    is_etf: bool,
    config: StrategyConfig,
    prices: pd.Series,
) -> TickerAnalysis:
    """Run strategy on prices, compute metrics, build reasoning."""
    prices = prices.ffill().dropna()
    if len(prices) < config.slow_ema_span + config.vol_lookback:
        return TickerAnalysis(
            symbol=symbol,
            sector_id=sector_id,
            is_etf=is_etf,
            current_price=float(prices.iloc[-1]),
            fast_ema=0.0,
            slow_ema=0.0,
            trend_signal="unknown",
            position_size=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            annual_volatility=0.0,
            max_drawdown=0.0,
            n_observations=len(prices),
            reasoning="Insufficient data for analysis.",
        )

    fast_ema = prices.ewm(span=config.fast_ema_span, adjust=False).mean()
    slow_ema = prices.ewm(span=config.slow_ema_span, adjust=False).mean()
    daily_returns = prices.pct_change().fillna(0.0)
    rolling_vol = daily_returns.rolling(config.vol_lookback).std().replace(0.0, np.nan)
    vol_target = config.target_vol / (rolling_vol * np.sqrt(252))
    position_size_series = vol_target.clip(upper=config.max_leverage).fillna(0.0)

    fe = float(fast_ema.iloc[-1])
    se = float(slow_ema.iloc[-1])
    trend = "bullish" if fe > se else "bearish"
    ps_last = position_size_series.dropna()
    ps_val = float(ps_last.iloc[-1]) if len(ps_last) else 0.0
    ps_val = min(max(ps_val, 0.0), config.max_leverage)
    pos = (1.0 if fe > se else 0.0) * ps_val

    strat_returns = compute_strategy_returns(prices, config)
    perf = calculate_performance(strat_returns)

    ann_ret = perf.annual_return
    sharpe = perf.sharpe_ratio
    ann_vol = perf.annual_volatility
    mdd = perf.max_drawdown

    # Mathematical reasoning
    reason_parts = [
        f"**Price & trend:** Latest close = ${prices.iloc[-1]:.2f}. "
        f"Fast EMA ({config.fast_ema_span}d) = {fe:.2f}, Slow EMA ({config.slow_ema_span}d) = {se:.2f}. "
        f"Trend is **{trend}** (fast {'>' if fe > se else '<'} slow).",
        f"**Volatility targeting:** Realized vol (annualized) = {ann_vol:.1%}. "
        f"Target vol = {config.target_vol:.0%}. Position size scaled to {pos:.2f} (capped at {config.max_leverage}x).",
        f"**Backtest (≈{len(prices)} days):** Annual return = {ann_ret:.1%}, "
        f"Sharpe = {sharpe:.2f}, Max drawdown = {mdd:.1%}. "
        f"Transaction costs = {config.transaction_cost_bps} bps.",
    ]
    if sharpe > 0.5 and ann_ret > 0:
        reason_parts.append(
            f"**Recommendation:** Favorable risk-adjusted profile (Sharpe > 0.5, positive return). "
            f"Consider exposure if trend remains {trend}."
        )
    elif trend == "bearish":
        reason_parts.append(
            "**Recommendation:** No long allocation; trend is bearish. "
            "Strategy waits for fast EMA > slow EMA before going long."
        )
    else:
        reason_parts.append(
            "**Recommendation:** Mixed signals. Evaluate alongside other sectors and risk constraints."
        )

    reasoning = " ".join(reason_parts)

    return TickerAnalysis(
        symbol=symbol,
        sector_id=sector_id,
        is_etf=is_etf,
        current_price=float(prices.iloc[-1]),
        fast_ema=fe,
        slow_ema=se,
        trend_signal=trend,
        position_size=pos,
        annual_return=ann_ret,
        sharpe_ratio=sharpe,
        annual_volatility=ann_vol,
        max_drawdown=mdd,
        n_observations=len(prices),
        reasoning=reasoning,
        raw_metrics={
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "annual_volatility": ann_vol,
            "max_drawdown": mdd,
        },
    )


def analyze_sector(
    sector: Sector,
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> SectorRecommendation:
    """Analyze sector ETF and representative stocks; produce recommendation."""
    config = config or StrategyConfig()
    rec = SectorRecommendation(sector=sector)

    # ETF
    etf_prices = _fetch_prices(sector.etf, lookback_days)
    if etf_prices is not None:
        rec.etf_analysis = _analyze_ticker(
            sector.etf, sector.id, True, config, etf_prices
        )

    # Stocks
    for sym in sector.stocks:
        prices = _fetch_prices(sym, lookback_days)
        if prices is not None:
            rec.stock_analyses.append(
                _analyze_ticker(sym, sector.id, False, config, prices)
            )

    # Sector score: avg Sharpe of ETF + stocks (bullish only, else 0)
    sharpes: List[float] = []
    if rec.etf_analysis and rec.etf_analysis.trend_signal == "bullish":
        sharpes.append(rec.etf_analysis.sharpe_ratio)
    for a in rec.stock_analyses:
        if a.trend_signal == "bullish":
            sharpes.append(a.sharpe_ratio)
    rec.sector_score = float(np.mean(sharpes)) if sharpes else 0.0

    # Sector-level reasoning
    n_bull = sum(1 for a in rec.stock_analyses if a.trend_signal == "bullish")
    if rec.etf_analysis and rec.etf_analysis.trend_signal == "bullish":
        n_bull += 1
    n_total = len(rec.stock_analyses) + (1 if rec.etf_analysis else 0)
    rec.reasoning = (
        f"**{sector.name}** ({sector.etf}): {n_bull}/{n_total} tickers bullish. "
        f"Sector score (avg Sharpe, bullish only) = {rec.sector_score:.2f}. "
        f"{sector.description}"
    )

    return rec


def analyze_all_sectors(
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> List[SectorRecommendation]:
    """Analyze all sectors and return sorted by sector score (desc)."""
    results = []
    for s in SECTORS:
        try:
            rec = analyze_sector(s, config=config, lookback_days=lookback_days)
            results.append(rec)
        except Exception:
            continue
    results.sort(key=lambda r: r.sector_score, reverse=True)
    return results


def analyze_ticker(
    symbol: str,
    sector_id: Optional[str] = None,
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> Optional[TickerAnalysis]:
    """Analyze a single ticker. sector_id used only for labeling."""
    config = config or StrategyConfig()
    prices = _fetch_prices(symbol, lookback_days)
    if prices is None:
        return None
    sector_id = sector_id or "unknown"
    return _analyze_ticker(symbol, sector_id, False, config, prices)

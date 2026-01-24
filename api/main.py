"""FastAPI backend for sector recommendations and ticker analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.sector_analysis import (
    SectorRecommendation,
    TickerAnalysis,
    analyze_all_sectors,
    analyze_sector,
    analyze_ticker,
)
from src.sectors import SECTORS, get_sector
from src.trading_bot import StrategyConfig

app = FastAPI(
    title="Quantitative Trading Bot – Sector Recommendations",
    description="Sector-based stock/ETF recommendations with mathematical reasoning.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ticker_to_dict(a: TickerAnalysis) -> Dict[str, Any]:
    return {
        "symbol": a.symbol,
        "sector_id": a.sector_id,
        "is_etf": a.is_etf,
        "current_price": a.current_price,
        "fast_ema": a.fast_ema,
        "slow_ema": a.slow_ema,
        "trend_signal": a.trend_signal,
        "position_size": a.position_size,
        "annual_return": a.annual_return,
        "sharpe_ratio": a.sharpe_ratio,
        "annual_volatility": a.annual_volatility,
        "max_drawdown": a.max_drawdown,
        "n_observations": a.n_observations,
        "reasoning": a.reasoning,
    }


def _sector_rec_to_dict(r: SectorRecommendation) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "sector": {
            "id": r.sector.id,
            "name": r.sector.name,
            "description": r.sector.description,
            "etf": r.sector.etf,
            "stocks": list(r.sector.stocks),
        },
        "sector_score": r.sector_score,
        "reasoning": r.reasoning,
        "etf_analysis": _ticker_to_dict(r.etf_analysis) if r.etf_analysis else None,
        "stock_analyses": [_ticker_to_dict(a) for a in r.stock_analyses],
    }
    return d


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Sector Recommendations API", "docs": "/docs"}


@app.get("/sectors")
def list_sectors() -> List[Dict[str, Any]]:
    """List all sectors (metadata only)."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "etf": s.etf,
            "stocks": list(s.stocks),
        }
        for s in SECTORS
    ]


@app.get("/sectors/analyze")
def analyze_sectors_api(
    lookback_days: int = 504,
) -> List[Dict[str, Any]]:
    """Analyze all sectors; returns recommendations sorted by sector score."""
    config = StrategyConfig()
    recs = analyze_all_sectors(config=config, lookback_days=lookback_days)
    return [_sector_rec_to_dict(r) for r in recs]


@app.get("/sectors/{sector_id}")
def sector_detail(sector_id: str) -> Dict[str, Any]:
    """Sector metadata by id."""
    s = get_sector(sector_id)
    if not s:
        raise HTTPException(status_code=404, detail="Sector not found")
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "etf": s.etf,
        "stocks": list(s.stocks),
    }


@app.get("/sectors/{sector_id}/analyze")
def sector_analyze_api(
    sector_id: str,
    lookback_days: int = 504,
) -> Dict[str, Any]:
    """Analyze a single sector (ETF + stocks) with reasoning."""
    s = get_sector(sector_id)
    if not s:
        raise HTTPException(status_code=404, detail="Sector not found")
    config = StrategyConfig()
    rec = analyze_sector(s, config=config, lookback_days=lookback_days)
    return _sector_rec_to_dict(rec)


@app.get("/tickers/{symbol}")
def ticker_analyze_api(
    symbol: str,
    sector_id: Optional[str] = None,
    lookback_days: int = 504,
) -> Dict[str, Any]:
    """Analyze a single ticker with reasoning."""
    config = StrategyConfig()
    a = analyze_ticker(symbol, sector_id=sector_id, config=config, lookback_days=lookback_days)
    if not a:
        raise HTTPException(status_code=404, detail="Ticker not found or insufficient data")
    return _ticker_to_dict(a)

"""Sector definitions and ticker mappings for sector-based recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Sector:
    """A market sector with ETF and representative stocks."""

    id: str
    name: str
    description: str
    etf: str  # Sector ETF
    stocks: List[str]  # Representative stocks


# All 11 GICS sectors with ETFs and representative tickers
SECTORS: List[Sector] = [
    Sector(
        id="financials",
        name="Financials",
        description="Banks, insurance, diversified financials, real estate investment trusts.",
        etf="XLF",
        stocks=["JPM", "BAC", "GS", "WFC", "MS"],
    ),
    Sector(
        id="information_technology",
        name="Information Technology",
        description="Software, hardware, semiconductors, IT services.",
        etf="XLK",
        stocks=["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    ),
    Sector(
        id="health_care",
        name="Health Care",
        description="Pharma, biotech, health care equipment, providers.",
        etf="XLV",
        stocks=["UNH", "JNJ", "LLY", "PFE", "ABBV"],
    ),
    Sector(
        id="consumer_discretionary",
        name="Consumer Discretionary",
        description="Retail, autos, hotels, leisure, durable household goods.",
        etf="XLY",
        stocks=["AMZN", "TSLA", "HD", "MCD", "NKE"],
    ),
    Sector(
        id="consumer_staples",
        name="Consumer Staples",
        description="Food, beverages, tobacco, household products.",
        etf="XLP",
        stocks=["PG", "KO", "PEP", "WMT", "COST"],
    ),
    Sector(
        id="energy",
        name="Energy",
        description="Oil, gas, coal, consumable fuels, energy equipment.",
        etf="XLE",
        stocks=["XOM", "CVX", "COP", "SLB", "EOG"],
    ),
    Sector(
        id="industrials",
        name="Industrials",
        description="Aerospace, machinery, construction, transportation.",
        etf="XLI",
        stocks=["HON", "UNP", "UPS", "CAT", "RTX"],
    ),
    Sector(
        id="materials",
        name="Materials",
        description="Chemicals, metals, mining, construction materials.",
        etf="XLB",
        stocks=["LIN", "APD", "SHW", "ECL", "FCX"],
    ),
    Sector(
        id="communication_services",
        name="Communication Services",
        description="Media, telecom, interactive media, entertainment.",
        etf="XLC",
        stocks=["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
    ),
    Sector(
        id="real_estate",
        name="Real Estate",
        description="REITs and real estate management.",
        etf="XLRE",
        stocks=["PLD", "AMT", "EQIX", "PSA", "O"],
    ),
    Sector(
        id="utilities",
        name="Utilities",
        description="Electric, gas, water utilities, independent power.",
        etf="XLU",
        stocks=["NEE", "DUK", "SO", "D", "AEP"],
    ),
]


def get_sector(sector_id: str) -> Sector | None:
    """Return sector by id, or None."""
    for s in SECTORS:
        if s.id == sector_id:
            return s
    return None


def get_all_tickers() -> List[str]:
    """All unique tickers (ETFs + stocks) across sectors."""
    seen = set()
    out: List[str] = []
    for s in SECTORS:
        if s.etf not in seen:
            seen.add(s.etf)
            out.append(s.etf)
        for t in s.stocks:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out

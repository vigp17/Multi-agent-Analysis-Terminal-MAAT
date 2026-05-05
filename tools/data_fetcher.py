"""MAAT — Data Fetcher

Fetches market and macro data for agents to consume.
All agents call these functions — never fetch data inside an agent directly.

Sources:
    - yfinance  : price data, financials, company info
    - FRED API  : macro indicators (optional, falls back to yfinance proxies)
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config.settings import settings


# ── Price Data ─────────────────────────────────────────────────────────

def fetch_price_data(ticker: str) -> dict:
    """Fetch OHLCV price history for a ticker.

    Returns:
        dict with keys: prices (DataFrame), returns, latest_price,
        52w_high, 52w_low, avg_volume
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(
            period=settings.price_history_period,
            interval=settings.price_history_interval,
        )

        if df.empty:
            return {"error": f"No price data found for {ticker}"}

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        returns = df["Close"].pct_change().dropna()

        return {
            "prices": df,
            "returns": returns,
            "latest_price": round(float(df["Close"].iloc[-1]), 2),
            "52w_high": round(float(df["High"].tail(252).max()), 2),
            "52w_low": round(float(df["Low"].tail(252).min()), 2),
            "avg_volume": int(df["Volume"].tail(30).mean()),
            "ticker": ticker,
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
        }

    except Exception as e:
        return {"error": str(e)}


# ── Fundamental Data ───────────────────────────────────────────────────

def fetch_fundamental_data(ticker: str) -> dict:
    """Fetch company financials and valuation metrics.

    Returns:
        dict with P/E, P/B, revenue growth, margins, debt ratios, etc.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "revenue_growth": info.get("revenueGrowth"),       # YoY
            "earnings_growth": info.get("earningsGrowth"),     # YoY
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cashflow": info.get("freeCashflow"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "analyst_target": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
        }

    except Exception as e:
        return {"error": str(e)}


# ── Macro Data ─────────────────────────────────────────────────────────

def fetch_macro_data() -> dict:
    """Fetch macro economic indicators.

    Uses yfinance proxies (Treasury ETFs, VIX, dollar index) so
    no FRED API key is required. If FRED key is set, it enriches with
    official series.

    Returns:
        dict with yield curve, VIX, dollar strength, market breadth
    """
    try:
        macro = {}

        # Treasury yields via ETFs / yfinance tickers
        tickers = {
            "^TNX": "yield_10y",     # 10-Year Treasury
            "^FVX": "yield_5y",      # 5-Year Treasury
            "^IRX": "yield_3m",      # 3-Month T-Bill
            "^VIX": "vix",           # Volatility index
            "DX-Y.NYB": "dxy",       # US Dollar Index
            "GC=F": "gold",          # Gold futures
            "CL=F": "oil",           # Crude oil futures
            "SPY": "spy",            # S&P 500 ETF (market proxy)
        }

        for yf_ticker, key in tickers.items():
            try:
                data = yf.Ticker(yf_ticker).history(period="5d")
                if not data.empty:
                    macro[key] = round(float(data["Close"].iloc[-1]), 2)
            except Exception:
                macro[key] = None

        # Derived: yield curve spread (10Y - 3M)
        if macro.get("yield_10y") and macro.get("yield_3m"):
            macro["yield_curve_spread"] = round(
                macro["yield_10y"] - macro["yield_3m"], 3
            )
            macro["yield_curve_inverted"] = macro["yield_curve_spread"] < 0
        else:
            macro["yield_curve_spread"] = None
            macro["yield_curve_inverted"] = None

        # SPY momentum (proxy for market trend)
        try:
            spy_hist = yf.Ticker("SPY").history(period="1y")
            if not spy_hist.empty:
                spy_close = spy_hist["Close"]
                macro["spy_above_200ma"] = bool(
                    spy_close.iloc[-1] > spy_close.tail(200).mean()
                )
                macro["spy_1m_return"] = round(
                    float((spy_close.iloc[-1] / spy_close.iloc[-22] - 1) * 100), 2
                )
                macro["spy_3m_return"] = round(
                    float((spy_close.iloc[-1] / spy_close.iloc[-66] - 1) * 100), 2
                )
        except Exception:
            pass

        # FRED enrichment (optional)
        if settings.fred_api_key:
            macro.update(_fetch_fred_data())

        return macro

    except Exception as e:
        return {"error": str(e)}


def _fetch_fred_data() -> dict:
    """Fetch official FRED series if API key is available."""
    try:
        import requests
        base = "https://api.stlouisfed.org/fred/series/observations"
        key = settings.fred_api_key
        fred = {}

        series = {
            "FEDFUNDS": "fed_funds_rate",
            "CPIAUCSL": "cpi",
            "UNRATE": "unemployment_rate",
            "GDP": "gdp_growth",
        }

        for series_id, field in series.items():
            try:
                resp = requests.get(base, params={
                    "series_id": series_id,
                    "api_key": key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 1,
                }, timeout=5)
                obs = resp.json().get("observations", [])
                if obs:
                    fred[field] = float(obs[0]["value"])
            except Exception:
                continue

        return fred

    except Exception:
        return {}


# ── Combined Fetch ─────────────────────────────────────────────────────

def fetch_all(ticker: str) -> dict:
    """Single call to fetch everything needed for a full analysis.

    Returns:
        dict with keys: price_data, fundamental_data, macro_data
    """
    return {
        "price_data": fetch_price_data(ticker),
        "fundamental_data": fetch_fundamental_data(ticker),
        "macro_data": fetch_macro_data(),
    }
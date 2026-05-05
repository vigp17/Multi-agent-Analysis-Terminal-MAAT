"""MAAT — Technical Indicators

Pure functions that compute technical indicators from price DataFrames.
The Technical Analyst agent calls these before sending data to Claude.

All functions accept a pandas DataFrame with a 'Close' column
and return scalar values or labeled strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend ──────────────────────────────────────────────────────────────

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def trend_label(prices: pd.DataFrame) -> str:
    """Classify price trend using 50MA vs 200MA relationship."""
    close = prices["Close"]
    ma50  = moving_average(close, 50).iloc[-1]
    ma200 = moving_average(close, 200).iloc[-1]
    price = close.iloc[-1]

    if price > ma50 > ma200:
        return "uptrend"
    elif price < ma50 < ma200:
        return "downtrend"
    elif abs(ma50 - ma200) / ma200 < 0.02:
        return "ranging"
    else:
        return "mixed"


def above_200ma(prices: pd.DataFrame) -> bool:
    """True if current price is above 200-day moving average."""
    close = prices["Close"]
    ma200 = moving_average(close, 200).iloc[-1]
    return bool(close.iloc[-1] > ma200)


# ── RSI ────────────────────────────────────────────────────────────────

def rsi(prices: pd.DataFrame, period: int = 14) -> float:
    """Relative Strength Index (0–100).

    > 70 = overbought, < 30 = oversold
    """
    close = prices["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return round(float(rsi_series.iloc[-1]), 2)


# ── MACD ───────────────────────────────────────────────────────────────

def macd(prices: pd.DataFrame) -> dict:
    """MACD line, signal line, and histogram.

    Standard params: 12 EMA - 26 EMA, signal = 9 EMA of MACD
    """
    close = prices["Close"]
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line   = ema12 - ema26
    signal_line = ema(macd_line, 9)
    histogram   = macd_line - signal_line

    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
    }


def macd_signal_label(prices: pd.DataFrame) -> str:
    """Human-readable MACD interpretation."""
    m = macd(prices)
    macd_val = m["macd"]
    hist     = m["histogram"]
    prev_hist = float(
        (prices["Close"].ewm(span=12).mean() - prices["Close"].ewm(span=26).mean())
        .diff()
        .iloc[-2]
    ) if len(prices) > 2 else 0

    if macd_val > 0 and hist > 0:
        return "bullish crossover" if hist > prev_hist else "bullish weakening"
    elif macd_val < 0 and hist < 0:
        return "bearish crossover" if hist < prev_hist else "bearish weakening"
    elif macd_val > 0 and hist < 0:
        return "bearish divergence"
    else:
        return "bullish divergence"


# ── Bollinger Bands ────────────────────────────────────────────────────

def bollinger_bands(prices: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    """Upper, middle, lower Bollinger Bands + %B position."""
    close = prices["Close"]
    mid   = moving_average(close, period)
    band  = std * close.rolling(period).std()
    upper = mid + band
    lower = mid - band
    pct_b = (close - lower) / (upper - lower)   # 0 = at lower, 1 = at upper

    return {
        "upper": round(float(upper.iloc[-1]), 2),
        "middle": round(float(mid.iloc[-1]), 2),
        "lower": round(float(lower.iloc[-1]), 2),
        "pct_b": round(float(pct_b.iloc[-1]), 3),
    }


# ── Support & Resistance ───────────────────────────────────────────────

def support_resistance(prices: pd.DataFrame, lookback: int = 90) -> dict:
    """Estimate support and resistance from recent price extremes."""
    recent = prices.tail(lookback)
    support    = round(float(recent["Low"].min()), 2)
    resistance = round(float(recent["High"].max()), 2)
    return {"support": support, "resistance": resistance}


# ── Volume ─────────────────────────────────────────────────────────────

def volume_trend(prices: pd.DataFrame) -> str:
    """Compare recent volume to 30-day average."""
    avg_vol    = prices["Volume"].tail(30).mean()
    recent_vol = prices["Volume"].tail(5).mean()
    ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

    if ratio > 1.5:
        return "high volume"
    elif ratio < 0.7:
        return "low volume"
    else:
        return "normal volume"


# ── Momentum ───────────────────────────────────────────────────────────

def price_momentum(prices: pd.DataFrame) -> dict:
    """Return % price change over 1W, 1M, 3M, 6M, 1Y."""
    close = prices["Close"]
    def pct(n): 
        return round(float((close.iloc[-1] / close.iloc[-n] - 1) * 100), 2) if len(close) > n else None

    return {
        "1w":  pct(5),
        "1m":  pct(22),
        "3m":  pct(66),
        "6m":  pct(132),
        "1y":  pct(252),
    }


# ── All-in-One ─────────────────────────────────────────────────────────

def compute_all(prices: pd.DataFrame) -> dict:
    """Compute all indicators in one call. Used by Technical Analyst agent."""
    return {
        "trend":       trend_label(prices),
        "above_200ma": above_200ma(prices),
        "rsi":         rsi(prices),
        "macd":        macd(prices),
        "macd_signal": macd_signal_label(prices),
        "bollinger":   bollinger_bands(prices),
        "support_resistance": support_resistance(prices),
        "volume_trend": volume_trend(prices),
        "momentum":    price_momentum(prices),
    }
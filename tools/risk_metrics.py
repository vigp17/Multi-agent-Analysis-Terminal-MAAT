"""MAAT — Risk Metrics

Computes quantitative risk measures from price/returns data.
The Risk Manager agent calls these before sending data to Claude.

Metrics:
    - Value at Risk (VaR) — historical simulation
    - Maximum Drawdown
    - Volatility (annualized)
    - Beta vs S&P 500
    - Kelly Criterion position sizing
    - Sharpe Ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from config.settings import settings


# ── Value at Risk ──────────────────────────────────────────────────────

def value_at_risk(returns: pd.Series, confidence: float = None) -> float:
    """Historical simulation VaR.

    Args:
        returns   : daily return series
        confidence: e.g. 0.95 for 95% VaR

    Returns:
        VaR as a positive percentage (e.g. 2.5 means 2.5% daily loss at 95%)
    """
    confidence = confidence or settings.var_confidence
    var = float(np.percentile(returns.dropna(), (1 - confidence) * 100))
    return round(abs(var) * 100, 3)   # Return as positive %


# ── Maximum Drawdown ───────────────────────────────────────────────────

def max_drawdown(prices: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown over the full price history.

    Returns:
        Drawdown as a positive percentage (e.g. 35.2 means -35.2%)
    """
    close        = prices["Close"]
    rolling_max  = close.cummax()
    drawdown     = (close - rolling_max) / rolling_max
    return round(abs(float(drawdown.min())) * 100, 2)


# ── Volatility ─────────────────────────────────────────────────────────

def annualized_volatility(returns: pd.Series, window: int = 30) -> float:
    """Annualized volatility based on recent N-day window.

    Returns:
        Volatility as a percentage (e.g. 28.5 means 28.5% annualized)
    """
    recent_vol = returns.tail(window).std()
    annualized = recent_vol * np.sqrt(252)
    return round(float(annualized) * 100, 2)


# ── Beta ───────────────────────────────────────────────────────────────

def compute_beta(returns: pd.Series, period: str = "2y") -> float:
    """Beta of the asset vs S&P 500 (SPY).

    Returns:
        Beta coefficient. 1.0 = market-like, >1 = more volatile, <1 = defensive
    """
    try:
        spy = yf.Ticker("SPY").history(period=period)["Close"].pct_change().dropna()

        # Align dates
        combined   = pd.concat([returns, spy], axis=1, join="inner").dropna()
        asset_ret  = combined.iloc[:, 0]
        market_ret = combined.iloc[:, 1]

        cov    = np.cov(asset_ret, market_ret)
        beta   = cov[0, 1] / cov[1, 1]
        return round(float(beta), 3)

    except Exception:
        return 1.0   # Default to market beta on failure


# ── Sharpe Ratio ───────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Annualized Sharpe Ratio.

    Args:
        returns        : daily return series
        risk_free_rate : annual risk-free rate (default 5% = current T-bill approx)

    Returns:
        Sharpe ratio. >1 = good, >2 = excellent, <0 = poor
    """
    daily_rf    = risk_free_rate / 252
    excess_ret  = returns - daily_rf
    sharpe      = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    return round(float(sharpe), 3)


# ── Kelly Criterion ────────────────────────────────────────────────────

def kelly_criterion(returns: pd.Series) -> float:
    """Kelly Criterion position sizing.

    Uses a simplified continuous Kelly formula.
    Applies fractional Kelly (settings.kelly_fraction) for safety.

    Returns:
        Suggested position size as % of portfolio (capped at max_position_size_pct)
    """
    mu    = float(returns.mean()) * 252          # Annualized mean return
    sigma = float(returns.std()) * np.sqrt(252)  # Annualized volatility

    if sigma == 0:
        return 0.0

    kelly_full     = mu / (sigma ** 2)
    kelly_fraction = kelly_full * settings.kelly_fraction
    kelly_pct      = kelly_fraction * 100

    # Cap between 0 and max allowed position
    kelly_pct = max(0.0, min(kelly_pct, settings.max_position_size_pct))
    return round(kelly_pct, 2)


# ── Risk Level Label ───────────────────────────────────────────────────

def risk_level_label(volatility: float, beta: float, var: float) -> str:
    """Classify overall risk level based on multiple metrics.

    Returns:
        "low" | "medium" | "high" | "very high"
    """
    score = 0

    # Volatility scoring
    if volatility < 15:   score += 1
    elif volatility < 25: score += 2
    elif volatility < 40: score += 3
    else:                 score += 4

    # Beta scoring
    if beta < 0.8:   score += 1
    elif beta < 1.2: score += 2
    elif beta < 1.8: score += 3
    else:            score += 4

    # VaR scoring
    if var < 1.5:  score += 1
    elif var < 2.5: score += 2
    elif var < 4.0: score += 3
    else:           score += 4

    avg = score / 3
    if avg <= 1.5:  return "low"
    elif avg <= 2.5: return "medium"
    elif avg <= 3.5: return "high"
    else:            return "very high"


# ── All-in-One ─────────────────────────────────────────────────────────

def compute_all(prices: pd.DataFrame, returns: pd.Series) -> dict:
    """Compute all risk metrics in one call. Used by Risk Manager agent."""
    vol  = annualized_volatility(returns)
    beta = compute_beta(returns)
    var  = value_at_risk(returns)

    return {
        "var_95":                  var,
        "max_drawdown":            max_drawdown(prices),
        "volatility_30d":          vol,
        "beta":                    beta,
        "sharpe_ratio":            sharpe_ratio(returns),
        "kelly_criterion":         kelly_criterion(returns),
        "recommended_position":    kelly_criterion(returns),
        "risk_level":              risk_level_label(vol, beta, var),
    }
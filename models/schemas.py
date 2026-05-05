"""MAAT — Pydantic Schemas

Shared data contracts between all agents and the graph.
Every agent reads and writes these types — no raw dicts passed around.

Structure:
    Enums → Agent Inputs → Agent Reports → Debate → Synthesis → Graph State
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class Signal(str, Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Confidence(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class MarketRegime(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class TimeHorizon(str, Enum):
    SHORT  = "1-3 months"
    MEDIUM = "3-6 months"
    LONG   = "6-12 months"


# ── Shared Agent Report Base ───────────────────────────────────────────

class AgentReport(BaseModel):
    """Base class for all agent reports."""
    agent_name: str
    ticker: str
    signal: Signal
    confidence: Confidence
    summary: str
    key_points: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


# ── Agent-Specific Reports ─────────────────────────────────────────────

class MacroReport(AgentReport):
    agent_name: str = "Macro Analyst"
    regime: MarketRegime
    fed_stance: str                        # e.g. "hawkish", "dovish", "neutral"
    yield_curve: str                       # e.g. "inverted", "normal", "flat"
    inflation_trend: str                   # e.g. "rising", "falling", "stable"
    recession_probability: float = Field(ge=0.0, le=1.0)


class TechnicalReport(AgentReport):
    agent_name: str = "Technical Analyst"
    regime: MarketRegime                   # HMM-detected regime
    trend: str                             # e.g. "uptrend", "downtrend", "ranging"
    rsi: float = Field(ge=0.0, le=100.0)
    macd_signal: str                       # e.g. "bullish crossover", "bearish"
    support_level: float
    resistance_level: float
    above_200ma: bool


class FundamentalReport(AgentReport):
    agent_name: str = "Fundamental Analyst"
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None  # YoY %
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    valuation: str                          # e.g. "overvalued", "fair", "undervalued"
    earnings_trend: str                     # e.g. "improving", "declining", "stable"


class RiskReport(AgentReport):
    agent_name: str = "Risk Manager"
    var_95: float                           # 1-day 95% Value at Risk (%)
    max_drawdown: float                     # Historical max drawdown (%)
    volatility_30d: float                   # 30-day annualized volatility (%)
    beta: float                             # Beta vs S&P 500
    recommended_position_size: float        # % of portfolio
    kelly_criterion: float                  # Kelly-suggested position size (%)
    risk_level: str                         # "low", "medium", "high", "very high"


# ── Debate ─────────────────────────────────────────────────────────────

class DebateArgument(BaseModel):
    agent_name: str
    position: Signal
    argument: str
    counter_to: Optional[str] = None       # Which agent it's responding to
    revised_confidence: Optional[Confidence] = None


class DebateResult(BaseModel):
    round_number: int = 1
    arguments: list[DebateArgument] = Field(default_factory=list)
    consensus_reached: bool = False
    consensus_signal: Optional[Signal] = None


# ── Final Synthesis ────────────────────────────────────────────────────

class SynthesisReport(BaseModel):
    ticker: str
    final_signal: Signal
    final_confidence: Confidence
    position_size_pct: float               # Recommended portfolio allocation %
    time_horizon: TimeHorizon
    executive_summary: str
    agreement_score: float = Field(ge=0.0, le=1.0)   # 1.0 = all agents agree
    key_risks: list[str] = Field(default_factory=list)
    dissenting_views: list[str] = Field(default_factory=list)
    macro_report: Optional[MacroReport] = None
    technical_report: Optional[TechnicalReport] = None
    fundamental_report: Optional[FundamentalReport] = None
    risk_report: Optional[RiskReport] = None
    debate_result: Optional[DebateResult] = None


# ── Graph State ────────────────────────────────────────────────────────

class GraphState(BaseModel):
    """Shared state object passed through every node in the LangGraph workflow."""

    # Input
    ticker: str
    timeframe: str = "3-6 months"
    user_query: Optional[str] = None

    # Raw market data (populated by data fetcher)
    market_data: dict = Field(default_factory=dict)
    macro_data: dict = Field(default_factory=dict)

    # Agent reports (populated by agent nodes)
    macro_report: Optional[MacroReport] = None
    technical_report: Optional[TechnicalReport] = None
    fundamental_report: Optional[FundamentalReport] = None
    risk_report: Optional[RiskReport] = None

    # Conflict + debate
    needs_debate: bool = False
    conflicts: list[str] = Field(default_factory=list)
    debate_result: Optional[DebateResult] = None

    # Final output
    synthesis: Optional[SynthesisReport] = None

    # Meta
    errors: list[str] = Field(default_factory=list)
    step: str = "init"
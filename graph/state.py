"""MAAT — Graph State

The single shared state object that flows through every node
in the LangGraph workflow. Every node reads from and writes to this.

LangGraph requires state to be a TypedDict or dataclass.
We use a dataclass wrapping our Pydantic GraphState for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from models.schemas import (
    MacroReport, TechnicalReport, FundamentalReport,
    RiskReport, SynthesisReport, DebateResult
)


@dataclass
class MaatState:
    """Shared state passed through every node in the MAAT workflow.

    Fields are populated progressively as nodes execute:
        init        → ticker, timeframe, user_query
        fetch_data  → market_data, macro_data
        run_agents  → macro_report, technical_report, fundamental_report, risk_report
        check       → conflicts, needs_debate
        debate      → debate_result
        synthesize  → synthesis
    """

    # ── Inputs ─────────────────────────────────────────────────────────
    ticker:     str = ""
    timeframe:  str = "3-6 months"
    user_query: Optional[str] = None

    # ── Raw Market Data ─────────────────────────────────────────────────
    market_data:     dict = field(default_factory=dict)   # price + fundamentals
    macro_data:      dict = field(default_factory=dict)   # macro indicators

    # ── Agent Reports ───────────────────────────────────────────────────
    macro_report:       Optional[MacroReport]       = None
    technical_report:   Optional[TechnicalReport]   = None
    fundamental_report: Optional[FundamentalReport] = None
    risk_report:        Optional[RiskReport]        = None

    # ── Conflict & Debate ───────────────────────────────────────────────
    conflicts:    list = field(default_factory=list)
    needs_debate: bool = False
    debate_result: Optional[DebateResult] = None

    # ── Final Output ────────────────────────────────────────────────────
    synthesis: Optional[SynthesisReport] = None

    # ── Metadata ────────────────────────────────────────────────────────
    errors: list = field(default_factory=list)
    step:   str  = "init"
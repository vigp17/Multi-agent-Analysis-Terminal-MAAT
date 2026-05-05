"""MAAT — Graph Nodes

Each function here is a node in the LangGraph workflow.
Nodes receive MaatState, do one job, and return updated MaatState.

Node execution order:
    fetch_data → run_agents → check_conflicts → [debate] → synthesize
"""

from __future__ import annotations

import traceback
from graph.state import MaatState
from tools.data_fetcher import fetch_all
from agents.macro_analyst import MacroAnalyst
from agents.technical_analyst import TechnicalAnalyst
from agents.fundamental_analyst import FundamentalAnalyst
from agents.risk_manager import RiskManager
from agents.cio_synthesizer import CIOSynthesizer
from models.schemas import DebateResult, DebateArgument
from config.settings import settings

# ── Instantiate agents once (reused across nodes) ──────────────────────
macro_agent     = MacroAnalyst()
technical_agent = TechnicalAnalyst()
fundamental_agent = FundamentalAnalyst()
risk_agent      = RiskManager()
cio             = CIOSynthesizer()


# ── Node 1: Fetch Data ─────────────────────────────────────────────────

def fetch_data(state: MaatState) -> MaatState:
    """Fetch all market and macro data for the ticker."""
    print(f"\n[1/5] Fetching data for {state.ticker}...")

    try:
        all_data = fetch_all(state.ticker)
        state.market_data = {
            "price_data":       all_data["price_data"],
            "fundamental_data": all_data["fundamental_data"],
        }
        state.macro_data = all_data["macro_data"]
        state.step = "data_fetched"
        print(f"      ✓ Price data: {all_data['price_data'].get('start_date')} → "
              f"{all_data['price_data'].get('end_date')}")
        print(f"      ✓ Latest price: ${all_data['price_data'].get('latest_price')}")

    except Exception as e:
        state.errors.append(f"fetch_data failed: {e}")
        print(f"      ✗ Data fetch failed: {e}")

    return state


# ── Node 2: Run Agents ─────────────────────────────────────────────────

def run_agents(state: MaatState) -> MaatState:
    """Run all 4 specialist agents in sequence."""
    print(f"\n[2/5] Running specialist agents...")

    data = {**state.market_data, "macro_data": state.macro_data}

    # Macro Analyst
    try:
        print("      → Macro Analyst...")
        state.macro_report = macro_agent.analyze(state.ticker, data)
        print(f"        ✓ Signal: {state.macro_report.signal.value} "
              f"({state.macro_report.confidence.value})")
    except Exception as e:
        state.errors.append(f"macro_agent failed: {e}")
        print(f"        ✗ Failed: {e}")

    # Technical Analyst
    try:
        print("      → Technical Analyst...")
        state.technical_report = technical_agent.analyze(state.ticker, data)
        print(f"        ✓ Signal: {state.technical_report.signal.value} "
              f"({state.technical_report.confidence.value}) | "
              f"Regime: {state.technical_report.regime.value}")
    except Exception as e:
        state.errors.append(f"technical_agent failed: {e}")
        print(f"        ✗ Failed: {e}")

    # Fundamental Analyst
    try:
        print("      → Fundamental Analyst...")
        state.fundamental_report = fundamental_agent.analyze(state.ticker, data)
        print(f"        ✓ Signal: {state.fundamental_report.signal.value} "
              f"({state.fundamental_report.confidence.value}) | "
              f"Valuation: {state.fundamental_report.valuation}")
    except Exception as e:
        state.errors.append(f"fundamental_agent failed: {e}")
        print(f"        ✗ Failed: {e}")

    # Risk Manager
    try:
        print("      → Risk Manager...")
        state.risk_report = risk_agent.analyze(state.ticker, data)
        print(f"        ✓ Signal: {state.risk_report.signal.value} "
              f"({state.risk_report.confidence.value}) | "
              f"Risk: {state.risk_report.risk_level}")
    except Exception as e:
        state.errors.append(f"risk_agent failed: {e}")
        print(f"        ✗ Failed: {e}")

    state.step = "agents_complete"
    return state


# ── Node 3: Check Conflicts ────────────────────────────────────────────

def check_conflicts(state: MaatState) -> MaatState:
    """Detect conflicts between agent signals."""
    print(f"\n[3/5] Checking for conflicts...")

    reports = [
        state.macro_report,
        state.technical_report,
        state.fundamental_report,
        state.risk_report,
    ]

    # Skip if any agent failed
    if any(r is None for r in reports):
        print("      ⚠ Some agents failed — skipping conflict detection")
        state.needs_debate = False
        return state

    conflicts = cio.detect_conflicts(
        state.macro_report,
        state.technical_report,
        state.fundamental_report,
        state.risk_report,
    )

    state.conflicts    = conflicts
    state.needs_debate = len(conflicts) > 0

    if conflicts:
        print(f"      ✗ {len(conflicts)} conflict(s) detected:")
        for c in conflicts:
            print(f"        - {c}")
    else:
        print("      ✓ No conflicts — agents are broadly aligned")

    state.step = "conflicts_checked"
    return state


# ── Node 4: Debate (conditional) ───────────────────────────────────────

def run_debate(state: MaatState) -> MaatState:
    """Run a debate round between conflicting agents."""
    print(f"\n[4/5] Running debate round...")

    arguments = []

    agent_map = {
        "Macro":       (macro_agent,       state.macro_report),
        "Technical":   (technical_agent,   state.technical_report),
        "Fundamental": (fundamental_agent, state.fundamental_report),
        "Risk":        (risk_agent,        state.risk_report),
    }

    for conflict in state.conflicts:
        for name, (agent, report) in agent_map.items():
            if report and name.lower() in conflict.lower():
                try:
                    rebuttal = agent.rebut(
                        report.model_dump(),
                        conflict,
                    )
                    arguments.append(DebateArgument(
                        agent_name=name,
                        position=report.signal,
                        argument=rebuttal,
                        counter_to=conflict,
                    ))
                    print(f"        ✓ {name}: rebuttal submitted")
                except Exception as e:
                    print(f"        ✗ {name}: rebuttal failed — {e}")

    state.debate_result = DebateResult(
        round_number=1,
        arguments=arguments,
        consensus_reached=False,
    )
    state.step = "debate_complete"
    return state


# ── Node 5: Synthesize ─────────────────────────────────────────────────

def synthesize(state: MaatState) -> MaatState:
    """CIO synthesizes all reports into a final recommendation."""
    print(f"\n[5/5] CIO synthesizing final recommendation...")

    required = [
        state.macro_report,
        state.technical_report,
        state.fundamental_report,
        state.risk_report,
    ]

    if any(r is None for r in required):
        state.errors.append("Synthesis skipped — one or more agent reports missing")
        print("      ✗ Cannot synthesize — missing agent reports")
        return state

    try:
        state.synthesis = cio.synthesize(
            ticker=state.ticker,
            timeframe=state.timeframe,
            macro=state.macro_report,
            technical=state.technical_report,
            fundamental=state.fundamental_report,
            risk=state.risk_report,
            conflicts=state.conflicts,
            debate_result=state.debate_result,
        )
        state.step = "complete"
        print(f"      ✓ Final signal: {state.synthesis.final_signal.value} "
              f"({state.synthesis.final_confidence.value})")
        print(f"      ✓ Position size: {state.synthesis.position_size_pct}%")
        print(f"      ✓ Agreement score: {state.synthesis.agreement_score:.0%}")

    except Exception as e:
        state.errors.append(f"synthesis failed: {e}")
        print(f"      ✗ Synthesis failed: {e}")
        traceback.print_exc()

    return state


# ── Routing Function ───────────────────────────────────────────────────

def should_debate(state: MaatState) -> str:
    """Route to debate node if conflicts exist, else skip to synthesis."""
    return "debate" if state.needs_debate else "synthesize"
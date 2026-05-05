from __future__ import annotations
from langgraph.graph import StateGraph, END
from graph.state import MaatState
from graph.nodes import fetch_data, run_agents, check_conflicts, run_debate, synthesize, should_debate

def build_workflow():
    workflow = StateGraph(MaatState)
    workflow.add_node("fetch_data",      fetch_data)
    workflow.add_node("run_agents",      run_agents)
    workflow.add_node("check_conflicts", check_conflicts)
    workflow.add_node("debate",          run_debate)
    workflow.add_node("synthesize",      synthesize)
    workflow.set_entry_point("fetch_data")
    workflow.add_edge("fetch_data",      "run_agents")
    workflow.add_edge("run_agents",      "check_conflicts")
    workflow.add_edge("debate",          "synthesize")
    workflow.add_edge("synthesize",      END)
    workflow.add_conditional_edges("check_conflicts", should_debate, {"debate": "debate", "synthesize": "synthesize"})
    return workflow.compile()

def run_analysis(ticker: str, timeframe: str = "3-6 months", user_query: str = None) -> MaatState:
    from config.settings import settings
    settings.validate()
    print(f"\n{'='*55}")
    print(f"  MAAT — Multi-Agent Analysis")
    print(f"  Ticker   : {ticker.upper()}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Model    : {settings.claude_model}")
    print(f"{'='*55}")
    graph = build_workflow()
    initial_state = MaatState(ticker=ticker.upper(), timeframe=timeframe, user_query=user_query)
    result = graph.invoke(initial_state)
    if isinstance(result, dict):
        final_state = MaatState(**result)
    else:
        final_state = result
    print(f"\n{'='*55}")
    if final_state.synthesis:
        s = final_state.synthesis
        print(f"  FINAL SIGNAL   : {s.final_signal.value}")
        print(f"  CONFIDENCE     : {s.final_confidence.value}")
        print(f"  POSITION SIZE  : {s.position_size_pct}%")
        print(f"  TIME HORIZON   : {s.time_horizon.value}")
        print(f"  AGREEMENT      : {s.agreement_score:.0%}")
        print(f"\n  {s.executive_summary}")
    else:
        print("  Analysis incomplete — check errors below")
        for err in final_state.errors:
            print(f"  ✗ {err}")
    print(f"{'='*55}\n")
    return final_state

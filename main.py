"""MAAT — CLI Entry Point

Run a full multi-agent analysis from the terminal.

Usage:
    python main.py                          # Analyze default ticker (AAPL)
    python main.py --ticker TSLA            # Analyze specific ticker
    python main.py --ticker NVDA --timeframe "6-12 months"
    python main.py --ticker MSFT --query "Should I add to my position?"
    python main.py --ticker AMZN --json     # Output raw JSON
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="MAAT — Multi-Agent Analysis Terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ticker AAPL
  python main.py --ticker TSLA --timeframe "1-3 months"
  python main.py --ticker NVDA --query "Earnings next week, should I hold?"
  python main.py --ticker MSFT --json
        """
    )

    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    parser.add_argument(
        "--timeframe", "-tf",
        type=str,
        default="3-6 months",
        choices=["1-3 months", "3-6 months", "6-12 months"],
        help="Investment horizon (default: 3-6 months)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Optional context or question for the agents"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full synthesis as JSON"
    )

    return parser.parse_args()


def print_full_report(state) -> None:
    """Print a detailed breakdown of all agent reports."""
    s = state.synthesis

    print("\n── AGENT REPORTS ──────────────────────────────────────")

    if state.macro_report:
        m = state.macro_report
        print(f"\n📊 Macro Analyst       : {m.signal.value} ({m.confidence.value})")
        print(f"   Regime             : {m.regime.value}")
        print(f"   Fed Stance         : {m.fed_stance}")
        print(f"   Yield Curve        : {m.yield_curve}")
        print(f"   Recession Prob     : {m.recession_probability:.0%}")
        print(f"   Summary            : {m.summary}")

    if state.technical_report:
        t = state.technical_report
        print(f"\n📈 Technical Analyst   : {t.signal.value} ({t.confidence.value})")
        print(f"   Regime (HMM)       : {t.regime.value}")
        print(f"   Trend              : {t.trend}")
        print(f"   RSI                : {t.rsi}")
        print(f"   MACD               : {t.macd_signal}")
        print(f"   Above 200MA        : {t.above_200ma}")
        print(f"   Support / Resist   : ${t.support_level} / ${t.resistance_level}")
        print(f"   Summary            : {t.summary}")

    if state.fundamental_report:
        f = state.fundamental_report
        print(f"\n🏦 Fundamental Analyst : {f.signal.value} ({f.confidence.value})")
        print(f"   Valuation          : {f.valuation}")
        print(f"   P/E Ratio          : {f.pe_ratio}")
        print(f"   Revenue Growth     : {f.revenue_growth}")
        print(f"   Earnings Trend     : {f.earnings_trend}")
        print(f"   Summary            : {f.summary}")

    if state.risk_report:
        r = state.risk_report
        print(f"\n⚠️  Risk Manager        : {r.signal.value} ({r.confidence.value})")
        print(f"   Risk Level         : {r.risk_level}")
        print(f"   VaR (95%)          : {r.var_95}%")
        print(f"   Max Drawdown       : -{r.max_drawdown}%")
        print(f"   Volatility (30d)   : {r.volatility_30d}%")
        print(f"   Beta               : {r.beta}")
        print(f"   Kelly Position     : {r.kelly_criterion}%")
        print(f"   Summary            : {r.summary}")

    if state.conflicts:
        print(f"\n⚡ Conflicts Detected  :")
        for c in state.conflicts:
            print(f"   - {c}")

    if state.debate_result and state.debate_result.arguments:
        print(f"\n🗣  Debate Round 1     :")
        for arg in state.debate_result.arguments:
            print(f"   [{arg.agent_name}] {arg.argument[:200]}...")

    print(f"\n── SYNTHESIS ──────────────────────────────────────────")
    print(f"   Final Signal       : {s.final_signal.value}")
    print(f"   Confidence         : {s.final_confidence.value}")
    print(f"   Position Size      : {s.position_size_pct}%")
    print(f"   Time Horizon       : {s.time_horizon.value}")
    print(f"   Agreement Score    : {s.agreement_score:.0%}")
    print(f"\n   Executive Summary  :")
    print(f"   {s.executive_summary}")

    if s.key_risks:
        print(f"\n   Key Risks          :")
        for risk in s.key_risks:
            print(f"   • {risk}")

    if s.dissenting_views:
        print(f"\n   Dissenting Views   :")
        for view in s.dissenting_views:
            print(f"   • {view}")

    if state.errors:
        print(f"\n   ⚠ Errors           :")
        for err in state.errors:
            print(f"   - {err}")

    print(f"\n{'='*55}\n")


def main():
    args = parse_args()

    try:
        from graph.workflow import run_analysis
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Make sure you've installed requirements: pip install -r requirements.txt")
        sys.exit(1)

    # Run the analysis
    state = run_analysis(
        ticker=args.ticker,
        timeframe=args.timeframe,
        user_query=args.query,
    )

    if not state.synthesis:
        print("✗ Analysis failed — no synthesis produced")
        sys.exit(1)

    # Output mode
    if args.json:
        print(state.synthesis.model_dump_json(indent=2, default=str))
    else:
        print_full_report(state)


if __name__ == "__main__":
    main()
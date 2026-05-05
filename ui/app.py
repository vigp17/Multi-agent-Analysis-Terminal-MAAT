"""MAAT — Streamlit Dashboard

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="MAAT — Multi-Agent Analysis Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .signal-STRONG_BUY  { color: #00c853; font-weight: bold; font-size: 1.2em; }
    .signal-BUY         { color: #69f0ae; font-weight: bold; font-size: 1.2em; }
    .signal-HOLD        { color: #ffd740; font-weight: bold; font-size: 1.2em; }
    .signal-SELL        { color: #ff6d00; font-weight: bold; font-size: 1.2em; }
    .signal-STRONG_SELL { color: #ff1744; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

SIGNAL_COLORS = {
    "STRONG_BUY":  "#00c853",
    "BUY":         "#69f0ae",
    "HOLD":        "#ffd740",
    "SELL":        "#ff6d00",
    "STRONG_SELL": "#ff1744",
}

SIGNAL_SCORES = {
    "STRONG_BUY": 2, "BUY": 1, "HOLD": 0, "SELL": -1, "STRONG_SELL": -2
}

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ MAAT Settings")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, NVDA").upper().strip()
    timeframe = st.selectbox("Investment Horizon", ["1-3 months", "3-6 months", "6-12 months"], index=1)
    user_query = st.text_area("Additional Context (optional)", placeholder="e.g. Earnings next week. Should I add?", height=80)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    st.markdown("---")
    st.caption("**Agent Weights**")
    st.caption("Macro: 25% | Technical: 25%")
    st.caption("Fundamental: 30% | Risk: 20%")
    st.markdown("---")
    st.caption("Powered by Claude Sonnet 4.6")

# ── Main ───────────────────────────────────────────────────────────────
st.title("📊 MAAT — Multi-Agent Analysis Terminal")
st.markdown("*Four specialized AI agents debate and synthesize a final investment recommendation.*")

if not run_btn:
    st.info("👈 Enter a ticker and click **Run Analysis** to begin.")
    st.markdown("""
**How it works:**
1. 📡 **Data Fetch** — 2 years of price data, fundamentals, and macro indicators
2. 🤖 **4 Agents Run** — Macro, Technical, Fundamental, and Risk analysts each produce a signal
3. ⚡ **Conflict Check** — if agents disagree, a debate round is triggered
4. 🎯 **CIO Synthesis** — Chief Investment Officer produces the final recommendation
    """)
    st.stop()

with st.spinner(f"Running MAAT analysis for **{ticker}**..."):
    try:
        from graph.workflow import run_analysis
        state = run_analysis(ticker=ticker, timeframe=timeframe, user_query=user_query or None)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if not state.synthesis:
    st.error("Analysis did not produce a synthesis. Check your API key and try again.")
    for err in state.errors:
        st.warning(err)
    st.stop()

s = state.synthesis
signal_val = s.final_signal.value

# ── Top Bar ────────────────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown("**Final Signal**")
    st.markdown(f"<span class='signal-{signal_val}'>{signal_val.replace('_',' ')}</span>", unsafe_allow_html=True)
with c2:
    st.metric("Confidence", s.final_confidence.value)
with c3:
    st.metric("Position Size", f"{s.position_size_pct}%")
with c4:
    st.metric("Agreement", f"{s.agreement_score:.0%}")
with c5:
    st.metric("Time Horizon", s.time_horizon.value)

st.markdown("---")

# ── Executive Summary ──────────────────────────────────────────────────
st.subheader("📋 Executive Summary")
st.info(s.executive_summary)

# ── Agent Signals ──────────────────────────────────────────────────────
st.subheader("🤖 Agent Signals")
reports = [
    ("📊 Macro",       state.macro_report),
    ("📈 Technical",   state.technical_report),
    ("🏦 Fundamental", state.fundamental_report),
    ("⚠️ Risk",        state.risk_report),
]

for col, (label, report) in zip(st.columns(4), reports):
    with col:
        if report:
            sig = report.signal.value
            color = SIGNAL_COLORS.get(sig, "#fff")
            st.markdown(f"**{label}**")
            st.markdown(f"<span style='color:{color};font-weight:bold'>{sig.replace('_',' ')}</span>", unsafe_allow_html=True)
            st.caption(f"Confidence: {report.confidence.value}")
            st.caption(report.summary[:120] + "..." if len(report.summary) > 120 else report.summary)
        else:
            st.markdown(f"**{label}**")
            st.warning("No report")

# ── Charts ─────────────────────────────────────────────────────────────
st.subheader("📡 Signal Overview")
col_chart, col_detail = st.columns([1, 1])

with col_chart:
    agent_names  = ["Macro", "Technical", "Fundamental", "Risk", "CIO Final"]
    agent_signals = [r.signal.value if r else "HOLD" for _, r in reports] + [signal_val]
    agent_scores  = [SIGNAL_SCORES.get(sig, 0) for sig in agent_signals]
    colors        = [SIGNAL_COLORS.get(sig, "#ffd740") for sig in agent_signals]

    fig = go.Figure(go.Bar(
        x=agent_names, y=agent_scores,
        marker_color=colors,
        text=[sig.replace("_", " ") for sig in agent_signals],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis=dict(range=[-2.5, 2.5], tickvals=[-2,-1,0,1,2],
                   ticktext=["STRONG SELL","SELL","HOLD","BUY","STRONG BUY"]),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=320, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_detail:
    if state.technical_report and state.risk_report and state.macro_report:
        t, r, m = state.technical_report, state.risk_report, state.macro_report
        df = pd.DataFrame({
            "Metric": ["RSI", "Trend", "HMM Regime", "VaR (95%)", "Max Drawdown", "Volatility", "Beta", "Recession Prob"],
            "Value":  [f"{t.rsi:.1f}", t.trend.title(), t.regime.value,
                       f"{r.var_95}%", f"-{r.max_drawdown}%", f"{r.volatility_30d}%",
                       f"{r.beta:.2f}", f"{m.recession_probability:.0%}"],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

# ── Price Chart ────────────────────────────────────────────────────────
st.subheader(f"📉 Price History — {ticker}")
try:
    prices_df = state.market_data.get("price_data", {}).get("prices")
    if prices_df is not None and not prices_df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Candlestick(
            x=prices_df.index, open=prices_df["Open"], high=prices_df["High"],
            low=prices_df["Low"], close=prices_df["Close"], name="Price",
        ))
        fig2.add_trace(go.Scatter(x=prices_df.index, y=prices_df["Close"].rolling(50).mean(),
                                  name="50 MA", line=dict(color="#ffd740", width=1.5)))
        fig2.add_trace(go.Scatter(x=prices_df.index, y=prices_df["Close"].rolling(200).mean(),
                                  name="200 MA", line=dict(color="#ff6d00", width=1.5)))
        fig2.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            xaxis_rangeslider_visible=False, height=400, margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.warning(f"Price chart unavailable: {e}")

# ── Conflicts & Debate ─────────────────────────────────────────────────
if state.conflicts:
    st.subheader("⚡ Conflicts & Debate")
    for c in state.conflicts:
        st.warning(f"**Conflict:** {c}")
    if state.debate_result and state.debate_result.arguments:
        for arg in state.debate_result.arguments:
            with st.expander(f"🗣 {arg.agent_name} — {arg.position.value}"):
                st.write(arg.argument)

# ── Risks & Dissent ────────────────────────────────────────────────────
col_risk, col_dissent = st.columns(2)
with col_risk:
    if s.key_risks:
        st.subheader("🔴 Key Risks")
        for risk in s.key_risks:
            st.markdown(f"• {risk}")
with col_dissent:
    if s.dissenting_views:
        st.subheader("💬 Dissenting Views")
        for view in s.dissenting_views:
            st.markdown(f"• {view}")

# ── Full Agent Reports ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Full Agent Reports")
for label, report in reports:
    if report:
        with st.expander(f"{label} — {report.signal.value} ({report.confidence.value})"):
            st.write(f"**Summary:** {report.summary}")
            if report.key_points:
                st.write("**Key Points:**")
                for pt in report.key_points:
                    st.markdown(f"  • {pt}")
            if report.risks:
                st.write("**Risks:**")
                for r in report.risks:
                    st.markdown(f"  • {r}")

# ── Errors ─────────────────────────────────────────────────────────────
if state.errors:
    with st.expander("⚠️ Errors / Warnings"):
        for err in state.errors:
            st.warning(err)
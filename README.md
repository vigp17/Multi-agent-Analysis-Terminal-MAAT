# MAAT — Multi-Agent Analysis Terminal

MAAT is a multi-agent AI system that analyzes stocks using four specialized agents — Macro, Technical, Fundamental, and Risk — powered by Claude Sonnet 4.6. When agents disagree, a live debate round is triggered before the Chief Investment Officer synthesizes a final recommendation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)

---

## How It Works

```
fetch_data → run_agents → check_conflicts → [debate] → synthesize
```

1. **Data Fetch** — pulls 2 years of price data, fundamentals, and macro indicators via yfinance
2. **4 Agents Run in parallel** — each produces a signal (STRONG_BUY → STRONG_SELL) with confidence
3. **Conflict Detection** — if agents are on opposite sides, a debate round is triggered
4. **Debate** — conflicting agents argue their position in plain prose
5. **CIO Synthesis** — the Chief Investment Officer weighs all signals and produces a final recommendation

---

## Agents

| Agent | Domain | Key Inputs |
|-------|--------|-----------|
| **Macro Analyst** | Fed policy, yield curve, inflation | VIX, treasury yields, SPY momentum |
| **Technical Analyst** | Price action, momentum, regime | RSI, MACD, Bollinger Bands, HMM regime |
| **Fundamental Analyst** | Valuation, earnings quality | P/E, P/B, revenue growth, margins |
| **Risk Manager** | Portfolio risk, position sizing | VaR, drawdown, beta, Kelly criterion |
| **CIO Synthesizer** | Final recommendation | All agent reports + debate results |

---

## HMM Regime Detection

The Technical Analyst uses a trained Gaussian Hidden Markov Model from the [market-regime-detection](https://github.com/vigp17/market-regime-detection) project to classify the current market into 5 regimes:

| Regime | Days | Ann. Return | Volatility |
|--------|------|-------------|------------|
| Strong Bull | 19.5% | +34.4% | 12.1% |
| Calm Bull | 31.2% | +12.5% | 8.1% |
| Neutral | 25.3% | +7.7% | 15.3% |
| Bear / High Vol | 18.6% | -7.1% | 23.7% |
| Crisis | 5.3% | -51.5% | 51.1% |

---

## Project Structure

```
maat/
├── main.py                  # CLI entry point
├── config/
│   └── settings.py          # API keys, model config, agent weights
├── models/
│   └── schemas.py           # Pydantic data contracts
├── agents/
│   ├── base_agent.py        # Claude API base class
│   ├── macro_analyst.py
│   ├── technical_analyst.py
│   ├── fundamental_analyst.py
│   ├── risk_manager.py
│   └── cio_synthesizer.py
├── tools/
│   ├── data_fetcher.py      # yfinance + FRED wrappers
│   ├── technical_indicators.py
│   ├── hmm_regime.py        # HMM integration
│   └── risk_metrics.py      # VaR, Kelly, Sharpe
├── graph/
│   ├── state.py             # LangGraph state
│   ├── nodes.py             # Graph nodes
│   └── workflow.py          # Graph assembly + routing
└── ui/
    └── app.py               # Streamlit dashboard
```

---

## Setup

```bash
git clone https://github.com/vigp17/Multi-Agent-Analysis-Terminal-MAAT.git
cd Multi-Agent-Analysis-Terminal-MAAT

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your-key-here
CLAUDE_MODEL=claude-sonnet-4-6
```

---

## Usage

**CLI:**
```bash
python main.py --ticker AAPL
python main.py --ticker TSLA --timeframe "6-12 months"
python main.py --ticker NVDA --query "Earnings next week, should I hold?"
python main.py --ticker MSFT --json
```

**Streamlit UI:**
```bash
streamlit run ui/app.py
```

---

## Sample Output

```
=======================================================
  MAAT — Multi-Agent Analysis
  Ticker   : TSLA
  Timeframe: 3-6 months
  Model    : claude-sonnet-4-6
=======================================================

[1/5] Fetching data for TSLA...
      ✓ Latest price: $392.51

[2/5] Running specialist agents...
      ✓ Macro Analyst    : BUY (MEDIUM)
      ✓ Technical Analyst: HOLD (LOW) | Regime: SIDEWAYS
      ✓ Fundamental      : SELL (HIGH) | Valuation: overvalued
      ✓ Risk Manager     : HOLD (MEDIUM) | Risk: very high

[3/5] Conflicts detected: Macro is BUY but Fundamental is SELL
[4/5] Debate round triggered...
[5/5] CIO synthesizing...

  FINAL SIGNAL  : SELL
  CONFIDENCE    : MEDIUM
  POSITION SIZE : 0.0%
  AGREEMENT     : 50%
=======================================================
```

---

## Tech Stack

- **Claude Sonnet 4.6** — powers all 5 agents
- **LangGraph** — orchestrates the multi-agent workflow
- **hmmlearn** — Gaussian HMM for market regime detection
- **yfinance** — market data and fundamentals
- **Streamlit + Plotly** — interactive dashboard
- **Pydantic** — typed data contracts between agents

---

## Author

**Vignesh Pai**   
[GitHub](https://github.com/vigp17)

---

## License

MIT

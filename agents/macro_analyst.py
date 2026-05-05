"""MAAT — Macro Analyst Agent

Analyzes macroeconomic conditions: Fed policy, yield curve,
inflation, market regime, and their implications for the ticker.

Output: MacroReport (Pydantic schema)
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.schemas import MacroReport, Signal, Confidence, MarketRegime


class MacroAnalyst(BaseAgent):

    def __init__(self):
        super().__init__(name="Macro Analyst")

    # ── System Prompt ──────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return """You are a Senior Macro Analyst at a top-tier hedge fund.
Your job is to assess the macroeconomic environment and its implications
for a specific stock or asset.

You specialize in:
- Federal Reserve policy and interest rate cycles
- Yield curve analysis (normal, flat, inverted)
- Inflation trends and their sector impact
- Business cycle positioning (expansion, peak, contraction, trough)
- Risk-on vs risk-off market sentiment

You always respond with a valid JSON object — no markdown, no preamble.
Your signal reflects whether macro conditions SUPPORT or OPPOSE holding this asset.

JSON format:
{
  "ticker": "string",
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": "HIGH | MEDIUM | LOW",
  "regime": "BULL | BEAR | SIDEWAYS",
  "fed_stance": "string (hawkish | dovish | neutral | on hold)",
  "yield_curve": "string (normal | flat | inverted | steepening)",
  "inflation_trend": "string (rising | falling | stable | volatile)",
  "recession_probability": float between 0.0 and 1.0,
  "summary": "string (2-3 sentences)",
  "key_points": ["string", "string", "string"],
  "risks": ["string", "string"]
}"""

    # ── Analysis ───────────────────────────────────────────────────────

    def analyze(self, ticker: str, data: dict) -> MacroReport:
        """Run macro analysis and return a validated MacroReport.

        Args:
            ticker : stock ticker symbol
            data   : dict containing macro_data from data_fetcher
        """
        macro_data = data.get("macro_data", {})
        price_data = data.get("price_data", {})

        user_prompt = f"""
Analyze the macroeconomic environment for {ticker}.

## Current Macro Data
{self._format_data_block(macro_data)}

## Asset Context
- Ticker: {ticker}
- Latest Price: {price_data.get('latest_price', 'N/A')}
- 52-Week High: {price_data.get('52w_high', 'N/A')}
- 52-Week Low: {price_data.get('52w_low', 'N/A')}

## Your Task
1. Assess the current macroeconomic regime (Bull / Bear / Sideways)
2. Evaluate Fed policy stance based on yield levels
3. Interpret the yield curve spread ({macro_data.get('yield_curve_spread', 'N/A')})
4. Determine if macro conditions support or oppose holding {ticker}
5. Assign a signal and confidence level

Respond ONLY with the JSON object. No explanation outside the JSON.
"""

        raw = self._call_claude_json(user_prompt)

        # Normalize enums
        raw["signal"]     = Signal(raw["signal"])
        raw["confidence"] = Confidence(raw["confidence"])
        raw["regime"]     = MarketRegime(raw["regime"])
        raw["ticker"]     = ticker
        raw["agent_name"] = "Macro Analyst"

        # Clamp recession probability
        raw["recession_probability"] = max(0.0, min(1.0, float(raw.get("recession_probability", 0.3))))

        return MacroReport(**raw)
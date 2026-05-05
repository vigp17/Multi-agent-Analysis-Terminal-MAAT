"""MAAT — Technical Analyst Agent

Analyzes price action, technical indicators, and HMM regime detection.
Combines quantitative signals with Claude's pattern interpretation.

Output: TechnicalReport (Pydantic schema)
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.schemas import TechnicalReport, Signal, Confidence, MarketRegime
from tools.technical_indicators import compute_all
from tools.hmm_regime import detect_regime


class TechnicalAnalyst(BaseAgent):

    def __init__(self):
        super().__init__(name="Technical Analyst")

    # ── System Prompt ──────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return """You are a Senior Technical Analyst at a quantitative hedge fund.
Your job is to analyze price action, technical indicators, and market regimes
to determine the likely short-to-medium term direction of an asset.

You specialize in:
- Trend analysis using moving averages (50MA, 200MA)
- Momentum indicators: RSI, MACD
- Volatility: Bollinger Bands
- Support and resistance levels
- Hidden Markov Model regime detection (Bull / Bear / Sideways)
- Volume confirmation

You always respond with a valid JSON object — no markdown, no preamble.
Base your signal purely on price action and technical structure.

JSON format:
{
  "ticker": "string",
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": "HIGH | MEDIUM | LOW",
  "regime": "BULL | BEAR | SIDEWAYS",
  "trend": "string (uptrend | downtrend | ranging | mixed)",
  "rsi": float between 0 and 100,
  "macd_signal": "string (e.g. bullish crossover, bearish weakening)",
  "support_level": float,
  "resistance_level": float,
  "above_200ma": boolean,
  "summary": "string (2-3 sentences)",
  "key_points": ["string", "string", "string"],
  "risks": ["string", "string"]
}"""

    # ── Analysis ───────────────────────────────────────────────────────

    def analyze(self, ticker: str, data: dict) -> TechnicalReport:
        """Run technical analysis and return a validated TechnicalReport.

        Args:
            ticker : stock ticker symbol
            data   : dict containing price_data from data_fetcher
        """
        price_data = data.get("price_data", {})
        prices     = price_data.get("prices")
        returns    = price_data.get("returns")

        if prices is None or prices.empty:
            raise ValueError(f"No price data available for {ticker}")

        # Compute all technical indicators
        indicators = compute_all(prices)

        # Run HMM regime detection
        regime_result = detect_regime(prices)
        current_regime = regime_result.get("current_regime", MarketRegime.SIDEWAYS)
        regime_confidence = regime_result.get("confidence", 0.0)
        regime_stable = regime_result.get("regime_stable", False)

        user_prompt = f"""
Analyze the technical setup for {ticker}.

## Price Summary
- Latest Price   : {price_data.get('latest_price', 'N/A')}
- 52-Week High   : {price_data.get('52w_high', 'N/A')}
- 52-Week Low    : {price_data.get('52w_low', 'N/A')}

## Technical Indicators
- Trend          : {indicators['trend']}
- Above 200MA    : {indicators['above_200ma']}
- RSI (14)       : {indicators['rsi']}
- MACD Signal    : {indicators['macd_signal']}
- MACD Values    : {indicators['macd']}
- Bollinger Bands: {indicators['bollinger']}
- Support        : {indicators['support_resistance']['support']}
- Resistance     : {indicators['support_resistance']['resistance']}
- Volume Trend   : {indicators['volume_trend']}

## Price Momentum
{self._format_data_block(indicators['momentum'])}

## HMM Regime Detection
- Current Regime : {current_regime.value if hasattr(current_regime, 'value') else current_regime}
- Regime Confidence: {regime_confidence:.1%}
- Regime Stable  : {regime_stable}
- Recent Regimes : {regime_result.get('recent_regimes', [])}
- Regime Distribution: {regime_result.get('regime_counts', {})}

## Your Task
1. Interpret the technical indicators holistically
2. Weight the HMM regime detection in your analysis
3. Identify the key levels to watch (support/resistance)
4. Determine if the technical setup is bullish, bearish, or neutral
5. Assign a signal and confidence level

Respond ONLY with the JSON object. No explanation outside the JSON.
"""

        raw = self._call_claude_json(user_prompt)

        # Normalize enums
        raw["signal"]     = Signal(raw["signal"])
        raw["confidence"] = Confidence(raw["confidence"])
        raw["regime"]     = MarketRegime(raw.get("regime", current_regime.value if hasattr(current_regime, "value") else "SIDEWAYS"))
        raw["ticker"]     = ticker
        raw["agent_name"] = "Technical Analyst"

        # Ensure numeric fields
        raw["rsi"]              = float(raw.get("rsi", indicators["rsi"]))
        raw["support_level"]    = float(raw.get("support_level", indicators["support_resistance"]["support"]))
        raw["resistance_level"] = float(raw.get("resistance_level", indicators["support_resistance"]["resistance"]))
        raw["above_200ma"]      = bool(raw.get("above_200ma", indicators["above_200ma"]))

        return TechnicalReport(**raw)
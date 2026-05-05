"""MAAT — Risk Manager Agent

Computes quantitative risk metrics and determines appropriate
position sizing. Acts as the portfolio's risk gatekeeper.

Output: RiskReport (Pydantic schema)
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.schemas import RiskReport, Signal, Confidence
from tools.risk_metrics import compute_all


class RiskManager(BaseAgent):

    def __init__(self):
        super().__init__(name="Risk Manager")

    # ── System Prompt ──────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return """You are a Senior Risk Manager at a multi-strategy hedge fund.
Your job is to assess the risk profile of an asset and recommend
an appropriate position size given the quantitative risk metrics.

You specialize in:
- Value at Risk (VaR) interpretation
- Drawdown analysis and tail risk
- Volatility regimes and beta exposure
- Kelly Criterion position sizing
- Sharpe Ratio and risk-adjusted returns
- Correlation and portfolio concentration risk

You always respond with a valid JSON object — no markdown, no preamble.
Your signal reflects whether the RISK PROFILE supports taking a position.
A high-risk asset may warrant HOLD even if other agents are bullish.

JSON format:
{
  "ticker": "string",
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": "HIGH | MEDIUM | LOW",
  "var_95": float,
  "max_drawdown": float,
  "volatility_30d": float,
  "beta": float,
  "recommended_position_size": float,
  "kelly_criterion": float,
  "risk_level": "string (low | medium | high | very high)",
  "summary": "string (2-3 sentences)",
  "key_points": ["string", "string", "string"],
  "risks": ["string", "string"]
}"""

    # ── Analysis ───────────────────────────────────────────────────────

    def analyze(self, ticker: str, data: dict) -> RiskReport:
        """Run risk analysis and return a validated RiskReport.

        Args:
            ticker : stock ticker symbol
            data   : dict containing price_data from data_fetcher
        """
        price_data = data.get("price_data", {})
        prices     = price_data.get("prices")
        returns    = price_data.get("returns")

        if prices is None or returns is None:
            raise ValueError(f"No price/returns data available for {ticker}")

        # Compute all risk metrics locally
        metrics = compute_all(prices, returns)

        user_prompt = f"""
Assess the risk profile and position sizing for {ticker}.

## Asset Summary
- Ticker        : {ticker}
- Latest Price  : {price_data.get('latest_price', 'N/A')}
- 52-Week High  : {price_data.get('52w_high', 'N/A')}
- 52-Week Low   : {price_data.get('52w_low', 'N/A')}
- Avg Volume    : {price_data.get('avg_volume', 'N/A')}

## Quantitative Risk Metrics
- VaR (95%, 1-day)     : {metrics['var_95']}%
- Max Drawdown         : -{metrics['max_drawdown']}%
- Volatility (30d ann) : {metrics['volatility_30d']}%
- Beta vs S&P 500      : {metrics['beta']}
- Sharpe Ratio         : {metrics['sharpe_ratio']}
- Kelly Criterion      : {metrics['kelly_criterion']}%
- Risk Level           : {metrics['risk_level']}

## Your Task
1. Interpret the VaR — is a {metrics['var_95']}% daily loss at 95% confidence acceptable?
2. Assess the max drawdown of -{metrics['max_drawdown']}% — is it within tolerable limits?
3. Evaluate beta of {metrics['beta']} — how much market risk does this add to a portfolio?
4. Validate or adjust the Kelly position size of {metrics['kelly_criterion']}%
5. Assign a risk-based signal and confidence level
6. Set recommended_position_size (0–10% of portfolio)

Note: Even if the asset looks attractive, assign SELL or HOLD if risk is excessive.
Respond ONLY with the JSON object. No explanation outside the JSON.
"""

        raw = self._call_claude_json(user_prompt)

        # Normalize enums
        raw["signal"]     = Signal(raw["signal"])
        raw["confidence"] = Confidence(raw["confidence"])
        raw["ticker"]     = ticker
        raw["agent_name"] = "Risk Manager"

        # Fallback to computed metrics if Claude omits fields
        raw["var_95"]                   = float(raw.get("var_95", metrics["var_95"]))
        raw["max_drawdown"]             = float(raw.get("max_drawdown", metrics["max_drawdown"]))
        raw["volatility_30d"]           = float(raw.get("volatility_30d", metrics["volatility_30d"]))
        raw["beta"]                     = float(raw.get("beta", metrics["beta"]))
        raw["kelly_criterion"]          = float(raw.get("kelly_criterion", metrics["kelly_criterion"]))
        raw["recommended_position_size"]= float(raw.get("recommended_position_size", metrics["recommended_position"]))
        raw["risk_level"]               = raw.get("risk_level", metrics["risk_level"])

        return RiskReport(**raw)
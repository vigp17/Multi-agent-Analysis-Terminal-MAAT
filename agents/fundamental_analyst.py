"""MAAT — Fundamental Analyst Agent

Analyzes company financials, valuation metrics, and earnings quality
to determine if the stock is fairly priced relative to its fundamentals.

Output: FundamentalReport (Pydantic schema)
"""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.schemas import FundamentalReport, Signal, Confidence


class FundamentalAnalyst(BaseAgent):

    def __init__(self):
        super().__init__(name="Fundamental Analyst")

    # ── System Prompt ──────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return """You are a Senior Fundamental Analyst at a value-oriented hedge fund.
Your job is to assess a company's intrinsic value, financial health,
and earnings quality to determine if the stock is worth buying, holding, or selling.

You specialize in:
- Valuation multiples: P/E, Forward P/E, P/B, P/S, PEG
- Profitability: margins, ROE, ROA
- Growth: revenue growth, earnings growth trends
- Financial health: debt-to-equity, current ratio, free cash flow
- Analyst sentiment: price targets vs current price

You always respond with a valid JSON object — no markdown, no preamble.
Your signal reflects whether the stock is attractively valued given its fundamentals.

JSON format:
{
  "ticker": "string",
  "signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "confidence": "HIGH | MEDIUM | LOW",
  "pe_ratio": float or null,
  "pb_ratio": float or null,
  "revenue_growth": float or null,
  "profit_margin": float or null,
  "debt_to_equity": float or null,
  "valuation": "string (overvalued | fair | undervalued)",
  "earnings_trend": "string (improving | stable | declining)",
  "summary": "string (2-3 sentences)",
  "key_points": ["string", "string", "string"],
  "risks": ["string", "string"]
}"""

    # ── Analysis ───────────────────────────────────────────────────────

    def analyze(self, ticker: str, data: dict) -> FundamentalReport:
        """Run fundamental analysis and return a validated FundamentalReport.

        Args:
            ticker : stock ticker symbol
            data   : dict containing fundamental_data from data_fetcher
        """
        fund_data  = data.get("fundamental_data", {})
        price_data = data.get("price_data", {})

        if fund_data.get("error"):
            raise ValueError(f"Fundamental data error for {ticker}: {fund_data['error']}")

        # Compute upside/downside vs analyst target
        latest_price   = price_data.get("latest_price")
        analyst_target = fund_data.get("analyst_target")
        upside_pct     = None
        if latest_price and analyst_target:
            upside_pct = round((analyst_target / latest_price - 1) * 100, 1)

        user_prompt = f"""
Analyze the fundamental value of {ticker}.

## Company Overview
- Name     : {fund_data.get('company_name', ticker)}
- Sector   : {fund_data.get('sector', 'Unknown')}
- Industry : {fund_data.get('industry', 'Unknown')}
- Market Cap: {fund_data.get('market_cap', 'N/A')}

## Valuation Multiples
- Trailing P/E   : {fund_data.get('pe_ratio', 'N/A')}
- Forward P/E    : {fund_data.get('forward_pe', 'N/A')}
- Price/Book     : {fund_data.get('pb_ratio', 'N/A')}
- Price/Sales    : {fund_data.get('ps_ratio', 'N/A')}
- PEG Ratio      : {fund_data.get('peg_ratio', 'N/A')}

## Growth Metrics
- Revenue Growth (YoY) : {fund_data.get('revenue_growth', 'N/A')}
- Earnings Growth (YoY): {fund_data.get('earnings_growth', 'N/A')}

## Profitability
- Profit Margin   : {fund_data.get('profit_margin', 'N/A')}
- Operating Margin: {fund_data.get('operating_margin', 'N/A')}
- Return on Equity: {fund_data.get('roe', 'N/A')}
- Return on Assets: {fund_data.get('roa', 'N/A')}

## Financial Health
- Debt / Equity  : {fund_data.get('debt_to_equity', 'N/A')}
- Current Ratio  : {fund_data.get('current_ratio', 'N/A')}
- Free Cash Flow : {fund_data.get('free_cashflow', 'N/A')}
- Dividend Yield : {fund_data.get('dividend_yield', 'N/A')}

## Analyst Consensus
- Recommendation  : {fund_data.get('recommendation', 'N/A')}
- Price Target    : {analyst_target}
- Current Price   : {latest_price}
- Implied Upside  : {upside_pct}%

## Your Task
1. Assess whether the stock is overvalued, fairly valued, or undervalued
2. Evaluate the quality and sustainability of earnings growth
3. Check financial health (debt load, cash generation)
4. Factor in analyst consensus vs your own view
5. Assign a signal and confidence level

Respond ONLY with the JSON object. No explanation outside the JSON.
"""

        raw = self._call_claude_json(user_prompt)

        # Normalize enums
        raw["signal"]     = Signal(raw["signal"])
        raw["confidence"] = Confidence(raw["confidence"])
        raw["ticker"]     = ticker
        raw["agent_name"] = "Fundamental Analyst"

        # Safe float conversion for optional fields
        for field in ["pe_ratio", "pb_ratio", "revenue_growth", "profit_margin", "debt_to_equity"]:
            val = raw.get(field)
            raw[field] = float(val) if val is not None else None

        return FundamentalReport(**raw)
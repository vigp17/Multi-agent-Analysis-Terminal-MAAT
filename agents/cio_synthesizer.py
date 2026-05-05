"""MAAT — CIO Synthesizer Agent

The Chief Investment Officer that synthesizes all 4 agent reports
into a final investment recommendation.

Responsibilities:
    - Weight each agent's signal by configured weights
    - Detect conflicts between agents
    - Incorporate debate results if a debate round occurred
    - Produce a final signal, position size, and executive summary

Output: SynthesisReport (Pydantic schema)
"""

from __future__ import annotations

import json
from agents.base_agent import BaseAgent
from models.schemas import (
    SynthesisReport, Signal, Confidence, TimeHorizon,
    MacroReport, TechnicalReport, FundamentalReport, RiskReport,
    DebateResult
)
from config.settings import settings


# ── Signal Scoring ─────────────────────────────────────────────────────

SIGNAL_SCORES = {
    Signal.STRONG_BUY:  2,
    Signal.BUY:         1,
    Signal.HOLD:        0,
    Signal.SELL:       -1,
    Signal.STRONG_SELL:-2,
}

CONFIDENCE_MULTIPLIERS = {
    Confidence.HIGH:   1.0,
    Confidence.MEDIUM: 0.7,
    Confidence.LOW:    0.4,
}


def score_to_signal(score: float) -> Signal:
    if score >= 1.5:   return Signal.STRONG_BUY
    elif score >= 0.5: return Signal.BUY
    elif score >= -0.5: return Signal.HOLD
    elif score >= -1.5: return Signal.SELL
    else:              return Signal.STRONG_SELL


def compute_agreement_score(signals: list[Signal]) -> float:
    scores = [SIGNAL_SCORES[s] for s in signals]
    spread = max(scores) - min(scores)
    return round(1.0 - (spread / 4.0), 3)


class CIOSynthesizer(BaseAgent):

    def __init__(self):
        super().__init__(name="CIO Synthesizer")

    # ── System Prompt ──────────────────────────────────────────────────

    def system_prompt(self) -> str:
        return """You are the Chief Investment Officer of a multi-strategy hedge fund.
You receive reports from four specialized analysts — Macro, Technical,
Fundamental, and Risk — and synthesize them into a final investment recommendation.

Your responsibilities:
- Weigh each analyst's view based on their confidence and the current market context
- Resolve disagreements through reasoned judgment, not averaging
- Produce a clear, actionable recommendation with position sizing
- Write an executive summary a portfolio manager can act on immediately

You always respond with a valid JSON object — no markdown, no preamble.

JSON format:
{
  "ticker": "string",
  "final_signal": "STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL",
  "final_confidence": "HIGH | MEDIUM | LOW",
  "position_size_pct": float between 0.0 and 10.0,
  "time_horizon": "1-3 months | 3-6 months | 6-12 months",
  "executive_summary": "string (3-5 sentences, actionable)",
  "key_risks": ["string", "string", "string"],
  "dissenting_views": ["string"]
}"""

    # ── Required abstract method stub ──────────────────────────────────

    def analyze(self, ticker: str, data: dict) -> None:
        """Not used — CIO uses synthesize() instead."""
        raise NotImplementedError("CIOSynthesizer uses synthesize(), not analyze()")

    # ── Conflict Detection ─────────────────────────────────────────────

    def detect_conflicts(
        self,
        macro: MacroReport,
        technical: TechnicalReport,
        fundamental: FundamentalReport,
        risk: RiskReport,
    ) -> list[str]:
        reports = {
            "Macro":       macro,
            "Technical":   technical,
            "Fundamental": fundamental,
            "Risk":        risk,
        }
        conflicts = []
        names  = list(reports.keys())
        values = list(reports.values())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score_i = SIGNAL_SCORES[values[i].signal]
                score_j = SIGNAL_SCORES[values[j].signal]
                if score_i > 0 and score_j < 0:
                    conflicts.append(
                        f"{names[i]} is {values[i].signal.value} but "
                        f"{names[j]} is {values[j].signal.value}"
                    )
                elif score_i < 0 and score_j > 0:
                    conflicts.append(
                        f"{names[j]} is {values[j].signal.value} but "
                        f"{names[i]} is {values[i].signal.value}"
                    )
        return conflicts

    # ── Weighted Score ─────────────────────────────────────────────────

    def compute_weighted_score(
        self,
        macro: MacroReport,
        technical: TechnicalReport,
        fundamental: FundamentalReport,
        risk: RiskReport,
    ) -> float:
        def weighted(report, weight):
            score      = SIGNAL_SCORES[report.signal]
            multiplier = CONFIDENCE_MULTIPLIERS[report.confidence]
            return score * multiplier * weight

        total = (
            weighted(macro,       settings.weight_macro)       +
            weighted(technical,   settings.weight_technical)   +
            weighted(fundamental, settings.weight_fundamental) +
            weighted(risk,        settings.weight_risk)
        )
        total_weight = (
            settings.weight_macro + settings.weight_technical +
            settings.weight_fundamental + settings.weight_risk
        )
        return round(total / total_weight, 4)

    # ── Synthesis ──────────────────────────────────────────────────────

    def synthesize(
        self,
        ticker: str,
        timeframe: str,
        macro: MacroReport,
        technical: TechnicalReport,
        fundamental: FundamentalReport,
        risk: RiskReport,
        conflicts: list[str],
        debate_result: DebateResult | None = None,
    ) -> SynthesisReport:

        weighted_score  = self.compute_weighted_score(macro, technical, fundamental, risk)
        quant_signal    = score_to_signal(weighted_score)
        agreement_score = compute_agreement_score([
            macro.signal, technical.signal, fundamental.signal, risk.signal
        ])

        debate_section = ""
        if debate_result and debate_result.arguments:
            debate_section = f"""
## Debate Round Results
Consensus reached: {debate_result.consensus_reached}
Arguments:
{json.dumps([a.model_dump() for a in debate_result.arguments], indent=2, default=str)}
"""

        user_prompt = f"""
Synthesize the following analyst reports for {ticker} into a final recommendation.

## Quantitative Pre-Score
Weighted signal score : {weighted_score:+.3f} → suggests {quant_signal.value}
Agreement score       : {agreement_score:.1%}
Conflicts detected    : {len(conflicts)}

## Agent Signals Summary
- Macro Analyst    : {macro.signal.value} ({macro.confidence.value}) — {macro.summary}
- Technical Analyst: {technical.signal.value} ({technical.confidence.value}) — {technical.summary}
- Fundamental      : {fundamental.signal.value} ({fundamental.confidence.value}) — {fundamental.summary}
- Risk Manager     : {risk.signal.value} ({risk.confidence.value}) — {risk.summary}

## Conflicts
{chr(10).join(f'- {c}' for c in conflicts) if conflicts else 'None — agents are broadly aligned'}

{debate_section}

## Key Data Points
- Current Regime     : {technical.regime.value}
- RSI                : {technical.rsi}
- VaR (95%)          : {risk.var_95}%
- Max Drawdown       : {risk.max_drawdown}%
- Recession Prob     : {macro.recession_probability:.0%}
- Valuation          : {fundamental.valuation}
- Kelly Position     : {risk.kelly_criterion}%
- Investment Horizon : {timeframe}

## Your Task
1. Start from the quantitative score ({weighted_score:+.3f}) but override it with judgment if warranted
2. Resolve any conflicts — explain which view you side with and why
3. Set a final position size (0–10% of portfolio), informed by Risk Manager's recommendation
4. Write a clear executive summary a PM can act on today

Respond ONLY with the JSON object. No explanation outside the JSON.
"""

        raw = self._call_claude_json(user_prompt)

        raw["final_signal"]     = Signal(raw["final_signal"])
        raw["final_confidence"] = Confidence(raw["final_confidence"])
        raw["ticker"]           = ticker

        horizon_map = {
            "1-3 months":  TimeHorizon.SHORT,
            "3-6 months":  TimeHorizon.MEDIUM,
            "6-12 months": TimeHorizon.LONG,
            "1-2 years":   TimeHorizon.LONG,
        }
        raw["time_horizon"] = horizon_map.get(
            raw.get("time_horizon", timeframe), TimeHorizon.MEDIUM
        )

        raw["position_size_pct"] = max(0.0, min(10.0, float(raw.get("position_size_pct", 0.0))))
        raw["agreement_score"]   = agreement_score
        raw["macro_report"]      = macro
        raw["technical_report"]  = technical
        raw["fundamental_report"]= fundamental
        raw["risk_report"]       = risk
        raw["debate_result"]     = debate_result

        return SynthesisReport(**raw)
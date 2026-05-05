"""MAAT — Configuration & Settings"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── API Keys ────────────────────────────────────────────────────────
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")

    # ── Claude Model ────────────────────────────────────────────────────
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6-20250929")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # ── Analysis Defaults ───────────────────────────────────────────────
    default_ticker: str = "AAPL"
    default_timeframe: str = "3-6 months"
    price_history_period: str = "2y"
    price_history_interval: str = "1d"

    # ── HMM Regime Detection ────────────────────────────────────────────
    hmm_n_states: int = 3
    hmm_n_iter: int = 100
    hmm_covariance_type: str = "full"

    # ── Agent Weights (CIO synthesis) ───────────────────────────────────
    weight_macro: float = 0.25
    weight_technical: float = 0.25
    weight_fundamental: float = 0.30
    weight_risk: float = 0.20

    # ── Conflict Detection ──────────────────────────────────────────────
    conflict_threshold: float = 0.4

    # ── Risk Defaults ───────────────────────────────────────────────────
    var_confidence: float = 0.95
    max_position_size_pct: float = 10.0
    kelly_fraction: float = 0.5

    def validate(self):
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        return True


settings = Settings()
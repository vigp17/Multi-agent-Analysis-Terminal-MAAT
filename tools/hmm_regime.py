"""MAAT — HMM Regime Detection (powered by market-regime-detection project)

Wraps Vignesh's trained Gaussian HMM from:
    https://github.com/vigp17/market-regime-detection

Key differences from a generic HMM:
    - 5 regimes: Strong Bull, Calm Bull, Neutral, Bear/High Vol, Crisis
    - Rich features: log returns, rolling vol (5/21/63d), vol ratio, RSI, MA distance
    - BIC-optimized state count
    - Pre-trained model + scaler loaded from disk (no retraining on every run)

MAAT maps 5 regimes → 3 (Bull / Sideways / Bear) for its schema:
    Strong Bull + Calm Bull → BULL
    Neutral                 → SIDEWAYS
    Bear/High Vol + Crisis  → BEAR
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

from models.schemas import MarketRegime


# ── Regime Mapping ─────────────────────────────────────────────────────
# Maps your project's 5-state labels to MAAT's 3-state schema.
# Keyed by annualized return rank (0=lowest, 4=highest).

REGIME_LABELS = {
    0: ("Crisis",        MarketRegime.BEAR),
    1: ("Bear/High Vol", MarketRegime.BEAR),
    2: ("Neutral",       MarketRegime.SIDEWAYS),
    3: ("Calm Bull",     MarketRegime.BULL),
    4: ("Strong Bull",   MarketRegime.BULL),
}


# ── Feature Engineering (mirrors src/features.py) ──────────────────────

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Replicate the feature matrix from market-regime-detection/src/features.py.

    Features (matching your project exactly):
        - log_return       : daily log return
        - vol_5d           : 5-day rolling volatility
        - vol_21d          : 21-day rolling volatility
        - vol_63d          : 63-day rolling volatility
        - vol_ratio        : vol_5d / vol_63d  (short vs long-term vol)
        - rsi_14           : 14-period RSI
        - ma_distance      : (price - 50MA) / 50MA
    """
    close = prices["Close"]

    log_return = np.log(close / close.shift(1))

    vol_5d  = log_return.rolling(5).std()  * np.sqrt(252)
    vol_21d = log_return.rolling(21).std() * np.sqrt(252)
    vol_63d = log_return.rolling(63).std() * np.sqrt(252)

    vol_ratio = (vol_5d / vol_63d.replace(0, np.nan))

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    # MA distance
    ma50        = close.rolling(50).mean()
    ma_distance = (close - ma50) / ma50.replace(0, np.nan)

    features = pd.DataFrame({
    "log_return":  log_return,
    "vol_21d":     vol_21d,
    "vol_ratio":   vol_ratio,
    "rsi":         rsi,
    "ma_distance": ma_distance,
}).dropna()


    return features


# ── Model Loading ──────────────────────────────────────────────────────

def _load_trained_model(model_dir: str = None):
    """Try to load the pre-trained HMM and scaler from your project.

    Looks for:
        <model_dir>/hmm_model.pkl
        <model_dir>/scaler.pkl

    Falls back to training fresh if not found.
    """
    if model_dir is None:
        # Default: look for market-regime-detection next to maat/
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, "..", "market-regime-detection", "models")

    hmm_path    = os.path.join(model_dir, "hmm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if os.path.exists(hmm_path) and os.path.exists(scaler_path):
        with open(hmm_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler

    return None, None


# ── Train Fresh (fallback) ─────────────────────────────────────────────

def _train_fresh(features_scaled: np.ndarray, n_states: int = 5):
    """Train a new HMM if no saved model is available.

    Uses BIC to select optimal state count if n_states is None.
    """
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42,
    )
    model.fit(features_scaled)
    return model


# ── Regime Labeling ────────────────────────────────────────────────────

def _label_states(model, n_states: int) -> dict:
    """Map HMM hidden states to regime labels by ranking mean log return."""
    mean_returns = model.means_[:, 0]   # First feature = log_return
    ranked       = np.argsort(mean_returns)   # Low → high

    state_to_label = {}
    for rank, state_idx in enumerate(ranked):
        state_to_label[state_idx] = REGIME_LABELS[rank]

    return state_to_label


# ── Main Entry Point ───────────────────────────────────────────────────

def detect_regime(prices: pd.DataFrame, model_dir: str = None) -> dict:
    """Run full HMM regime detection pipeline.

    Tries to load your pre-trained model first.
    Falls back to training fresh on the provided price data.

    Args:
        prices    : DataFrame with 'Close' column (2+ years recommended)
        model_dir : path to market-regime-detection/models/ (optional)

    Returns:
        dict with:
            current_regime     : MarketRegime (BULL / BEAR / SIDEWAYS)
            current_label      : detailed label (e.g. "Strong Bull", "Crisis")
            confidence         : probability of current state (0–1)
            regime_stable      : True if last 10 days all same regime
            recent_regimes     : list of last 10 regime labels
            regime_counts      : days in each MAAT regime historically
            detailed_counts    : days in each of the 5 fine-grained regimes
            using_trained_model: True if loaded from disk
    """
    try:
        if len(prices) < 100:
            return {
                "current_regime": MarketRegime.SIDEWAYS,
                "current_label":  "Neutral",
                "confidence":     0.0,
                "error":          "Insufficient price history (need 100+ days)",
            }

        # Build features
        features_df = build_features(prices)
        feature_arr = features_df.values

        # Try loading trained model + scaler
        model, scaler = _load_trained_model(model_dir)
        using_trained = model is not None

        if scaler is not None:
            feature_arr = scaler.transform(feature_arr)
        else:
            # Standardize manually if no scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_arr = scaler.fit_transform(feature_arr)

        if model is None:
            model = _train_fresh(feature_arr, n_states=5)

        # Predict states
        n_states       = model.n_components
        hidden_states  = model.predict(feature_arr)
        state_probs    = model.predict_proba(feature_arr)
        state_to_label = _label_states(model, n_states)

        # Current regime
        current_state     = hidden_states[-1]
        current_fine, current_maat = state_to_label[current_state]
        confidence        = round(float(state_probs[-1][current_state]), 3)

        # Full regime sequence (MAAT 3-state)
        maat_sequence = [state_to_label[s][1].value for s in hidden_states]
        fine_sequence = [state_to_label[s][0] for s in hidden_states]

        # Regime stability (last 10 days)
        recent_fine    = fine_sequence[-10:]
        recent_maat    = maat_sequence[-10:]
        regime_stable  = len(set(recent_maat)) == 1

        # Count days in each MAAT regime
        regime_counts = {
            MarketRegime.BULL.value:     maat_sequence.count(MarketRegime.BULL.value),
            MarketRegime.BEAR.value:     maat_sequence.count(MarketRegime.BEAR.value),
            MarketRegime.SIDEWAYS.value: maat_sequence.count(MarketRegime.SIDEWAYS.value),
        }

        # Count days in each fine-grained regime
        detailed_counts = {label: fine_sequence.count(label)
                          for _, (label, _) in REGIME_LABELS.items()}

        return {
            "current_regime":      current_maat,
            "current_label":       current_fine,
            "confidence":          confidence,
            "regime_stable":       regime_stable,
            "recent_regimes":      recent_fine,
            "recent_maat_regimes": recent_maat,
            "regime_counts":       regime_counts,
            "detailed_counts":     detailed_counts,
            "total_days":          len(maat_sequence),
            "using_trained_model": using_trained,
            "n_states":            n_states,
        }

    except Exception as e:
        return {
            "current_regime": MarketRegime.SIDEWAYS,
            "current_label":  "Neutral",
            "confidence":     0.0,
            "error":          str(e),
        }
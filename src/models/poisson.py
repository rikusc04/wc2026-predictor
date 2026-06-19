"""Baseline Poisson regression for match-outcome prediction.

We fit two GLMs (generalized linear models) on top of the same feature
table:
  - one predicts log(lambda_home) — expected home goals
  - one predicts log(lambda_away) — expected away goals

For each future match we want to predict, we use the two trained models
to estimate (lambda_home, lambda_away), then derive everything else
(W/D/L probs, exact-score probs) from two independent Poissons.

Independence between home/away goal counts is a known simplification
(real matches have some correlation between team scores, which the
Dixon-Coles correction will address later). Good enough for v1 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import PROJECT_ROOT


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"

TRAIN_START = pd.Timestamp("1990-01-01")

# Maximum score we'll bother computing probabilities for. With λ up to ~12
# (extreme mismatches), Poisson has non-trivial mass past 10 goals.
# Setting to 20 keeps total prob mass within ~1e-6 of 1.0 even for λ=12.
MAX_GOALS = 20

# Features that MUST be non-null in training (filtered via dropna before fit).
# v2: graded host_advantage_{home,away} ∈ [0,1] replaces v1's binary `neutral`.
# v2 Item 2: altitude_native_{home,away} captures lifelong-adaptation
# advantage at altitude venues that 2-week visitor acclimation can't match.
NUMERIC_FEATURES_REQUIRED = [
    "home_elo_pre", "away_elo_pre",
    "home_form_scored", "home_form_conceded",
    "away_form_scored", "away_form_conceded",
    "home_days_since_last", "away_days_since_last",
    "host_advantage_home", "host_advantage_away",
    "altitude_native_home", "altitude_native_away",
]

# Features that may legitimately be missing (e.g., squad value for pre-2006
# matches or for nations TM doesn't track).
# Handled via median imputation inside the pipeline rather than row filtering.
NUMERIC_FEATURES_IMPUTED = [
    "home_squad_value", "away_squad_value",
    # v2 Phase 2.1: starting-XI value from StatsBomb lineups. Only available
    # for ~300 international matches (WC 2018/22, Euro 20/24, Copa 24, AFCON 23).
    # NaN for everything else (including WC 2014 — no StatsBomb coverage).
    # Imputer fills with median, so the model still trains on all rows.
    "lineup_value_home", "lineup_value_away",
    # v2 Phase 2.2d ran a diagnostic A/B against `lineup_elo_home/away`
    # (clubelo.com per-starter Elo, position-weighted). On WC 2022 backtest:
    #   value only:        1.091  (this config)
    #   value + Elo:       1.118  (+0.027 — sparse-feature multicollinearity)
    #   Elo only:          1.115  (+0.024 — Elo strictly worse than value)
    # Club Elo turned out too coarse a proxy compared to per-player TM value,
    # so we kept the 2.1 feature here. The 2.2d infrastructure is preserved
    # (lineup_elo.csv, wc2026_predicted_lineup_elo.csv etc. still build) for
    # future use as e.g. a fallback when TM value is missing, or as a
    # multiplicative adjustment on value.
    # NOTE: home_pts_before / away_pts_before were tried but removed — they
    # fight the is_dead_rubber signal because in dead rubbers the favorite
    # has 6 points (high) but should be softened, exactly opposite of what
    # the model learns from points_before across all matches.
]

NUMERIC_FEATURES = NUMERIC_FEATURES_REQUIRED + NUMERIC_FEATURES_IMPUTED
# v2: `neutral` was dropped — host_advantage_{home,away} subsumes it
# (neutral==True ⇔ both host_advantage sides < 1.0).
BOOL_FEATURES = ["is_dead_rubber"]
CATEGORICAL_FEATURES = ["tournament_class"]

ALL_FEATURES = NUMERIC_FEATURES + BOOL_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainedModels:
    home_model: Pipeline
    away_model: Pipeline
    train_cutoff: pd.Timestamp
    n_train: int
    rho: float = 0.0   # Dixon-Coles correction (0 = pure Poisson)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lambda_home, lambda_away) arrays for each row in X."""
        lam_h = self.home_model.predict(X[ALL_FEATURES])
        lam_a = self.away_model.predict(X[ALL_FEATURES])
        return lam_h, lam_a


def _make_pipeline() -> Pipeline:
    # Numeric features go through (impute median for NaN) → (standardize to
    # zero mean / unit variance). Imputation must come BEFORE scaling.
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
    )
    # alpha is L2 regularization. Some shrinkage prevents the model from
    # over-fitting noisy team-pair effects, especially for rarely-played teams.
    return Pipeline([
        ("pre", preprocessor),
        ("reg", PoissonRegressor(alpha=0.1, max_iter=500)),
    ])


def load_training_frame(cutoff: pd.Timestamp, features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(features_path, parse_dates=["date"])
    mask = (df["date"] >= TRAIN_START) & (df["date"] < cutoff)
    df = df[mask].copy()
    # Required features must be non-null; imputed features may be NaN
    # (handled by the SimpleImputer inside the pipeline).
    df = df.dropna(subset=NUMERIC_FEATURES_REQUIRED + ["home_score", "away_score"])
    # Backstop: if any imputed numeric feature is missing entirely from
    # features.csv (e.g., file built before that feature was added), add NaN
    # columns so the pipeline still works.
    for col in NUMERIC_FEATURES_IMPUTED:
        if col not in df.columns:
            df[col] = float("nan")
    for col in BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
    return df


def train(cutoff: pd.Timestamp, fit_dc: bool = True) -> TrainedModels:
    """Train Poisson models on all data from 1990 up to (but not including) cutoff.

    If `fit_dc` is True (default), also fit the Dixon-Coles ρ correction
    on the same training data.
    """
    df = load_training_frame(cutoff)

    home_model = _make_pipeline()
    home_model.fit(df[ALL_FEATURES], df["home_score"])

    away_model = _make_pipeline()
    away_model.fit(df[ALL_FEATURES], df["away_score"])

    rho = fit_rho(home_model, away_model, df) if fit_dc else 0.0

    return TrainedModels(
        home_model=home_model,
        away_model=away_model,
        train_cutoff=cutoff,
        n_train=len(df),
        rho=rho,
    )


def score_matrix(lam_h: float, lam_a: float, max_goals: int = MAX_GOALS, rho: float = 0.0) -> np.ndarray:
    """P(home=i, away=j) under (optionally Dixon-Coles-corrected) Poisson.

    rho=0 gives the pure independent-Poisson score matrix.
    Non-zero rho applies the Dixon-Coles τ multiplier to cells
    (0,0), (1,0), (0,1), (1,1) — negative rho boosts draws.

    The corrected matrix is renormalized so all probabilities sum to 1
    (the τ correction technically perturbs the total mass slightly).
    """
    ph = poisson.pmf(np.arange(max_goals + 1), lam_h)
    pa = poisson.pmf(np.arange(max_goals + 1), lam_a)
    M = np.outer(ph, pa)
    if rho != 0.0:
        M[0, 0] *= 1.0 - lam_h * lam_a * rho
        M[1, 0] *= 1.0 + lam_a * rho
        M[0, 1] *= 1.0 + lam_h * rho
        M[1, 1] *= 1.0 - rho
        # τ can drive cells negative for extreme λ; clip and renormalize
        M = np.maximum(M, 0.0)
        M = M / M.sum()
    return M


def _tau(h: int, a: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles τ for the four affected scorelines; 1 elsewhere."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def fit_rho(home_model: Pipeline, away_model: Pipeline, train_df: pd.DataFrame) -> float:
    """Estimate the Dixon-Coles ρ by maximum likelihood on training data.

    Two-stage fit: λ's come from the already-trained Poisson models.
    We then find the ρ that maximizes Σ log τ(h, a; λ_h, λ_a, ρ) across
    matches. Only the four low-score cells contribute non-trivially.
    """
    X = train_df[ALL_FEATURES]
    lam_h = home_model.predict(X)
    lam_a = away_model.predict(X)
    h_scores = train_df["home_score"].astype(int).to_numpy()
    a_scores = train_df["away_score"].astype(int).to_numpy()

    # Only rows where (h, a) is in {(0,0), (1,0), (0,1), (1,1)} have τ ≠ 1.
    low_score_mask = (
        ((h_scores == 0) & (a_scores == 0))
        | ((h_scores == 1) & (a_scores == 0))
        | ((h_scores == 0) & (a_scores == 1))
        | ((h_scores == 1) & (a_scores == 1))
    )
    h_low = h_scores[low_score_mask]
    a_low = a_scores[low_score_mask]
    lh_low = lam_h[low_score_mask]
    la_low = lam_a[low_score_mask]

    def neg_log_lik(rho: float) -> float:
        # Vectorized τ for each affected match
        tau_vals = np.ones_like(lh_low)
        m00 = (h_low == 0) & (a_low == 0)
        m10 = (h_low == 1) & (a_low == 0)
        m01 = (h_low == 0) & (a_low == 1)
        m11 = (h_low == 1) & (a_low == 1)
        tau_vals[m00] = 1.0 - lh_low[m00] * la_low[m00] * rho
        tau_vals[m10] = 1.0 + la_low[m10] * rho
        tau_vals[m01] = 1.0 + lh_low[m01] * rho
        tau_vals[m11] = 1.0 - rho
        # If any τ <= 0, this rho is infeasible
        if (tau_vals <= 0).any():
            return 1e10
        return -np.log(tau_vals).sum()

    result = minimize_scalar(neg_log_lik, bounds=(-0.3, 0.3), method="bounded")
    return float(result.x)


def outcome_probs(lam_h: float, lam_a: float, max_goals: int = MAX_GOALS, rho: float = 0.0) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) from a (possibly DC-corrected) score matrix."""
    M = score_matrix(lam_h, lam_a, max_goals, rho)
    p_home = np.tril(M, -1).sum()  # rows > cols => home goals > away goals
    p_draw = np.trace(M)
    p_away = np.triu(M, 1).sum()
    return float(p_home), float(p_draw), float(p_away)


def most_likely_score(lam_h: float, lam_a: float, max_goals: int = MAX_GOALS, rho: float = 0.0) -> tuple[int, int, float]:
    """Argmax of the (possibly DC-corrected) score matrix and its probability."""
    M = score_matrix(lam_h, lam_a, max_goals, rho)
    idx = np.unravel_index(np.argmax(M), M.shape)
    return int(idx[0]), int(idx[1]), float(M[idx])


def _format_prediction(row: pd.Series, lam_h: float, lam_a: float) -> str:
    p_h, p_d, p_a = outcome_probs(lam_h, lam_a)
    sh, sa, sp = most_likely_score(lam_h, lam_a)
    actual = f"{int(row['home_score'])}-{int(row['away_score'])}"
    return (
        f"{row['date'].date()}  {row['home_team']:>15s} {actual:>5s} {row['away_team']:<15s}  "
        f"λ=({lam_h:.2f},{lam_a:.2f})  "
        f"W/D/L=({p_h:.0%}/{p_d:.0%}/{p_a:.0%})  "
        f"top score: {sh}-{sa} ({sp:.0%})"
    )


def _demo(models: TrainedModels, sample: pd.DataFrame, label: str) -> None:
    print(f"=== {label} ===")
    lam_h, lam_a = models.predict(sample)
    for i, (_, row) in enumerate(sample.iterrows()):
        print(_format_prediction(row, lam_h[i], lam_a[i]))
    print()


if __name__ == "__main__":
    # Smoke test: train through end of 2023, then predict a handful of
    # recent matches just before the cutoff (these are *in-sample* — they
    # were in the training set — so this is not a real evaluation, just a
    # check that the model produces sensible numbers. Real eval is task #7.
    cutoff = pd.Timestamp("2024-01-01")
    models = train(cutoff)
    print(f"trained on {models.n_train:,} matches (1990-01-01 → 2023-12-31)")
    print()

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.dropna(subset=NUMERIC_FEATURES + ["home_score", "away_score"])

    # A few interesting recent-pre-cutoff matches
    in_sample = df[(df["date"] >= "2023-11-01") & (df["date"] < "2024-01-01")]
    interesting = pd.concat([
        in_sample[in_sample["tournament_class"] == "world_cup"].tail(3),
        in_sample[in_sample["tournament_class"] == "qualifier"].tail(3),
        in_sample[in_sample["tournament_class"] == "continental"].tail(3),
    ])
    _demo(models, interesting, "predictions on a few late-2023 matches")

    # Reasonability check using a matchup the model has seen plenty:
    # Argentina vs Bolivia. Argentina is a strong favorite but the gap is
    # within the training distribution (~700 Elo), so the prediction
    # shouldn't be an absurd extrapolation.
    print("=== quick reasonability check ===")
    arg_last = df[(df["home_team"] == "Argentina") | (df["away_team"] == "Argentina")].tail(1).iloc[0]
    bol_last = df[(df["home_team"] == "Bolivia") | (df["away_team"] == "Bolivia")].tail(1).iloc[0]
    arg_elo = arg_last["home_elo_post"] if arg_last["home_team"] == "Argentina" else arg_last["away_elo_post"]
    bol_elo = bol_last["home_elo_post"] if bol_last["home_team"] == "Bolivia" else bol_last["away_elo_post"]

    hypo = pd.DataFrame([{
        "home_elo_pre": arg_elo,
        "away_elo_pre": bol_elo,
        "home_form_scored": 2.3,
        "home_form_conceded": 0.6,
        "away_form_scored": 0.9,
        "away_form_conceded": 2.0,
        "home_days_since_last": 5,
        "away_days_since_last": 5,
        "host_advantage_home": 1.0,   # Argentina at home
        "host_advantage_away": 0.7,   # Bolivia same confederation (CONMEBOL)
        "altitude_native_home": 0.0,  # Buenos Aires is sea level
        "altitude_native_away": 0.0,  # Bolivia not native at sea level
        "tournament_class": "qualifier",
    }])
    lam_h, lam_a = models.predict(hypo)
    p_h, p_d, p_a = outcome_probs(lam_h[0], lam_a[0])
    sh, sa, sp = most_likely_score(lam_h[0], lam_a[0])
    print(f"Argentina ({arg_elo:.0f}) vs Bolivia ({bol_elo:.0f}) at home, qualifier")
    print(f"  λ_home={lam_h[0]:.2f}, λ_away={lam_a[0]:.2f}")
    print(f"  P(Argentina wins)={p_h:.1%}  P(draw)={p_d:.1%}  P(Bolivia wins)={p_a:.1%}")
    print(f"  most likely score: {sh}-{sa} ({sp:.1%})")
    print(f"  (probs sum: {p_h + p_d + p_a:.4f} — should be ≈ 1.0)")
    print()

    # Extreme extrapolation check: Brazil vs San Marino.
    # Elo gap ~1100 — much larger than anything in training data.
    # The model will produce a large λ_home; the value isn't really
    # meaningful (model is extrapolating beyond its training distribution)
    # but we want to confirm that (a) probs still sum to 1.0 with MAX_GOALS=20,
    # and (b) the directional answer (Brazil dominates) is right.
    print("=== extreme-extrapolation check (out-of-distribution) ===")
    brazil_last = df[(df["home_team"] == "Brazil") | (df["away_team"] == "Brazil")].tail(1).iloc[0]
    sm_last = df[(df["home_team"] == "San Marino") | (df["away_team"] == "San Marino")].tail(1).iloc[0]
    brazil_elo = brazil_last["home_elo_post"] if brazil_last["home_team"] == "Brazil" else brazil_last["away_elo_post"]
    sm_elo = sm_last["home_elo_post"] if sm_last["home_team"] == "San Marino" else sm_last["away_elo_post"]

    hypo = pd.DataFrame([{
        "home_elo_pre": brazil_elo,
        "away_elo_pre": sm_elo,
        "home_form_scored": 2.5,
        "home_form_conceded": 0.8,
        "away_form_scored": 0.3,
        "away_form_conceded": 3.5,
        "home_days_since_last": 5,
        "away_days_since_last": 5,
        "host_advantage_home": 1.0,   # Brazil at home
        "host_advantage_away": 0.0,   # San Marino: UEFA in CONMEBOL country
        "altitude_native_home": 0.0,
        "altitude_native_away": 0.0,
        "tournament_class": "friendly",
    }])
    lam_h, lam_a = models.predict(hypo)
    p_h, p_d, p_a = outcome_probs(lam_h[0], lam_a[0])
    sh, sa, sp = most_likely_score(lam_h[0], lam_a[0])
    print(f"Brazil ({brazil_elo:.0f}) vs San Marino ({sm_elo:.0f}) at Brazil's home, friendly")
    print(f"  λ_home={lam_h[0]:.2f}, λ_away={lam_a[0]:.2f}")
    print(f"  (λ_home of {lam_h[0]:.1f} is unrealistic — model is extrapolating")
    print(f"   beyond its training distribution. Directionally correct, magnitude isn't.)")
    print(f"  P(Brazil wins)={p_h:.1%}  P(draw)={p_d:.1%}  P(San Marino wins)={p_a:.1%}")
    print(f"  most likely score: {sh}-{sa} ({sp:.1%})")
    print(f"  (probs sum: {p_h + p_d + p_a:.4f} — should be ≈ 1.0)")

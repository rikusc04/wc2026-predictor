"""Backtest the Poisson model against three past World Cups.

For each WC year, train the model on everything from 1990 up to the day
before kickoff, then predict every match of that tournament and measure
quality with log-loss, accuracy, and RPS. Compare to a naive Elo-only
baseline so we can tell whether the full feature set actually helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.data.loader import PROJECT_ROOT
from src.models.poisson import (
    NUMERIC_FEATURES_REQUIRED,
    TrainedModels,
    outcome_probs,
    train,
)


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"


@dataclass
class WCConfig:
    year: int
    cutoff: str   # kickoff date — train uses < cutoff, so training data
                  # ends on (kickoff − 1 day)
    end: str      # last match of tournament (inclusive)


WC_CONFIGS = [
    WCConfig(2014, "2014-06-12", "2014-07-13"),
    WCConfig(2018, "2018-06-14", "2018-07-15"),
    WCConfig(2022, "2022-11-20", "2022-12-18"),
]


# Outcome encoding for log_loss/RPS: H=0, D=1, A=2.
HOME, DRAW, AWAY = 0, 1, 2


def outcome_int(home_goals: float, away_goals: float) -> int:
    if home_goals > away_goals:
        return HOME
    if home_goals < away_goals:
        return AWAY
    return DRAW


def rps(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Ranked Probability Score for ordered [H, D, A] outcomes.

    For each match, compute the squared error between the cumulative
    predicted probabilities and the cumulative one-hot of the true class,
    then average across matches.

    Lower is better. A random uniform predictor scores ~0.22.
    """
    cum_pred = np.cumsum(probs, axis=1)
    cum_true = np.zeros_like(probs)
    for i, y in enumerate(y_true):
        cum_true[i, y:] = 1.0
    # Last cumulative element is always 1 for both → contributes 0; we can
    # skip it but it doesn't matter numerically.
    return float(np.mean(np.sum((cum_pred - cum_true) ** 2, axis=1)))


def model_predict_probs(models: TrainedModels, wc: pd.DataFrame) -> np.ndarray:
    lam_h, lam_a = models.predict(wc)
    out = np.zeros((len(wc), 3))
    for i in range(len(wc)):
        p_h, p_d, p_a = outcome_probs(lam_h[i], lam_a[i])
        out[i] = [p_h, p_d, p_a]
    # tiny numerical drift can push sums slightly off 1.0; renormalize so
    # log_loss is happy
    out = out / out.sum(axis=1, keepdims=True)
    return out


def naive_elo_probs(wc: pd.DataFrame, draw_rate: float = 0.28) -> np.ndarray:
    """Elo-only baseline.

    expected_home_score = sigmoid-like Elo function in [0, 1]. We reserve
    `draw_rate` (28% for WC matches) for draws and split the remaining
    72% between home and away in proportion to the Elo expected score.
    No Poisson math, no form features, no tournament/neutral effects.
    """
    elo_diff = wc["elo_diff"].to_numpy()
    e_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    p_draw = np.full_like(e_home, draw_rate)
    non_draw = 1.0 - p_draw
    p_home = non_draw * e_home
    p_away = non_draw * (1.0 - e_home)
    return np.column_stack([p_home, p_draw, p_away])


@dataclass
class Metrics:
    n_matches: int
    log_loss: float
    accuracy: float
    rps: float
    extra: dict = field(default_factory=dict)


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> Metrics:
    preds = np.argmax(probs, axis=1)
    return Metrics(
        n_matches=len(y_true),
        log_loss=float(log_loss(y_true, probs, labels=[0, 1, 2])),
        accuracy=float(accuracy_score(y_true, preds)),
        rps=rps(y_true, probs),
    )


def _confusion(y_true: np.ndarray, probs: np.ndarray) -> dict[str, int]:
    """Predicted-vs-actual class breakdown (handy for spotting bias)."""
    preds = np.argmax(probs, axis=1)
    labels = {HOME: "H", DRAW: "D", AWAY: "A"}
    counts: dict[str, int] = {}
    for actual_label in labels.values():
        for pred_label in labels.values():
            counts[f"actual={actual_label} pred={pred_label}"] = 0
    for t, p in zip(y_true, preds):
        counts[f"actual={labels[int(t)]} pred={labels[int(p)]}"] += 1
    return counts


def backtest_wc(config: WCConfig, features: pd.DataFrame) -> dict:
    cutoff = pd.Timestamp(config.cutoff)
    end = pd.Timestamp(config.end)

    print(f"=== WC {config.year} ===")
    print(f"training on 1990-01-01 → {(cutoff - pd.Timedelta(days=1)).date()}")
    models = train(cutoff)
    print(f"  trained on {models.n_train:,} matches")

    wc = features[
        (features["tournament"] == "FIFA World Cup")
        & (features["date"] >= cutoff)
        & (features["date"] <= end)
    ].copy()
    # Use NUMERIC_FEATURES_REQUIRED (not NUMERIC_FEATURES) so we don't drop
    # rows with NaN in imputed features like squad_value or pts_before — those
    # NaNs are handled by the SimpleImputer inside the model pipeline.
    wc = wc.dropna(subset=NUMERIC_FEATURES_REQUIRED + ["home_score", "away_score"])
    print(f"  predicting {len(wc)} matches")

    y_true = np.array([outcome_int(h, a) for h, a in zip(wc["home_score"], wc["away_score"])])

    model_probs = model_predict_probs(models, wc)
    naive_probs = naive_elo_probs(wc)

    model_m = evaluate(y_true, model_probs)
    naive_m = evaluate(y_true, naive_probs)

    print()
    print(f"  {'metric':12s}  {'model':>10s}  {'naive Elo':>10s}  {'delta':>10s}")
    for name, fmt in [("log_loss", ".4f"), ("accuracy", ".1%"), ("rps", ".4f")]:
        m = getattr(model_m, name)
        n = getattr(naive_m, name)
        delta = m - n
        better = "  ✓" if (name == "accuracy" and delta > 0) or (name != "accuracy" and delta < 0) else "   "
        print(f"  {name:12s}  {format(m, fmt):>10s}  {format(n, fmt):>10s}  {format(delta, fmt):>10s}{better}")

    print()
    print("  confusion (model):")
    confusion = _confusion(y_true, model_probs)
    for actual in ["H", "D", "A"]:
        line = "    "
        for pred in ["H", "D", "A"]:
            line += f"{confusion[f'actual={actual} pred={pred}']:>4d}  "
        print(line + f"  ← actual={actual}")
    print(f"    {'H':>4s}  {'D':>4s}  {'A':>4s}     ← predicted")
    print()

    return {
        "year": config.year,
        "n": model_m.n_matches,
        "model_logloss": model_m.log_loss,
        "naive_logloss": naive_m.log_loss,
        "model_acc": model_m.accuracy,
        "naive_acc": naive_m.accuracy,
        "model_rps": model_m.rps,
        "naive_rps": naive_m.rps,
    }


if __name__ == "__main__":
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])

    rows = [backtest_wc(c, features) for c in WC_CONFIGS]

    print("=== summary across all three backtests ===")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print("interpretation guide:")
    print("  log_loss: lower better. Random = 1.099. Good models ~0.95-1.05.")
    print("  accuracy: higher better. Random = 33%. WC matches are noisy;")
    print("            even strong models rarely exceed 55-60%.")
    print("  rps:      lower better. Random ≈ 0.22. Good models ~0.18-0.20.")

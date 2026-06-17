"""Per-match prediction diagnostics for the backtest.

Runs the same three backtests as src.evaluation.backtest, but instead of
aggregate metrics, dumps two diagnostics per tournament:

  1. The N worst-loss matches — which specific games are hurting log-loss most?
  2. A calibration table — do the model's probability outputs match observed
     frequencies? I.e., when the model says "70%", does it actually happen 70%
     of the time?

These answer: "Is the model uniformly over-confident, or are a few catastrophic
predictions dragging the average down?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.evaluation.backtest import (
    WC_CONFIGS,
    FEATURES_PATH,
    outcome_int,
    model_predict_probs,
)
from src.models.poisson import NUMERIC_FEATURES_REQUIRED, train


# Numerical floor to keep log(0) from happening when the model assigns vanishing
# probability to an outcome that actually occurred (e.g., a huge upset).
_LOG_FLOOR = 1e-15


def per_match_diagnostic(wc: pd.DataFrame, probs: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Per-match prediction details + log-loss contribution."""
    label_names = {0: "H", 1: "D", 2: "A"}
    rows = []
    for i in range(len(wc)):
        m = wc.iloc[i]
        p_actual = float(probs[i, int(y_true[i])])
        ll = -np.log(max(p_actual, _LOG_FLOOR))
        rows.append({
            "date": str(m["date"])[:10],
            "matchup": f"{m['home_team']} vs {m['away_team']}",
            "score": f"{int(m['home_score'])}-{int(m['away_score'])}",
            "actual": label_names[int(y_true[i])],
            "P_H": float(probs[i, 0]),
            "P_D": float(probs[i, 1]),
            "P_A": float(probs[i, 2]),
            "P_actual": p_actual,
            "log_loss": ll,
        })
    return pd.DataFrame(rows)


def multiclass_calibration_table(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Calibration across all (match, outcome-class) probability pairs.

    For each match, the model emits 3 probabilities (one per class). For
    each (match, class) pair we record:
      - predicted probability the model assigned to that class
      - whether that class actually happened (0 or 1)

    Then we bin those pairs by predicted probability and compute the
    actual frequency within each bin. A well-calibrated model has
    predicted_mean ≈ actual_rate in every bin.
    """
    p_flat = []
    y_flat = []
    for i in range(len(y_true)):
        actual = int(y_true[i])
        for c in range(probs.shape[1]):
            p_flat.append(float(probs[i, c]))
            y_flat.append(1.0 if actual == c else 0.0)
    p_flat = np.array(p_flat)
    y_flat = np.array(y_flat)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(p_flat, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        pred_mean = float(p_flat[mask].mean())
        actual_rate = float(y_flat[mask].mean())
        rows.append({
            "bin_lo": float(bins[b]),
            "bin_hi": float(bins[b + 1]),
            "n_pairs": int(mask.sum()),
            "predicted_mean": pred_mean,
            "actual_rate": actual_rate,
            "over_under": pred_mean - actual_rate,
        })
    return pd.DataFrame(rows)


def _format_row_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = df[c].apply(lambda x: f"{x:.1%}")
    return df


def diagnose_wc(config, features: pd.DataFrame, top_n: int = 10) -> None:
    cutoff = pd.Timestamp(config.cutoff)
    end = pd.Timestamp(config.end)

    print(f"\n{'=' * 70}")
    print(f"WC {config.year}")
    print(f"{'=' * 70}")

    models = train(cutoff)

    wc = features[
        (features["tournament"] == "FIFA World Cup")
        & (features["date"] >= cutoff)
        & (features["date"] <= end)
    ].copy()
    wc = wc.dropna(subset=NUMERIC_FEATURES_REQUIRED + ["home_score", "away_score"])

    y_true = np.array([outcome_int(h, a) for h, a in zip(wc["home_score"], wc["away_score"])])
    probs = model_predict_probs(models, wc)

    diag = per_match_diagnostic(wc, probs, y_true)
    diag_sorted = diag.sort_values("log_loss", ascending=False)

    avg_ll = diag["log_loss"].mean()
    print(f"\noverall log-loss: {avg_ll:.4f}")
    print(f"sum of top-{top_n} match log-losses: "
          f"{diag_sorted['log_loss'].head(top_n).sum():.3f} "
          f"({100 * diag_sorted['log_loss'].head(top_n).sum() / diag['log_loss'].sum():.0f}% of total)")

    print(f"\n--- {top_n} worst-loss matches (model was most wrong here) ---")
    worst = _format_row_pct(diag_sorted.head(top_n), ["P_H", "P_D", "P_A", "P_actual"])
    worst["log_loss"] = worst["log_loss"].apply(lambda x: f"{x:.2f}")
    print(worst.to_string(index=False))

    print(f"\n--- calibration: do predicted probabilities match observed rates? ---")
    cal = multiclass_calibration_table(probs, y_true)
    cal_disp = cal.copy()
    cal_disp["bin"] = cal_disp.apply(lambda r: f"{r['bin_lo']:.0%}-{r['bin_hi']:.0%}", axis=1)
    cal_disp = cal_disp[["bin", "n_pairs", "predicted_mean", "actual_rate", "over_under"]]
    cal_disp = _format_row_pct(cal_disp, ["predicted_mean", "actual_rate", "over_under"])
    print(cal_disp.to_string(index=False))
    print("\n  reading: 'over_under' = predicted_mean − actual_rate")
    print("           positive means model over-confident in that bin")
    print("           negative means model under-confident in that bin")


if __name__ == "__main__":
    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    for c in WC_CONFIGS:
        diagnose_wc(c, features)

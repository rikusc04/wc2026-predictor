"""Generate WC 2026 predictions.

Pipeline:
  1. Train the production model on data 1990-01-01 → 2026-06-10 (the day
     before WC 2026 kickoff). This is the same `train()` we use for
     backtests, just with the latest cutoff.
  2. Build a "team state at cutoff" snapshot — each team's Elo, recent
     form, days since last match, squad value as of 2026-06-10.
  3. For each WC 2026 match in the fixture list, construct a feature
     row from the snapshot and predict (λ_home, λ_away).
  4. Derive W/D/L probabilities, most-likely score, and full score-grid
     probability from the model output.

This is a STATIC pre-tournament forecast: every match is predicted using
features frozen at 2026-06-10. Earlier WC 2026 results don't update the
features for later matches in this version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT, load_results
from src.features.altitude import altitude_native_advantage
from src.features.confederations import host_advantage
from src.features.tournaments import classify_tournament
from src.models.poisson import (
    ALL_FEATURES,
    most_likely_score,
    outcome_probs,
    train,
)


CUTOFF = pd.Timestamp("2026-06-11")
FORM_WINDOW = 10
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_PATH = PROCESSED_DIR / "wc2026_predictions.csv"


def compute_team_state_at_cutoff() -> dict[str, dict]:
    """For each team, compute their feature state as of the cutoff date.

    Returns: {team_name: {"elo": float, "form_scored": float, ...}, ...}

    Reads:
      - results.csv (cutoff-applied — drops WC 2026 matches)
      - final_elo.csv (each team's Elo after all pre-cutoff matches)
      - squad_values.csv (year=2026 snapshot)
    """
    results, _ = load_results(apply_cutoff=True)
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)

    # Walk every team's match history, track last FORM_WINDOW matches
    team_recent: dict[str, list[tuple[pd.Timestamp, float, float]]] = {}
    team_last_date: dict[str, pd.Timestamp] = {}

    for _, row in results.iterrows():
        for team, gf, ga in [
            (row["home_team"], row["home_score"], row["away_score"]),
            (row["away_team"], row["away_score"], row["home_score"]),
        ]:
            team_recent.setdefault(team, []).append((row["date"], gf, ga))
            if len(team_recent[team]) > FORM_WINDOW:
                team_recent[team] = team_recent[team][-FORM_WINDOW:]
            team_last_date[team] = row["date"]

    # Build the per-team state dict
    state: dict[str, dict] = {}
    for team, matches in team_recent.items():
        gfs = [m[1] for m in matches if pd.notna(m[1])]
        gas = [m[2] for m in matches if pd.notna(m[2])]
        state[team] = {
            "form_scored": float(np.mean(gfs)) if gfs else np.nan,
            "form_conceded": float(np.mean(gas)) if gas else np.nan,
            "last_match_date": team_last_date.get(team),
        }

    # Layer in Elo from final_elo.csv (computed over all pre-cutoff matches)
    elo_df = pd.read_csv(PROCESSED_DIR / "final_elo.csv")
    for _, row in elo_df.iterrows():
        team = row["team"]
        if team not in state:
            state[team] = {
                "form_scored": np.nan,
                "form_conceded": np.nan,
                "last_match_date": None,
            }
        state[team]["elo"] = float(row["elo"])

    # Layer in 2026 squad values
    sv = pd.read_csv(PROCESSED_DIR / "squad_values.csv")
    sv_2026 = sv[sv["year"] == 2026]
    for _, row in sv_2026.iterrows():
        team = row["team_name"]
        if team not in state:
            state[team] = {
                "elo": 1500.0,
                "form_scored": np.nan,
                "form_conceded": np.nan,
                "last_match_date": None,
            }
        state[team]["squad_value"] = float(row["squad_value_eur"])

    # v2 Phase 2.2a: layer in predicted starting-XI value for each qualifier.
    # File is produced by `python -m src.features.lineup_predictor`. If missing,
    # we just leave lineup_value as NaN and the model's imputer fills it.
    predicted_lineups_path = PROCESSED_DIR / "wc2026_predicted_lineup_values.csv"
    if predicted_lineups_path.exists():
        plv = pd.read_csv(predicted_lineups_path)
        for _, row in plv.iterrows():
            team = row["team"]
            if team not in state:
                continue
            v = row["lineup_value_eur"]
            if pd.notna(v):
                state[team]["lineup_value"] = float(v)

    return state


def build_match_features(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    match_country: str,
    state: dict[str, dict],
    match_city: str | None = None,
    days_since_override: float | None = None,
) -> dict:
    """Construct a single feature row for one match using the team state snapshot.

    `match_country` drives the v2 Item 1 graded host-advantage features.
    `match_city` drives the v2 Item 2 altitude-native feature (default None
    = no city info → no altitude effect, safe for sea-level venues and for
    the knockout cache which doesn't know individual venues).
    `days_since_override` lets callers (e.g., the knockout simulator) supply
    a fixed days-since value instead of computing it from each team's last
    pre-tournament match. Useful when predicting matches in an unfolding
    tournament where the "last match" is some other unsampled match.
    """
    h = state.get(home_team, {})
    a = state.get(away_team, {})

    def days_since(team_state):
        if days_since_override is not None:
            return float(days_since_override)
        last = team_state.get("last_match_date")
        if last is None:
            return np.nan
        return float((match_date - last).days)

    return {
        "home_elo_pre": h.get("elo", 1500.0),
        "away_elo_pre": a.get("elo", 1500.0),
        "home_form_scored": h.get("form_scored", np.nan),
        "home_form_conceded": h.get("form_conceded", np.nan),
        "away_form_scored": a.get("form_scored", np.nan),
        "away_form_conceded": a.get("form_conceded", np.nan),
        "home_days_since_last": days_since(h),
        "away_days_since_last": days_since(a),
        "home_squad_value": h.get("squad_value", np.nan),
        "away_squad_value": a.get("squad_value", np.nan),
        "host_advantage_home": host_advantage(home_team, match_country),
        "host_advantage_away": host_advantage(away_team, match_country),
        "altitude_native_home": altitude_native_advantage(home_team, match_city),
        "altitude_native_away": altitude_native_advantage(away_team, match_city),
        # v2 Phase 2.2a: predicted lineup_value from
        # `src/features/lineup_predictor.py` (modal starting XI from each team's
        # last 5 StatsBomb matches, or citizenship-top-11 fallback for the
        # ~16 qualifiers with no StatsBomb coverage). If the predictor wasn't
        # run, this falls back to NaN and the imputer handles it.
        "lineup_value_home": h.get("lineup_value", np.nan),
        "lineup_value_away": a.get("lineup_value", np.nan),
        "tournament_class": "world_cup",
        "is_dead_rubber": False,  # we don't know future qualification states
    }


def predict_wc_2026() -> tuple[pd.DataFrame, object]:
    """Train production model and predict every WC 2026 match."""
    last_train_day = (CUTOFF - pd.Timedelta(days=1)).date()
    print(f"training production model (1990-01-01 → {last_train_day})...")
    models = train(CUTOFF)
    print(f"  trained on {models.n_train:,} matches")
    print(f"  Dixon-Coles ρ = {models.rho:+.4f}")

    print()
    print("loading WC 2026 fixture list...")
    results, _ = load_results(apply_cutoff=False)
    results["date"] = pd.to_datetime(results["date"])
    wc26 = results[
        (results["tournament"] == "FIFA World Cup") & (results["date"] >= CUTOFF)
    ].copy().sort_values("date").reset_index(drop=True)
    print(f"  {len(wc26)} WC 2026 matches")

    print()
    print("snapshotting team state at cutoff...")
    state = compute_team_state_at_cutoff()
    print(f"  {len(state)} teams in state lookup")

    print()
    print("building per-match feature rows...")
    # v2 Phase 2.2b: for the 12 already-played matches, override the predicted
    # lineup_value with the actual starting-XI value from Wikipedia.
    actual_lv_path = PROCESSED_DIR / "wc2026_actual_lineup_values.csv"
    actual_lv: dict[tuple, tuple[float | None, float | None]] = {}
    if actual_lv_path.exists():
        alv = pd.read_csv(actual_lv_path)
        for (mdate, h, a), grp in alv.groupby(["match_date", "home_team", "away_team"]):
            home_row = grp[grp["side"] == "home"]
            away_row = grp[grp["side"] == "away"]
            actual_lv[(str(mdate), h, a)] = (
                float(home_row["lineup_value_eur"].iloc[0]) if len(home_row) and pd.notna(home_row["lineup_value_eur"].iloc[0]) else None,
                float(away_row["lineup_value_eur"].iloc[0]) if len(away_row) and pd.notna(away_row["lineup_value_eur"].iloc[0]) else None,
            )

    feature_rows = []
    for _, m in wc26.iterrows():
        feats = build_match_features(
            m["home_team"], m["away_team"], m["date"], m["country"], state,
            match_city=m["city"],
        )
        # Override predicted lineup_value with actual when this match has been played
        key = (m["date"].strftime("%Y-%m-%d"), m["home_team"], m["away_team"])
        if key in actual_lv:
            actual_h, actual_a = actual_lv[key]
            if actual_h is not None:
                feats["lineup_value_home"] = actual_h
            if actual_a is not None:
                feats["lineup_value_away"] = actual_a
        feature_rows.append(feats)
    features_df = pd.DataFrame(feature_rows)

    print()
    print("predicting...")
    lam_h, lam_a = models.predict(features_df)

    predictions = []
    for i, (_, m) in enumerate(wc26.iterrows()):
        p_h, p_d, p_a = outcome_probs(lam_h[i], lam_a[i], rho=models.rho)
        sh, sa, sp = most_likely_score(lam_h[i], lam_a[i], rho=models.rho)
        actual = ""
        if pd.notna(m["home_score"]) and pd.notna(m["away_score"]):
            actual = f"{int(m['home_score'])}-{int(m['away_score'])}"
        predictions.append({
            "date": m["date"].date(),
            "home": m["home_team"],
            "away": m["away_team"],
            "neutral": bool(m["neutral"]),
            "actual_score": actual,
            "expected_goals_home": float(lam_h[i]),
            "expected_goals_away": float(lam_a[i]),
            "prob_home_win": p_h,
            "prob_draw": p_d,
            "prob_away_win": p_a,
            "most_likely_score": f"{sh}-{sa}",
            "most_likely_score_prob": sp,
        })
    return pd.DataFrame(predictions), models


def report(df: pd.DataFrame) -> None:
    """Print a tidy summary of predictions."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_PATH, index=False)
    print()
    print(f"wrote {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print()

    # Format probabilities as percentages for display
    disp = df.copy()
    for col in ["prob_home_win", "prob_draw", "prob_away_win", "most_likely_score_prob"]:
        disp[col] = disp[col].apply(lambda x: f"{x:.1%}")

    print("=== all 72 match predictions (chronological) ===")
    pd.set_option("display.max_rows", None)
    print(disp[[
        "date", "home", "away", "actual_score",
        "prob_home_win", "prob_draw", "prob_away_win",
        "most_likely_score", "most_likely_score_prob",
    ]].to_string(index=False))

    # If any matches are already played, evaluate
    played = df[df["actual_score"] != ""].copy()
    if len(played) > 0:
        print()
        print(f"=== quick eval on {len(played)} already-played matches ===")
        # Compute log-loss
        from src.evaluation.backtest import outcome_int
        ll_sum = 0.0
        n_correct = 0
        rows = []
        for _, p in played.iterrows():
            h, a = p["actual_score"].split("-")
            y = outcome_int(int(h), int(a))
            probs = [p["prob_home_win"], p["prob_draw"], p["prob_away_win"]]
            ll_sum += -np.log(max(probs[y], 1e-15))
            pred_argmax = int(np.argmax(probs))
            if pred_argmax == y:
                n_correct += 1
            rows.append({
                "date": p["date"], "home": p["home"], "away": p["away"],
                "actual": p["actual_score"],
                "prob_home_win": f"{probs[0]:.1%}",
                "prob_draw": f"{probs[1]:.1%}",
                "prob_away_win": f"{probs[2]:.1%}",
                "right": "✓" if pred_argmax == y else " ",
            })
        avg_ll = ll_sum / len(played)
        acc = n_correct / len(played)
        print(f"  log-loss: {avg_ll:.4f}   accuracy: {acc:.1%}   (n={len(played)})")
        print()
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    df, models = predict_wc_2026()
    report(df)

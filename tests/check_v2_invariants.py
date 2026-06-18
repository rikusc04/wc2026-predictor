"""End-to-end invariant checker for the v2 model.

Run after any model/feature change to catch "the numbers are off" silently.
Plain script (no pytest dependency). Reads the saved output CSVs plus
re-derives a few structural things. Exits non-zero on any failure.

Usage:
    .venv/bin/python tests/check_v2_invariants.py

Wall time: ~2 seconds on saved CSVs; +~15 seconds if --backtest is passed
to also re-run the 3-WC backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import PROJECT_ROOT, load_results  # noqa: E402
from src.features.altitude import altitude_native_advantage  # noqa: E402
from src.features.confederations import host_advantage  # noqa: E402
from src.features.group_standings import identify_groups  # noqa: E402
from src.prediction.bracket import (  # noqa: E402
    THIRD_PLACE_ELIGIBILITY,
    assign_third_place_slots,
)
from src.prediction.simulate_wc2026 import derive_fifa_group_labels  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  ←  {detail}")


# ----------------------------------------------------------------------
# Group 1: per-match prediction invariants
# ----------------------------------------------------------------------

def check_predictions_csv() -> None:
    print("\n=== wc2026_predictions.csv ===")
    path = PROCESSED_DIR / "wc2026_predictions.csv"
    if not path.exists():
        check("predictions file exists", False, f"missing {path}")
        return

    df = pd.read_csv(path)

    check("predictions: 72 rows", len(df) == 72, f"got {len(df)}")

    sums = df["prob_home_win"] + df["prob_draw"] + df["prob_away_win"]
    max_dev = (sums - 1.0).abs().max()
    check(
        "predictions: W/D/L probabilities sum to 1.0",
        max_dev < 1e-3,
        f"max deviation = {max_dev:.6f}",
    )

    for col in ["prob_home_win", "prob_draw", "prob_away_win"]:
        bad = ((df[col] < 0) | (df[col] > 1)).any()
        check(f"predictions: {col} in [0, 1]", not bad)

    check(
        "predictions: no NaN in probability columns",
        not df[["prob_home_win", "prob_draw", "prob_away_win"]].isna().any().any(),
    )

    bad_lambda = ((df["expected_goals_home"] < 0) | (df["expected_goals_home"] > 15)).sum()
    check(
        "predictions: expected_goals_home in (0, 15]",
        bad_lambda == 0,
        f"{bad_lambda} rows out of bound",
    )


# ----------------------------------------------------------------------
# Group 2: tournament-simulation invariants
# ----------------------------------------------------------------------

def check_simulation_csv() -> None:
    print("\n=== wc2026_simulation.csv ===")
    path = PROCESSED_DIR / "wc2026_simulation.csv"
    if not path.exists():
        check("simulation file exists", False, f"missing {path}")
        return

    sim = pd.read_csv(path)
    check("simulation: 48 teams", len(sim) == 48, f"got {len(sim)}")

    # Exactly one champion per simulation → sum of p_win_wc across all teams = 1.0
    total = sim["p_win_wc"].sum()
    check(
        "simulation: Σ p_win_wc across 48 teams ≈ 1.0",
        abs(total - 1.0) < 0.01,
        f"got {total:.4f}",
    )

    # Each round prob in [0, 1]
    round_cols = [c for c in sim.columns if c.startswith("p_")]
    for c in round_cols:
        bad = ((sim[c] < 0) | (sim[c] > 1)).any()
        check(f"simulation: {c} in [0, 1]", not bad)

    check(
        "simulation: no NaN in probability columns",
        not sim[round_cols].isna().any().any(),
    )

    # Σ p_advance across all teams = 32 (32 teams advance per sim)
    if "p_advance" in sim.columns:
        total_adv = sim["p_advance"].sum()
        check(
            "simulation: Σ p_advance ≈ 32",
            abs(total_adv - 32.0) < 0.5,
            f"got {total_adv:.4f}",
        )

    # Top 5 by P(win WC) must include the obvious favorites — if Spain or
    # Argentina vanish from the top 5, something is structurally broken.
    top5 = set(sim.sort_values("p_win_wc", ascending=False).head(5)["team"])
    must_include = {"Spain", "Argentina"}
    missing = must_include - top5
    check(
        "simulation: top 5 includes Spain and Argentina",
        not missing,
        f"missing: {missing}",
    )

    # Monotone-down round probabilities (per team): p_advance ≥ p_reach_r16 ≥ p_reach_qf ≥ ...
    chain = ["p_advance", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final", "p_win_wc"]
    chain = [c for c in chain if c in sim.columns]
    violations = 0
    for _, row in sim.iterrows():
        for a, b in zip(chain, chain[1:]):
            if row[a] + 1e-6 < row[b]:
                violations += 1
    check(
        "simulation: per-team round probs are monotone non-increasing",
        violations == 0,
        f"{violations} (team, round-pair) violations",
    )


# ----------------------------------------------------------------------
# Group 3: features.csv structural invariants
# ----------------------------------------------------------------------

def check_features_csv() -> None:
    print("\n=== features.csv ===")
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        check("features file exists", False, f"missing {path}")
        return

    df = pd.read_csv(path, nrows=5000)
    required = [
        "host_advantage_home", "host_advantage_away",
        "altitude_native_home", "altitude_native_away",
    ]
    for col in required:
        check(f"features: {col} column present", col in df.columns)

    if "neutral" in df.columns:
        # OK that it's still there (it's a data-source column), but shouldn't
        # appear in NUMERIC_FEATURES or BOOL_FEATURES anymore
        from src.models.poisson import ALL_FEATURES
        check(
            "model: 'neutral' is no longer in ALL_FEATURES",
            "neutral" not in ALL_FEATURES,
            f"ALL_FEATURES contains 'neutral': {ALL_FEATURES}",
        )

    for col in required:
        if col in df.columns:
            vals = df[col].dropna()
            in_range = ((vals >= 0) & (vals <= 1)).all()
            check(f"features: {col} all values in [0, 1]", bool(in_range))


# ----------------------------------------------------------------------
# Group 4: lookup-table sanity (catches "I forgot to rebuild features.csv")
# ----------------------------------------------------------------------

def check_lookup_tables() -> None:
    print("\n=== feature-lookup sanity ===")

    # Direct asserts on the canonical answers we expect
    cases_host = [
        ("Mexico", "Mexico", 1.0, "Mexico at home"),
        ("Brazil", "United States", 0.3, "Americas adjacency"),
        ("Morocco", "Qatar", 0.3, "CAF↔AFC adjacency (refinement)"),
        ("Japan", "United States", 0.0, "AFC in US — no adjacency"),
    ]
    for team, country, expected, desc in cases_host:
        actual = host_advantage(team, country)
        check(
            f"host_advantage: {desc}",
            abs(actual - expected) < 1e-9,
            f"got {actual}, expected {expected}",
        )

    cases_alt = [
        ("Mexico", "Mexico City", 1.0, "Mexico at Azteca"),
        ("Spain", "Mexico City", 0.0, "Spain at Azteca (acclimated visitor)"),
        ("Mexico", "Inglewood", 0.0, "Mexico at sea-level US"),
        ("Bolivia", "La Paz", 1.0, "Bolivia at La Paz"),
    ]
    for team, city, expected, desc in cases_alt:
        actual = altitude_native_advantage(team, city)
        check(
            f"altitude_native: {desc}",
            abs(actual - expected) < 1e-9,
            f"got {actual}, expected {expected}",
        )


# ----------------------------------------------------------------------
# Group 5: bracket structural invariants
# ----------------------------------------------------------------------

def check_bracket() -> None:
    print("\n=== bracket structure ===")

    # 3rd-place eligibility: every group letter A-L appears at least once
    union = set().union(*THIRD_PLACE_ELIGIBILITY.values())
    expected_letters = set("ABCDEFGHIJKL")
    missing = expected_letters - union
    check(
        "bracket: every group letter (A-L) appears in some 3rd-place slot eligibility",
        not missing,
        f"missing: {missing}",
    )

    # Three sample bipartite matchings: each gives a valid assignment
    test_scenarios = [
        ["E", "F", "G", "H", "I", "J", "K", "L"],
        ["A", "B", "C", "D", "E", "F", "G", "H"],
        ["A", "C", "D", "E", "G", "I", "J", "L"],
    ]
    for qualifiers in test_scenarios:
        matching = assign_third_place_slots(qualifiers)
        ok_size = set(matching.values()) == set(qualifiers)
        ok_eligible = all(
            g in THIRD_PLACE_ELIGIBILITY[slot] for slot, g in matching.items()
        )
        check(
            f"bracket: scenario {qualifiers} → valid assignment",
            ok_size and ok_eligible,
            f"size_ok={ok_size}, eligibility_ok={ok_eligible}",
        )


# ----------------------------------------------------------------------
# Group 6: FIFA group label inference (catches the inverted Group E/F bug)
# ----------------------------------------------------------------------

def check_fifa_group_labels() -> None:
    print("\n=== FIFA group label inference ===")

    results_df, _ = load_results(apply_cutoff=False)
    results_df["date"] = pd.to_datetime(results_df["date"])
    wc26 = results_df[
        (results_df["tournament"] == "FIFA World Cup")
        & (results_df["date"] >= "2026-06-11")
    ].copy()

    auto_groups = identify_groups(wc26)
    auto_to_fifa = derive_fifa_group_labels(wc26)
    groups = {auto_to_fifa[a]: teams for a, teams in auto_groups.items()}

    # FIFA convention: host countries in the first groups.
    # Mexico (opening match) → Group A; Canada → Group B; USA → Group C.
    check(
        "groups: Mexico in Group A (host opener)",
        "Mexico" in groups.get("A", set()),
        f"Group A contains: {sorted(groups.get('A', []))}",
    )
    check(
        "groups: Canada in Group B",
        "Canada" in groups.get("B", set()),
        f"Group B contains: {sorted(groups.get('B', []))}",
    )
    check(
        "groups: United States in Group C",
        "United States" in groups.get("C", set()),
        f"Group C contains: {sorted(groups.get('C', []))}",
    )
    check(
        "groups: every letter A-L is assigned to a 4-team group",
        all(len(groups.get(L, [])) == 4 for L in "ABCDEFGHIJKL"),
        f"sizes: { {L: len(groups.get(L, [])) for L in 'ABCDEFGHIJKL'} }",
    )


# ----------------------------------------------------------------------
# Group 7: backtest log-loss in plausible range
# ----------------------------------------------------------------------

def check_backtest() -> None:
    """Run the 3-WC backtest in-process and assert log-loss is in plausible range."""
    print("\n=== backtest log-loss range ===")
    from src.evaluation.backtest import FEATURES_PATH, WC_CONFIGS, backtest_wc

    features = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    for config in WC_CONFIGS:
        result = backtest_wc(config, features)
        ll = float(result["model_logloss"])
        check(
            f"backtest: WC {result['year']} log-loss in [0.80, 1.20]",
            0.80 <= ll <= 1.20,
            f"got {ll:.4f}",
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    do_backtest = "--backtest" in sys.argv

    print("=" * 70)
    print("v2 invariant check")
    print("=" * 70)

    check_predictions_csv()
    check_simulation_csv()
    check_features_csv()
    check_lookup_tables()
    check_bracket()
    check_fifa_group_labels()
    if do_backtest:
        check_backtest()
    else:
        print("\n  (skipping backtest check; pass --backtest to include it, ~15 sec)")

    total = len(PASSED) + len(FAILED)
    print()
    print("=" * 70)
    print(f"  {len(PASSED)}/{total} passed")
    if FAILED:
        print(f"  FAILURES ({len(FAILED)}):")
        for name, detail in FAILED:
            print(f"    - {name}: {detail}")
        return 1
    print("  all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Predict starting XI + lineup value for WC 2026 qualifiers.

v2 Phase 2.2a. Unlocks the lineup_value feature for WC 2026 predictions —
without this, every WC 2026 match got lineup_value=NaN (imputer median →
zero contribution to predictions, per issues #51).

Two paths per team:

1. **Modal-XI path** (used for 37 of 48 WC 2026 qualifiers): pull the team's
   most recent LAST_K matches from StatsBomb data (statsbomb_lineups.csv).
   Count appearances per player across those matches. Top 11 by appearance
   count = predicted starting XI. Sum their Transfermarkt valuations as of
   the WC 2026 kickoff date.

2. **Citizenship-fallback path** (used for ~11 teams with no StatsBomb
   coverage: Curaçao, Cape Verde, Haiti, Iraq, Jordan, New Zealand, Norway,
   Uzbekistan, Bosnia, plus some others when name-mapping fails): sum top-11
   Transfermarkt valuations among players whose `country_of_citizenship`
   matches the qualifier. Same fallback shape as `squad_values._citizenship_top26_at`,
   just at n=11.

The StatsBomb → results.csv team-name normalization handles the 3 known
mismatches (Côte d'Ivoire ↔ Ivory Coast, Cape Verde Islands ↔ Cape Verde,
Congo DR ↔ DR Congo).

Output: data/processed/wc2026_predicted_lineup_values.csv with one row per
qualifier and columns (team, snapshot_date, lineup_value_eur, n_starters,
source). The `source` field is "modal_xi" or "citizenship_top11" for audit.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.squad_values import (
    _build_player_value_lookup,
    _build_tm_name_index,
    _citizenship_top26_at,
    _normalize,
    _player_value_at,
)
from src.features.lineup_values import _match_starter
from src.features.club_lookup import _load_appearances_for, player_club_at_via_app
from src.features.lineup_elo import EloLookup, position_group, weighted_lineup_elo


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
ACTUAL_LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_actual_lineups.csv"
SB_TO_TM_PATH = PROJECT_ROOT / "data" / "processed" / "sb_player_to_tm.csv"
SQUAD_TO_SB_PATH = PROJECT_ROOT / "data" / "processed" / "wc2026_squad_to_sb.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "wc2026_predicted_lineup_values.csv"
ACTUAL_LINEUP_VALUES_PATH = PROCESSED_DIR / "wc2026_actual_lineup_values.csv"
PREDICTED_LINEUP_ELO_PATH = PROCESSED_DIR / "wc2026_predicted_lineup_elo.csv"
ACTUAL_LINEUP_ELO_PATH = PROCESSED_DIR / "wc2026_actual_lineup_elo.csv"

# Phase 2.2c: minimum in-squad SB players required to trust a squad-filtered
# modal XI. Below this, the modal XI is dominated by recently-retired or
# uncalled-up StatsBomb starters — better to fall through to the unfiltered
# top-11 (or the citizenship fallback if the team has thin SB coverage too).
MIN_IN_SQUAD_FOR_FILTER = 6

WC_2026_KICKOFF = date(2026, 6, 11)
LAST_K_MATCHES = 5
MIN_STARTERS_FROM_SB = 8  # need at least this many matched to TM to trust modal

# StatsBomb uses different spellings than results.csv for a few teams.
# Apply this map when reading SB data so team names align with results.csv.
SB_TO_RESULTS: dict[str, str] = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}

# results.csv (or other inputs) → Transfermarkt `country_of_citizenship`
# spelling. Reused from squad_values.py for the small federations.
RESULTS_TO_CITIZENSHIP: dict[str, str] = {
    "Curaçao": "Curacao",
    "Ivory Coast": "Cote d'Ivoire",
    "DR Congo": "Congo DR",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "South Korea": "Korea, South",
}


def _build_sb_to_tm_cache(
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    rebuild: bool = False,
) -> dict[int, int | None]:
    """Build (or load from cache) the StatsBomb → Transfermarkt player_id map.

    The fuzzy match is slow (~4 min for 1,800 unique starters). We cache the
    result so subsequent runs reuse it.
    """
    if SB_TO_TM_PATH.exists() and not rebuild:
        cached = pd.read_csv(SB_TO_TM_PATH)
        return dict(zip(cached["sb_player_id"], cached["tm_player_id"]))

    print(f"  building SB→TM cache (one-time, ~3-4 min)...")
    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())

    mapping: dict[int, int | None] = {}
    for sb_id, group in lineups.groupby("player_id"):
        first = group.iloc[0]
        tm_id = _match_starter(
            first["player_name"], first["player_nickname"] or "",
            tm_index, fuzzy_candidates,
        )
        mapping[int(sb_id)] = tm_id

    cache_df = pd.DataFrame([
        {"sb_player_id": k, "tm_player_id": v} for k, v in mapping.items()
    ])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(SB_TO_TM_PATH, index=False)
    print(f"  wrote {SB_TO_TM_PATH.relative_to(PROJECT_ROOT)}")
    return mapping


def predict_starting_xi(
    team: str,
    snapshot_date: date,
    lineups: pd.DataFrame,
    squad_filter: set[int] | None = None,
) -> list[int] | None:
    """Return predicted starting XI as list of StatsBomb player_ids.

    None if the team has fewer than 3 StatsBomb matches before snapshot_date.

    `squad_filter` is the set of SB player_ids for the team's current 26-man
    WC 2026 squad (from wc2026_squad_to_sb.csv). When provided and the team
    has at least MIN_IN_SQUAD_FOR_FILTER matches in `recent` who are also in
    `squad_filter`, the modal XI is restricted to those players — pads with
    unfiltered top appearance-makers when fewer than 11 in-squad players are
    available. When fewer than MIN_IN_SQUAD_FOR_FILTER in-squad players exist,
    the filter is dropped (modal XI is too thinly anchored to the current
    squad to trust) and the function falls back to the unfiltered top-11.
    """
    sb_team = next((k for k, v in SB_TO_RESULTS.items() if v == team), team)
    team_lineups = lineups[lineups["team"] == sb_team]
    snapshot_ts = pd.Timestamp(snapshot_date)
    team_lineups = team_lineups[pd.to_datetime(team_lineups["match_date"]) < snapshot_ts]

    n_matches = team_lineups["match_id"].nunique()
    if n_matches < 3:
        return None

    recent_match_ids = (
        team_lineups.drop_duplicates("match_id")
        .sort_values("match_date")
        .tail(LAST_K_MATCHES)["match_id"]
    )
    recent = team_lineups[team_lineups["match_id"].isin(recent_match_ids)]

    counts = recent.groupby("player_id").size().reset_index(name="n_starts")
    counts = counts.sort_values(["n_starts", "player_id"], ascending=[False, True])

    if squad_filter:
        in_squad = counts[counts["player_id"].isin(squad_filter)]
        if len(in_squad) >= MIN_IN_SQUAD_FOR_FILTER:
            picked = in_squad.head(11)["player_id"].astype(int).tolist()
            if len(picked) < 11:
                out_squad = counts[~counts["player_id"].isin(squad_filter)]
                picked.extend(
                    out_squad.head(11 - len(picked))["player_id"].astype(int).tolist()
                )
            return picked

    return counts.head(11)["player_id"].astype(int).tolist()


def load_squad_filters() -> dict[str, set[int]]:
    """Load wc2026_squad_to_sb.csv → {team → set of SB player_ids in squad}.

    Returns {} if the file doesn't exist yet (lets predict_starting_xi run
    pre-Phase-2.2c without crashing).
    """
    if not SQUAD_TO_SB_PATH.exists():
        return {}
    df = pd.read_csv(SQUAD_TO_SB_PATH)
    df = df[df["sb_player_id"].notna()]
    out: dict[str, set[int]] = {}
    for team, grp in df.groupby("team"):
        out[team] = set(grp["sb_player_id"].astype(int).tolist())
    return out


def predict_lineup_value(
    team: str,
    snapshot_date: date,
    lineups: pd.DataFrame,
    sb_to_tm: dict[int, int | None],
    valuations_by_player: dict[int, pd.DataFrame],
    players: pd.DataFrame,
    squad_filter: set[int] | None = None,
) -> tuple[float | None, int, str]:
    """Return (lineup_value_eur, n_matched_starters, source).

    source is "modal_xi" or "citizenship_top11".
    """
    sb_xi = predict_starting_xi(team, snapshot_date, lineups, squad_filter=squad_filter)

    if sb_xi is not None:
        # Modal-XI path
        values: list[float] = []
        for sb_pid in sb_xi:
            tm_pid = sb_to_tm.get(int(sb_pid))
            if tm_pid is None or pd.isna(tm_pid):
                continue
            v = _player_value_at(int(tm_pid), snapshot_date, valuations_by_player)
            if v is not None:
                values.append(v)

        if len(values) >= MIN_STARTERS_FROM_SB:
            return float(sum(values)), len(values), "modal_xi"

    # Citizenship fallback (top-11 instead of top-26)
    citizenship_name = RESULTS_TO_CITIZENSHIP.get(team, team)
    val = _citizenship_top26_at(
        citizenship_name, players, valuations_by_player, snapshot_date, top_n=11,
    )
    return (float(val) if val is not None else None), 11, "citizenship_top11"


def _modal_player_position(
    sb_player_id: int,
    recent_lineups: pd.DataFrame,
) -> str:
    """Most-common StatsBomb position for a player across the rows in `recent_lineups`.

    Used to bucket predicted-modal-XI players into GK/DEF/MID/FWD for the
    Phase 2.2d position-weighted Elo aggregation. Falls back to "MID" if
    the player never appears in `recent_lineups`.
    """
    rows = recent_lineups[recent_lineups["player_id"] == sb_player_id]
    if rows.empty:
        return "MID"
    pos = rows["position"].mode().iat[0]
    return position_group(pos)


def predict_lineup_elo(
    team: str,
    snapshot_date: date,
    lineups: pd.DataFrame,
    sb_to_tm: dict[int, int | None],
    elo: EloLookup,
    app_index: pd.DataFrame,
    squad_filter: set[int] | None = None,
) -> tuple[float | None, int, str]:
    """Return (lineup_elo_weighted, n_starters_with_elo, source) for a qualifier.

    source ∈ {modal_xi, no_xi}. There's no citizenship fallback here — Elo is
    a club-level signal and we have no clean way to synthesize a club from
    a citizenship row, so teams with no SB coverage emit None and the
    downstream imputer fills with the dataset median.
    """
    sb_xi = predict_starting_xi(team, snapshot_date, lineups, squad_filter=squad_filter)
    if sb_xi is None:
        return None, 0, "no_xi"

    sb_team = next((k for k, v in SB_TO_RESULTS.items() if v == team), team)
    snapshot_ts = pd.Timestamp(snapshot_date)
    team_lineups = lineups[
        (lineups["team"] == sb_team)
        & (pd.to_datetime(lineups["match_date"]) < snapshot_ts)
    ]
    recent_match_ids = (
        team_lineups.drop_duplicates("match_id")
        .sort_values("match_date")
        .tail(LAST_K_MATCHES)["match_id"]
    )
    recent = team_lineups[team_lineups["match_id"].isin(recent_match_ids)]

    starters_data: list[tuple[float, str]] = []
    for sb_pid in sb_xi:
        tm_pid = sb_to_tm.get(int(sb_pid))
        if tm_pid is None or pd.isna(tm_pid):
            continue
        club_id = player_club_at_via_app(int(tm_pid), snapshot_ts, app_index)
        if club_id is None:
            continue
        club_elo = elo.elo_at(club_id, snapshot_ts)
        if club_elo is None:
            continue
        starters_data.append((club_elo, _modal_player_position(int(sb_pid), recent)))

    if not starters_data:
        return None, 0, "modal_xi"
    return weighted_lineup_elo(starters_data), len(starters_data), "modal_xi"


def build_wc2026_predicted_lineup_elo(
    snapshot_date: date = WC_2026_KICKOFF,
) -> pd.DataFrame:
    """Per-qualifier modal-XI weighted Elo for WC 2026 (Phase 2.2d)."""
    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])
    players = pd.read_csv(TM_DIR / "players.csv")

    sb_to_tm = _build_sb_to_tm_cache(lineups, players)
    squad_filters = load_squad_filters()
    elo = EloLookup()

    needed_tm_pids = {int(v) for v in sb_to_tm.values() if v is not None and not pd.isna(v)}
    print(f"  loading appearances for {len(needed_tm_pids):,} TM players...")
    app_index = _load_appearances_for(needed_tm_pids)
    print(f"  {len(app_index):,} appearance rows")

    results = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "results.csv")
    results["date"] = pd.to_datetime(results["date"])
    wc26 = results[(results["date"] >= "2026-06-11") & (results["tournament"] == "FIFA World Cup")]
    qualifiers = sorted(set(wc26["home_team"]) | set(wc26["away_team"]))

    rows = []
    for team in qualifiers:
        elo_val, n, source = predict_lineup_elo(
            team, snapshot_date, lineups, sb_to_tm, elo, app_index,
            squad_filter=squad_filters.get(team),
        )
        rows.append({
            "team": team,
            "snapshot_date": snapshot_date.isoformat(),
            "lineup_elo_weighted": elo_val,
            "n_starters_with_elo": n,
            "source": source,
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PREDICTED_LINEUP_ELO_PATH, index=False)
    return out


def build_actual_lineup_elo() -> pd.DataFrame:
    """Per-(match, side) weighted Elo for the 12 played WC 2026 matches.

    Uses actual starters from data/raw/wc2026_actual_lineups.csv (which carries
    abbreviated position strings 'GK'/'CB'/'RB'/...). Mirrors
    `build_actual_lineup_values` for the Phase 2.2b lineup_value pipeline.
    """
    if not ACTUAL_LINEUPS_PATH.exists():
        raise FileNotFoundError(
            f"{ACTUAL_LINEUPS_PATH} not found. "
            f"Run `python -m src.data.wc2026_actual_lineups` first."
        )

    actual = pd.read_csv(ACTUAL_LINEUPS_PATH)
    actual["match_date"] = pd.to_datetime(actual["match_date"])
    print(f"loaded {len(actual):,} actual starter rows from "
          f"{actual['match_id'].nunique()} played WC 2026 matches")

    players = pd.read_csv(TM_DIR / "players.csv")
    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())
    elo = EloLookup()

    actual["tm_player_id"] = [
        _match_starter(r["player_name"], r["player_nickname"] or "", tm_index, fuzzy_candidates)
        for _, r in actual.iterrows()
    ]
    needed_tm_pids = {int(v) for v in actual["tm_player_id"].dropna().tolist()}
    app_index = _load_appearances_for(needed_tm_pids)

    rows = []
    for (mdate, home, away, side), grp in actual.groupby(
        ["match_date", "home_team", "away_team", "side"]
    ):
        starters_data: list[tuple[float, str]] = []
        for _, r in grp.iterrows():
            tm_pid = r["tm_player_id"]
            if tm_pid is None or pd.isna(tm_pid):
                continue
            club_id = player_club_at_via_app(int(tm_pid), pd.Timestamp(mdate), app_index)
            if club_id is None:
                continue
            club_elo = elo.elo_at(club_id, pd.Timestamp(mdate))
            if club_elo is None:
                continue
            starters_data.append((club_elo, position_group(r["position"])))

        rows.append({
            "match_date": mdate.date() if hasattr(mdate, "date") else mdate,
            "home_team": home,
            "away_team": away,
            "side": side,
            "lineup_elo_weighted": weighted_lineup_elo(starters_data),
            "n_starters_with_elo": len(starters_data),
            "n_starters_total": len(grp),
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ACTUAL_LINEUP_ELO_PATH, index=False)
    return out


def build_wc2026_predictions(snapshot_date: date = WC_2026_KICKOFF) -> pd.DataFrame:
    """Compute predicted lineup_value for every WC 2026 qualifier."""
    # Load inputs
    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])

    players = pd.read_csv(TM_DIR / "players.csv")
    valuations = pd.read_csv(TM_DIR / "player_valuations.csv")
    valuations_by_player = _build_player_value_lookup(valuations)

    sb_to_tm = _build_sb_to_tm_cache(lineups, players)
    squad_filters = load_squad_filters()

    # Get WC 2026 qualifiers from results.csv
    results = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "results.csv")
    results["date"] = pd.to_datetime(results["date"])
    wc26 = results[(results["date"] >= "2026-06-11") & (results["tournament"] == "FIFA World Cup")]
    qualifiers = sorted(set(wc26["home_team"]) | set(wc26["away_team"]))

    rows = []
    for team in qualifiers:
        v, n, source = predict_lineup_value(
            team, snapshot_date, lineups, sb_to_tm, valuations_by_player, players,
            squad_filter=squad_filters.get(team),
        )
        rows.append({
            "team": team,
            "snapshot_date": snapshot_date.isoformat(),
            "lineup_value_eur": v,
            "n_matched_starters": n,
            "source": source,
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


def build_actual_lineup_values(snapshot_date: date = WC_2026_KICKOFF) -> pd.DataFrame:
    """Compute per-(match, side) lineup_value for the 12 played WC 2026 matches.

    Uses actual lineups from data/raw/wc2026_actual_lineups.csv (Wikipedia-sourced,
    hardcoded in src/data/wc2026_actual_lineups.py). Same TM-matching shape as
    lineup_values.py for StatsBomb data.
    """
    if not ACTUAL_LINEUPS_PATH.exists():
        raise FileNotFoundError(
            f"{ACTUAL_LINEUPS_PATH} not found. "
            f"Run `python -m src.data.wc2026_actual_lineups` first."
        )

    actual = pd.read_csv(ACTUAL_LINEUPS_PATH)
    print(f"loaded {len(actual):,} actual starter rows from "
          f"{actual['match_id'].nunique()} played WC 2026 matches")

    players = pd.read_csv(TM_DIR / "players.csv")
    valuations = pd.read_csv(TM_DIR / "player_valuations.csv")
    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())
    valuations_by_player = _build_player_value_lookup(valuations)

    # Match each actual starter to a TM player_id and look up their value
    rows = []
    for (date_str, home, away, side), grp in actual.groupby(
        ["match_date", "home_team", "away_team", "side"]
    ):
        values: list[float] = []
        starter_names = []
        for _, r in grp.iterrows():
            tm_id = _match_starter(
                r["player_name"], r["player_nickname"] or "",
                tm_index, fuzzy_candidates,
            )
            starter_names.append(r["player_name"])
            if tm_id is None:
                continue
            v = _player_value_at(int(tm_id), snapshot_date, valuations_by_player)
            if v is not None:
                values.append(v)

        rows.append({
            "match_date": date_str,
            "home_team": home,
            "away_team": away,
            "side": side,
            "lineup_value_eur": sum(values) if values else None,
            "n_starters_matched": len(values),
            "n_starters_total": len(grp),
        })

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ACTUAL_LINEUP_VALUES_PATH, index=False)
    return out


def overlap_diagnostic() -> pd.DataFrame:
    """For each played match, compare predicted XI to actual XI at the SB player_id level.

    Both sides are mapped through wc2026_squad_to_sb.csv first:
      - predicted XI already comes back as SB player_ids
      - actual XI names get mapped to SB player_ids via the same squad→SB
        matcher used elsewhere (sorted-tokens for Asian name swaps, last-name
        / token-subset for mononyms)
    Overlap = |predicted_ids ∩ actual_ids|. Comparing at id level sidesteps
    the name-format mismatch that depressed the Phase 2.2b diagnostic (e.g.
    Korea read 0/11 only because "Heung-Min Son" ≠ "Son Heung-min" by last
    token, not because the predictor was actually missing the player).
    """
    actual = pd.read_csv(ACTUAL_LINEUPS_PATH)
    actual["match_date"] = actual["match_date"].astype(str)

    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])

    squad_filters = load_squad_filters()

    # Build squad→SB lookup keyed by (team, normalized squad_name) for matching
    # actual-lineup names to SB player_ids. Uses the same wc2026_squad_to_sb.csv
    # cache built in Phase 2.2c.
    squad_to_sb_df = pd.read_csv(SQUAD_TO_SB_PATH) if SQUAD_TO_SB_PATH.exists() else pd.DataFrame()
    squad_to_sb: dict[tuple[str, str], int] = {}
    for _, r in squad_to_sb_df.iterrows():
        if pd.notna(r["sb_player_id"]):
            squad_to_sb[(r["team"], _normalize(r["squad_name"]))] = int(r["sb_player_id"])

    rows = []
    for (date_str, home, away), grp in actual.groupby(
        ["match_date", "home_team", "away_team"]
    ):
        snapshot = pd.Timestamp(date_str).date()
        for team_label, team in [("home", home), ("away", away)]:
            actual_names = (
                grp[grp["side"] == team_label]["player_name"].astype(str).tolist()
            )
            actual_pids: set[int] = set()
            for n in actual_names:
                pid = squad_to_sb.get((team, _normalize(n)))
                if pid is not None:
                    actual_pids.add(pid)

            sb_pids = predict_starting_xi(
                team, snapshot, lineups, squad_filter=squad_filters.get(team),
            )
            source = "modal_xi" if sb_pids is not None else "fallback"
            predicted_pids = set(sb_pids) if sb_pids else set()

            overlap = len(predicted_pids & actual_pids)

            rows.append({
                "match_date": date_str,
                "team": team,
                "actual_count": len(actual_names),
                "actual_mapped": len(actual_pids),
                "predicted_count": len(predicted_pids),
                "overlap": overlap,
                "source": source,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_wc2026_predictions()
    print(f"\nwrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} qualifiers)")
    print()
    print("=== sources ===")
    print(df["source"].value_counts().to_string())
    print()
    print("=== teams with no value computed (will stay NaN) ===")
    print(df[df["lineup_value_eur"].isna()][["team", "source"]].to_string(index=False))
    print()
    print("=== top 10 by predicted lineup value ===")
    top10 = df.dropna(subset=["lineup_value_eur"]).sort_values("lineup_value_eur", ascending=False).head(10)
    top10 = top10.copy()
    top10["lineup_value_eur"] = top10["lineup_value_eur"].apply(lambda v: f"€{v:,.0f}")
    print(top10.to_string(index=False))
    print()
    print("=== bottom 10 ===")
    bot10 = df.dropna(subset=["lineup_value_eur"]).sort_values("lineup_value_eur").head(10)
    bot10 = bot10.copy()
    bot10["lineup_value_eur"] = bot10["lineup_value_eur"].apply(lambda v: f"€{v:,.0f}")
    print(bot10.to_string(index=False))

    # v2 Phase 2.2d: also emit lineup_elo predictions if the clubelo cache is
    # ready. Skipped silently otherwise — first-time setup is: run lineup_value
    # (this script) first, then `python -m src.data.clubelo_loader`, then
    # re-run this script.
    tm_to_clubelo_path = PROCESSED_DIR / "tm_club_to_clubelo.csv"
    if tm_to_clubelo_path.exists():
        print()
        print("=" * 60)
        print("Phase 2.2d: building WC 2026 lineup_elo predictions...")
        elo_df = build_wc2026_predicted_lineup_elo()
        print(f"  wrote {PREDICTED_LINEUP_ELO_PATH.relative_to(PROJECT_ROOT)} "
              f"({len(elo_df)} qualifiers, "
              f"{elo_df['lineup_elo_weighted'].notna().sum()} with Elo)")
        print()
        print("=== sources ===")
        print(elo_df["source"].value_counts().to_string())
        print()
        print("=== top 10 by predicted lineup Elo ===")
        top_elo = (
            elo_df.dropna(subset=["lineup_elo_weighted"])
            .sort_values("lineup_elo_weighted", ascending=False)
            .head(10)
        )
        print(top_elo.to_string(index=False))

        if ACTUAL_LINEUPS_PATH.exists():
            print()
            print("building actual lineup_elo for 12 played WC 2026 matches...")
            actual_df = build_actual_lineup_elo()
            print(f"  wrote {ACTUAL_LINEUP_ELO_PATH.relative_to(PROJECT_ROOT)} "
                  f"({len(actual_df)} rows)")
    else:
        print()
        print("(skipping Phase 2.2d outputs — run "
              "`python -m src.data.clubelo_loader` first)")

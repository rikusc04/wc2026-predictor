"""Map TM players to their club at any given date.

v2 Phase 2.2d. Two related lookups built from Transfermarkt's
appearances.csv + games.csv:

  - `attach_club_to_lineups`: vectorized merge_asof to add a `player_club_id`
    column to a lineups DataFrame, where club_id is from the player's most
    recent TM appearance ≤ match_date.

  - `attach_club_to_starters`: scalar version used by lineup_predictor.py
    when looking up predicted/actual WC 2026 starters one at a time.

  - `build_club_id_to_name`: harvest `club_id → club_name` from games.csv.

TM `appearances.csv` only covers club football (~1.9M rows, all major
European leagues + UCL/UEL etc.). National-team matches are not in it,
which is fine — we use national-team match_date as the *query date* and
find the player's CLUB on that date. Players outside TM's coverage
(e.g. MLS until recently, Liga MX, Saudi PL, smaller AFC leagues) get
NaN — handled downstream by the Elo lookup returning None.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
APPEARANCES_PATH = TM_DIR / "appearances.csv"
GAMES_PATH = TM_DIR / "games.csv"
LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
SB_TO_TM_PATH = PROJECT_ROOT / "data" / "processed" / "sb_player_to_tm.csv"


def build_club_id_to_name() -> dict[int, str]:
    """Return {club_id → most recent observed club_name} from games.csv.

    Most-recent so reorganized/renamed clubs land on their current name —
    that's the name the clubelo mapping audit reads.
    """
    games = pd.read_csv(
        GAMES_PATH,
        usecols=["home_club_id", "away_club_id", "home_club_name", "away_club_name", "date"],
        parse_dates=["date"],
    )
    games = games.sort_values("date")  # newer rows overwrite older

    out: dict[int, str] = {}
    for _, r in games.iterrows():
        if pd.notna(r["home_club_id"]) and isinstance(r["home_club_name"], str):
            out[int(r["home_club_id"])] = r["home_club_name"]
        if pd.notna(r["away_club_id"]) and isinstance(r["away_club_name"], str):
            out[int(r["away_club_id"])] = r["away_club_name"]
    return out


def _load_appearances_for(player_ids: set[int]) -> pd.DataFrame:
    """Return appearances rows for the given player_ids, sorted by date globally.

    Date-only sort serves both consumers: pd.merge_asof (which requires the
    `on` column globally monotonic on both sides — `by` groups for matching,
    it does NOT relax the sort requirement), and the scalar lookups in
    `player_club_at_via_app` which filter by player_id then take `.iloc[-1]`
    (still correct because date order is preserved through the filter).
    """
    app = pd.read_csv(
        APPEARANCES_PATH,
        usecols=["player_id", "player_club_id", "date"],
        parse_dates=["date"],
    )
    app = app[app["player_id"].isin(player_ids)]
    return app.sort_values("date").reset_index(drop=True)


def attach_club_to_lineups(
    lineups: pd.DataFrame,
    tm_player_id_col: str = "tm_player_id",
    match_date_col: str = "match_date",
) -> pd.DataFrame:
    """Add `player_club_id` to a lineups DataFrame via merge_asof.

    Caller must have already added `tm_player_id_col` (the TM player_id for
    each starter — see lineup_values.py for the matching logic). Rows where
    that column is NaN get NaN club_id, as do rows for players with no TM
    appearance history.
    """
    needed_pids = set(
        lineups[tm_player_id_col].dropna().astype(int).tolist()
    )
    if not needed_pids:
        out = lineups.copy()
        out["player_club_id"] = pd.NA
        return out

    app = _load_appearances_for(needed_pids)

    # merge_asof requires both frames globally sorted by the `on` key. The
    # `by` parameter groups rows for matching but doesn't relax that sort
    # requirement, so the left side gets sorted by match_date alone (not
    # [tm_player_id, match_date]). We restore caller row order at the end.
    lin = lineups.copy()
    lin["_orig_order"] = range(len(lin))
    lin[match_date_col] = pd.to_datetime(lin[match_date_col])

    with_pid = lin[lin[tm_player_id_col].notna()].copy()
    with_pid[tm_player_id_col] = with_pid[tm_player_id_col].astype(int)
    with_pid = with_pid.sort_values(match_date_col)

    merged = pd.merge_asof(
        with_pid,
        app.rename(columns={"date": "_app_date", "player_id": tm_player_id_col}),
        left_on=match_date_col,
        right_on="_app_date",
        by=tm_player_id_col,
        direction="backward",
    )
    merged = merged.drop(columns=["_app_date"])

    without_pid = lin[lin[tm_player_id_col].isna()].copy()
    without_pid["player_club_id"] = pd.NA

    out = pd.concat([merged, without_pid], ignore_index=True)
    out = out.sort_values("_orig_order").drop(columns=["_orig_order"]).reset_index(drop=True)
    return out


def player_club_at_via_app(
    tm_player_id: int,
    target_date: pd.Timestamp,
    app: pd.DataFrame,
) -> int | None:
    """Scalar: most recent player_club_id from a pre-filtered, sorted appearances df.

    `app` is expected to come from `_load_appearances_for` (sorted by
    player_id, date). Use the vectorized `attach_club_to_lineups` when
    processing more than a handful of rows.
    """
    rows = app[(app["player_id"] == tm_player_id) & (app["date"] <= target_date)]
    if rows.empty:
        return None
    return int(rows.iloc[-1]["player_club_id"])


def clubs_needed_for_lineups() -> set[int]:
    """Return the set of TM club_ids referenced by any starter in our pipeline.

    Used by clubelo_loader to decide which clubs to scrape. Covers both
    historical StatsBomb starters (backtest matches) and the WC 2026 squad.
    """
    lineups = pd.read_csv(LINEUPS_PATH)
    lineups["match_date"] = pd.to_datetime(lineups["match_date"])

    # SB → TM cache (built by lineup_values.py); skip if it doesn't exist yet.
    if not SB_TO_TM_PATH.exists():
        raise FileNotFoundError(
            f"{SB_TO_TM_PATH} not found. "
            f"Run lineup_values build first (it builds the SB→TM cache)."
        )
    sb_to_tm = pd.read_csv(SB_TO_TM_PATH)
    sb_to_tm = sb_to_tm[sb_to_tm["tm_player_id"].notna()]
    mapping = dict(zip(sb_to_tm["sb_player_id"], sb_to_tm["tm_player_id"].astype(int)))

    lineups["tm_player_id"] = lineups["player_id"].map(mapping)
    annotated = attach_club_to_lineups(lineups)
    club_ids = annotated["player_club_id"].dropna().astype(int).unique().tolist()
    return set(club_ids)

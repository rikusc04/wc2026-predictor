"""StatsBomb open-data lineup loader.

v2 Phase 2.1: pulls starting-XI data from StatsBomb's free public dataset
(github.com/statsbomb/open-data) for the men's international tournaments
they cover. Lineups feed the `lineup_value` feature in src/features/.

Coverage (as of repo audit):
    competition_id=43,  season=3    → FIFA World Cup 2018
    competition_id=43,  season=106  → FIFA World Cup 2022
    competition_id=55,  season=43   → UEFA Euro 2020 (played 2021)
    competition_id=55,  season=282  → UEFA Euro 2024
    competition_id=223, season=282  → Copa America 2024
    competition_id=1267, season=107 → AFCON 2023 (played early 2024)

Notable gap: WC 2014 isn't in StatsBomb's open data, so the lineup-value
feature will be NaN for our WC 2014 backtest and the imputer will handle it.

Output: data/raw/statsbomb_lineups.csv with one row per (match, starter).
Idempotent — skips download if the file already exists. Pass `force=True`
to refresh.

Wall time: ~3-5 minutes on first run (one HTTP request per match,
~400 matches, with a small polite delay).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.data.loader import PROJECT_ROOT, RAW_DIR


LINEUPS_PATH = RAW_DIR / "statsbomb_lineups.csv"

OPEN_DATA_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

COMPETITIONS: list[tuple[int, int, str]] = [
    (43, 3,    "FIFA World Cup 2018"),
    (43, 106,  "FIFA World Cup 2022"),
    (55, 43,   "UEFA Euro 2020"),
    (55, 282,  "UEFA Euro 2024"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "AFCON 2023"),
]


def _fetch_json(url: str) -> dict | list:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_matches(comp_id: int, season_id: int) -> list[dict]:
    return _fetch_json(f"{OPEN_DATA_BASE}/matches/{comp_id}/{season_id}.json")


def _fetch_lineup(match_id: int) -> list[dict]:
    return _fetch_json(f"{OPEN_DATA_BASE}/lineups/{match_id}.json")


def _extract_starting_xi(side_data: dict) -> list[dict]:
    """Return list of starter dicts for one side of one match."""
    starters = []
    for p in side_data["lineup"]:
        for pos in p.get("positions", []):
            if pos.get("start_reason") == "Starting XI":
                starters.append({
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "player_nickname": p.get("player_nickname") or "",
                    "jersey_number": p["jersey_number"],
                    "position_id": pos["position_id"],
                    "position": pos["position"],
                })
                break
    return starters


def build_lineups_table(force: bool = False, polite_delay_sec: float = 0.05) -> pd.DataFrame:
    """Fetch all lineups across the covered competitions; write CSV; return df."""
    if LINEUPS_PATH.exists() and not force:
        print(f"  already exists: {LINEUPS_PATH.relative_to(PROJECT_ROOT)}")
        return pd.read_csv(LINEUPS_PATH)

    rows: list[dict] = []
    for comp_id, season_id, label in COMPETITIONS:
        print(f"\nfetching {label} (comp={comp_id}, season={season_id})...")
        matches = _fetch_matches(comp_id, season_id)
        print(f"  {len(matches)} matches")

        for i, m in enumerate(matches):
            match_id = m["match_id"]
            date = m["match_date"]
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]

            try:
                lineup_data = _fetch_lineup(match_id)
            except Exception as e:
                print(f"  ! failed match {match_id}: {e}")
                continue

            for side_data in lineup_data:
                team = side_data["team_name"]
                is_home = team == home
                starters = _extract_starting_xi(side_data)
                for s in starters:
                    rows.append({
                        "match_id": match_id,
                        "match_date": date,
                        "competition": label,
                        "home_team": home,
                        "away_team": away,
                        "side": "home" if is_home else "away",
                        "team": team,
                        **s,
                    })

            if polite_delay_sec > 0:
                time.sleep(polite_delay_sec)

            if (i + 1) % 16 == 0:
                print(f"    fetched {i + 1}/{len(matches)} matches in {label}")

    df = pd.DataFrame(rows)
    LINEUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LINEUPS_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_lineups_table()
    print(f"\nwrote {LINEUPS_PATH.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")
    print(f"\ncoverage: {df['competition'].value_counts().to_string()}")
    print(f"\ndistinct matches: {df['match_id'].nunique()}")
    print(f"distinct teams: {df['team'].nunique()}")
    print(f"distinct players: {df['player_id'].nunique()}")

    print(f"\nstarters per side (expect 11): {df.groupby(['match_id', 'side']).size().describe()}")

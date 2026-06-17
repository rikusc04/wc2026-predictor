"""Build a per-(team, year) squad market value table.

Data sources:
  - Fjelstul WC database (data/raw/wc_squads_fjelstul.csv) — actual rosters
    for every WC 1930-2022 with player names by team.
  - Transfermarkt mirror (data/raw/transfermarkt/) — players.csv,
    player_valuations.csv, national_teams.csv. Gives us each player's
    market value over time.

Pipeline:
  1. Take each (WC year, team, player) from Fjelstul.
  2. Match the player name to a Transfermarkt player_id by normalized
     full-name lookup (with fuzzy fallback for tricky matches).
  3. For matched players, look up their valuation at WC kickoff date.
  4. Sum top SQUAD_SIZE values → squad market value.
  5. For WC 2026 (post-Fjelstul), use the current national_teams snapshot.

Output: data/processed/squad_values.csv with columns:
    year, team_name, squad_value_eur, n_matched, n_total
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import PROJECT_ROOT


TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
FJELSTUL_PATH = PROJECT_ROOT / "data" / "raw" / "wc_squads_fjelstul.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "squad_values.csv"

# WC kickoff dates — we look up valuations as-of-or-before this date.
# Fjelstul covers Men's WCs 2006-2022; we add 2026 separately.
WC_KICKOFFS = {
    2006: date(2006, 6, 9),
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
    2022: date(2022, 11, 20),
}

# Standard WC squad size. Was 23 pre-2022; 26 from 2022 onward.
# Generous cap — we take min(n_matched, SQUAD_SIZE) per team.
SQUAD_SIZE = 26

# Fuzzy-match threshold (0-1). Anything below this is considered no match.
FUZZY_THRESHOLD = 0.85


def _normalize(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _build_tm_name_index(players: pd.DataFrame) -> dict[str, list[int]]:
    """Map normalized full name → list of TM player_ids (handles duplicates)."""
    index: dict[str, list[int]] = defaultdict(list)
    for _, row in players.iterrows():
        for key in (row.get("name"), f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()):
            norm = _normalize(key)
            if norm:
                index[norm].append(int(row["player_id"]))
    return dict(index)


def _fuzzy_lookup(target: str, candidates: list[str]) -> str | None:
    """Return best match from candidates if score >= threshold, else None."""
    best_name: str | None = None
    best_score = 0.0
    for cand in candidates:
        score = SequenceMatcher(None, target, cand).ratio()
        if score > best_score:
            best_score = score
            best_name = cand
    if best_score >= FUZZY_THRESHOLD:
        return best_name
    return None


def _match_player(
    given: str,
    family: str,
    tm_index: dict[str, list[int]],
    fuzzy_candidates: list[str],
) -> int | None:
    """Find a TM player_id for this Fjelstul name. None if no confident match.

    Handles mononyms (common for Brazilian players): Fjelstul stores them as
    family_name="Neymar", given_name="not applicable" — we use family_name
    alone in that case.
    """
    # Brazilian-style mononyms: given_name is "not applicable" sentinel
    if not isinstance(given, str) or given.strip().lower() in ("not applicable", "n/a", "", "nan"):
        given = ""
    if not isinstance(family, str):
        family = ""

    candidates_to_try = [
        f"{given} {family}".strip(),
        f"{family} {given}".strip(),
        family.strip() if not given else "",  # mononym fallback
        given.strip() if not family else "",
    ]
    candidates_to_try = [c for c in candidates_to_try if c]

    # Direct hits first
    for cand in candidates_to_try:
        norm = _normalize(cand)
        if norm and norm in tm_index:
            return tm_index[norm][0]

    # Fuzzy fallback on the most-complete name we have
    primary = candidates_to_try[0]
    best = _fuzzy_lookup(_normalize(primary), fuzzy_candidates)
    if best is not None:
        return tm_index[best][0]
    return None


def _build_player_value_lookup(valuations: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group valuations by player_id, sorted by date. Indexed for fast lookup."""
    val = valuations.sort_values(["player_id", "date"])
    return {pid: g for pid, g in val.groupby("player_id")}


def _player_value_at(
    player_id: int,
    target_date: date,
    valuations_by_player: dict[int, pd.DataFrame],
) -> float | None:
    """Most recent valuation on or before target_date. None if no data."""
    g = valuations_by_player.get(player_id)
    if g is None:
        return None
    target_str = str(target_date)
    before = g[g["date"] <= target_str]
    if before.empty:
        return None
    return float(before["market_value_in_eur"].iloc[-1])


def _citizenship_top26_at(country_name: str, players: pd.DataFrame,
                          valuations_by_player: dict[int, pd.DataFrame],
                          target_date: date, top_n: int = SQUAD_SIZE) -> float | None:
    """Sum top-N player valuations as of `target_date` for players with this
    citizenship. Used for historical snapshots of non-WC-participant teams."""
    sub = players[players["country_of_citizenship"] == country_name]
    values: list[float] = []
    for pid in sub["player_id"]:
        v = _player_value_at(int(pid), target_date, valuations_by_player)
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort(reverse=True)
    return float(sum(values[:top_n]))


def build_historical(
    fjelstul: pd.DataFrame,
    tm_index: dict[str, list[int]],
    fuzzy_candidates: list[str],
    valuations_by_player: dict[int, pd.DataFrame],
    players: pd.DataFrame,
    national_teams: pd.DataFrame,
    cal_factor: float,
) -> pd.DataFrame:
    rows = []
    nt_names = national_teams["name"].tolist()

    for year, kickoff in WC_KICKOFFS.items():
        tournament_id = f"WC-{year}"
        wc_rows = fjelstul[fjelstul["tournament_id"] == tournament_id]
        wc_team_names: set[str] = set()
        n_fjelstul = 0
        n_citizenship = 0

        # Pass 1: WC participants via Fjelstul rosters (accurate)
        for team_name, team_group in wc_rows.groupby("team_name"):
            values: list[float] = []
            n_total = len(team_group)
            n_matched = 0
            for _, p in team_group.iterrows():
                pid = _match_player(p["given_name"], p["family_name"], tm_index, fuzzy_candidates)
                if pid is None:
                    continue
                v = _player_value_at(pid, kickoff, valuations_by_player)
                if v is None:
                    continue
                values.append(v)
                n_matched += 1
            values.sort(reverse=True)
            squad_value = float(sum(values[:SQUAD_SIZE]))
            rows.append({
                "year": year,
                "team_name": team_name,
                "squad_value_eur": squad_value,
                "source": "fjelstul_roster",
                "n_matched": n_matched,
                "n_total": n_total,
            })
            wc_team_names.add(team_name)
            n_fjelstul += 1

        # Pass 2: every other TM-tracked team via citizenship (approximate)
        for tm_name in nt_names:
            team_name = TM_NAME_FIXUPS.get(tm_name, tm_name)
            if team_name in wc_team_names:
                continue
            cit_value = _citizenship_top26_at(
                tm_name, players, valuations_by_player, kickoff
            )
            if cit_value is None:
                continue
            rows.append({
                "year": year,
                "team_name": team_name,
                "squad_value_eur": float(cit_value * cal_factor),
                "source": "citizenship_calibrated_historical",
                "n_matched": SQUAD_SIZE,
                "n_total": SQUAD_SIZE,
            })
            n_citizenship += 1

        print(f"  WC {year}: {n_fjelstul} Fjelstul + {n_citizenship} citizenship "
              f"= {n_fjelstul + n_citizenship} teams")

    return pd.DataFrame(rows)


def _wc_2026_participants() -> set[str]:
    """The 48 teams actually playing in WC 2026.

    Read from the WC 2026 fixture rows in results.csv (those have NaN scores
    for unplayed games but home_team / away_team are populated).
    """
    from src.data.loader import load_results
    df, _ = load_results(apply_cutoff=False)
    df["date"] = pd.to_datetime(df["date"])
    wc26 = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-11")]
    return set(wc26["home_team"]).union(set(wc26["away_team"]))


# Some team-name spellings differ between Transfermarkt's national_teams.csv
# and our results.csv. Map TM → results convention.
TM_NAME_FIXUPS = {
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# For WC 2026 qualifiers NOT in national_teams.csv at all (small federations),
# we fall back to citizenship-based lookup. These teams may use a different
# spelling in `players.country_of_citizenship` than in results.csv.
# Map results.csv name → citizenship-column spelling.
RESULTS_TO_CITIZENSHIP = {
    "Curaçao": "Curacao",
    "Ivory Coast": "Cote d'Ivoire",
    # Cape Verde, DR Congo, Haiti match directly — no fixup needed
}


def _citizenship_top26(country_name: str, players: pd.DataFrame, top_n: int = 26) -> float | None:
    """Sum top-N market values for all players with this country_of_citizenship.

    Overestimates the actual national-team value (it includes every eligible
    player, not just the call-up squad). Used as a fallback with calibration
    when the official national-team aggregate is missing.
    """
    sub = players[players["country_of_citizenship"] == country_name]
    values = sub["market_value_in_eur"].dropna().astype(float)
    if values.empty:
        return None
    return float(values.sort_values(ascending=False).head(top_n).sum())


def _calibration_factor(national_teams: pd.DataFrame, players: pd.DataFrame) -> float:
    """Ratio of (TM aggregate) / (citizenship top-26), computed for teams
    with deep eligibility pools where the "calling-up haircut" actually
    matters.

    For most countries (Estonia, Bolivia, etc.) the top-26 by citizenship
    IS the active squad, so the ratio is ~1.0. For big football nations
    (England, France, Germany) the citizenship pool is much larger than
    the squad, so the ratio is ~0.6-0.7. We need the big-nation ratio
    because that's what applies to teams missing the aggregate.
    """
    # Collect (tm_value, cit_value) pairs for all teams with both signals
    pairs = []
    for _, row in national_teams.iterrows():
        tm_value = row["total_market_value"]
        if pd.isna(tm_value):
            continue
        cit_value = _citizenship_top26(row["name"], players)
        if cit_value is None or cit_value == 0:
            continue
        pairs.append((float(tm_value), float(cit_value)))

    if not pairs:
        return 1.0

    # Restrict to teams with deep talent pools — the top half by citizenship_top26.
    # These are the teams where the "eligible but not called up" gap is large,
    # which is what we expect for the missing teams (England, France, Spain).
    pairs.sort(key=lambda p: p[1], reverse=True)
    big = pairs[: max(1, len(pairs) // 2)]
    # Use total-over-total (magnitude-weighted) rather than median of ratios —
    # gives more weight to teams whose absolute values are large.
    total_tm = sum(p[0] for p in big)
    total_cit = sum(p[1] for p in big)
    return float(total_tm / total_cit) if total_cit > 0 else 1.0


def build_current_2026(national_teams: pd.DataFrame, players: pd.DataFrame,
                       cal_factor: float | None = None) -> pd.DataFrame:
    """For WC 2026, use the current TM snapshot, with two extensions:

    1. Filter to actual WC 2026 qualifiers (read from results.csv fixtures).
    2. For qualifiers missing `total_market_value` (e.g., England, France,
       Spain in the public mirror), fall back to a calibrated estimate
       based on top-26 player market values filtered by `country_of_citizenship`.
    """
    participants = _wc_2026_participants()
    cal = cal_factor if cal_factor is not None else _calibration_factor(national_teams, players)

    nt = national_teams.copy()
    nt["team_name"] = nt["name"].replace(TM_NAME_FIXUPS)
    nt = nt[nt["team_name"].isin(participants)].copy()

    rows = []
    for _, row in nt.iterrows():
        value = row["total_market_value"]
        source = "tm_aggregate"
        if pd.isna(value):
            # Fall back to citizenship-based estimate, calibrated
            cit_value = _citizenship_top26(row["name"], players)
            if cit_value is not None:
                value = cit_value * cal
                source = "citizenship_calibrated"
        if value is None or pd.isna(value):
            continue
        rows.append({
            "year": 2026,
            "team_name": row["team_name"],
            "squad_value_eur": float(value),
            "source": source,
            "n_matched": int(row["squad_size"]) if pd.notna(row["squad_size"]) else 26,
            "n_total": int(row["squad_size"]) if pd.notna(row["squad_size"]) else 26,
        })

    # Recover qualifiers that aren't in national_teams.csv at all by going
    # straight to country_of_citizenship in players.csv (with optional
    # spelling fixup), then applying the same calibration factor.
    already_have = {r["team_name"] for r in rows}
    for team_name in participants - already_have:
        citizenship_name = RESULTS_TO_CITIZENSHIP.get(team_name, team_name)
        cit_value = _citizenship_top26(citizenship_name, players)
        if cit_value is None:
            continue
        rows.append({
            "year": 2026,
            "team_name": team_name,
            "squad_value_eur": float(cit_value * cal),
            "source": "citizenship_only",
            "n_matched": 26,
            "n_total": 26,
        })

    return pd.DataFrame(rows)


def build_all() -> pd.DataFrame:
    print("loading sources...")
    fjelstul = pd.read_csv(FJELSTUL_PATH)
    players = pd.read_csv(TM_DIR / "players.csv", low_memory=False)
    valuations = pd.read_csv(TM_DIR / "player_valuations.csv", low_memory=False)
    national_teams = pd.read_csv(TM_DIR / "national_teams.csv", low_memory=False)
    print(f"  fjelstul squads:       {len(fjelstul):>9,}")
    print(f"  TM players:            {len(players):>9,}")
    print(f"  TM valuations:         {len(valuations):>9,}")
    print(f"  TM national_teams:     {len(national_teams):>9,}")

    print()
    print("building TM name index...")
    tm_index = _build_tm_name_index(players)
    fuzzy_candidates = list(tm_index.keys())
    print(f"  {len(tm_index):,} distinct normalized names")

    print()
    print("indexing valuations by player...")
    valuations_by_player = _build_player_value_lookup(valuations)
    print(f"  {len(valuations_by_player):,} players with at least one valuation")

    print()
    print("computing calibration factor (used by citizenship-based fallback)...")
    cal = _calibration_factor(national_teams, players)
    print(f"  calibration factor: {cal:.3f}")

    print()
    print("computing historical squad values...")
    historical = build_historical(
        fjelstul, tm_index, fuzzy_candidates, valuations_by_player,
        players, national_teams, cal,
    )

    print()
    print("adding WC 2026 current snapshot...")
    current = build_current_2026(national_teams, players, cal_factor=cal)
    print(f"  {len(current)} teams")

    return pd.concat([historical, current], ignore_index=True)


if __name__ == "__main__":
    df = build_all()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df):,} rows)")
    print()
    for year in [2006, 2010, 2014, 2018, 2022, 2026]:
        if year == 2026:
            label = "WC 2026 qualifiers — top 10 squads"
        else:
            label = (
                f"All teams snapshotted at {year} (incl. non-WC participants); "
                f"WC participants in this view marked with ✓"
            )
        print(f"=== {label} ===")
        year_view = df[df["year"] == year].copy()
        year_view["squad_value_eur_M"] = (year_view["squad_value_eur"] / 1_000_000).round(1)
        if year != 2026:
            year_view["wc"] = year_view["source"].apply(
                lambda s: "✓" if s == "fjelstul_roster" else " "
            )
            view = year_view.sort_values("squad_value_eur", ascending=False).head(10)
            cols = ["wc", "team_name", "squad_value_eur_M", "n_matched", "n_total"]
        else:
            view = year_view.sort_values("squad_value_eur", ascending=False).head(10)
            cols = ["team_name", "squad_value_eur_M"]
        print(view[cols].to_string(index=False))
        print()

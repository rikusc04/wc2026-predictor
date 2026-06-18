"""Map WC 2026 squad players to StatsBomb player_ids.

v2 Phase 2.2c. Drives the squad filter in lineup_predictor.predict_starting_xi:
modal-XI is restricted to players whose StatsBomb player_id is in the team's
current 26-man squad. Without this filter the predictor often picks players
who aren't even called up (e.g. Neymar for Brazil).

The naive normalized-name match misses two important cases:

1. **Asian name-order swap**: Wikipedia uses traditional family-first order
   ("Kim Min-jae"); StatsBomb uses given-first ("Min-jae Kim"). Per the
   Phase 2.2b diagnostic this drove South Korea from a plausible ~6/11
   overlap down to 0/11. We fix it with a sorted-tokens equality check.

2. **Brazilian / Portuguese mononyms**: Wikipedia has "Vinícius Júnior",
   StatsBomb has "Vinícius José Paixão de Oliveira Júnior". Same person.
   We fix it with a last-token (surname) match, accepted only when unique
   among the team's StatsBomb starters.

Output: data/processed/wc2026_squad_to_sb.csv with columns
    team, squad_name, sb_player_id, sb_player_name, match_strategy
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.squad_values import _normalize


SQUADS_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_squads.csv"
SB_LINEUPS_PATH = PROJECT_ROOT / "data" / "raw" / "statsbomb_lineups.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "wc2026_squad_to_sb.csv"

# Same mapping as in lineup_predictor.py — StatsBomb's spellings differ
# from results.csv (and our wc2026_squads.csv) for a few teams.
SB_TO_RESULTS: dict[str, str] = {
    "Côte d'Ivoire": "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR": "DR Congo",
}


def _dehyphen(name: str) -> str:
    """Normalize + replace hyphens with spaces.

    Korean players differ in hyphenation between sources: Wikipedia writes
    "Min-jae", StatsBomb writes "Min Jae". Treating both as space-separated
    tokens lets the matcher line them up.
    """
    return _normalize(name).replace("-", " ")


def _sorted_tokens(name: str) -> str:
    """Normalize + dehyphenate + sort tokens — order-invariant key.

    Order-invariant (handles Asian family-first vs given-first swap) and
    hyphen-invariant (handles Korean / Vietnamese hyphenation differences).
    """
    return " ".join(sorted(_dehyphen(name).split()))


def _last_token(name: str) -> str:
    """Last whitespace-separated token of the normalized name."""
    toks = _normalize(name).split()
    return toks[-1] if toks else ""


def _build_sb_team_index(sb: pd.DataFrame) -> dict[str, list[dict]]:
    """For each team, a list of {player_id, name, nickname} unique entries.

    `team` keys use the results.csv spelling (after SB→results normalization).
    """
    out: dict[str, list[dict]] = {}
    for team_sb, grp in sb.groupby("team"):
        team_results = SB_TO_RESULTS.get(team_sb, team_sb)
        seen: set[int] = set()
        entries: list[dict] = []
        for _, r in grp.iterrows():
            pid = int(r["player_id"])
            if pid in seen:
                continue
            seen.add(pid)
            entries.append({
                "player_id": pid,
                "name": str(r.get("player_name") or ""),
                "nickname": str(r.get("player_nickname") or ""),
            })
        out.setdefault(team_results, []).extend(entries)
    return out


def _match_one(
    squad_name: str,
    sb_entries: list[dict],
    claimed: set[int],
) -> tuple[int | None, str]:
    """Return (sb_player_id, strategy). strategy='none' on miss.

    `claimed` is mutated: matched SB player_ids are added so a later squad
    member with the same surname won't claim the same SB player (e.g. USA's
    Miles Robinson + Antonee Robinson — only one should resolve to each SB id).
    Exact matches still claim — earlier squad rows win contested SB ids.
    """
    norm_squad = _normalize(squad_name)
    if not norm_squad:
        return None, "none"

    sb_norms = []
    for e in sb_entries:
        if e["player_id"] in claimed:
            continue
        sb_norms.append({
            "player_id": e["player_id"],
            "name_norm": _normalize(e["name"]),
            "nick_norm": _normalize(e["nickname"]),
            "name_sorted": _sorted_tokens(e["name"]),
            "nick_sorted": _sorted_tokens(e["nickname"]),
            "name_last": _last_token(e["name"]),
            "nick_last": _last_token(e["nickname"]),
        })

    for e in sb_norms:
        if norm_squad == e["name_norm"] or norm_squad == e["nick_norm"]:
            claimed.add(e["player_id"])
            return e["player_id"], "exact"

    sorted_squad = _sorted_tokens(squad_name)
    for e in sb_norms:
        if sorted_squad == e["name_sorted"] or sorted_squad == e["nick_sorted"]:
            claimed.add(e["player_id"])
            return e["player_id"], "sorted_tokens"

    last_squad = _last_token(squad_name)
    if last_squad:
        cands = [
            e for e in sb_norms
            if last_squad == e["name_last"] or last_squad == e["nick_last"]
        ]
        if len(cands) == 1:
            claimed.add(cands[0]["player_id"])
            return cands[0]["player_id"], "last_name"

    squad_toks = set(norm_squad.split())
    if squad_toks:
        cands = []
        for e in sb_norms:
            name_toks = set(e["name_norm"].split())
            nick_toks = set(e["nick_norm"].split())
            if squad_toks.issubset(name_toks) or (nick_toks and squad_toks.issubset(nick_toks)):
                cands.append(e)
        if len(cands) == 1:
            claimed.add(cands[0]["player_id"])
            return cands[0]["player_id"], "token_subset"

    return None, "none"


def build_squad_to_sb() -> pd.DataFrame:
    """Match every WC 2026 squad player to a StatsBomb player_id (if possible)."""
    squads = pd.read_csv(SQUADS_PATH)
    sb = pd.read_csv(SB_LINEUPS_PATH)

    sb_by_team = _build_sb_team_index(sb)

    rows: list[dict] = []
    for team, team_squads in squads.groupby("team"):
        sb_entries = sb_by_team.get(team, [])
        claimed: set[int] = set()
        for _, r in team_squads.iterrows():
            squad_name = r["player_name"]
            pid, strategy = _match_one(squad_name, sb_entries, claimed)
            sb_name = ""
            if pid is not None:
                sb_name = next((e["name"] for e in sb_entries if e["player_id"] == pid), "")
            rows.append({
                "team": team,
                "squad_name": squad_name,
                "sb_player_id": pid,
                "sb_player_name": sb_name,
                "match_strategy": strategy,
            })

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    df = build_squad_to_sb()
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print()
    print("=== match strategy distribution ===")
    print(df["match_strategy"].value_counts().to_string())
    print()
    matched = df[df["sb_player_id"].notna()]
    print(f"\nmatched: {len(matched)}/{len(df)} ({len(matched)/len(df):.1%})")
    print()
    print("=== teams with lowest match rate ===")
    per_team = (
        df.assign(matched=df["sb_player_id"].notna())
          .groupby("team")["matched"].agg(["sum", "count"])
    )
    per_team["rate"] = per_team["sum"] / per_team["count"]
    print(per_team.sort_values("rate").head(15).to_string())

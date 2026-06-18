"""Scrape WC 2026 26-man squads from Wikipedia.

v2 Phase 2.2c. The modal-XI predictor in lineup_predictor.py picks the top 11
StatsBomb appearance-makers from each team's last 5 matches — but StatsBomb
data is 1-4 years old, so it often picks players who aren't even in the
current WC 2026 squad (e.g. predicts Neymar starts for Brazil although he
wasn't called up). This module pulls the published 26-man squads so we can
filter the modal XI to currently-rostered players only.

Source: en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads. The page uses
{{nat fs g player|no=X|pos=Y|name=[[Z]]|...}} MediaWiki templates per player,
one block per team under a `===TeamName===` heading.

Output: data/raw/wc2026_squads.csv with columns
    team, jersey, position, player_name, sortname, club
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_squads.csv"
WIKI_PAGE = "2026_FIFA_World_Cup_squads"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Wikipedia spells a couple of teams differently than results.csv.
# Keep results.csv spelling on the output side so downstream joins line up.
WIKI_TO_RESULTS: dict[str, str] = {
    "Republic of Ireland": "Ireland",
}


def fetch_wikitext() -> str:
    """Fetch the WC 2026 squads page wikitext via Wikipedia's MediaWiki API."""
    cmd = [
        "curl", "-sS", "--max-time", "30",
        "-A", "wc2026-predictor-research/0.1 (research; rs8057@nyu.edu)",
        f"{WIKI_API}?action=parse&page={WIKI_PAGE}&format=json&prop=wikitext",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    return payload["parse"]["wikitext"]["*"]


_PLAYER_TEMPLATE_START_RE = re.compile(r"\{\{\s*nat fs g player\b", re.IGNORECASE)
_TEAM_HEADING_RE = re.compile(r"^===\s*([^=]+?)\s*===\s*$", re.MULTILINE)
_GROUP_HEADING_RE = re.compile(r"^==\s*Group\s+[A-L]\s*==\s*$", re.MULTILINE)


def _find_player_templates(body: str) -> list[str]:
    """Return the inner body of every `{{nat fs g player|...}}` in `body`.

    Walks the string and tracks `{{...}}` depth so nested templates
    (e.g. `{{birth date and age2|...}}`) don't terminate the outer match.
    """
    results: list[str] = []
    for m in _PLAYER_TEMPLATE_START_RE.finditer(body):
        # Find the `|` after `nat fs g player` (skipping whitespace)
        i = m.end()
        while i < len(body) and body[i] not in "|}":
            i += 1
        if i >= len(body) or body[i] != "|":
            continue
        i += 1  # past the `|`
        depth = 1
        start = i
        while i < len(body) - 1 and depth > 0:
            if body[i] == "{" and body[i + 1] == "{":
                depth += 1
                i += 2
            elif body[i] == "}" and body[i + 1] == "}":
                depth -= 1
                if depth == 0:
                    results.append(body[start:i])
                    i += 2
                    break
                i += 2
            else:
                i += 1
    return results


def _parse_template_args(body: str) -> dict[str, str]:
    """Parse pipe-separated key=value args from a MediaWiki template body.

    Splits on `|` but respects `[[...]]` and `{{...}}` nesting depths.
    """
    args: dict[str, str] = {}
    depth_sq = 0  # [[...]]
    depth_cu = 0  # {{...}}
    current: list[str] = []
    parts: list[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "[" and i + 1 < len(body) and body[i + 1] == "[":
            depth_sq += 1
            current.append("[[")
            i += 2
            continue
        if c == "]" and i + 1 < len(body) and body[i + 1] == "]":
            depth_sq -= 1
            current.append("]]")
            i += 2
            continue
        if c == "{" and i + 1 < len(body) and body[i + 1] == "{":
            depth_cu += 1
            current.append("{{")
            i += 2
            continue
        if c == "}" and i + 1 < len(body) and body[i + 1] == "}":
            depth_cu -= 1
            current.append("}}")
            i += 2
            continue
        if c == "|" and depth_sq == 0 and depth_cu == 0:
            parts.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        parts.append("".join(current).strip())

    for p in parts:
        if "=" not in p:
            continue
        k, _, v = p.partition("=")
        args[k.strip().lower()] = v.strip()
    return args


def _clean_wikilink(text: str) -> str:
    """Extract the display text from a wikilink: [[Foo|Bar]] → Bar, [[Foo]] → Foo."""
    text = text.strip()
    # Strip surrounding [[ ]]
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2]
    # Take the display side of any pipe
    if "|" in text:
        text = text.rsplit("|", 1)[1]
    # Strip any leftover wikitext formatting
    text = re.sub(r"'''?", "", text).strip()
    return text


def parse_squads(wikitext: str) -> pd.DataFrame:
    """Walk the wikitext and pull out every (team, player) row."""
    # Drop everything before the first Group heading to avoid lead-section noise.
    first_group = _GROUP_HEADING_RE.search(wikitext)
    if first_group:
        wikitext = wikitext[first_group.start():]

    # Find team-heading positions; the slice between two consecutive headings
    # is the body for the first one.
    headings = [(m.start(), _clean_wikilink(m.group(1))) for m in _TEAM_HEADING_RE.finditer(wikitext)]
    headings.append((len(wikitext), "__END__"))

    rows: list[dict[str, object]] = []
    for (start, team), (end, _) in zip(headings, headings[1:]):
        body = wikitext[start:end]
        team_clean = WIKI_TO_RESULTS.get(team, team)
        for inner in _find_player_templates(body):
            args = _parse_template_args(inner)
            name = _clean_wikilink(args.get("name", ""))
            if not name:
                continue
            jersey_raw = args.get("no", "")
            try:
                jersey = int(jersey_raw) if jersey_raw.strip().isdigit() else None
            except ValueError:
                jersey = None
            rows.append({
                "team": team_clean,
                "jersey": jersey,
                "position": args.get("pos", "").upper(),
                "player_name": name,
                "sortname": args.get("sortname", ""),
                "club": _clean_wikilink(args.get("club", "")),
            })

    return pd.DataFrame(rows)


def build_wc2026_squads() -> pd.DataFrame:
    wikitext = fetch_wikitext()
    df = parse_squads(wikitext)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_wc2026_squads()
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print()
    print("=== rows per team ===")
    counts = df.groupby("team").size().sort_values()
    print(counts.to_string())
    print()
    print(f"teams: {df['team'].nunique()}")
    print(f"mean squad size: {counts.mean():.1f}")
    print(f"min/max squad size: {counts.min()} / {counts.max()}")

# 06 — Advanced features: multi-source data integration

You've validated the baseline model. It's beating naive Elo by a slim margin. EDA suggests the limitation is your feature set — most of your engineered features (recent form, days since last match, tournament class) are partly redundant with Elo. To meaningfully improve, you need genuinely new information.

This chapter walks through adding **squad market value** as a feature — the case study for any "integrate a second data source" task in ML. We cover:

- Where to find data when the obvious source blocks scraping.
- How to combine multiple sources (rosters from one, valuations from another).
- The name-matching problem and how to solve it.
- Sentinel-value gotchas in real-world data.
- Why some of your "fix-the-bug-and-it-works" intuitions don't apply to data engineering.

## Why squad market value?

Elo captures *team identity* — how strong is the country labeled "Brazil." But teams are made of players, and squads change. The 2014 Brazil that lost 7-1 to Germany was missing Neymar (injured) and Thiago Silva (suspended) — neither absence is visible to Elo, which sees only the team label.

A "squad market value" feature represents *the available talent at a given point in time*. It's correlated with Elo for stable teams (Germany has been rich both in 2014 and 2024), but de-correlated for teams with major squad turnover (Italy 2014 vs. Italy 2024 are very different teams in terms of star players).

Adding this feature gives the model information Elo cannot extract from match results alone.

## The data we need

For each (team, date) in our training set, we want: **what was the total market value of this team's available squad at that date?**

### Time-series data and "snapshots"

A team's squad value changes over time — players gain or lose value, new players break through, veterans retire. To use squad value as a feature, we need to know each team's value *at the time of each match in our training data*, not just "today's value."

We could in principle track every team's value every day, but that's expensive and unnecessary — squad values change slowly between tournaments. So we record values at **anchor dates** spaced ~4 years apart, called **snapshots**.

A snapshot is just "the value of something at a specific moment in time." For us, the natural anchor dates are the kickoffs of past World Cups:

- 2006 snapshot — June 9, 2006
- 2010 snapshot — June 11, 2010
- 2014 snapshot — June 12, 2014
- 2018 snapshot — June 14, 2018
- 2022 snapshot — November 20, 2022
- 2026 snapshot — June 11, 2026

Six snapshots × ~100 teams per snapshot ≈ 600 (team, value) data points. Much more manageable than tracking values daily.

### How snapshots get joined to matches

For any match in our training data, we need to pick a snapshot. The rule: **the most recent snapshot whose date is ≤ the match's date.** This is called **forward-filling** — we extend each snapshot's value forward in time until the next snapshot replaces it.

Examples:

- A 2015 friendly → use the 2014 snapshot (most recent before the match).
- A 2019 qualifier → use the 2018 snapshot.
- A 2024 Nations League match → use the 2022 snapshot.

The downstream feature table will look like this:

```
year  team           squad_value_eur
2006  Brazil         200_000_000
2006  Germany        180_000_000
...
2022  Argentina      630_000_000
2026  Portugal       864_500_000
```

And for each row in our training data (each match), we attach two columns — `home_squad_value` and `away_squad_value` — looked up from this table via forward-fill.

## Finding the data — the practical approach

The obvious source is **Transfermarkt** (`transfermarkt.com`) — the de facto market-value publisher. Each player has a current valuation; teams have aggregate squad valuations. Tournament-specific archive pages show per-team squad values at the time.

The catch: Transfermarkt actively blocks automated traffic. Any direct scrape attempt gets a CAPTCHA or 403. Even browser-emulated scrapes are unreliable.

**Triage path when the primary source is blocked:**

1. **Check if a public dataset already mirrors the data.** Search GitHub, Kaggle, and academic archives. Someone has probably already done this work.
2. **Check if the data can be reconstructed from related sources.** For us: even if we can't get tournament-snapshot squad values directly, we can compute them from (roster lists) + (player valuations over time) if both are available separately.
3. **Last resort: scrape from secondary sources** (Wikipedia, news archives, etc.). These are usually permissive but the data is less structured.

For our project:

- **Roster lists per WC** — available from [jfjelstul/worldcup on GitHub](https://github.com/jfjelstul/worldcup), a comprehensive WC database with squads.csv covering every men's WC 1930-2022.
- **Player valuations over time** — available from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), a weekly-updated mirror of Transfermarkt's data including `player_valuations.csv.gz` (~500k records).
- **Current snapshot (for WC 2026)** — same dcaribou mirror's `national_teams.csv` with current `total_market_value`.

Combining (1) and (2) gives us historical squad values for past WCs. (3) covers our prediction target.

## Downloading the data

Both sources are downloadable as CSVs without API auth. For dcaribou (gzipped files):

```python
import gzip
import shutil
import requests
from pathlib import Path

TM_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

FILES = [
    "players.csv.gz",
    "player_valuations.csv.gz",
    "national_teams.csv.gz",
]

def _download_and_decompress(filename: str) -> Path:
    TM_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = TM_DIR / filename
    csv_path = gz_path.with_suffix("")

    if csv_path.exists():
        return csv_path

    resp = requests.get(f"{BASE_URL}/{filename}", timeout=60)
    resp.raise_for_status()
    gz_path.write_bytes(resp.content)

    with gzip.open(gz_path, "rb") as f_in, csv_path.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return csv_path
```

For Fjelstul (plain CSV from GitHub):

```python
def download_fjelstul() -> Path:
    url = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv"
    path = PROJECT_ROOT / "data" / "raw" / "wc_squads_fjelstul.csv"
    if not path.exists():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path
```

For the full file see `src/data/squad_value_loader.py`.

## Inspect before integrating

Before writing aggregation code, look at the raw data:

```bash
# Fjelstul squads schema
head -1 data/raw/wc_squads_fjelstul.csv
# key_id,tournament_id,tournament_name,team_id,team_name,team_code,
# player_id,family_name,given_name,shirt_number,position_name,position_code

# Sample rows
grep "WC-2014" data/raw/wc_squads_fjelstul.csv | head -3
```

```bash
# Transfermarkt player_valuations schema
head -1 data/raw/transfermarkt/player_valuations.csv
# player_id,date,market_value_in_eur,current_club_name,current_club_id,
# player_club_domestic_competition_id

# How many records?
wc -l data/raw/transfermarkt/player_valuations.csv
# 507816 (≈500k valuations spanning years 2000 to current)

# Date range?
awk -F',' 'NR>1 {print $2}' data/raw/transfermarkt/player_valuations.csv \
  | sort | head -1
# 2000-01-20
```

Key facts to note:

- Fjelstul covers all WCs 1930–2022, but Transfermarkt's valuations only start ~2000. So we can only compute squad values for WC 2002 onward.
- WC 1990, 1994, 1998 are out of reach for this feature. Those training matches will get NaN squad value, which is fine — the model can handle missing values via imputation.

This is the kind of thing you discover before writing the integration code; otherwise you'd write a pipeline assuming complete coverage and then debug confusing zeros for older WCs.

## The name-matching problem

We have:

- **Fjelstul**: `family_name="Klose", given_name="Miroslav"` (or similar split).
- **Transfermarkt**: `name="Miroslav Klose", first_name="Miroslav", last_name="Klose"`.

To compute squad value, we need to find each Fjelstul player's `player_id` in Transfermarkt so we can look up their valuation. **This requires matching names across two independent datasets.**

This sounds easy but is the single biggest source of work in any multi-source data integration. Two reasons:

1. **Name conventions differ.** First/last order varies. Accents may or may not be present. Multiple players may share a name.
2. **Sentinel values.** Some datasets use placeholders like `"not applicable"` or empty strings for fields that don't fit the schema.

### The normalization step

The basic recipe: normalize both names, then compare.

```python
import unicodedata

def _normalize(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())
```

This handles:

- Case (`MIROSLAV` → `miroslav`).
- Accents (`Cédric` → `cedric`).
- Whitespace (`"  miroslav  klose"` → `"miroslav klose"`).

After normalization, build a lookup from full name to player_id:

```python
from collections import defaultdict

def _build_tm_name_index(players: pd.DataFrame) -> dict[str, list[int]]:
    """Map normalized full name → list of TM player_ids."""
    index = defaultdict(list)
    for _, row in players.iterrows():
        # Try multiple constructions
        for key in (row.get("name"), f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()):
            norm = _normalize(key)
            if norm:
                index[norm].append(int(row["player_id"]))
    return dict(index)
```

Note we add both `row["name"]` (TM's preferred display name) and a constructed `first + last` version. This catches cases where the display name and the structured fields disagree slightly.

### The matching function

Now the matcher itself:

```python
def _match_player(
    given: str,
    family: str,
    tm_index: dict[str, list[int]],
    fuzzy_candidates: list[str],
) -> int | None:
    """Find a TM player_id for this Fjelstul name."""

    # Sentinel detection: Fjelstul uses "not applicable" for mononym players
    if not isinstance(given, str) or given.strip().lower() in ("not applicable", "n/a", "", "nan"):
        given = ""
    if not isinstance(family, str):
        family = ""

    candidates_to_try = [
        f"{given} {family}".strip(),
        f"{family} {given}".strip(),        # reversed order
        family.strip() if not given else "", # mononym fallback
        given.strip() if not family else "",
    ]
    candidates_to_try = [c for c in candidates_to_try if c]

    # Direct hits first
    for cand in candidates_to_try:
        norm = _normalize(cand)
        if norm and norm in tm_index:
            return tm_index[norm][0]

    # Fuzzy fallback on the most-complete name
    primary = candidates_to_try[0]
    best = _fuzzy_lookup(_normalize(primary), fuzzy_candidates)
    if best is not None:
        return tm_index[best][0]
    return None
```

The four candidates handle:

1. `"miroslav klose"` (given then family — most common).
2. `"klose miroslav"` (reversed — some datasets/conventions).
3. Family alone (if `given` is empty / sentinel).
4. Given alone (rare but happens for some names).

### The fuzzy fallback

If no exact normalized match works, use a fuzzy comparison:

```python
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.85

def _fuzzy_lookup(target: str, candidates: list[str]) -> str | None:
    best_name, best_score = None, 0.0
    for cand in candidates:
        score = SequenceMatcher(None, target, cand).ratio()
        if score > best_score:
            best_score = score
            best_name = cand
    return best_name if best_score >= FUZZY_THRESHOLD else None
```

`SequenceMatcher` from the standard library is good enough. For very large candidate lists, switch to `rapidfuzz` or `thefuzz` for speed.

The 0.85 threshold is conservative — names that match below this are usually different people. Tune by inspecting false positives and false negatives on a sample.

### Sentinel values: the bug that costs hours

Fjelstul stores mononym players (common in Brazilian football: Neymar, Hulk, Fred, Marcelo) as:

```
given_name: "not applicable"
family_name: "Neymar"
```

A naive matcher constructs `"not applicable Neymar"` and looks it up — finds nothing. The Brazilian players don't match.

You'll notice this not at the per-row level (each individual lookup just returns None) but at the **aggregate level**: Brazil's total squad value comes out 50% lower than expected for that year. Specifically, Brazil 2014 with only ~10/23 players matched.

**Lesson:** when integrating two datasets, always sample the output by entity (here, by team) and sanity-check that:

- The match rate per entity is reasonable (15+/23 for non-obscure teams).
- The aggregate values are in the right ballpark.
- Known outliers (Brazil with Neymar) appear in the expected position.

The fix in this case is to detect the `"not applicable"` sentinel and use the family name alone (the matcher above does this).

## The aggregation pipeline

Once players match, the rest is straightforward:

```python
WC_KICKOFFS = {
    2006: date(2006, 6, 9),
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
    2022: date(2022, 11, 20),
}

SQUAD_SIZE = 26  # generous cap; FIFA squads were 23 pre-2022, 26 from 2022

def _player_value_at(player_id: int, target_date: date,
                     valuations_by_player: dict[int, pd.DataFrame]) -> float | None:
    """Most recent valuation on or before target_date. None if no data."""
    g = valuations_by_player.get(player_id)
    if g is None:
        return None
    target_str = str(target_date)
    before = g[g["date"] <= target_str]
    if before.empty:
        return None
    return float(before["market_value_in_eur"].iloc[-1])


def compute_squad_value(team_players: list[tuple[str, str]],
                        kickoff: date,
                        tm_index, fuzzy_candidates, valuations_by_player) -> tuple[float, int]:
    values = []
    for given, family in team_players:
        pid = _match_player(given, family, tm_index, fuzzy_candidates)
        if pid is None:
            continue
        v = _player_value_at(pid, kickoff, valuations_by_player)
        if v is None:
            continue
        values.append(v)
    values.sort(reverse=True)
    return float(sum(values[:SQUAD_SIZE])), len(values)
```

The result is a per-(team, year) squad value table. For the full file, see `src/features/squad_values.py`.

## When the primary aggregate column is missing for some rows

Multi-source datasets often have **partial aggregates** — a summary column that *should* exist for every entity but turns out to be NaN for some, even important ones.

A real example from this project: the `total_market_value` column in `national_teams.csv` is populated for ~115 of 118 national teams. But the three missing teams are **England, France, and Spain** — among the most-tracked national teams in the world. Why those? Probably a data-quality gap in the public mirror — perhaps the scraper failed on those pages on the dataset's last refresh. The reason is unknowable; the fact is what matters.

When this happens, you have three options:

1. **Drop the affected rows.** Wrong here — England, France, Spain are major contenders. Dropping them gives them NaN squad value, which the model imputes with the *median*. For top-5 teams, the median is a huge underestimate.

2. **Manually patch the values.** Look them up from another source and hard-code. Fragile (numbers go stale) and tedious.

3. **Fall back to row-level computation.** The aggregate is just a sum of individuals. We *also* have player-level data (`players.csv` with `current_national_team_id` and `market_value_in_eur`). For any team missing the aggregate, sum the top-N player values to compute the aggregate ourselves.

Option 3 is right in principle. But the actual implementation has two layers of subtlety worth walking through.

### First attempt: filter atomic rows by the team's ID

```python
def _team_value_from_players(team_id: int, players: pd.DataFrame, top_n: int = 26) -> float | None:
    team_players = players[players["current_national_team_id"] == team_id]
    values = team_players["market_value_in_eur"].dropna().astype(float)
    if values.empty:
        return None
    return float(values.sort_values(ascending=False).head(top_n).sum())
```

This *seems* right — find the players whose current national team is X, sum their top-N values. But when you run it on England, France, Spain in this dataset, you get **zero matched rows**. The data-quality gap goes deeper than just the aggregate column: those three teams have no players tagged with their team ID either. The upstream scrape failed in two places at once.

**Lesson:** the obvious atomic identifier (here: `current_national_team_id`) is sometimes co-broken with the aggregate column. Have a second-class identifier ready.

### Second attempt: use a categorical proxy

A different column — `country_of_citizenship` — is populated for ~all players in the dataset. This gives us a workable fallback:

```python
def _citizenship_top26(country_name: str, players: pd.DataFrame, top_n: int = 26) -> float | None:
    sub = players[players["country_of_citizenship"] == country_name]
    values = sub["market_value_in_eur"].dropna().astype(float)
    if values.empty:
        return None
    return float(values.sort_values(ascending=False).head(top_n).sum())
```

Filter by citizenship instead of national-team-id. Top 26 most-valued players with that passport. Gets numbers — for England, ~€1780M; France, ~€1750M; Spain, ~€1255M.

But these are **overstated** compared to TM's official national-team aggregates. For Germany (where we have both signals), TM aggregate is €773M while citizenship top-26 is €1166M — a 1.5× overstatement. The reason: citizenship includes *eligible* players, while TM's national-team aggregate is closer to the *actually called-up* squad. Many top European nations have dozens of pros eligible for the national team but only ~25 actually called up.

### Third attempt: calibration

#### What a calibration factor is, in plain terms

When you have two ways to measure the same thing and they give different numbers, a calibration factor is the multiplier you need to convert one to the other.

Here we have:

- **Method 1 — Official aggregate.** `total_market_value` from TM's national-teams page. For Germany: €773M. Based on the ~26 players actually called up.
- **Method 2 — Citizenship top-26.** Sum the 26 most valuable players who hold that country's passport. For Germany: €1166M. Includes every eligible player, called up or not.

Both methods are measuring "Germany's national-team value," but they disagree by 50% because Method 2 includes players who never actually get called up (4th-string goalkeepers, talented kids who haven't broken into the senior squad, retired-from-internationals veterans). For a deep football nation, the gap is real.

The calibration factor `c` is the number such that `Method 1 ≈ c × Method 2`. For Germany: `c = 773 / 1166 = 0.66`. Once we learn `c` from teams where we have both signals (Method 1 and Method 2 available), we can apply it to teams where we only have Method 2: `England_estimate = c × England_citizenship_top26`.

This puts all teams on the **same scale** (Method 1's scale), which is what the model needs — otherwise England would look artificially better than other teams just because we used a different measurement method.

#### The computation

To make the citizenship-based number comparable to the TM aggregate (for use as a feature, where units matter for the model), we calibrate:

```python
def _calibration_factor(national_teams: pd.DataFrame, players: pd.DataFrame) -> float:
    """Ratio (TM aggregate) / (citizenship top-26), computed only from
    teams with deep talent pools — the top half by citizenship_top26."""
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

    # Top half by citizenship — these are the teams where the "calling-up
    # haircut" applies, and they're the teams the fallback is FOR.
    pairs.sort(key=lambda p: p[1], reverse=True)
    big = pairs[: max(1, len(pairs) // 2)]
    total_tm = sum(p[0] for p in big)
    total_cit = sum(p[1] for p in big)
    return float(total_tm / total_cit) if total_cit > 0 else 1.0
```

### Why the top-half restriction matters

You might be tempted to use the median ratio across *all* teams. That gives the wrong answer here.

Most countries Transfermarkt tracks (Estonia, Bolivia, Vietnam, etc.) have small enough talent pools that "top-26 by citizenship" basically IS the active national team. For them, the ratio is ~1.0. The big-football-nation pattern (Germany, England, France with ratios of ~0.65) is the minority.

If you take the median across all 105 teams, you get ~1.0 — which is the right answer for the typical team but the wrong answer for the teams the fallback is for. The fallback only fires on missing teams, and the missing teams (England, France, Spain) are big football nations — exactly the kind where the calibration matters most.

Restricting to the top half by `citizenship_top26` is a pragmatic fix: it gives extra weight to the big nations where the haircut applies, and ignores the small nations where it doesn't.

A magnitude-weighted ratio (`total_tm / total_cit`, summed across the top half) gives an answer of ~0.78 in this dataset.

### Final pipeline

```python
def build_current_2026(national_teams: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    cal = _calibration_factor(national_teams, players)
    print(f"  calibration factor: {cal:.3f}")
    ...
    for _, row in nt.iterrows():
        value = row["total_market_value"]
        if pd.isna(value):
            cit_value = _citizenship_top26(row["name"], players)
            if cit_value is not None:
                value = cit_value * cal
        ...
```

### Fourth tier: entities not in the aggregate table at all

Even after calibrating the citizenship-based fallback, some entities aren't in the aggregate table to begin with — they have *no* row in `national_teams.csv`, so the loop that iterates over national-team rows never touches them. In this project that was Cape Verde, Curaçao, DR Congo, Haiti, and Ivory Coast (5 of 48 WC 2026 qualifiers).

The solution is to **iterate over the target set, not the source table**. After processing every team that *is* in `national_teams.csv`, do a second pass over the qualifier set and process anything still missing:

```python
# After the main loop fills `rows` from national_teams.csv...
already_have = {r["team_name"] for r in rows}
for team_name in participants - already_have:
    # Direct citizenship lookup — bypasses the national_teams table entirely
    citizenship_name = RESULTS_TO_CITIZENSHIP.get(team_name, team_name)
    cit_value = _citizenship_top26(citizenship_name, players)
    if cit_value is None:
        continue
    rows.append({
        "team_name": team_name,
        "squad_value_eur": float(cit_value * cal),
        "source": "citizenship_only",
        ...
    })
```

Notice the `RESULTS_TO_CITIZENSHIP` map — for a handful of small federations, the citizenship column uses a different spelling than the canonical name (e.g., "Curacao" in citizenship vs. "Curaçao" in results, "Cote d'Ivoire" in citizenship vs. "Ivory Coast" in results). A small dict handles this without a fuzzy-match dependency.

### Generalizing this pattern

The same multi-tier fallback arc shows up in many multi-source integrations:

1. **Primary** — use the source's aggregate column when available.
2. **Atomic by primary identifier** — filter the atomic rows by team_id or equivalent. Often fails because the data gap is wider than the aggregate alone.
3. **Atomic by categorical proxy** — filter by a categorical proxy (here: `country_of_citizenship`). Usually works but produces values on a different scale, so calibrate.
4. **Atomic by categorical proxy, no aggregate table needed** — iterate over the target set rather than the source. Use the same categorical proxy, with a spelling map for cases where columns disagree.

Each tier handles a strictly larger set of edge cases than the previous one. Each tier should be tagged in the output (`source` column) so you can audit which entities came from which path. If a downstream model misbehaves on entities tagged with a deeper tier, you've found a calibration mismatch.

The pattern generalizes beyond squad values: any time you fill in missing aggregates with a fallback computation, you need to think about whether the fallback's units match the original. A calibration factor — learned from where both signals are present — is the standard solution.

## Sparse features and training-coverage extension

Even after the multi-tier fallback handles every WC 2026 qualifier, there's a separate problem: **the training set has 49k matches but the snapshot table covers ~190 historical (team, year) rows.** Almost every match involves a team-pair where at least one team isn't a WC participant in that year's snapshot — and so doesn't have a value to look up.

Concretely: an Estonia vs. Latvia friendly in 2015 gets NaN for both squad values, because neither team was a WC participant in 2014. Most of the training set looks like this. Median imputation handles it numerically, but the model can only *learn* the feature's relationship to outcomes from the small fraction of rows where the feature has a real value.

In our project, the initial integration produced **~88% NaN** in `home_squad_value` and `away_squad_value` — far too sparse for the model to learn from. The backtest improvements were modest and inconsistent.

The fix: **extend the snapshot table to cover non-participant teams too.** For each historical snapshot year, run a two-pass computation:

1. **Pass 1 (accurate, ~32 teams):** WC participants via their actual roster (Fjelstul) + per-player valuations at the WC date.
2. **Pass 2 (approximate, ~70 teams):** every other TM-tracked national team via citizenship-based top-26 at the WC date, multiplied by the calibration factor we learned earlier.

```python
def build_historical(fjelstul, ..., players, national_teams, cal_factor):
    rows = []
    for year, kickoff in WC_KICKOFFS.items():
        wc_team_names = set()

        # Pass 1: WC participants via roster (existing logic)
        for team_name, group in wc_rows.groupby("team_name"):
            # ... sum top-26 player valuations from actual roster ...
            rows.append({"source": "fjelstul_roster", ...})
            wc_team_names.add(team_name)

        # Pass 2: everyone else via citizenship
        for tm_name in national_teams["name"]:
            if tm_name in wc_team_names:
                continue
            cit_value = _citizenship_top26_at(tm_name, players,
                                              valuations_by_player, kickoff)
            if cit_value is not None:
                rows.append({
                    "team_name": tm_name,
                    "squad_value_eur": cit_value * cal_factor,
                    "source": "citizenship_calibrated_historical",
                    ...
                })
    return pd.DataFrame(rows)
```

Two practical notes:

- **Cost.** Pass 2 is the expensive part — ~70 teams × ~150 players per team × 5 historical years × per-player valuation lookups. With smart indexing (group valuations by player_id once, reuse), it runs in 30-60 seconds total. Without indexing, minutes.
- **Calibration must be passed in.** We learn the calibration factor once (from current data) and apply it to historical citizenship-based estimates. Don't re-compute per year — current TM data is the only place we can ground the calibration, and the "calling-up haircut" is roughly stable across eras.

After this expansion, the missing-value rate in the integrated features drops from ~88% to ~76% — less dramatic than naively expected, because two structural gaps remain:

- **Pre-snapshot matches.** If the dataset's earliest snapshot is from 2006, then 1990-2005 matches (a third of historical training data) inherently have no value to forward-fill from. The gap is data-coverage, not code.
- **Teams the dataset doesn't track at all.** Tiny federations (e.g., San Marino, Andorra, Bhutan in our case) aren't in the source's team list, so they get NaN regardless of how many extra passes we run. These teams play a lot of qualifying matches, so they add up.

The *informative-row* count (rows where both squad-value features are non-null) roughly doubles from 12% → 24% of training data — going from ~6k to ~12k matches the model can learn the feature's effect from. Real improvement, just smaller than the headline percentage suggests.

The lesson: extending a sparse feature can be worth it even when the headline improvement is modest, because *informative-row count* is what actually drives feature usefulness.

### Asymmetric scoping: historical vs. future snapshots

A subtle design point: historical snapshots are scoped *wider* than the future snapshot. The reason is they're used for different things.

- **Historical snapshots** (2006, 2010, 2014, 2018, 2022) feed *training data*. Training matches span 30+ years and many teams — qualifiers, friendlies, Nations League, regional cups — most of which involve teams that never played in any World Cup. For the squad-value feature to be informative on those rows, every team needs to be snapshotted, not just WC participants. So historical snapshots are scoped to **every team the dataset tracks (~100 per year)**.

- **The future snapshot** (2026) feeds *predictions*. The only matches we'll predict are WC 2026 matches, which involve exactly the 48 qualified teams. Snapshotting Italy 2026 wouldn't be used anywhere — Italy doesn't play in WC 2026, and training data is cut off at 2026-06-10 by leakage policy, so non-WC matches in 2026 don't enter the pipeline either. So the 2026 snapshot is scoped tightly to **just the 48 actual qualifiers**.

The asymmetry isn't laziness; it's discipline. Each snapshot is scoped to the matches that will actually consume it. If you ever generalize this project to predict a different tournament, the same rule applies: scope the future snapshot to that tournament's participants, and keep the historical snapshots as wide as possible.

## Coverage gaps and how to handle them

After running the full pipeline (primary → calibrated fallback → citizenship-only fallback), you'll see:

- WC 2002, 2006, 2010, 2014, 2018, 2022: clean per-team squad values, derived from explicit rosters + historical valuations.
- WC 1990, 1994, 1998: no data — Transfermarkt didn't track valuations before ~2000.
- WC 2026: 48/48 qualifiers covered through the multi-tier fallback chain.

For training matches in 1990–1998 (a few thousand rows), the squad value feature is NaN. The downstream pipeline imputes with the column median.

The model handles NaN via **imputation** — fill missing values with a sensible default before training:

```python
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer(transformers=[
    ("num_with_impute", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), NUMERIC_FEATURES + ["home_squad_value", "away_squad_value"]),
    ...
])
```

The imputer fills NaN with the median of the column (computed from training data). This is a "do no harm" default — the missing values don't disturb the model's parameter estimates beyond adding noise.

## Verifying the feature added signal

After integrating the feature, re-run your backtest. Compare log-loss before and after.

If the feature added meaningful information, you'll see:

- Log-loss drops by 0.01–0.05 per backtest.
- Accuracy may tick up modestly (1–3 percentage points).
- Confusion matrix becomes slightly more balanced (less systematic bias toward favorites).

If you see *no* improvement, possible causes:

- The feature is redundant with Elo (rich teams are also high-Elo, by historical performance).
- Missing values dominate (most training rows are NaN-imputed, so the feature has little variance).
- Match rate is low (the team_id → player_id matching is failing for many teams; squad values are systematically underestimated).

Each of these is debuggable. The diagnostic is to print the feature's correlation with the target on training data, the number of non-NaN rows, and a few example rows for sanity.

## Common pitfalls in multi-source integration

**Trusting that names match exactly.** They almost never do. Always run a sample inspection: print 20 random matches, eyeball them.

**Sentinel values.** "not applicable", "N/A", "NaN" (as string!), empty string, "Unknown". Each is a possible placeholder you need to detect.

**Time-zone issues in date columns.** If one source uses UTC and the other uses local time, your "before" comparisons may be wrong. For sports, always use the kickoff date as a `date` (not a `datetime`) and document the time zone.

**Joining on team name when team IDs are available.** Names are unreliable; IDs are usually stable. If both datasets have IDs (or you can derive them), join on those.

**Building a one-shot integration script.** Treat the integration as a reusable, idempotent pipeline. The data sources will update; you'll want to re-run cleanly without modifying code each time.

**Ignoring data freshness.** The squad value snapshot for WC 2026 is "current as of dataset update time," not "current as of kickoff." For a model running months later, this is fine. For a model deployed during the tournament, you'd want a more recent snapshot.

## What's next

You now have a richer feature set. The model is trained and evaluated. The next step is using it for actual predictions — applying it to the WC 2026 fixtures and producing the bracket simulation. That's `07_prediction.md`.

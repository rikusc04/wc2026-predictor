# 01 — Data acquisition

The data you choose determines the ceiling on what your model can do. Before training a single model, you need to:

1. Find the right datasets.
2. Validate they cover what you need (date range, completeness, quality).
3. Establish a leakage-prevention discipline at the load level.
4. Set up a reproducible download/cache pipeline.

This chapter covers all four for a soccer-prediction project. The same principles apply to any time-series prediction problem (financial forecasting, weather, demand prediction).

## What data does this model need?

Our model predicts **goal counts in international soccer matches**. To do this, it needs to learn the relationship between team strength, recent form, venue, and the goal count. So at a minimum we need:

- **A long history of international matches** with dates, teams, scores, tournament, venue, neutral-flag. This is the spine — every other feature joins to this.
- **Team identity continuity** across renames (West Germany → Germany, Yugoslavia → Serbia, etc.).
- **(Optional, used later)** Player-level data for richer features like squad market value.

We don't need:

- Club-level data (this is international football, not Premier League / Champions League).
- Goalscorer-level data (final scores are what matter for our model; *who* scored doesn't help unless we move to player-level modeling).
- Match commentary, expected-goals (xG), shot data — none of this exists going back to 1872, and our model performs fine without it.

## Finding good datasets

For any ML project, your first move is to search for **pre-compiled datasets** before writing scrapers. The Pareto principle here is brutal: 80% of your time goes into data engineering, but 80% of that data engineering is already done by someone, somewhere. Search before you build.

**Where to look:**

- **Kaggle datasets** (`kaggle.com/datasets`) — searchable, often with sample notebooks showing usage. Most cleaned community-maintained data lives here.
- **GitHub repos** (`github.com/datasets`, `github.com/openfootball`, etc.) — open data initiatives with CSV/JSON drops.
- **Academic data archives** — for older or niche data (FIFA archives, UN data, etc.).
- **Project-specific aggregators** — for soccer: [martj42/international_results](https://github.com/martj42/international_results), [openfootball](https://github.com/openfootball), [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup).

**What to look for in a candidate dataset:**

- **License** — must allow your intended use. ODbL, MIT, CC-BY-SA are good. "All rights reserved" with no API license is bad. For research/learning, most public datasets are fine.
- **Maintenance** — when was it last updated? A 2017-dated dataset described as "current" probably isn't. Look at the commit history.
- **Schema documentation** — clear column descriptions, not just a CSV dump.
- **Date range** — covers your target prediction period? With enough lead time to train on?
- **Completeness** — anything weird about the coverage (skipped years, restricted geography)?

For this project, the primary dataset is:

**`martj42/international_results`** on GitHub — every international men's match from 1872 to present, continuously updated. ODbL license. Fields: date, home_team, away_team, home_score, away_score, tournament, city, country, neutral. Also publishes `shootouts.csv` (penalty shootouts) and `former_names.csv` (country renames).

The Kaggle mirror of the same data is called `international-football-results-from-1872-to-2017` but the maintainer keeps updating it — the title just hasn't changed. Either source works.

## Data leakage at the source level

**Data leakage** is when information from the future (or from your test set) sneaks into training data. It's the #1 cause of "my model looked amazing in development but is useless in production." It happens at many levels — feature engineering, train/test splits, even cross-validation. The most fundamental level is the data source itself.

**The principle:** if you're predicting the 2026 World Cup, **every piece of data your model can see must be available before the 2026 World Cup begins.** Including information your dataset includes by default but shouldn't be used:

- Match outcomes that happened after your prediction date.
- Player market values that reflect post-target performance.
- FIFA rankings published after your prediction date.

The fix is a **hard cutoff date**, baked into the data loader as default behavior. Not as a reminder. Not as a comment in the code. As the *default*, so future code can't accidentally leak.

Concretely, here's the cutoff pattern (real code from this project):

```python
# src/data/loader.py

from datetime import date

# WC 2026 group stage began 2026-06-11, so the cutoff is the day before.
# Anything on or after 2026-06-11 is "the future" relative to our model.
WORLD_CUP_2026_CUTOFF = date(2026, 6, 10)


def load_results(apply_cutoff: bool = True) -> tuple[pd.DataFrame, DataReport]:
    """Load match results from disk.

    apply_cutoff=True (default!) drops matches on/after the WC 2026 cutoff
    to prevent data leakage. Pass apply_cutoff=False only when you have
    a specific need to inspect the dropped rows (e.g., for the prediction
    pipeline that fills in 2026 fixtures).
    """
    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date

    if apply_cutoff:
        df = df[df["date"] <= WORLD_CUP_2026_CUTOFF].reset_index(drop=True)

    return df, report
```

Two things to notice:

1. **The default behavior is safe.** Calling `load_results()` with no arguments returns leak-free data. To get future data, you have to *opt in* with `apply_cutoff=False` — and that explicit opt-in makes leakage harder to introduce by accident.

2. **The cutoff lives in a single named constant.** When the date changes (we hold out a different event), we change it in one place. The principle is: encode invariants in code, not in your head.

For multi-event backtesting (we'll predict WC 2014, 2018, 2022 each with their own cutoff), the pattern is the same: each model's training cutoff is the day before *that* tournament's kickoff, applied as a filter parameter:

```python
def train(cutoff: pd.Timestamp) -> TrainedModels:
    df = load_results(apply_cutoff=True)  # primary safety
    df = df[df["date"] < cutoff]          # backtest-specific cutoff
    ...
```

The match-level cutoff (`apply_cutoff`) and the model-level cutoff are *both* applied. Belt and suspenders.

## A reproducible loader pattern

Every data source should have a loader that's idempotent (running it twice produces the same result) and cached (running it twice doesn't re-download). Here's the canonical pattern:

```python
# src/data/loader.py

from pathlib import Path
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_REPO_RAW = "https://raw.githubusercontent.com/martj42/international_results/master"
RESULTS_URL = f"{_REPO_RAW}/results.csv"
RESULTS_PATH = RAW_DIR / "results.csv"


def _download(url: str, path: Path, force: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path                # cached — don't re-download
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def download_results(force: bool = False) -> Path:
    return _download(RESULTS_URL, RESULTS_PATH, force)
```

The `force=False` default means re-running your scripts won't slam the source repo with re-downloads. `force=True` is for when the source data has been updated and you want a fresh copy.

A `__main__` block lets you run the loader as a script to do a one-time download and validation:

```python
if __name__ == "__main__":
    path = download_results()
    df, report = load_results(apply_cutoff=False)
    print(report.summary())
    print(df.head(3))
    print(df.tail(3))
```

For the full file, see `src/data/loader.py` in the repo.

## Validating data at load time

Don't trust data. Validate it. The minimum-viable validation is:

1. **Date range** — does it cover what you expect? Is it fresh enough?
2. **Row counts** — does the total match what you'd expect from the schema? Big drops or spikes year-over-year usually indicate a data issue.
3. **Missing values** — are there nulls in critical columns? Especially in the training slice (i.e., after cutoff).
4. **Schema** — column names match documentation? Types as expected?

A simple `DataReport` dataclass makes this self-documenting:

```python
@dataclass
class DataReport:
    n_rows: int
    min_date: date
    max_date: date
    n_after_cutoff: int

    def summary(self) -> str:
        return (
            f"rows: {self.n_rows:,}\n"
            f"date range: {self.min_date} → {self.max_date}\n"
            f"rows on/after cutoff: {self.n_after_cutoff:,}"
        )
```

When you run `python -m src.data.loader`, this report prints automatically. The reflexive habit you want is: **before doing anything with new data, eyeball the report and confirm it matches your expectations.** This catches the kind of "wait, the dataset only goes to 2017?" bugs before they propagate.

For our project, the expected output is roughly:

```
rows: 49,477
date range: 1872-11-30 → 2026-06-27
rows on/after 2026-06-11 (WC 2026 + future): 72
```

The 72 rows after cutoff are exactly the WC 2026 group stage matches (12 groups × 4 teams × 3 games each ÷ 2 = 72). If you saw a different number, something's off — and you'd want to investigate before going further.

## Common pitfalls in data acquisition

A few traps to avoid:

**Trusting dataset metadata over the data itself.** A dataset titled "1872 to 2017" might actually be current (the title is stale). A dataset titled "current" might be a year out of date (no recent commits). Always look at `min(date)` and `max(date)` of the actual data.

**Assuming completeness.** "International matches since 1872" may have gaps in WWI, WWII, or specific country boycotts. Check year-by-year match counts to see if anything looks suspicious.

**Identifier instability.** Team names change over time (renames, splits, merges). What looks like one team in old data may be two teams in new data, or vice versa. We address this with `former_names.csv` and a careful inspection in EDA.

**Pre-cleaning that hides edge cases.** Some maintainers pre-canonicalize data ("Germany" used continuously since 1909 even when it was "West Germany" historically). This is convenient but loses information. Read the dataset's README to understand what's been canonicalized.

**Web scraping where you shouldn't.** Some sites (Transfermarkt, certain stats sites) actively block automated traffic. Even if technically possible, scraping commercial sites violates their ToS and may legally expose you. Prefer pre-compiled datasets, paid APIs (if budget allows), or scientific archives.

## Going further

For richer modeling later, you may want additional datasets:

- **Player rosters** — for each WC, which players were on which national team's squad? The [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) database has this for every men's WC 1930–2022.
- **Player market values over time** — Transfermarkt-mirrored datasets like [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) publish `player_valuations.csv.gz` with ~500k historical valuation records. (Note: their `appearances.csv` is club-only — doesn't include internationals.)
- **FIFA rankings** — official monthly rankings, available on Kaggle in various forms. Generally weaker signal than Elo, but useful as a baseline feature.
- **Player club performance** — much harder to integrate; player → club → club performance requires joining many tables. Save for v2.

We integrate the first two later (chapter 6). For now, the international match history is enough to build a strong v1.

## What's next

Once you have data loaded and validated, the next step is to *look* at it carefully — exploratory data analysis. That's `02_eda.md`.

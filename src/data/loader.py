"""
Load international match results.
Source: github.com/martj42/international_results — same dataset as the Kaggle mirror, refreshed continuously by the maintainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_REPO_RAW = "https://raw.githubusercontent.com/martj42/international_results/master"

RESULTS_URL = f"{_REPO_RAW}/results.csv"
SHOOTOUTS_URL = f"{_REPO_RAW}/shootouts.csv"
FORMER_NAMES_URL = f"{_REPO_RAW}/former_names.csv"

RESULTS_PATH = RAW_DIR / "results.csv"
SHOOTOUTS_PATH = RAW_DIR / "shootouts.csv"
FORMER_NAMES_PATH = RAW_DIR / "former_names.csv"

# Everything used to predict the 2026 World Cup must be frozen before kickoff.
# WC 2026 group stage began 2026-06-11, so the cutoff is the day before.
WORLD_CUP_2026_CUTOFF = date(2026, 6, 10)


@dataclass
class DataReport:
    n_rows: int
    min_date: date
    max_date: date
    n_after_cutoff: int

    def summary(self) -> str:
        wc_start = WORLD_CUP_2026_CUTOFF + timedelta(days=1)
        return (
            f"rows: {self.n_rows:,}\n"
            f"date range: {self.min_date} → {self.max_date}\n"
            f"rows on/after {wc_start} (WC 2026 + future): {self.n_after_cutoff:,}"
        )


def _download(url: str, path: Path, force: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def download_results(force: bool = False) -> Path:
    return _download(RESULTS_URL, RESULTS_PATH, force)


def download_shootouts(force: bool = False) -> Path:
    return _download(SHOOTOUTS_URL, SHOOTOUTS_PATH, force)


def download_former_names(force: bool = False) -> Path:
    return _download(FORMER_NAMES_URL, FORMER_NAMES_PATH, force)


def download_all(force: bool = False) -> list[Path]:
    return [
        download_results(force),
        download_shootouts(force),
        download_former_names(force),
    ]


def load_results(apply_cutoff: bool = True) -> tuple[pd.DataFrame, DataReport]:
    """
    Load match results from disk. apply_cutoff=True drops matches on/after the WC 2026 cutoff to prevent data leakage — those games are what we want to predict, not learn from.
    """
    if not RESULTS_PATH.exists():
        download_results()

    df = pd.read_csv(RESULTS_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date

    report = DataReport(
        n_rows=len(df),
        min_date=df["date"].min(),
        max_date=df["date"].max(),
        n_after_cutoff=int((df["date"] > WORLD_CUP_2026_CUTOFF).sum()),
    )

    if apply_cutoff:
        df = df[df["date"] <= WORLD_CUP_2026_CUTOFF].reset_index(drop=True)

    return df, report


def load_shootouts(apply_cutoff: bool = True) -> pd.DataFrame:
    """Load penalty-shootout outcomes for matches that ended drawn in knockouts.

    Not used for training (shootouts are essentially coin flips and recording
    them as wins would teach the model to predict noise). Kept for bracket
    simulation in task #9 to estimate baseline shootout probabilities.
    """
    if not SHOOTOUTS_PATH.exists():
        download_shootouts()
    df = pd.read_csv(SHOOTOUTS_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    if apply_cutoff:
        df = df[df["date"] <= WORLD_CUP_2026_CUTOFF].reset_index(drop=True)
    return df


def load_former_names() -> pd.DataFrame:
    """Load country-name change history.

    Used by the Elo pipeline to preserve rating continuity across renames
    (e.g., West Germany → Germany in 1990). No cutoff needed — these are
    historical name mappings, not match events.
    """
    if not FORMER_NAMES_PATH.exists():
        download_former_names()
    return pd.read_csv(FORMER_NAMES_PATH)


if __name__ == "__main__":
    for p in download_all():
        print(f"saved {p.relative_to(PROJECT_ROOT)}")

    df, report = load_results(apply_cutoff=False)
    print()
    print("=== results.csv ===")
    print(report.summary())
    print("columns:", list(df.columns))
    print("sample rows:")
    print(df.head(3).to_string(index=False))
    print(df.tail(3).to_string(index=False))

    shootouts = load_shootouts(apply_cutoff=False)
    print()
    print("=== shootouts.csv ===")
    print(f"rows: {len(shootouts):,}")
    print("columns:", list(shootouts.columns))
    print(shootouts.head(3).to_string(index=False))

    former = load_former_names()
    print()
    print("=== former_names.csv ===")
    print(f"rows: {len(former):,}")
    print("columns:", list(former.columns))
    print(former.head(10).to_string(index=False))

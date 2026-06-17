"""Download Transfermarkt historical data via the dcaribou public mirror.

Pulls gzipped CSVs from R2 storage (no auth needed) into data/raw/.
This gives us:
  - players: profile per player, including current national team
  - player_valuations: historical market-value records (~500k rows)
  - national_teams: team metadata

These are the building blocks for an approximate per-team, per-date
squad value. The actual aggregation happens in a separate module once
we've seen what's there.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import requests

from src.data.loader import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "transfermarkt"
BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

# Files we need from the mirror
FILES = [
    "players.csv.gz",
    "player_valuations.csv.gz",
    "national_teams.csv.gz",
]


def _download_and_decompress(filename: str, force: bool = False) -> Path:
    """Download a gzipped CSV from R2 and decompress to a .csv next to it."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = RAW_DIR / filename
    csv_path = gz_path.with_suffix("")  # strips .gz, leaves .csv

    if csv_path.exists() and not force:
        return csv_path

    url = f"{BASE_URL}/{filename}"
    print(f"  fetching {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    gz_path.write_bytes(resp.content)

    with gzip.open(gz_path, "rb") as f_in, csv_path.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()  # delete the .gz now that we have the .csv

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  → {csv_path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)")
    return csv_path


def download_all(force: bool = False) -> list[Path]:
    print(f"downloading {len(FILES)} tables to {RAW_DIR.relative_to(PROJECT_ROOT)}/")
    return [_download_and_decompress(f, force) for f in FILES]


if __name__ == "__main__":
    paths = download_all()
    print()
    print("done. files available:")
    for p in paths:
        print(f"  {p.relative_to(PROJECT_ROOT)}")

"""Generate a CSV template for squad market values.

For each World Cup since 1990, extracts the participating teams from
results.csv (so team names exactly match our dataset — important for
the later feature merge) and writes a CSV with year, team, and a blank
`total_value_eur_millions` column for the user to fill in.

Once filled in, save as `data/raw/squad_values.csv` (without `_template`
in the name) and the feature pipeline will pick it up.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import PROJECT_ROOT, load_results


WC_YEARS = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022, 2026]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEMPLATE_PATH = RAW_DIR / "squad_values_template.csv"


def build_template() -> pd.DataFrame:
    # apply_cutoff=False so we include WC 2026 participants too
    results, _ = load_results(apply_cutoff=False)
    results["year"] = pd.to_datetime(results["date"]).dt.year

    rows = []
    for year in WC_YEARS:
        wc_mask = (results["year"] == year) & (results["tournament"] == "FIFA World Cup")
        wc = results[wc_mask]
        teams = sorted(set(wc["home_team"]).union(set(wc["away_team"])))
        for team in teams:
            rows.append({
                "year": year,
                "team": team,
                "total_value_eur_millions": "",
            })

    df = pd.DataFrame(rows)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TEMPLATE_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_template()
    print(f"wrote {TEMPLATE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"total rows: {len(df)}")
    print()
    print("rows by WC year:")
    print(df.groupby("year").size().to_string())

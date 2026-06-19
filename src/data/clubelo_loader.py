"""Scrape and cache historical club Elo from clubelo.com.

v2 Phase 2.2d. clubelo.com publishes daily Elo ratings for every European
club back to ~1995, with historical periods exposed via a tiny CSV API:

  - http://api.clubelo.com/<YYYY-MM-DD>  → snapshot ranking for that date
  - http://api.clubelo.com/<Shortname>    → full historical Elo timeline
                                            for one club

clubelo uses unique short-names (no spaces) — e.g. ManCity, RealMadrid,
Bayern. The snapshot endpoint is how we discover those shortnames given
the long names Transfermarkt uses.

Cached layout:
    data/raw/clubelo/index_YYYY-MM-DD.csv     (snapshot ranking, one file)
    data/raw/clubelo/<Shortname>.csv          (per-club history)
    data/processed/tm_club_to_clubelo.csv     (mapping audit: TM→clubelo)

This module is read-only at feature-build time. The user runs it as a
script to refresh the cache:

    python -m src.data.clubelo_loader

Re-running is idempotent — only missing files are fetched.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT
from src.features.squad_values import _fuzzy_lookup, _normalize


CLUBELO_DIR = PROJECT_ROOT / "data" / "raw" / "clubelo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MAPPING_PATH = PROCESSED_DIR / "tm_club_to_clubelo.csv"

# Snapshot date used to discover clubelo shortnames. June 2026 puts us right
# at WC 2026 kickoff so we see every active top-flight club.
INDEX_DATE = "2026-06-01"

# Polite scrape: clubelo.com is a hobbyist site. 300ms between requests.
SCRAPE_DELAY_S = 0.3
USER_AGENT = "wc2026-predictor research (github.com/rikusc04/wc2026-predictor)"

# Hardcoded TM→clubelo translations for clubs the fuzzy matcher gets wrong.
# Keys are normalized TM club names (use _normalize); values are clubelo
# shortnames. Extend as needed when the audit CSV flags new misses.
HARDCODED_TM_TO_CLUBELO: dict[str, str] = {
    "manchester city": "ManCity",
    "manchester united": "ManUnited",
    "real madrid": "RealMadrid",
    "atletico madrid": "AtleticoMadrid",
    "atletico de madrid": "AtleticoMadrid",
    "bayern munich": "Bayern",
    "fc bayern munchen": "Bayern",
    "borussia dortmund": "Dortmund",
    "bayer 04 leverkusen": "Leverkusen",
    "bayer leverkusen": "Leverkusen",
    "rb leipzig": "RBLeipzig",
    "psv eindhoven": "PSV",
    "ajax amsterdam": "Ajax",
    "afc ajax": "Ajax",
    "paris saint germain": "Paris",
    "paris saint-germain": "Paris",
    "olympique de marseille": "Marseille",
    "olympique lyonnais": "Lyon",
    "as monaco": "Monaco",
    "fc barcelona": "Barcelona",
    "athletic club": "Athletic",
    "athletic bilbao": "Athletic",
    "real betis balompie": "Betis",
    "real sociedad": "RealSociedad",
    "villarreal cf": "Villarreal",
    "valencia cf": "Valencia",
    "sevilla fc": "Sevilla",
    "internazionale milano": "Inter",
    "inter milan": "Inter",
    "ac milan": "Milan",
    "as roma": "Roma",
    "ssc napoli": "Napoli",
    "juventus fc": "Juventus",
    "ss lazio": "Lazio",
    "atalanta bc": "Atalanta",
    "fc porto": "Porto",
    "sl benfica": "Benfica",
    "sporting cp": "Sporting",
    "tottenham hotspur": "Tottenham",
    "newcastle united": "Newcastle",
    "west ham united": "WestHam",
    "aston villa": "AstonVilla",
    "brighton hove albion": "Brighton",
    "brighton & hove albion": "Brighton",
    "crystal palace": "CrystalPalace",
    "nottingham forest": "Forest",
    "leicester city": "Leicester",
    # Clubs whose common name diverges from their legal name — pure
    # substring matching can't bridge these even after decoration-strip.
    "wolverhampton wanderers football club": "Wolves",
    "hamburger sport verein": "Hamburg",
    "heart of midlothian football club": "Hearts",
    "borussia verein fur leibesubungen 1900 monchengladbach": "Gladbach",
    "football club internazionale milano s.p.a.": "Inter",
    "1. fussballclub union berlin": "UnionBerlin",
    "queens park rangers": "QPR",
    "fudbalski klub crvena zvezda beograd": "CrvenaZvezda",
    "fk dinamo moskva": "DinamoMoskva",
    "panthessalonikios athlitikos omilos konstantinoupoliton": "PAOK",
    "omilos filathlon irakliou fc": "Iraklis",
}


def _fetch_text(url: str) -> str:
    """GET text from clubelo with a friendly UA. Raises on non-200."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def fetch_index(snapshot: str = INDEX_DATE) -> pd.DataFrame:
    """Fetch the full clubelo snapshot for a given date (cached on disk).

    Returns DataFrame with columns: Rank, Club, Country, Level, Elo, From, To.
    `Club` is the clubelo shortname we need for the per-club URL.
    """
    CLUBELO_DIR.mkdir(parents=True, exist_ok=True)
    path = CLUBELO_DIR / f"index_{snapshot}.csv"
    if not path.exists():
        text = _fetch_text(f"http://api.clubelo.com/{snapshot}")
        path.write_text(text)
    return pd.read_csv(path)


def fetch_club_history(shortname: str) -> pd.DataFrame:
    """Fetch the historical Elo CSV for a clubelo shortname (cached on disk).

    Returns DataFrame with columns: Rank, Club, Country, Level, Elo, From, To.
    Each row is a period during which the club held that Elo.
    """
    CLUBELO_DIR.mkdir(parents=True, exist_ok=True)
    path = CLUBELO_DIR / f"{shortname}.csv"
    if not path.exists():
        text = _fetch_text(f"http://api.clubelo.com/{urllib.parse.quote(shortname)}")
        path.write_text(text)
        time.sleep(SCRAPE_DELAY_S)
    return pd.read_csv(path, parse_dates=["From", "To"])


def _clubelo_token_form(shortname: str) -> str:
    """Convert clubelo's CamelCase shortname to space-separated lowercase tokens.

    RealMadrid → 'real madrid'; ManCity → 'man city'; Atalanta → 'atalanta';
    BayerLeverkusen → 'bayer leverkusen'. This is what we hunt for as a
    substring inside the verbose TM long names.
    """
    out: list[str] = []
    for i, c in enumerate(shortname):
        if i > 0 and c.isupper() and not shortname[i - 1].isupper():
            out.append(" ")
        out.append(c)
    return "".join(out).lower()


_SUFFIX_RE = re.compile(
    r"\b("
    r"s\s?a\s?d?|ag|gmbh|kft|"
    r"football\s+club|football|fussball(?:club)?|calcio|"
    r"sportvereniging|verein\s+fur\s+leibesubungen|"
    r"associazione\s+(?:calcio|sportiva)|"
    r"club\s+de\s+futbol|"
    r"sociedad\s+anonima\s+deportiva|"
    r"idratsforening|idrtsforening|"  # Danish "Idrætsforening" after partial strip
    r"sport\s+verein|sportclub|sport"
    r")\b"
)


def _normalize_tm_club(name: str) -> str:
    """Lowercase + accent-strip + drop common corporate decorations.

    Lets `clubelo_token_form ⊆ normalized_tm_name` substring match work for
    the common pattern where TM has the full legal name (`Arsenal Football
    Club`) and clubelo has the common name (`Arsenal`).

    NFKD (in `_normalize`) handles diaereses (ö→o, ü→u) but leaves a few
    atomic non-ASCII letters alone — ß, ø, æ, ł — that German / Danish /
    Polish club names rely on. We rewrite those manually so the substring
    match against clubelo's plain-ASCII shortnames lines up.
    """
    n = _normalize(name)
    n = (
        n.replace("ß", "ss")
        .replace("ø", "o")
        .replace("æ", "ae")
        .replace("ł", "l")
    )
    # Replace punctuation AND hyphens with spaces so that "fussball-club"
    # is treated as two words and the suffix regex can drop both halves.
    n = re.sub(r"[.,\-]", " ", n)
    n = _SUFFIX_RE.sub(" ", n)
    n = " ".join(n.split())
    return n


def map_tm_clubs_to_clubelo(
    tm_club_names: dict[int, str],
    index: pd.DataFrame,
) -> pd.DataFrame:
    """Build {tm_club_id, tm_name, clubelo_shortname, source} mapping.

    `source` ∈ {hardcoded, exact, substring, fuzzy, unmatched}. Caller should
    review rows with source=fuzzy / unmatched before trusting them. Match
    order: hardcoded → exact-on-tokenized-shortname → substring (longest
    clubelo token-form that's a substring of the cleaned TM name) → fuzzy.
    """
    clubelo_tokenforms = {
        _clubelo_token_form(c): c for c in index["Club"] if isinstance(c, str)
    }
    # For exact-match: also index by the token-form (covers "Arsenal" exact)
    clubelo_norm_to_short = {tf: sn for tf, sn in clubelo_tokenforms.items()}
    fuzzy_candidates = list(clubelo_norm_to_short.keys())

    def _ascii_norm(name: str) -> str:
        """_normalize plus the ß/ø/æ/ł character replacements that NFKD skips.
        Makes the ASCII-spelled HARDCODED keys actually match TM names that
        carry those literal characters."""
        n = _normalize(name)
        return (
            n.replace("ß", "ss").replace("ø", "o")
            .replace("æ", "ae").replace("ł", "l")
        )

    rows: list[dict] = []
    for club_id, tm_name in tm_club_names.items():
        if not isinstance(tm_name, str) or not tm_name.strip():
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": None, "source": "unmatched",
            })
            continue

        norm = _ascii_norm(tm_name)
        if norm in HARDCODED_TM_TO_CLUBELO:
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": HARDCODED_TM_TO_CLUBELO[norm],
                "source": "hardcoded",
            })
            continue

        if norm in clubelo_norm_to_short:
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": clubelo_norm_to_short[norm],
                "source": "exact",
            })
            continue

        # Substring pass — clubelo's token-form must appear inside the
        # decoration-stripped TM name. Longest match wins so that
        # "atletico madrid" beats just "madrid" when both could be substrings.
        cleaned = _normalize_tm_club(tm_name)
        best_short: str | None = None
        best_len = 0
        for token_form, shortname in clubelo_tokenforms.items():
            # Require word-boundary alignment so "atalanta" doesn't sneak
            # into "atletico de madrid" via "at..." overlap.
            if re.search(rf"\b{re.escape(token_form)}\b", cleaned):
                if len(token_form) > best_len:
                    best_len = len(token_form)
                    best_short = shortname
        if best_short is not None:
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": best_short, "source": "substring",
            })
            continue

        # Fuzzy fallback on the cleaned name
        best = _fuzzy_lookup(cleaned, fuzzy_candidates)
        if best is not None:
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": clubelo_norm_to_short[best],
                "source": "fuzzy",
            })
        else:
            rows.append({
                "tm_club_id": club_id, "tm_name": tm_name,
                "clubelo_shortname": None, "source": "unmatched",
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.features.club_lookup import build_club_id_to_name, clubs_needed_for_lineups

    print(f"=== fetch clubelo index ({INDEX_DATE}) ===")
    index = fetch_index()
    print(f"  {len(index):,} clubs in snapshot")

    print("\n=== identify clubs needed from TM lineup data ===")
    needed_club_ids = clubs_needed_for_lineups()
    club_id_to_name = build_club_id_to_name()
    needed = {cid: club_id_to_name.get(cid, "") for cid in needed_club_ids}
    print(f"  {len(needed):,} unique clubs to map")

    print("\n=== map TM → clubelo ===")
    mapping = map_tm_clubs_to_clubelo(needed, index)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(MAPPING_PATH, index=False)
    print(f"  wrote {MAPPING_PATH.relative_to(PROJECT_ROOT)}")
    print(mapping["source"].value_counts().to_string())

    print("\n=== fetch per-club histories ===")
    to_fetch = mapping.dropna(subset=["clubelo_shortname"])["clubelo_shortname"].unique()
    print(f"  {len(to_fetch):,} clubs to fetch (skipping already-cached)")
    for i, short in enumerate(sorted(to_fetch), 1):
        try:
            fetch_club_history(short)
        except Exception as e:
            print(f"  [{i}/{len(to_fetch)}] {short}: FAILED ({e})")
            continue
        if i % 50 == 0:
            print(f"  [{i}/{len(to_fetch)}] {short}: ok")

    print("\n=== done ===")
    print(f"clubelo cache at {CLUBELO_DIR.relative_to(PROJECT_ROOT)}")

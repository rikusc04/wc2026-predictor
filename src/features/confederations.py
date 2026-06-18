"""Country/team → FIFA confederation mapping and graded host-advantage scoring.

v2 multi-host feature. Replaces v1's single binary `neutral` flag with a
graded score in [0, 1] that captures how "at-home" a team is for a given
match. The grading is:

    1.0  team's country == match country (true home)
    0.7  team's confederation == match country's confederation (intra-conf)
    0.3  Americas adjacency only: CONMEBOL ↔ CONCACAF
    0.0  otherwise (true neutral / far from home)

The 0.3 level is the one explicit cross-confederation adjacency we model,
since WC 2026 (CONCACAF host) draws heavily from CONMEBOL. Brazil playing
in the US is *not* at home, but also not as neutral as Brazil playing in
Japan. Russia 2018 (UEFA host) and Qatar 2022 (AFC host) do not activate
this level — for those backtests the scheme effectively collapses to
0.0 / 0.7 / 1.0.

Teams or countries not in the table fall back to "OTHER", which always
scores 0.0. Safe for obscure entities (Sealand, Donetsk PR, ...) that
appear in friendlies but never WCs.
"""

from __future__ import annotations


UEFA = "UEFA"
CONMEBOL = "CONMEBOL"
CONCACAF = "CONCACAF"
AFC = "AFC"
CAF = "CAF"
OFC = "OFC"
OTHER = "OTHER"


# Single source of truth: every name (team or match country) → confederation.
# Includes historical synonyms (Soviet Union, Zaïre, Burma, …) so the lookup
# works regardless of which spelling appears in a given row.
CONFEDERATION: dict[str, str] = {
    # ---- UEFA ----
    "Albania": UEFA, "Andorra": UEFA, "Armenia": UEFA, "Austria": UEFA,
    "Azerbaijan": UEFA, "Belarus": UEFA, "Belgium": UEFA,
    "Bosnia and Herzegovina": UEFA, "Bulgaria": UEFA, "Croatia": UEFA,
    "Cyprus": UEFA, "Czech Republic": UEFA, "Czechoslovakia": UEFA,
    "Denmark": UEFA, "England": UEFA, "Estonia": UEFA,
    "Faroe Islands": UEFA, "Finland": UEFA, "France": UEFA,
    "Georgia": UEFA, "Germany": UEFA, "German DR": UEFA, "Gibraltar": UEFA,
    "Greece": UEFA, "Hungary": UEFA, "Iceland": UEFA,
    "Republic of Ireland": UEFA, "Ireland": UEFA, "Éire": UEFA,
    "Irish Free State": UEFA, "Israel": UEFA, "Italy": UEFA,
    "Kazakhstan": UEFA, "Kosovo": UEFA, "Latvia": UEFA,
    "Liechtenstein": UEFA, "Lithuania": UEFA, "Luxembourg": UEFA,
    "Malta": UEFA, "Moldova": UEFA, "Monaco": UEFA, "Montenegro": UEFA,
    "Netherlands": UEFA, "North Macedonia": UEFA, "Northern Ireland": UEFA,
    "Norway": UEFA, "Poland": UEFA, "Portugal": UEFA, "Romania": UEFA,
    "Russia": UEFA, "San Marino": UEFA, "Saarland": UEFA, "Scotland": UEFA,
    "Serbia": UEFA, "Serbia and Montenegro": UEFA, "Slovakia": UEFA,
    "Slovenia": UEFA, "Soviet Union": UEFA, "Spain": UEFA, "Sweden": UEFA,
    "Switzerland": UEFA, "Turkey": UEFA, "Ukraine": UEFA, "Wales": UEFA,
    "Yugoslavia": UEFA, "FR Yugoslavia": UEFA,
    "Bohemia": UEFA, "Bohemia and Moravia": UEFA,
    # Non-FIFA Euro sub-state / regional sides — assigned by geography
    "Catalonia": UEFA, "Basque Country": UEFA, "Galicia": UEFA,
    "Asturias": UEFA, "Madrid": UEFA, "Andalusia": UEFA, "Central Spain": UEFA,
    "Canary Islands": UEFA, "Menorca": UEFA, "Rhodes": UEFA,
    "Brittany": UEFA, "Corsica": UEFA, "Provence": UEFA, "Occitania": UEFA,
    "Saugeais": UEFA, "Seborga": UEFA, "County of Nice": UEFA,
    "Padania": UEFA, "Two Sicilies": UEFA, "Cilento": UEFA, "Elba Island": UEFA,
    "Ticino": UEFA, "Raetia": UEFA, "Republic of St. Pauli": UEFA,
    "Franconia": UEFA, "Silesia": UEFA,
    "Yorkshire": UEFA, "Surrey": UEFA, "Kernow": UEFA, "Shetland": UEFA,
    "Orkney": UEFA, "Isle of Wight": UEFA, "Western Isles": UEFA,
    "Ynys Môn": UEFA, "Ellan Vannin": UEFA, "Sark": UEFA,
    "Parishes of Jersey": UEFA, "Frøya": UEFA, "Hitra": UEFA,
    "Gotland": UEFA, "Sápmi": UEFA, "Sealand": UEFA, "Vatican City": UEFA,
    "Åland Islands": UEFA, "Saare County": UEFA, "Crimea": UEFA,
    "Guernsey": UEFA, "Jersey": UEFA, "Alderney": UEFA, "Isle of Man": UEFA,
    "Gozo": UEFA, "Northern Cyprus": UEFA,
    "Donetsk PR": UEFA, "Luhansk PR": UEFA, "South Ossetia": UEFA,
    "Abkhazia": UEFA, "Artsakh": UEFA, "Chechnya": UEFA,
    "Délvidék": UEFA, "Felvidék": UEFA, "Kárpátalja": UEFA,
    "Székely Land": UEFA, "Chameria": UEFA, "Găgăuzia": UEFA,
    "Greenland": UEFA,  # geographically Americas but governed by Denmark; rarely plays

    # ---- CONMEBOL ----
    "Argentina": CONMEBOL, "Bolivia": CONMEBOL, "Brazil": CONMEBOL,
    "Chile": CONMEBOL, "Colombia": CONMEBOL, "Ecuador": CONMEBOL,
    "Paraguay": CONMEBOL, "Peru": CONMEBOL, "Uruguay": CONMEBOL,
    "Venezuela": CONMEBOL,
    "Aymara": CONMEBOL, "Mapuche": CONMEBOL, "Maule Sur": CONMEBOL,

    # ---- CONCACAF ----
    "United States": CONCACAF, "Canada": CONCACAF, "Mexico": CONCACAF,
    "Costa Rica": CONCACAF, "Honduras": CONCACAF, "Panama": CONCACAF,
    "Jamaica": CONCACAF, "Trinidad and Tobago": CONCACAF,
    "El Salvador": CONCACAF, "Guatemala": CONCACAF, "Haiti": CONCACAF,
    "Cuba": CONCACAF, "Nicaragua": CONCACAF, "Belize": CONCACAF,
    "Dominican Republic": CONCACAF, "Bermuda": CONCACAF, "Barbados": CONCACAF,
    "Curaçao": CONCACAF, "Aruba": CONCACAF, "Bahamas": CONCACAF,
    "Suriname": CONCACAF, "Guyana": CONCACAF, "French Guiana": CONCACAF,
    "Cayman Islands": CONCACAF, "Saint Lucia": CONCACAF,
    "Antigua and Barbuda": CONCACAF,
    "Saint Vincent and the Grenadines": CONCACAF,
    "Saint Kitts and Nevis": CONCACAF, "Grenada": CONCACAF,
    "Dominica": CONCACAF, "Anguilla": CONCACAF, "Montserrat": CONCACAF,
    "Martinique": CONCACAF, "Guadeloupe": CONCACAF, "Puerto Rico": CONCACAF,
    "United States Virgin Islands": CONCACAF,
    "British Virgin Islands": CONCACAF, "Bonaire": CONCACAF,
    "Sint Maarten": CONCACAF, "Turks and Caicos Islands": CONCACAF,
    "Saint-Martin": CONCACAF, "Saint Martin": CONCACAF,
    "Saint Barthélemy": CONCACAF,
    "Dutch Guyana": CONCACAF, "British Guiana": CONMEBOL,  # now Guyana, CONCACAF, but match country pre-independence
    "Saint Pierre and Miquelon": CONCACAF, "Cascadia": CONCACAF,
    "Quebec": CONCACAF,

    # ---- AFC ----
    "Afghanistan": AFC, "Australia": AFC, "Bahrain": AFC, "Bangladesh": AFC,
    "Bhutan": AFC, "Brunei": AFC, "Cambodia": AFC, "China PR": AFC,
    "Chinese Taipei": AFC, "Timor-Leste": AFC, "East Timor": AFC,
    "Guam": AFC, "Hong Kong": AFC, "India": AFC, "Indonesia": AFC,
    "Iran": AFC, "Iraq": AFC, "Japan": AFC, "Jordan": AFC,
    "North Korea": AFC, "South Korea": AFC, "Kuwait": AFC,
    "Kyrgyzstan": AFC, "Laos": AFC, "Lebanon": AFC, "Macau": AFC,
    "Malaysia": AFC, "Maldives": AFC, "Mongolia": AFC, "Myanmar": AFC,
    "Nepal": AFC, "Northern Mariana Islands": AFC, "Oman": AFC,
    "Pakistan": AFC, "Palestine": AFC, "Philippines": AFC, "Qatar": AFC,
    "Saudi Arabia": AFC, "Singapore": AFC, "Sri Lanka": AFC,
    "Syria": AFC, "Tajikistan": AFC, "Thailand": AFC, "Turkmenistan": AFC,
    "United Arab Emirates": AFC, "Uzbekistan": AFC, "Vietnam": AFC,
    "Yemen": AFC, "Yemen AR": AFC, "South Yemen": AFC, "Yemen DPR": AFC,
    "Burma": AFC, "Malaya": AFC, "Ceylon": AFC,
    "Manchuria": AFC, "Manchukuo": AFC, "Mandatory Palestine": AFC,
    # Asian non-FIFA / sub-state
    "Tibet": AFC, "East Turkestan": AFC, "Ryūkyū": AFC, "Hmong": AFC,
    "Tamil Eelam": AFC, "West Papua": AFC, "Iraqi Kurdistan": AFC,
    "Kurdistan": AFC, "Western Armenia": AFC, "Panjab": AFC,
    "United Koreans in Japan": AFC, "Arameans Suryoye": AFC,
    "China": AFC, "Taiwan": AFC,  # alt spellings of China PR / Chinese Taipei
    "Vietnam Republic": AFC, "North Vietnam": AFC,

    # ---- CAF ----
    "Algeria": CAF, "Angola": CAF, "Benin": CAF, "Botswana": CAF,
    "Burkina Faso": CAF, "Upper Volta": CAF, "Burundi": CAF,
    "Cameroon": CAF, "Cape Verde": CAF, "Central African Republic": CAF,
    "Chad": CAF, "Comoros": CAF, "Congo": CAF, "DR Congo": CAF,
    "Belgian Congo": CAF, "Congo-Kinshasa": CAF, "Zaïre": CAF,
    "Djibouti": CAF, "French Somaliland": CAF,
    "Egypt": CAF, "United Arab Republic": CAF,
    "Equatorial Guinea": CAF, "Eritrea": CAF,
    "Eswatini": CAF, "Swaziland": CAF, "Ethiopia": CAF,
    "Gabon": CAF, "Gambia": CAF, "Ghana": CAF, "Gold Coast": CAF,
    "Guinea": CAF, "Guinea-Bissau": CAF, "Portuguese Guinea": CAF,
    "Ivory Coast": CAF, "Kenya": CAF, "Lesotho": CAF,
    "Liberia": CAF, "Libya": CAF, "Madagascar": CAF,
    "Malawi": CAF, "Nyasaland": CAF,
    "Mali": CAF, "Mauritania": CAF, "Mauritius": CAF,
    "Morocco": CAF, "Mozambique": CAF, "Namibia": CAF,
    "Niger": CAF, "Nigeria": CAF, "Rwanda": CAF,
    "São Tomé and Príncipe": CAF, "Senegal": CAF, "Seychelles": CAF,
    "Sierra Leone": CAF, "Somalia": CAF, "South Africa": CAF,
    "South Sudan": CAF, "Sudan": CAF, "Tanzania": CAF, "Tanganyika": CAF,
    "Zanzibar": CAF, "Togo": CAF, "Tunisia": CAF, "Uganda": CAF,
    "Zambia": CAF, "Northern Rhodesia": CAF, "Zimbabwe": CAF,
    "Southern Rhodesia": CAF, "Dahomey": CAF, "Mayotte": CAF, "Réunion": CAF,
    # African non-FIFA / breakaway regions
    "Western Sahara": CAF, "Darfur": CAF, "Biafra": CAF, "Ambazonia": CAF,
    "Kabylia": CAF, "Somaliland": CAF, "Matabeleland": CAF,
    "Yoruba Nation": CAF, "Saint Helena": CAF, "Romani people": UEFA,
    "Barawa": CAF,  # Somali diaspora team, CONIFA

    # ---- OFC ----
    "American Samoa": OFC, "Cook Islands": OFC, "Fiji": OFC,
    "Kiribati": OFC, "New Caledonia": OFC, "New Zealand": OFC,
    "Niue": OFC, "Papua New Guinea": OFC, "Samoa": OFC,
    "Western Samoa": OFC, "Solomon Islands": OFC, "Tahiti": OFC,
    "Tonga": OFC, "Tuvalu": OFC, "Vanuatu": OFC, "New Hebrides": OFC,
    "Micronesia": OFC, "Marshall Islands": OFC, "Palau": OFC,
    "Wallis Islands and Futuna": OFC, "Chagos Islands": OFC,
    "Falkland Islands": CONMEBOL,  # South Atlantic; closest to CONMEBOL
    "Western Australia": OFC,
}


# Historical renames and sub-state teams whose "home" is logically the same
# place as a different-string match country. e.g., Russia playing in Soviet
# Union (1990s data) should score 1.0 (true home), not 0.7. Similarly,
# Catalonia playing in Spain is at home.
#
# Direction is name → canonical home identity. Both sides of a comparison
# get normalized before the equality check, so {Russia, Soviet Union} →
# "Russia" makes both look identical for the 1.0 check.
HOME_ALIASES: dict[str, str] = {
    # Country renames (modern name kept as canonical)
    "Soviet Union": "Russia",
    "Swaziland": "Eswatini",
    "Dutch Guyana": "Suriname",
    "Zaïre": "DR Congo", "Congo-Kinshasa": "DR Congo", "Belgian Congo": "DR Congo",
    "Ireland": "Republic of Ireland", "Éire": "Republic of Ireland",
    "Irish Free State": "Republic of Ireland",
    "Malaya": "Malaysia",
    "British Guiana": "Guyana",
    "FR Yugoslavia": "Serbia", "Serbia and Montenegro": "Serbia",
    "Burma": "Myanmar",
    "Upper Volta": "Burkina Faso",
    "Southern Rhodesia": "Zimbabwe",
    "Dahomey": "Benin",
    "Tanganyika": "Tanzania",
    "Northern Rhodesia": "Zambia",
    "United Arab Republic": "Egypt",
    "Ceylon": "Sri Lanka",
    "Gold Coast": "Ghana",
    "Czechoslovakia": "Czech Republic",  # imperfect (also covers Slovak side)
    "Bohemia": "Czech Republic", "Bohemia and Moravia": "Czech Republic",
    "Yugoslavia": "Serbia",  # imperfect (covers multiple successor states)
    "Manchukuo": "China", "Manchuria": "China",
    "Mandatory Palestine": "Israel",
    "New Hebrides": "Vanuatu",
    "Portuguese Guinea": "Guinea-Bissau",
    "Nyasaland": "Malawi",
    "French Somaliland": "Djibouti",
    "Western Samoa": "Samoa",
    "Yemen AR": "Yemen", "Yemen DPR": "Yemen",
    "North Vietnam": "Vietnam", "Vietnam Republic": "Vietnam",
    "Taiwan": "Chinese Taipei", "China": "China PR",
    # Sub-state and minority teams within parent state
    "Catalonia": "Spain", "Basque Country": "Spain", "Galicia": "Spain",
    "Asturias": "Spain", "Madrid": "Spain", "Andalusia": "Spain",
    "Central Spain": "Spain", "Canary Islands": "Spain", "Menorca": "Spain",
    "Brittany": "France", "Corsica": "France", "Provence": "France",
    "Occitania": "France", "County of Nice": "France", "Saugeais": "France",
    "Padania": "Italy", "Two Sicilies": "Italy", "Cilento": "Italy",
    "Elba Island": "Italy", "Seborga": "Italy", "Vatican City": "Italy",
    "Sápmi": "Sweden",  # spans Nordic countries; pick one
    "Gotland": "Sweden",
    "Åland Islands": "Finland",
    "Frøya": "Norway", "Hitra": "Norway",
    "Yorkshire": "England", "Surrey": "England", "Kernow": "England",
    "Isle of Wight": "England", "Shetland": "Scotland", "Orkney": "Scotland",
    "Western Isles": "Scotland", "Sealand": "England",
    "Ynys Môn": "Wales",
    "Guernsey": "England", "Jersey": "England", "Alderney": "England",
    "Sark": "England", "Parishes of Jersey": "England",
    "Ellan Vannin": "England", "Isle of Man": "England",
    "Gozo": "Malta",
    "Saare County": "Estonia",
    "Abkhazia": "Georgia", "South Ossetia": "Georgia",
    "Chechnya": "Russia",
    "Crimea": "Ukraine", "Donetsk PR": "Ukraine", "Luhansk PR": "Ukraine",
    "Saarland": "Germany",
    "Northern Cyprus": "Cyprus",
    "Saint Barthélemy": "France",  # French overseas
}


def _canon_home(name: str) -> str:
    """Normalize a team or country name to its canonical 'home' identity."""
    return HOME_ALIASES.get(name, name)


def confederation_of(name: str) -> str:
    """Return the confederation for a team name or match country.

    Falls back to OTHER for entries we haven't catalogued (~55 obscure
    teams with <10 historical matches, plus any new entry that might
    appear in future data refreshes). OTHER always scores 0.0 in the
    host-advantage grading, which is the safe default.
    """
    return CONFEDERATION.get(name, OTHER)


# Cross-confederation adjacencies that earn a 0.3 boost. Each is a
# geography-driven proximity where confederations share substantial
# regional fan/travel patterns.
#   CONMEBOL ↔ CONCACAF: the Americas (added for WC 2026's US/Can/Mex host)
#   CAF ↔ AFC:           Mediterranean / Middle East / N. Africa
#                         (added in the v2 refinement pass to fix the
#                         Qatar 2022 backtest regression — Morocco/Tunisia/
#                         Egypt now get a 0.3 in Qatar instead of 0.0)
_CROSS_CONF_ADJACENCIES: tuple[frozenset, ...] = (
    frozenset({CONMEBOL, CONCACAF}),
    frozenset({CAF, AFC}),
)


def host_advantage(team_country: str, match_country: str) -> float:
    """Graded host-advantage score in {0.0, 0.3, 0.7, 1.0}.

    Args:
        team_country: the team's home nation (e.g., "Brazil")
        match_country: the country the match is played in (e.g., "United States")
    """
    if _canon_home(team_country) == _canon_home(match_country):
        return 1.0

    team_conf = confederation_of(team_country)
    match_conf = confederation_of(match_country)

    # If either side is OTHER (unrecognized), don't try to grade — return 0.0.
    # This keeps the score conservative for entries we don't have confidence in.
    if team_conf == OTHER or match_conf == OTHER:
        return 0.0

    if team_conf == match_conf:
        return 0.7
    if frozenset({team_conf, match_conf}) in _CROSS_CONF_ADJACENCIES:
        return 0.3
    return 0.0


if __name__ == "__main__":
    # Sanity checks against known WC 2026 fixtures and historical backtests.
    cases = [
        # Format: (team, match_country, expected, description)
        ("Mexico", "Mexico", 1.0, "Mexico at home in Mexico City"),
        ("South Africa", "Mexico", 0.0, "South Africa (CAF) in Mexico"),
        ("United States", "United States", 1.0, "USA at home"),
        ("Canada", "Canada", 1.0, "Canada at home"),
        ("Paraguay", "United States", 0.3, "Paraguay (CONMEBOL) in US (CONCACAF)"),
        ("Brazil", "United States", 0.3, "Brazil in 'neutral' US — Americas adjacency"),
        ("Morocco", "United States", 0.0, "Morocco (CAF) in US — true neutral"),
        ("Curaçao", "United States", 0.7, "Curaçao (CONCACAF) in US — same conf"),
        ("Japan", "United States", 0.0, "Japan in US — true neutral"),
        # 2014 backtest (Brazil, CONMEBOL)
        ("Brazil", "Brazil", 1.0, "Brazil 2014 at home"),
        ("Argentina", "Brazil", 0.7, "Argentina (CONMEBOL) in Brazil"),
        ("United States", "Brazil", 0.3, "USA (CONCACAF) in Brazil — Americas"),
        ("Germany", "Brazil", 0.0, "Germany (UEFA) in Brazil"),
        # 2018 backtest (Russia, UEFA) — Americas adjacency should not fire
        ("Russia", "Russia", 1.0, "Russia 2018 at home"),
        ("Germany", "Russia", 0.7, "Germany in Russia — same conf"),
        ("Brazil", "Russia", 0.0, "Brazil in Russia — no Americas adjacency"),
        # 2022 backtest (Qatar, AFC)
        ("Qatar", "Qatar", 1.0, "Qatar 2022 at home"),
        ("Japan", "Qatar", 0.7, "Japan (AFC) in Qatar"),
        ("Brazil", "Qatar", 0.0, "Brazil in Qatar — no adjacency to AFC"),
        ("Morocco", "Qatar", 0.3, "Morocco (CAF) in Qatar — CAF↔AFC adjacency"),
        # Historical entities (within-era only — cross-era pairs like Zaïre vs.
        # Belgian Congo can't co-occur in any real match)
        ("Czechoslovakia", "Soviet Union", 0.7, "Both UEFA"),
        ("Soviet Union", "Soviet Union", 1.0, "USSR at home"),
    ]
    passed, failed = 0, 0
    for team, match_country, expected, desc in cases:
        actual = host_advantage(team, match_country)
        ok = abs(actual - expected) < 1e-9
        marker = "ok " if ok else "FAIL"
        print(f"  {marker}  {desc}: got {actual}, expected {expected}")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")

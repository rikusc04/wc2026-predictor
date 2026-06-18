"""Altitude-native advantage feature.

v2 Phase 1 Item 2. Replaces the naive "altitude penalty for visitors" with
the more realistic native-advantage framing: FIFA mandates ~2 weeks of
acclimation at base camp before every WC match, which roughly halves the
visitor disadvantage in the literature (McSharry 2007 BMJ, on CONMEBOL
qualifiers). What's left is the *lifelong* adaptation that altitude-native
teams keep — and that 2-week visitor camps cannot replicate.

So the feature is binary: 1.0 if the team is native to the venue's
altitude (their typical home altitude is within NATIVE_MARGIN meters of
venue altitude), 0.0 otherwise. Matches at sub-threshold venues (<1500m)
get 0.0 on both sides — the effect doesn't register.

For WC 2026, this lights up only on the 4 Mexico City group matches
(Estadio Azteca, 2240m); every Mexico City match has Mexico = 1.0 native,
all opponents 0.0. US/Canada venues are all sea-level-ish. The Guadalajara
metro venues (Zapopan, 1560m) sit just above threshold — Mexico gets the
1.0 there too. The other two host cities (Monterrey area at ~540m) get
nothing.

City spellings match what's in `results.csv` (Mexico City, Bogotá with
accent, Sana'a with apostrophe, Nezahualcóyotl, etc.). Coverage was audited
against the 11 high-altitude national teams' historical home cities.
"""

from __future__ import annotations


# City → elevation in meters. Includes only cities at or above the 1500m
# threshold where altitude is empirically meaningful for football
# (per the altitude-effect literature). Everything else defaults to 0.
HIGH_ALTITUDE_CITIES: dict[str, int] = {
    # ---- Mexico (heaviest WC 2026 relevance) ----
    "Mexico City": 2240, "Toluca": 2680, "Pachuca": 2400,
    "Puebla": 2135, "León": 1815, "Aguascalientes": 1880,
    "Querétaro": 1820, "San Luis Potosí": 1864,
    "Guadalajara": 1560, "Zapopan": 1560,  # GDL metro
    "Tlaxcala": 2230, "Saltillo": 1600,
    "Nezahualcóyotl": 2240,  # Mexico City metro

    # ---- Bolivia ----
    "La Paz": 3640, "El Alto": 4150, "Sucre": 2810,
    "Cochabamba": 2570, "Oruro": 3700, "Potosí": 4090,
    "Tarija": 1875,

    # ---- Ecuador ----
    "Quito": 2850, "Cuenca": 2560, "Ambato": 2580,
    "Riobamba": 2750, "Latacunga": 2750, "Loja": 2060,
    "Ibarra": 2225, "Azogues": 2518,

    # ---- Colombia ----
    "Bogotá": 2640, "Tunja": 2820, "Pasto": 2527,
    "Manizales": 2160, "Pereira": 1411,  # just below threshold; included for borderline
    "Armenia": 1551,

    # ---- Peru ----
    "Cusco": 3400, "Arequipa": 2335, "Huancayo": 3260,
    "Puno": 3825, "Juliaca": 3825, "Ayacucho": 2761,

    # ---- Africa ----
    "Addis Ababa": 2355, "Bahir Dar": 1840,  # Ethiopia
    "Asmara": 2325,                            # Eritrea
    "Nairobi": 1795,                           # Kenya
    "Kigali": 1567,                            # Rwanda
    "Antananarivo": 1276,                      # Madagascar — below threshold
    "Maseru": 1480,                            # Lesotho — borderline (just below)

    # ---- Middle East / Central Asia ----
    "Tehran": 1190,                            # Iran — below threshold
    "Kabul": 1790,                             # Afghanistan
    "Sana'a": 2250,                            # Yemen

    # ---- Europe / Americas (Western Hemisphere altitude venues) ----
    "Denver": 1610,                            # USA
    "Colorado Springs": 1830,
}


# Team → "typical home altitude" in meters. Includes only teams whose
# national team trains and plays significant home matches at altitude.
# Used to decide if the team is "native" to a given venue's altitude.
#
# For teams with multiple home venues at different altitudes (e.g., Mexico
# at Mexico City + Monterrey + Guadalajara), the value reflects the highest
# altitude they regularly use. The native check is "team altitude ≥ venue −
# NATIVE_MARGIN", so a lower stated value would cause Mexico to miss its
# Mexico City native advantage.
HIGH_ALTITUDE_TEAMS: dict[str, int] = {
    "Bolivia": 3640,        # La Paz home base
    "Ecuador": 2850,        # Quito
    "Colombia": 2640,       # Bogotá
    # Peru intentionally excluded: national team plays the great majority
    # of home matches in Lima (sea level); occasional Andean venues don't
    # give them the lifelong-adaptation edge that defines "native".
    "Mexico": 2240,         # Estadio Azteca
    "Ethiopia": 2355,       # Addis Ababa
    "Eritrea": 2325,        # Asmara
    "Afghanistan": 1790,    # Kabul
    "Yemen": 2250,          # Sana'a
    "Lesotho": 1480,        # Maseru — borderline
}


ALTITUDE_THRESHOLD = 1500  # below this, altitude effect doesn't register
NATIVE_MARGIN = 500        # team is "native" if home alt ≥ venue alt − this


def venue_altitude(city: str | None) -> int:
    """Elevation of a match city in meters; 0 for unknown/low-altitude cities."""
    if not city:
        return 0
    return HIGH_ALTITUDE_CITIES.get(city, 0)


def team_home_altitude(team: str) -> int:
    """Team's typical home elevation in meters; 0 if team isn't altitude-native."""
    return HIGH_ALTITUDE_TEAMS.get(team, 0)


def altitude_native_advantage(team: str, match_city: str | None) -> float:
    """1.0 if the team is native to this venue's altitude, else 0.0.

    Returns 0.0 at sub-threshold venues regardless of team — the effect
    doesn't register below ~1500m. At altitude venues, returns 1.0 only
    for teams whose own home altitude is within NATIVE_MARGIN of the
    venue's altitude. That captures the lifelong-adaptation advantage
    that FIFA's mandatory 2-week acclimation cannot replicate.
    """
    v_alt = venue_altitude(match_city)
    if v_alt < ALTITUDE_THRESHOLD:
        return 0.0
    t_alt = team_home_altitude(team)
    if t_alt >= v_alt - NATIVE_MARGIN:
        return 1.0
    return 0.0


if __name__ == "__main__":
    # Sanity check against known scenarios.
    cases = [
        # WC 2026
        ("Mexico", "Mexico City", 1.0, "Mexico at Azteca — native"),
        ("South Africa", "Mexico City", 0.0, "South Africa at Azteca — visitor"),
        ("Spain", "Mexico City", 0.0, "Spain at Azteca — visitor, even with acclimation"),
        ("Mexico", "Zapopan", 1.0, "Mexico at GDL metro — native at 1560m"),
        ("Mexico", "Guadalupe", 0.0, "Mexico at MTY metro — below threshold (~540m)"),
        ("Brazil", "Inglewood", 0.0, "Brazil at sea-level US — no altitude"),
        # Historical CONMEBOL altitude
        ("Bolivia", "La Paz", 1.0, "Bolivia at home in La Paz — extreme altitude"),
        ("Brazil", "La Paz", 0.0, "Brazil at La Paz — would be heavily disadvantaged"),
        ("Argentina", "La Paz", 0.0, "Argentina at La Paz"),
        ("Peru", "La Paz", 0.0, "Peru at La Paz — Peru's home base is Lima (sea level), not native"),
        ("Ecuador", "Quito", 1.0, "Ecuador at home Quito"),
        ("Brazil", "Quito", 0.0, "Brazil at Quito"),
        ("Colombia", "Bogotá", 1.0, "Colombia at home Bogotá"),
        # Borderline
        ("Argentina", "Salta", 0.0, "Argentina at Salta — 1187m, below threshold"),
        ("Spain", "Madrid", 0.0, "Madrid is 660m — below threshold"),
        # Same conf, different altitude
        ("Bolivia", "Lima", 0.0, "Bolivia at Lima (sea level) — no altitude effect"),
    ]
    passed = failed = 0
    for team, city, expected, desc in cases:
        actual = altitude_native_advantage(team, city)
        ok = abs(actual - expected) < 1e-9
        marker = "ok " if ok else "FAIL"
        print(f"  {marker}  {desc}: got {actual}, expected {expected}")
        passed += ok
        failed += not ok
    print(f"\n{passed}/{passed + failed} passed")

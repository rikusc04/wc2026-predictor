"""Hardcoded actual starting XIs for the 12 played WC 2026 group-stage matches.

v2 Phase 2.2b. Source: Wikipedia per-group pages
(en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{A..F}). Until StatsBomb
publishes WC 2026 open data, these lineups have to live in source.

Running this module writes the same shape as `statsbomb_lineups.csv` to
`data/raw/wc2026_actual_lineups.csv`, which the existing lineup_values.py
matching pipeline can then consume as-is. Player names are kept exactly as
they appear on Wikipedia, including accents and diacritics — the Transfermarkt
name-matcher normalizes those.

Synthetic match_ids start at 90000000 to avoid collision with real StatsBomb
match_ids.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import PROJECT_ROOT


OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wc2026_actual_lineups.csv"

# Each entry: (date, home_team, away_team) -> {"home": [(jersey, name, position), ...], "away": ...}
LINEUPS: dict[tuple[str, str, str], dict[str, list[tuple[int, str, str]]]] = {
    # ---- Group A ----
    ("2026-06-11", "Mexico", "South Africa"): {
        "home": [
            (1, "Raúl Rangel", "GK"),
            (15, "Israel Reyes", "RB"),
            (3, "César Montes", "CB"),
            (5, "Johan Vásquez", "CB"),
            (23, "Jesús Gallardo", "LB"),
            (6, "Érik Lira", "DM"),
            (8, "Álvaro Fidalgo", "CM"),
            (26, "Brian Gutiérrez", "CM"),
            (25, "Roberto Alvarado", "RF"),
            (9, "Raúl Jiménez", "CF"),
            (16, "Julián Quiñones", "LF"),
        ],
        "away": [
            (1, "Ronwen Williams", "GK"),
            (21, "Ime Okon", "CB"),
            (19, "Nkosinathi Sibisi", "CB"),
            (14, "Mbekezeli Mbokazi", "CB"),
            (20, "Khuliso Mudau", "RWB"),
            (6, "Aubrey Modiba", "LWB"),
            (23, "Jayden Adams", "CM"),
            (13, "Sphephelo Sithole", "CM"),
            (4, "Teboho Mokoena", "CM"),
            (9, "Lyle Foster", "CF"),
            (15, "Iqraam Rayners", "CF"),
        ],
    },
    ("2026-06-11", "South Korea", "Czech Republic"): {
        "home": [
            (1, "Kim Seung-gyu", "GK"),
            (2, "Lee Han-beom", "CB"),
            (4, "Kim Min-jae", "CB"),
            (3, "Lee Gi-hyuk", "CB"),
            (22, "Seol Young-woo", "RM"),
            (6, "Hwang In-beom", "CM"),
            (8, "Paik Seung-ho", "CM"),
            (13, "Lee Tae-seok", "LM"),
            (19, "Lee Kang-in", "RF"),
            (7, "Son Heung-min", "CF"),
            (10, "Lee Jae-sung", "LF"),
        ],
        "away": [
            (1, "Matěj Kovář", "GK"),
            (6, "Štěpán Chaloupek", "CB"),
            (4, "Robin Hranáč", "CB"),
            (7, "Ladislav Krejčí", "CB"),
            (5, "Vladimír Coufal", "RWB"),
            (20, "Jaroslav Zelený", "LWB"),
            (22, "Tomáš Souček", "CM"),
            (17, "Lukáš Provod", "CM"),
            (24, "Alexandr Sojka", "RF"),
            (10, "Patrik Schick", "CF"),
            (15, "Pavel Šulc", "LF"),
        ],
    },
    # ---- Group B ----
    ("2026-06-12", "Canada", "Bosnia and Herzegovina"): {
        "home": [
            (16, "Maxime Crépeau", "GK"),
            (2, "Alistair Johnston", "RB"),
            (4, "Luc de Fougerolles", "CB"),
            (13, "Derek Cornelius", "CB"),
            (22, "Richie Laryea", "LB"),
            (17, "Tajon Buchanan", "RM"),
            (7, "Stephen Eustáquio", "CM"),
            (8, "Ismaël Koné", "CM"),
            (11, "Liam Millar", "LM"),
            (10, "Jonathan David", "CF"),
            (12, "Tani Oluwaseyi", "CF"),
        ],
        "away": [
            (1, "Nikola Vasilj", "GK"),
            (7, "Amar Dedić", "RB"),
            (18, "Nikola Katić", "CB"),
            (4, "Tarik Muharemović", "CB"),
            (5, "Sead Kolašinac", "LB"),
            (20, "Esmir Bajraktarević", "RM"),
            (13, "Ivan Bašić", "CM"),
            (6, "Benjamin Tahirović", "CM"),
            (15, "Amar Memić", "LM"),
            (10, "Ermedin Demirović", "CF"),
            (25, "Jovo Lukić", "CF"),
        ],
    },
    ("2026-06-13", "Qatar", "Switzerland"): {
        "home": [
            (1, "Mahmud Abunada", "GK"),
            (13, "Ayoub Al-Oui", "RB"),
            (2, "Pedro Miguel", "CB"),
            (16, "Boualem Khoukhi", "CB"),
            (14, "Homam Ahmed", "LB"),
            (5, "Jassem Gaber", "CM"),
            (4, "Issa Laye", "CM"),
            (8, "Edmilson Junior", "RW"),
            (23, "Assim Madibo", "AM"),
            (11, "Akram Afif", "LW"),
            (15, "Yusuf Abdurisag", "CF"),
        ],
        "away": [
            (1, "Gregor Kobel", "GK"),
            (6, "Denis Zakaria", "RB"),
            (5, "Manuel Akanji", "CB"),
            (4, "Nico Elvedi", "CB"),
            (13, "Ricardo Rodriguez", "LB"),
            (20, "Michel Aebischer", "CM"),
            (10, "Granit Xhaka", "CM"),
            (8, "Remo Freuler", "CM"),
            (11, "Dan Ndoye", "RF"),
            (7, "Breel Embolo", "CF"),
            (17, "Rubén Vargas", "LF"),
        ],
    },
    # ---- Group C (Wikipedia: Brazil's group) ----
    ("2026-06-13", "Brazil", "Morocco"): {
        "home": [
            (1, "Alisson", "GK"),
            (24, "Roger Ibañez", "RB"),
            (4, "Marquinhos", "CB"),
            (3, "Gabriel Magalhães", "CB"),
            (16, "Douglas Santos", "LB"),
            (20, "Lucas Paquetá", "RM"),
            (5, "Casemiro", "CM"),
            (8, "Bruno Guimarães", "CM"),
            (7, "Vinícius Júnior", "LM"),
            (25, "Igor Thiago", "CF"),
            (11, "Raphinha", "CF"),
        ],
        "away": [
            (1, "Yassine Bounou", "GK"),
            (2, "Achraf Hakimi", "RB"),
            (14, "Issa Diop", "CB"),
            (18, "Chadi Riad", "CB"),
            (3, "Noussair Mazraoui", "LB"),
            (24, "Neil El Aynaoui", "CM"),
            (6, "Ayyoub Bouaddi", "CM"),
            (10, "Brahim Díaz", "RW"),
            (8, "Azzedine Ounahi", "AM"),
            (23, "Bilal El Khannouss", "LW"),
            (11, "Ismael Saibari", "CF"),
        ],
    },
    ("2026-06-13", "Haiti", "Scotland"): {
        "home": [
            (1, "Johny Placide", "GK"),
            (2, "Carlens Arcus", "RB"),
            (4, "Ricardo Adé", "CB"),
            (5, "Hannes Delcroix", "CB"),
            (8, "Martin Expérience", "LB"),
            (11, "Louicius Deedson", "RM"),
            (17, "Danley Jean Jacques", "CM"),
            (10, "Jean-Ricner Bellegarde", "CM"),
            (15, "Ruben Providence", "LM"),
            (20, "Frantzdy Pierrot", "CF"),
            (18, "Wilson Isidor", "CF"),
        ],
        "away": [
            (1, "Angus Gunn", "GK"),
            (2, "Aaron Hickey", "RB"),
            (5, "Grant Hanley", "CB"),
            (13, "Jack Hendry", "CB"),
            (3, "Andy Robertson", "LB"),
            (17, "Ben Gannon-Doak", "RM"),
            (4, "Scott McTominay", "CM"),
            (19, "Lewis Ferguson", "CM"),
            (7, "John McGinn", "LM"),
            (20, "Lawrence Shankland", "CF"),
            (10, "Ché Adams", "CF"),
        ],
    },
    # ---- Group D (Wikipedia: USA's group) ----
    ("2026-06-12", "United States", "Paraguay"): {
        "home": [
            (24, "Matt Freese", "GK"),
            (16, "Alex Freeman", "RB"),
            (3, "Chris Richards", "CB"),
            (13, "Tim Ream", "CB"),
            (5, "Antonee Robinson", "LB"),
            (4, "Tyler Adams", "CM"),
            (17, "Malik Tillman", "CM"),
            (2, "Sergiño Dest", "RW"),
            (8, "Weston McKennie", "AM"),
            (10, "Christian Pulisic", "LW"),
            (20, "Folarin Balogun", "CF"),
        ],
        "away": [
            (12, "Orlando Gill", "GK"),
            (4, "Juan José Cáceres", "RB"),
            (15, "Gustavo Gómez", "CB"),
            (3, "Omar Alderete", "CB"),
            (6, "Júnior Alonso", "LB"),
            (8, "Diego Gómez", "RM"),
            (14, "Andrés Cubas", "CM"),
            (16, "Damián Bobadilla", "CM"),
            (10, "Miguel Almirón", "LM"),
            (9, "Antonio Sanabria", "CF"),
            (19, "Julio Enciso", "CF"),
        ],
    },
    ("2026-06-13", "Australia", "Turkey"): {
        "home": [
            (18, "Patrick Beach", "GK"),
            (3, "Alessandro Circati", "CB"),
            (19, "Harry Souttar", "CB"),
            (21, "Cameron Burgess", "CB"),
            (4, "Jacob Italiano", "RWB"),
            (5, "Jordan Bos", "LWB"),
            (8, "Connor Metcalfe", "RM"),
            (24, "Paul Okon-Engstler", "CM"),
            (13, "Aiden O'Neill", "CM"),
            (17, "Nestory Irankunda", "LM"),
            (9, "Mohamed Touré", "CF"),
        ],
        "away": [
            (23, "Uğurcan Çakır", "GK"),
            (2, "Zeki Çelik", "RB"),
            (3, "Merih Demiral", "CB"),
            (14, "Abdülkerim Bardakcı", "CB"),
            (20, "Ferdi Kadıoğlu", "LB"),
            (10, "Hakan Çalhanoğlu", "CM"),
            (16, "İsmail Yüksek", "CM"),
            (8, "Arda Güler", "RW"),
            (6, "Orkun Kökçü", "AM"),
            (21, "Barış Alper Yılmaz", "LW"),
            (7, "Kerem Aktürkoğlu", "CF"),
        ],
    },
    # ---- Group E ----
    ("2026-06-14", "Germany", "Curaçao"): {
        "home": [
            (1, "Manuel Neuer", "GK"),
            (6, "Joshua Kimmich", "CB"),
            (4, "Jonathan Tah", "CB"),
            (15, "Nico Schlotterbeck", "CB"),
            (23, "Felix Nmecha", "RM"),
            (10, "Jamal Musiala", "CM"),
            (5, "Aleksandar Pavlović", "CM"),
            (18, "Nathaniel Brown", "LM"),
            (19, "Leroy Sané", "RF"),
            (7, "Kai Havertz", "CF"),
            (17, "Florian Wirtz", "LF"),
        ],
        "away": [
            (1, "Eloy Room", "GK"),
            (5, "Sherel Floranus", "RB"),
            (23, "Riechedly Bazoer", "CB"),
            (18, "Armando Obispo", "CB"),
            (24, "Deveron Fonville", "LB"),
            (10, "Leandro Bacuna", "DM"),
            (8, "Livano Comenencia", "CM"),
            (21, "Tahith Chong", "CM"),
            (12, "Sontje Hansen", "RF"),
            (9, "Jürgen Locadia", "CF"),
            (7, "Juninho Bacuna", "LF"),
        ],
    },
    ("2026-06-14", "Ivory Coast", "Ecuador"): {
        "home": [
            (1, "Yahia Fofana", "GK"),
            (17, "Guéla Doué", "RB"),
            (5, "Wilfried Singo", "CB"),
            (20, "Emmanuel Agbadou", "CB"),
            (3, "Ghislain Konan", "LB"),
            (11, "Yan Diomande", "RM"),
            (8, "Franck Kessié", "CM"),
            (6, "Seko Fofana", "CM"),
            (24, "Bazoumana Touré", "LM"),
            (19, "Nicolas Pépé", "CF"),
            (12, "Elye Wahi", "CF"),
        ],
        "away": [
            (1, "Hernán Galíndez", "GK"),
            (21, "Alan Franco", "RB"),
            (4, "Joel Ordóñez", "CB"),
            (6, "Willian Pacho", "CB"),
            (3, "Piero Hincapié", "LB"),
            (9, "John Yeboah", "RM"),
            (23, "Moisés Caicedo", "CM"),
            (15, "Pedro Vite", "CM"),
            (14, "Alan Minda", "LM"),
            (19, "Gonzalo Plata", "CF"),
            (13, "Enner Valencia", "CF"),
        ],
    },
    # ---- Group F ----
    ("2026-06-14", "Netherlands", "Japan"): {
        "home": [
            (1, "Bart Verbruggen", "GK"),
            (22, "Denzel Dumfries", "RB"),
            (6, "Jan Paul van Hecke", "CB"),
            (4, "Virgil van Dijk", "CB"),
            (15, "Micky van de Ven", "LB"),
            (21, "Frenkie de Jong", "DM"),
            (8, "Ryan Gravenberch", "CM"),
            (14, "Tijjani Reijnders", "CM"),
            (24, "Crysencio Summerville", "RF"),
            (18, "Donyell Malen", "CF"),
            (11, "Cody Gakpo", "LF"),
        ],
        "away": [
            (1, "Zion Suzuki", "GK"),
            (16, "Tsuyoshi Watanabe", "CB"),
            (3, "Shōgo Taniguchi", "CB"),
            (21, "Hiroki Itō", "CB"),
            (10, "Ritsu Dōan", "RM"),
            (24, "Kaishū Sano", "CM"),
            (15, "Daichi Kamada", "CM"),
            (13, "Keito Nakamura", "LM"),
            (8, "Takefusa Kubo", "RF"),
            (18, "Ayase Ueda", "CF"),
            (11, "Daizen Maeda", "LF"),
        ],
    },
    ("2026-06-14", "Sweden", "Tunisia"): {
        "home": [
            (23, "Kristoffer Nordfeldt", "GK"),
            (2, "Gustaf Lagerbielke", "CB"),
            (4, "Isak Hien", "CB"),
            (3, "Victor Lindelöf", "CB"),
            (21, "Alexander Bernhardsson", "RM"),
            (16, "Jesper Karlström", "CM"),
            (18, "Yasin Ayari", "CM"),
            (5, "Gabriel Gudmundsson", "LM"),
            (10, "Benjamin Nygren", "AM"),
            (17, "Viktor Gyökeres", "CF"),
            (9, "Alexander Isak", "CF"),
        ],
        "away": [
            (1, "Mouhib Chamakh", "GK"),
            (4, "Omar Rekik", "CB"),
            (3, "Montassar Talbi", "CB"),
            (21, "Mohamed Amine Ben Hamida", "LB"),
            (20, "Yan Valery", "RWB"),
            (2, "Ali Abdi", "LWB"),
            (13, "Rani Khedira", "CM"),
            (17, "Ellyes Skhiri", "CM"),
            (10, "Hannibal Mejbri", "CM"),
            (8, "Elias Saad", "CF"),
            (25, "Anis Ben Slimane", "CF"),
        ],
    },
}


def to_csv() -> pd.DataFrame:
    """Flatten LINEUPS into the same CSV schema as statsbomb_lineups.csv."""
    rows = []
    synthetic_match_id = 90000000
    synthetic_player_id = 99000000

    for (date, home, away), sides in LINEUPS.items():
        synthetic_match_id += 1
        for side, lineup in sides.items():
            team = home if side == "home" else away
            for jersey, name, position in lineup:
                synthetic_player_id += 1
                rows.append({
                    "match_id": synthetic_match_id,
                    "match_date": date,
                    "competition": "FIFA World Cup 2026",
                    "home_team": home,
                    "away_team": away,
                    "side": side,
                    "team": team,
                    "player_id": synthetic_player_id,
                    "player_name": name,
                    "player_nickname": "",
                    "jersey_number": jersey,
                    "position_id": 0,
                    "position": position,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = to_csv()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(df):,} starter rows)")
    print(f"  matches: {df['match_id'].nunique()}")
    print(f"  per-side starters (expect 11): {df.groupby(['match_id', 'side']).size().describe()}")

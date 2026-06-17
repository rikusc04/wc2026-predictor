"""Coarse classification of the 200+ distinct tournament names in results.csv.

Used for weighting matches by importance (Elo) and as a model feature
(friendlies behave differently from competitive matches).
"""


TOURNAMENT_CLASSES = ("friendly", "qualifier", "world_cup", "continental", "other")


def classify_tournament(name: str) -> str:
    """Bucket a tournament name into a coarse class.

    Order matters: "qualif" is checked before "world cup" because
    "FIFA World Cup qualification" contains both substrings.
    """
    name = name.lower()
    if "friendly" in name:
        return "friendly"
    if "qualif" in name:
        return "qualifier"
    if "world cup" in name or "fifa" in name:
        return "world_cup"
    if any(k in name for k in (
        "euro", "copa", "africa", "asian", "concacaf",
        "uefa", "afcon", "gold cup", "nations league",
    )):
        return "continental"
    return "other"

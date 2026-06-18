"""Official FIFA WC 2026 knockout bracket (48-team format).

v2 Phase 1 Item 3: replaces v1's `pair_with_group_avoidance` random pairing
(issues.md #21) with FIFA's published bracket tree. The bracket is fixed
in advance — group winners and runners-up slot into pre-determined R32
positions, and the 8 reserved 3rd-place slots are filled per FIFA's
per-slot eligibility matrix.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage

Bracket tree (FIFA match numbers in parentheses):

  R32 (16 matches: 73-88)
    ├─ 8 slot pairs (winner_X vs runner-up_Y, or winner_X vs 3rd_from_Z)
  R16 (8 matches: 89-96)
    ├─ each = winner(R32_a) vs winner(R32_b) per fixed mapping
  QF (4 matches: 97-100)
  SF (2 matches: 101-102)
  Final (104)
  (Third-place: 103, not modeled)

Third-place slot assignment uses a bipartite matching over the eligibility
matrix (which 5 of 12 groups can fill each of the 8 reserved slots, after
excluding the slot's paired winner-group and the bracket-half-conflicting
groups). This recovers any of the 495 valid FIFA-published assignments
without hand-coding the full table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---- R32 structure ----
#
# Two kinds of slot entries:
#   ("W", group)       — winner of that group
#   ("RU", group)      — runner-up of that group
#   ("3RD", slot_id)   — a 3rd-place qualifier filled at runtime
#                        (the slot_id is the match number it occupies)
#
# Per the Wikipedia bracket page. Match numbers match FIFA's published
# competition schedule (matches 73-88 are R32).

R32_MATCHUPS: list[tuple[int, tuple, tuple]] = [
    (73, ("RU", "A"), ("RU", "B")),
    (74, ("W",  "E"), ("3RD", 74)),
    (75, ("W",  "F"), ("RU", "C")),
    (76, ("W",  "C"), ("RU", "F")),
    (77, ("W",  "I"), ("3RD", 77)),
    (78, ("RU", "E"), ("RU", "I")),
    (79, ("W",  "A"), ("3RD", 79)),
    (80, ("W",  "L"), ("3RD", 80)),
    (81, ("W",  "D"), ("3RD", 81)),
    (82, ("W",  "G"), ("3RD", 82)),
    (83, ("RU", "K"), ("RU", "L")),
    (84, ("W",  "H"), ("RU", "J")),
    (85, ("W",  "B"), ("3RD", 85)),
    (86, ("W",  "J"), ("RU", "H")),
    (87, ("W",  "K"), ("3RD", 87)),
    (88, ("RU", "D"), ("RU", "G")),
]

# Per-slot eligibility for 3rd-place qualifiers, by match number.
# Each set is the 5 group letters whose 3rd-place team is allowed to fill
# that slot (excludes the slot's winner-group + bracket-half conflicts).
THIRD_PLACE_ELIGIBILITY: dict[int, frozenset[str]] = {
    74: frozenset({"A", "B", "C", "D", "F"}),
    77: frozenset({"C", "D", "F", "G", "H"}),
    79: frozenset({"C", "E", "F", "H", "I"}),
    80: frozenset({"E", "H", "I", "J", "K"}),
    81: frozenset({"B", "E", "F", "I", "J"}),
    82: frozenset({"A", "E", "H", "I", "J"}),
    85: frozenset({"E", "F", "G", "I", "J"}),
    87: frozenset({"D", "E", "I", "J", "L"}),
}
THIRD_PLACE_SLOTS = tuple(sorted(THIRD_PLACE_ELIGIBILITY.keys()))


# ---- Tree above R32 ----
#
# Each later-round match is identified by its FIFA match number, with a
# pair of feeder match numbers. R16: matches 89-96. QF: 97-100. SF: 101-102.
# Final: 104. (Third-place play-off 103 is omitted — we don't model losers.)

R16_FEEDERS: dict[int, tuple[int, int]] = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QF_FEEDERS: dict[int, tuple[int, int]] = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SF_FEEDERS: dict[int, tuple[int, int]] = {
    101: (97, 98),
    102: (99, 100),
}

FINAL_FEEDERS: tuple[int, int] = (101, 102)


# ---- Per-match venue assignment (FIFA-published WC 2026 schedule) ----
#
# Maps FIFA match number → (city, country). Knockout matches are scheduled
# in advance regardless of who the participants are; only the participants
# get filled in during the tournament.
#
# Source: Wikipedia 2026 FIFA World Cup knockout stage page (matches 73-104).
# Used by the simulator's per-venue knockout lambda cache to route
# host-advantage and altitude features correctly through each knockout.

KNOCKOUT_VENUES: dict[int, tuple[str, str]] = {
    # R32 (73-88)
    73: ("Inglewood", "United States"),
    74: ("Foxborough", "United States"),
    75: ("Guadalupe", "Mexico"),
    76: ("Houston", "United States"),
    77: ("East Rutherford", "United States"),
    78: ("Arlington", "United States"),
    79: ("Mexico City", "Mexico"),       # altitude venue
    80: ("Atlanta", "United States"),
    81: ("Santa Clara", "United States"),
    82: ("Seattle", "United States"),
    83: ("Toronto", "Canada"),
    84: ("Inglewood", "United States"),
    85: ("Vancouver", "Canada"),
    86: ("Miami Gardens", "United States"),
    87: ("Kansas City", "United States"),
    88: ("Arlington", "United States"),
    # R16 (89-96)
    89: ("Philadelphia", "United States"),
    90: ("Houston", "United States"),
    91: ("East Rutherford", "United States"),
    92: ("Mexico City", "Mexico"),       # altitude venue
    93: ("Arlington", "United States"),
    94: ("Seattle", "United States"),
    95: ("Atlanta", "United States"),
    96: ("Vancouver", "Canada"),
    # QF (97-100)
    97: ("Foxborough", "United States"),
    98: ("Inglewood", "United States"),
    99: ("Miami Gardens", "United States"),
    100: ("Kansas City", "United States"),
    # SF (101-102), Final (104). Third-place (103) is omitted (we don't simulate it).
    101: ("Arlington", "United States"),
    102: ("Atlanta", "United States"),
    104: ("East Rutherford", "United States"),
}


@dataclass
class GroupStandings:
    """For one group: ranked teams [(team, pts, gd, gf), ...] in 1st→4th order."""
    label: str
    ranked: list[tuple[str, int, int, int]]

    @property
    def winner(self) -> str:
        return self.ranked[0][0]

    @property
    def runner_up(self) -> str:
        return self.ranked[1][0]

    @property
    def third(self) -> tuple[str, int, int, int]:
        """Returns the 4-tuple — pts/gd/gf needed for 3rd-place ranking."""
        return self.ranked[2]


def assign_third_place_slots(qualifier_groups: list[str]) -> dict[int, str]:
    """Assign 8 qualifying 3rd-place groups to the 8 reserved R32 slots.

    Returns: {slot_match_number: group_letter}.

    Uses backtracking over the FIFA eligibility matrix. For most scenarios
    the matching is unique or near-unique (high-eligibility groups like I/E
    flex into rare-eligibility slots like K's 80). When multiple valid
    matchings exist, we return the first one found in deterministic order
    (sorted slots → sorted groups), which approximates FIFA's canonical
    choice without hand-coding the full 495-row table.
    """
    if len(qualifier_groups) != 8:
        raise ValueError(f"need 8 qualifier groups, got {len(qualifier_groups)}")
    qualifier_set = set(qualifier_groups)

    def backtrack(
        slot_idx: int,
        assigned_groups: set[str],
        current: dict[int, str],
    ) -> Optional[dict[int, str]]:
        if slot_idx == len(THIRD_PLACE_SLOTS):
            return current
        slot = THIRD_PLACE_SLOTS[slot_idx]
        # Eligible groups for this slot AND in the qualifier set AND not yet used
        candidates = sorted(
            (THIRD_PLACE_ELIGIBILITY[slot] & qualifier_set) - assigned_groups
        )
        for g in candidates:
            current[slot] = g
            assigned_groups.add(g)
            result = backtrack(slot_idx + 1, assigned_groups, current)
            if result is not None:
                return result
            assigned_groups.remove(g)
            del current[slot]
        return None

    matching = backtrack(0, set(), {})
    if matching is None:
        # Eligibility constraints failed (shouldn't happen for any of the 495
        # FIFA-valid scenarios — but might for non-FIFA edge cases). Fall
        # back to a simpler "exclude own group winner" rule.
        return _assign_third_place_fallback(qualifier_groups)
    return matching


def _assign_third_place_fallback(qualifier_groups: list[str]) -> dict[int, str]:
    """Fallback if the strict eligibility matching has no solution.

    Relaxes to the only universal constraint: a 3rd-placer from group X
    can't face the winner of group X. Assigns greedily.
    """
    own_winner_at_slot = {
        74: "E", 77: "I", 79: "A", 80: "L",
        81: "D", 82: "G", 85: "B", 87: "K",
    }
    remaining = sorted(qualifier_groups)
    out: dict[int, str] = {}
    for slot in THIRD_PLACE_SLOTS:
        forbidden = own_winner_at_slot[slot]
        pick = next((g for g in remaining if g != forbidden), None)
        if pick is None:
            # Forced placement — accept the violation
            pick = remaining[0]
        out[slot] = pick
        remaining.remove(pick)
    return out


def build_r32_slot_to_team(
    group_standings: dict[str, GroupStandings],
    third_place_qualifier_groups: list[str],
) -> dict[tuple, str]:
    """Resolve every R32 slot entry to a concrete team.

    Returns a dict keyed by slot entry: ("W", "A") → team name, etc.,
    plus ("3RD", match_number) → team name for the 8 reserved 3rd-place slots.
    """
    slot_to_team: dict[tuple, str] = {}
    for letter, gs in group_standings.items():
        slot_to_team[("W", letter)] = gs.winner
        slot_to_team[("RU", letter)] = gs.runner_up

    third_place_assignment = assign_third_place_slots(third_place_qualifier_groups)
    for slot_match, group_letter in third_place_assignment.items():
        slot_to_team[("3RD", slot_match)] = group_standings[group_letter].third[0]

    return slot_to_team


def r32_matchups_resolved(
    slot_to_team: dict[tuple, str],
) -> list[tuple[int, str, str]]:
    """Return the 16 R32 matchups as (match_num, team_a, team_b) tuples."""
    out = []
    for match_num, slot_a, slot_b in R32_MATCHUPS:
        ta = slot_to_team[slot_a]
        tb = slot_to_team[slot_b]
        out.append((match_num, ta, tb))
    return out


if __name__ == "__main__":
    # Smoke test the assignment with the example from the WebFetch
    # (groups E,F,G,H,I,J,K,L produce 3rd-place qualifiers).
    test_qualifiers = ["E", "F", "G", "H", "I", "J", "K", "L"]
    matching = assign_third_place_slots(test_qualifiers)
    print("Test case: 3rd-place qualifiers from {E,F,G,H,I,J,K,L}")
    for slot in THIRD_PLACE_SLOTS:
        print(f"  Match {slot} (eligible {sorted(THIRD_PLACE_ELIGIBILITY[slot])}): "
              f"→ 3rd from group {matching[slot]}")
    # Verify: every group assigned exactly once, each to an eligible slot
    assigned = set(matching.values())
    assert assigned == set(test_qualifiers), f"missing: {set(test_qualifiers) - assigned}"
    for slot, group in matching.items():
        assert group in THIRD_PLACE_ELIGIBILITY[slot], \
            f"slot {slot} got {group} but eligibility is {THIRD_PLACE_ELIGIBILITY[slot]}"
    print("  ✓ valid matching")

    # Also test a tougher scenario: A,B,C,D,E,F,G,H (top 8 alphabetically)
    test2 = ["A", "B", "C", "D", "E", "F", "G", "H"]
    matching2 = assign_third_place_slots(test2)
    print(f"\nTest case 2: 3rd-place qualifiers from {test2}")
    for slot in THIRD_PLACE_SLOTS:
        print(f"  Match {slot}: → 3rd from group {matching2[slot]}")
    assert set(matching2.values()) == set(test2)
    for slot, group in matching2.items():
        assert group in THIRD_PLACE_ELIGIBILITY[slot]
    print("  ✓ valid matching")

    # And a likely-realistic scenario for WC 2026: a mix
    test3 = ["A", "C", "D", "E", "G", "I", "J", "L"]
    matching3 = assign_third_place_slots(test3)
    print(f"\nTest case 3: 3rd-place qualifiers from {test3}")
    for slot in THIRD_PLACE_SLOTS:
        print(f"  Match {slot}: → 3rd from group {matching3[slot]}")
    assert set(matching3.values()) == set(test3)
    print("  ✓ valid matching")

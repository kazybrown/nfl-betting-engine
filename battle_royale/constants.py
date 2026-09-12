"""Format constants: positions, roster rules, snake-draft math, Underdog scoring."""

from __future__ import annotations

import numpy as np

# Position encoding used across all array-based layers.
POSITIONS = ("QB", "RB", "WR", "TE")
POS_INDEX = {p: i for i, p in enumerate(POSITIONS)}

QB, RB, WR, TE = 0, 1, 2, 3

# Battle Royale rooms: six drafters, six rounds.
SEATS = 6
ROUNDS = 6
ROSTER_SIZE = 6
TOTAL_PICKS = SEATS * ROUNDS

# Roster construction: QB, RB, WR, WR, TE, FLEX (RB/WR/TE).
ROSTER_MIN = np.array([1, 1, 2, 1], dtype=np.int8)
ROSTER_MAX = np.array([1, 2, 3, 2], dtype=np.int8)

# Underdog NFL scoring (half PPR, no yardage bonuses).
UNDERDOG_SCORING = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -1.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "receptions": 0.50,
    "two_point_conversions": 2.0,
    "fumbles_lost": -2.0,
}


def seat_of_pick(pick: int) -> int:
    """Seat (0-5) holding overall pick number ``pick`` (1-36) in a snake draft."""
    if not 1 <= pick <= TOTAL_PICKS:
        raise ValueError(f"pick must be in 1..{TOTAL_PICKS}, got {pick}")
    rnd, k = divmod(pick - 1, SEATS)
    return k if rnd % 2 == 0 else SEATS - 1 - k


def picks_of_seat(seat: int) -> list[int]:
    """All overall pick numbers belonging to ``seat`` (0-5), in order."""
    if not 0 <= seat < SEATS:
        raise ValueError(f"seat must be in 0..{SEATS - 1}, got {seat}")
    return [p for p in range(1, TOTAL_PICKS + 1) if seat_of_pick(p) == seat]


def next_pick_of_seat(current_pick: int, seat: int) -> int | None:
    """First pick strictly after ``current_pick`` belonging to ``seat``, or None."""
    for p in range(current_pick + 1, TOTAL_PICKS + 1):
        if seat_of_pick(p) == seat:
            return p
    return None

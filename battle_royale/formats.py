"""Contest formats: structural rules of each Underdog 6-man weekly NFL draft.

Every layer reads structure from the :class:`ContestFormat` attached to the
slate, so switching from Battle Royale to Hurry Up (or any other weekly
room) is a ``--format`` flag plus that contest's ETR export. Formats are
data (``data/formats.json``): adding one needs no code change, and the
structural rules are validated on load. Battle Royale is the built-in
default, and module-level constants in :mod:`battle_royale.constants`
remain its (unchanged) definition for calibration code that is inherently
Battle-Royale-specific.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .constants import POSITIONS


@dataclass(frozen=True)
class ContestFormat:
    key: str
    name: str
    seats: int = 6
    rounds: int = 6
    # Per-position roster bounds in POSITIONS order (QB, RB, WR, TE).
    roster_min: tuple[int, ...] = (1, 1, 2, 1)
    roster_max: tuple[int, ...] = (1, 2, 3, 2)
    default_contest_size: int = 70_000
    payouts_file: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.seats < 2:
            raise ValueError(f"{self.key}: need at least 2 seats")
        if self.rounds < 1:
            raise ValueError(f"{self.key}: need at least 1 round")
        npos = len(POSITIONS)
        if len(self.roster_min) != npos or len(self.roster_max) != npos:
            raise ValueError(f"{self.key}: roster bounds must have {npos} entries (QB/RB/WR/TE)")
        if any(lo < 0 or lo > hi for lo, hi in zip(self.roster_min, self.roster_max)):
            raise ValueError(f"{self.key}: roster_min must satisfy 0 <= min <= max per position")
        # Every pick fills a roster slot in these rooms, so the bounds must be
        # able to absorb exactly `rounds` picks.
        if not sum(self.roster_min) <= self.rounds <= sum(self.roster_max):
            raise ValueError(
                f"{self.key}: {self.rounds} rounds cannot satisfy roster bounds "
                f"min={self.roster_min} max={self.roster_max}"
            )

    # ------------------------------------------------------------------
    # Derived structure
    # ------------------------------------------------------------------

    @property
    def total_picks(self) -> int:
        return self.seats * self.rounds

    @property
    def roster_size(self) -> int:
        return self.rounds

    @property
    def min_arr(self) -> np.ndarray:
        return np.array(self.roster_min, dtype=np.int8)

    @property
    def max_arr(self) -> np.ndarray:
        return np.array(self.roster_max, dtype=np.int8)

    def structure_key(self) -> str:
        """Stable string of everything that changes draft structure (for caches)."""
        return (
            f"{self.key}:{self.seats}x{self.rounds}"
            f":{','.join(map(str, self.roster_min))}:{','.join(map(str, self.roster_max))}"
        )

    # ------------------------------------------------------------------
    # Snake-draft math
    # ------------------------------------------------------------------

    def seat_of_pick(self, pick: int) -> int:
        """Seat (0-based) holding overall pick number ``pick`` (1-based)."""
        if not 1 <= pick <= self.total_picks:
            raise ValueError(f"pick must be in 1..{self.total_picks}, got {pick}")
        rnd, k = divmod(pick - 1, self.seats)
        return k if rnd % 2 == 0 else self.seats - 1 - k

    def picks_of_seat(self, seat: int) -> list[int]:
        """All overall pick numbers belonging to ``seat`` (0-based), in order."""
        if not 0 <= seat < self.seats:
            raise ValueError(f"seat must be in 0..{self.seats - 1}, got {seat}")
        return [p for p in range(1, self.total_picks + 1) if self.seat_of_pick(p) == seat]

    def next_pick_of_seat(self, current_pick: int, seat: int) -> int | None:
        """First pick strictly after ``current_pick`` belonging to ``seat``."""
        for p in range(current_pick + 1, self.total_picks + 1):
            if self.seat_of_pick(p) == seat:
                return p
        return None


BATTLE_ROYALE = ContestFormat(
    key="battle_royale",
    name="Battle Royale",
    notes="Underdog flagship weekly: 6 seats, 6 rounds, QB/RB/WR/WR/TE/FLEX.",
)


def load_formats(path: str | Path | None = None) -> dict[str, ContestFormat]:
    """Format registry from ``data/formats.json`` (or an explicit ``path``).

    Battle Royale is always present. Registry entries are validated on load;
    a structurally impossible entry fails immediately rather than mid-draft.
    """
    formats = {BATTLE_ROYALE.key: BATTLE_ROYALE}
    try:
        if path is not None:
            raw = Path(path).read_text()
        else:
            ref = resources.files("battle_royale") / "data" / "formats.json"
            raw = ref.read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        return formats
    for entry in json.loads(raw)["formats"]:
        fmt = ContestFormat(
            key=entry["key"],
            name=entry["name"],
            seats=int(entry.get("seats", 6)),
            rounds=int(entry.get("rounds", 6)),
            roster_min=tuple(int(entry["roster_min"][p]) for p in POSITIONS)
            if "roster_min" in entry
            else BATTLE_ROYALE.roster_min,
            roster_max=tuple(int(entry["roster_max"][p]) for p in POSITIONS)
            if "roster_max" in entry
            else BATTLE_ROYALE.roster_max,
            default_contest_size=int(entry.get("default_contest_size", 70_000)),
            payouts_file=entry.get("payouts_file"),
            notes=entry.get("notes", ""),
        )
        formats[fmt.key] = fmt
    return formats


def get_format(key: str, path: str | Path | None = None) -> ContestFormat:
    formats = load_formats(path)
    if key not in formats:
        known = ", ".join(sorted(formats))
        raise KeyError(f"unknown contest format {key!r}; known formats: {known}")
    return formats[key]

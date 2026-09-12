"""Draft room state: rosters, position counts, availability, legality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .slate import Slate


@dataclass(eq=False)
class DraftState:
    """Mutable room state. ``next_pick`` is the overall pick about to be made.

    All structural rules (seats, rounds, roster bounds, snake order) come
    from ``slate.fmt``, so the same state machinery runs any contest format.
    """

    slate: Slate
    rosters: list[list[int]] = None  # type: ignore[assignment]
    counts: np.ndarray = None  # type: ignore[assignment]
    avail: np.ndarray = None  # type: ignore[assignment]
    next_pick: int = 1

    def __post_init__(self) -> None:
        if self.rosters is None:
            self.rosters = [[] for _ in range(self.fmt.seats)]
        if self.counts is None:
            self.counts = np.zeros((self.fmt.seats, 4), dtype=np.int8)
        if self.avail is None:
            self.avail = np.ones(self.slate.n, dtype=bool)

    @property
    def fmt(self):
        return self.slate.fmt

    # ------------------------------------------------------------------
    # Construction / copying
    # ------------------------------------------------------------------

    @classmethod
    def from_history(cls, slate: Slate, picks: list[str | int]) -> DraftState:
        """Rebuild state from the ordered list of completed picks."""
        st = cls(slate)
        for x in picks:
            st.apply_pick(slate.idx(x))
        return st

    def clone(self) -> DraftState:
        st = DraftState.__new__(DraftState)
        st.slate = self.slate
        st.rosters = [list(r) for r in self.rosters]
        st.counts = self.counts.copy()
        st.avail = self.avail.copy()
        st.next_pick = self.next_pick
        return st

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def complete(self) -> bool:
        return self.next_pick > self.fmt.total_picks

    def seat_on_clock(self) -> int:
        return self.fmt.seat_of_pick(self.next_pick)

    def picks_left(self, seat: int) -> int:
        return self.fmt.roster_size - len(self.rosters[seat])

    def next_own_pick(self, seat: int) -> int | None:
        """The seat's next pick strictly after the one currently on the clock."""
        return self.fmt.next_pick_of_seat(self.next_pick, seat)

    def position_eligibility(self, seat: int) -> np.ndarray:
        """Bool[4]: positions this seat may legally draft with its next pick."""
        counts = self.counts[seat]
        left_after = self.picks_left(seat) - 1
        out = np.zeros(4, dtype=bool)
        if left_after < 0:
            return out
        rmin, rmax = self.fmt.min_arr, self.fmt.max_arr
        for pos in range(4):
            if counts[pos] >= rmax[pos]:
                continue
            after = counts.copy()
            after[pos] += 1
            needed = int(np.maximum(rmin - after, 0).sum())
            out[pos] = needed <= left_after
        return out

    def eligible_mask(self, seat: int) -> np.ndarray:
        """Bool[n_players]: available players this seat may legally draft."""
        pos_ok = self.position_eligibility(seat)
        return self.avail & pos_ok[self.slate.pos]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def apply_pick(self, idx: int) -> None:
        """Record ``idx`` as the pick for the seat currently on the clock."""
        if self.complete:
            raise ValueError("draft is complete")
        idx = int(idx)
        if not self.avail[idx]:
            raise ValueError(f"player already drafted: {self.slate.players[idx].name}")
        seat = self.seat_on_clock()
        pos = int(self.slate.pos[idx])
        if not self.position_eligibility(seat)[pos]:
            raise ValueError(
                f"illegal pick for seat {seat}: {self.slate.players[idx].name} "
                f"({self.slate.players[idx].position}) at pick {self.next_pick}"
            )
        self.rosters[seat].append(idx)
        self.counts[seat, pos] += 1
        self.avail[idx] = False
        self.next_pick += 1

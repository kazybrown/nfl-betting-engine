"""Simulated contest field: generation, combinatorial ownership, duplication.

Ownership here is exact counting inside a simulated (or real, if supplied)
field of entries — never products of marginal ownership, which historical
Battle Royale fields violate badly for stacks and negatively for same-slot
rivals (task-5 findings).
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np

from .constants import SEATS
from .opponents import OpponentPolicy
from .slate import Slate


def generate_field(
    policy: OpponentPolicy,
    n_entries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate fresh rooms until ``n_entries`` rosters exist; (entries, 6)."""
    rooms = math.ceil(n_entries / SEATS)
    out = np.empty((rooms * SEATS, SEATS), dtype=np.int16)
    k = 0
    for _ in range(rooms):
        st = policy.simulate_room(rng)
        for seat in range(SEATS):
            out[k] = st.rosters[seat]
            k += 1
    return out[:n_entries]


class RosterIndex:
    """Exact-roster copy counts only — the light duplication reference.

    Unlike :class:`FieldAnalytics`, this skips pair/triple counting, so it is
    cheap to build over the large field used for duplicate payout sharing and
    small enough to cache on disk.
    """

    def __init__(self, field: np.ndarray):
        field = np.asarray(field, dtype=np.int64)
        self.n_entries = len(field)
        self.roster_counts: Counter = Counter(
            tuple(sorted(int(x) for x in row)) for row in field
        )

    def roster_copies(self, rosters: np.ndarray) -> np.ndarray:
        out = np.zeros(len(rosters), dtype=np.int64)
        for k, row in enumerate(np.asarray(rosters, dtype=np.int64)):
            out[k] = self.roster_counts.get(tuple(sorted(int(x) for x in row)), 0)
        return out


class FieldAnalytics:
    """Exact marginal / pair / triple / full-roster counts for a field."""

    def __init__(self, slate: Slate, field: np.ndarray):
        field = np.asarray(field, dtype=np.int16)
        if field.ndim != 2 or field.shape[1] != 6:
            raise ValueError("field must have shape (entries, 6)")
        self.slate = slate
        self.field = field
        self.n_entries = len(field)

        self.marginal_counts = np.zeros(slate.n, dtype=np.int64)
        self.pair_counts: Counter = Counter()
        self.triple_counts: Counter = Counter()
        self.roster_counts: Counter = Counter()
        for row in field:
            sig = tuple(sorted(int(x) for x in row))
            self.roster_counts[sig] += 1
            self.marginal_counts[list(sig)] += 1
            self.pair_counts.update(combinations(sig, 2))
            self.triple_counts.update(combinations(sig, 3))
        self.entry_ownership = self.marginal_counts / max(self.n_entries, 1)
        # One room per six entries; a player appears at most once per room.
        self.room_drafted_rate = self.marginal_counts / max(self.n_entries / SEATS, 1)

    # ------------------------------------------------------------------

    def _ids(self, players) -> tuple[int, ...]:
        return tuple(sorted(self.slate.idx(p) for p in players))

    def combo_count(self, players) -> int:
        ids = self._ids(players)
        k = len(ids)
        if k == 1:
            return int(self.marginal_counts[ids[0]])
        if k == 2:
            return int(self.pair_counts.get(ids, 0))
        if k == 3:
            return int(self.triple_counts.get(ids, 0))
        if k == 6:
            return int(self.roster_counts.get(ids, 0))
        target = set(ids)
        return int(sum(target.issubset(set(int(x) for x in row)) for row in self.field))

    def combo_metrics(self, players) -> dict:
        ids = self._ids(players)
        count = self.combo_count(players)
        p = count / max(self.n_entries, 1)
        p_ind = float(np.prod(self.entry_ownership[list(ids)]))
        return {
            "players": self.slate.names(ids),
            "count": count,
            "field_ownership": p,
            "independence_product": p_ind,
            "affinity_vs_independence": (p / p_ind) if p_ind > 0 else None,
        }

    def roster_copies(self, rosters: np.ndarray) -> np.ndarray:
        """Exact-copy counts in this field for each roster row."""
        out = np.zeros(len(rosters), dtype=np.int64)
        for k, row in enumerate(np.asarray(rosters, dtype=np.int64)):
            out[k] = self.roster_counts.get(tuple(sorted(int(x) for x in row)), 0)
        return out

    def duplication_summary(self) -> dict:
        hist = Counter(self.roster_counts.values())
        unique_entries = sum(v for v in self.roster_counts.values() if v == 1)
        return {
            "entries": self.n_entries,
            "distinct_rosters": len(self.roster_counts),
            "unique_entry_share": unique_entries / max(self.n_entries, 1),
            "duplicated_entry_share": 1.0 - unique_entries / max(self.n_entries, 1),
            "max_exact_copies": max(self.roster_counts.values()) if self.roster_counts else 0,
            "copy_count_histogram": dict(sorted(hist.items())),
        }

    def calibration_vs_slate(self) -> dict:
        """How closely simulated room-drafted rates match the slate's Own %."""
        err = np.abs(self.room_drafted_rate - self.slate.room_drafted_rate)
        top36 = np.argsort(self.slate.rank)[:36]
        return {
            "entries": self.n_entries,
            "mae_all_pct": float(err.mean() * 100),
            "mae_top36_pct": float(err[top36].mean() * 100),
            "corr_room_rate": float(
                np.corrcoef(self.room_drafted_rate, self.slate.room_drafted_rate)[0, 1]
            ),
        }


def positional_construction(slate: Slate, field: np.ndarray) -> dict:
    """FLEX construction shares across field entries (RB/WR/TE flex rates)."""
    pos = slate.pos[np.asarray(field, dtype=np.int64)]
    counts = np.stack([(pos == k).sum(axis=1) for k in range(4)], axis=1)
    return {
        "avg_positions": {p: float(c) for p, c in zip(["QB", "RB", "WR", "TE"], counts.mean(0))},
        "flex_share": {
            "RB": float(np.mean(counts[:, 1] == 2)),
            "WR": float(np.mean(counts[:, 2] == 3)),
            "TE": float(np.mean(counts[:, 3] == 2)),
        },
    }

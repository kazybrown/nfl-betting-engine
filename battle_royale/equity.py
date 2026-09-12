"""Tournament payout equity for candidate rosters.

The contest is modeled as ``contest_size`` entries drawn from the same
field-generating process as the simulated reference field. Conditional on a
player-outcome simulation, every entry's score is determined by its roster, so
the sampled field gives an estimate of the score distribution among entrants.
A roster's expected rank then follows from the fraction of field entries that
outscore it, scaled to contest size, with exact-duplicate ties sharing the top
of the payout curve.

Payout curves are step functions on the fractional rank (rank / contest size)
expressed in entry-fee multiples. The default curve is a generic top-heavy
weekly-tournament shape — replace it with the actual contest structure via
:class:`PayoutCurve` when known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist

# (top fraction of field, payout in entry-fee multiples). A rank counts for a
# tier when rank/contest_size <= top_fraction. Roughly: ~15% of entries paid,
# strong concentration at the very top.
DEFAULT_CURVE_POINTS = [
    (0.000002, 25000.0),
    (0.00001, 2500.0),
    (0.0001, 400.0),
    (0.001, 100.0),
    (0.01, 20.0),
    (0.05, 5.0),
    (0.10, 2.5),
    (0.15, 1.5),
]


@dataclass
class PayoutCurve:
    points: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_CURVE_POINTS)
    )

    def __post_init__(self) -> None:
        self.points = sorted(self.points)
        self._fracs = np.array([p[0] for p in self.points])
        self._pays = np.array([p[1] for p in self.points])

    def payout(self, rank_frac: np.ndarray) -> np.ndarray:
        """Vectorized payout (entry multiples) for fractional ranks."""
        idx = np.searchsorted(self._fracs, rank_frac, side="left")
        out = np.zeros_like(np.asarray(rank_frac, dtype=float))
        inside = idx < len(self._pays)
        out[inside] = self._pays[idx[inside]]
        return out


@dataclass
class TournamentModel:
    contest_size: int = 150_000
    curve: PayoutCurve = field(default_factory=PayoutCurve)
    # Sims where fewer than this many sampled field entries outscore a roster
    # get their payout integrated over the Beta posterior of the true
    # exceedance probability, instead of a point estimate. This smooths the
    # jackpot region, which is otherwise invisible below 1/n_entries
    # resolution and dominated by lucky "beat the whole sample" events.
    tail_quadrature_below: int = 12
    _quad_q: np.ndarray = field(
        default_factory=lambda: (np.arange(64) + 0.5) / 64.0, repr=False
    )

    def _top_payout_table(self, n_entries: int) -> np.ndarray:
        """Expected payout (no tie sharing) for exceedance counts 0..K-1,
        integrating rank over the Beta(count+0.5, n-count+0.5) posterior."""
        f = float(self.contest_size)
        k = min(self.tail_quadrature_below, max(n_entries - 1, 1))
        out = np.empty(k)
        for c in range(k):
            p = beta_dist.ppf(self._quad_q, c + 0.5, n_entries - c + 0.5)
            rank_frac = ((f - 1.0) * p + 1.0) / f
            out[c] = float(self.curve.payout(rank_frac).mean())
        return out

    def evaluate_rosters(
        self,
        scores: np.ndarray,
        field_entries: np.ndarray,
        rosters: np.ndarray,
        roster_dup_counts: np.ndarray | None = None,
        chunk: int = 250,
    ) -> list[dict]:
        """Equity metrics for each roster against the sampled field.

        Args:
            scores: player-outcome sims, shape (n_sims, n_players).
            field_entries: reference field, shape (n_entries, 6) player indices.
            rosters: rosters to evaluate, shape (n_rosters, 6).
            roster_dup_counts: for each roster, its exact-copy count within the
                reference field (0 if absent). Used for top-of-field payout
                sharing; computed as 0s when omitted.
            chunk: sims per block (bounds the (chunk, n_entries) work array).

        Returns:
            One dict per roster with expected payout and rank-tail metrics.
        """
        n_sims = scores.shape[0]
        n_entries = len(field_entries)
        n_rosters = len(rosters)
        f = float(self.contest_size)
        dup = (
            np.zeros(n_rosters)
            if roster_dup_counts is None
            else np.asarray(roster_dup_counts, dtype=float)
        )
        # Expected exact copies of each roster among the other contest entries.
        expected_copies = dup * (f - 1.0) / max(n_entries, 1)

        pay_sum = np.zeros(n_rosters)
        beat_sample_max = np.zeros(n_rosters)
        top_counts = {0.001: np.zeros(n_rosters), 0.01: np.zeros(n_rosters), 0.10: np.zeros(n_rosters)}
        score_sum = np.zeros(n_rosters)
        score_all = np.empty((n_sims, n_rosters), dtype=np.float32)
        top_table = self._top_payout_table(n_entries)
        k_quad = len(top_table)

        done = 0
        while done < n_sims:
            m = min(chunk, n_sims - done)
            x = scores[done : done + m]
            fs = x[:, field_entries].sum(axis=2)
            fs_sorted = np.sort(fs, axis=1)
            rs = x[:, rosters].sum(axis=2)  # (m, n_rosters)
            score_all[done : done + m] = rs
            score_sum += rs.sum(axis=0)

            for t in range(m):
                row = fs_sorted[t]
                s = rs[t]
                # Field entries strictly above / exactly equal to each roster.
                hi = np.searchsorted(row, s, side="right")
                lo = np.searchsorted(row, s, side="left")
                n_gt = n_entries - hi
                n_eq = hi - lo
                p_gt = n_gt / n_entries
                p_eq = n_eq / n_entries
                exp_better = (f - 1.0) * p_gt
                exp_equal = np.maximum((f - 1.0) * p_eq, expected_copies)
                rank_frac = (exp_better + 0.5 * exp_equal + 1.0) / f
                pay = self.curve.payout(rank_frac)
                # Near the top of the field the sample resolution (1/n_entries)
                # cannot see jackpot tiers; integrate over the Beta posterior
                # of the true exceedance probability instead.
                near = n_gt < k_quad
                pay[near] = top_table[n_gt[near]]
                # Top-of-field tie sharing: when essentially nothing in the
                # contest beats the roster, exact co-holders split the payout.
                near_top = exp_better < 1.0
                pay = np.where(near_top, pay / (1.0 + exp_equal), pay)
                pay_sum += pay
                beat_sample_max += s > row[-1]
                for frac, acc in top_counts.items():
                    acc += p_gt <= frac
            done += m

        out = []
        for k in range(n_rosters):
            sc = score_all[:, k].astype(float)
            out.append(
                {
                    "expected_payout": float(pay_sum[k] / n_sims),
                    "win_rate_vs_sample": float(beat_sample_max[k] / n_sims),
                    "p_top_0_1pct": float(top_counts[0.001][k] / n_sims),
                    "p_top_1pct": float(top_counts[0.01][k] / n_sims),
                    "p_top_10pct": float(top_counts[0.10][k] / n_sims),
                    "mean_score": float(score_sum[k] / n_sims),
                    "p90_score": float(np.quantile(sc, 0.90)),
                    "p99_score": float(np.quantile(sc, 0.99)),
                    "field_copies_in_sample": int(dup[k]),
                    "expected_contest_copies": float(expected_copies[k]),
                }
            )
        return out

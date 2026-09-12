"""Tournament payout equity for candidate rosters.

The contest is modeled as ``contest_size`` entries drawn from the same
field-generating process as the simulated reference field. Conditional on a
player-outcome simulation, every entry's score is determined by its roster, so
the sampled field estimates the score distribution among entrants. A roster's
rank distribution then follows from the fraction of field entries that
outscore it, scaled to contest size; exact co-holders of the same roster share
the tied rank block's pooled prizes.

Payout curves are rank-based step functions in entry-fee multiples: each tier
is (threshold, payout) where a threshold >= 1 is an absolute rank cutoff
("ranks 1..10 pay 1000x"; 1 alone is the winner) and a threshold < 1 is a
fraction of the contest ("top 1% pays 15x"). Rank-based top tiers guarantee
first place pays first place at any contest size. The default curve is a
generic top-heavy weekly tournament shape - replace it with the actual
contest structure when known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import norm

# (threshold, payout multiple). Thresholds > 1 are absolute ranks; <= 1 are
# contest fractions. Roughly: winner-heavy top, ~15% of entries paid.
DEFAULT_CURVE_POINTS = [
    (1, 20000.0),
    (5, 2500.0),
    (25, 500.0),
    (100, 120.0),
    (0.005, 25.0),
    (0.02, 8.0),
    (0.06, 4.0),
    (0.10, 2.5),
    (0.15, 1.5),
]


@dataclass
class PayoutCurve:
    points: list[tuple[float, float]] = field(
        default_factory=lambda: list(DEFAULT_CURVE_POINTS)
    )

    def resolve(self, contest_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Absolute-rank cutoffs and payouts, sorted by rank.

        Thresholds >= 1 are absolute ranks (1 = the winner); thresholds < 1
        are contest fractions.
        """
        cutoffs = []
        for threshold, pay in self.points:
            rank = float(threshold) if threshold >= 1 else max(threshold * contest_size, 1.0)
            cutoffs.append((rank, float(pay)))
        cutoffs.sort()
        ranks = np.array([c[0] for c in cutoffs])
        pays = np.array([c[1] for c in cutoffs])
        return ranks, pays


class _RankPayout:
    """Piecewise-constant payout by rank with O(1) range averages.

    ``payout(r)`` is the prize for rank r; ``range_mean(lo, hi)`` is the mean
    prize over the continuous rank interval [lo, hi] - the pooled-prize share
    for a tied block occupying those ranks.
    """

    def __init__(self, curve: PayoutCurve, contest_size: int):
        self.cutoffs, self.pays = curve.resolve(contest_size)

    def payout(self, ranks: np.ndarray) -> np.ndarray:
        # Continuous rank r in [k, k+1) means integer rank k.
        idx = np.searchsorted(self.cutoffs, np.floor(ranks), side="left")
        out = np.zeros_like(np.asarray(ranks, dtype=float))
        inside = idx < len(self.pays)
        out[inside] = self.pays[idx[inside]]
        return out

    def range_mean(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Mean payout over rank interval [lo, hi] (hi >= lo >= 1)."""
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        width = np.maximum(hi - lo, 0.0)
        point = self.payout(lo)
        # Integral of the step function from rank 1 to r.
        def cum(r):
            total = np.zeros_like(r)
            prev = np.ones_like(r)
            for cutoff, pay in zip(self.cutoffs, self.pays):
                seg = np.clip(np.minimum(r, cutoff + 1.0) - prev, 0.0, None)
                total += seg * pay
                prev = np.maximum(prev, np.minimum(r, cutoff + 1.0))
            return total

        avg = np.where(width > 1e-9, (cum(hi) - cum(lo)) / np.maximum(width, 1e-9), point)
        return avg


@dataclass
class TournamentModel:
    contest_size: int = 70_000
    curve: PayoutCurve = field(default_factory=PayoutCurve)
    # Sims where fewer than this many sampled field entries outscore a roster
    # get their payout integrated over the Beta posterior of the true
    # exceedance probability (and the Binomial rank spread around it), instead
    # of a point estimate. This resolves the jackpot region, which is invisible
    # below 1/n_entries sample resolution.
    tail_quadrature_below: int = 12
    _quad_q: np.ndarray = field(
        default_factory=lambda: (np.arange(64) + 0.5) / 64.0, repr=False
    )
    # Binomial rank spread nodes (normal approximation quantiles).
    _rank_z: np.ndarray = field(
        default_factory=lambda: norm.ppf([0.1, 0.3, 0.5, 0.7, 0.9]), repr=False
    )

    def _top_payout_table(self, n_entries: int, rp: _RankPayout) -> np.ndarray:
        """Expected payout for exceedance counts 0..K-1 in the sampled field.

        Integrates the true exceedance probability p over its Beta posterior
        and the contest rank over a 5-node normal approximation of
        Binomial(contest_size - 1, p).
        """
        f = float(self.contest_size)
        k = min(self.tail_quadrature_below, max(n_entries - 1, 1))
        out = np.empty(k)
        for c in range(k):
            p = beta_dist.ppf(self._quad_q, c + 0.5, n_entries - c + 0.5)
            mean_better = (f - 1.0) * p
            sd_better = np.sqrt(np.maximum((f - 1.0) * p * (1.0 - p), 0.0))
            ranks = 1.0 + mean_better[:, None] + sd_better[:, None] * self._rank_z[None, :]
            ranks = np.maximum(ranks, 1.0)
            out[c] = float(rp.payout(ranks.ravel()).mean())
        return out

    def evaluate_rosters(
        self,
        scores: np.ndarray,
        field_entries: np.ndarray,
        rosters: np.ndarray,
        roster_dup_counts: np.ndarray | None = None,
        dup_reference_entries: int | None = None,
        chunk: int = 250,
    ) -> list[dict]:
        """Equity metrics for each roster against the sampled field.

        Args:
            scores: player-outcome sims, shape (n_sims, n_players).
            field_entries: reference field, shape (n_entries, 6) player indices.
            rosters: rosters to evaluate, shape (n_rosters, 6).
            roster_dup_counts: exact-copy counts for each roster within a
                duplication reference field (0 if absent). Pass counts from a
                LARGER field than ``field_entries`` (via
                ``dup_reference_entries``) to avoid quantizing the sharing
                penalty at (contest_size / n_entries) copies per sampled copy.
            dup_reference_entries: size of the field the dup counts came from;
                defaults to ``len(field_entries)``.
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
        dup_ref = float(dup_reference_entries or n_entries)
        # Expected exact copies of each roster among the other contest entries.
        expected_copies = dup * (f - 1.0) / max(dup_ref, 1.0)

        rp = _RankPayout(self.curve, self.contest_size)
        pay_sum = np.zeros(n_rosters)
        beat_sample_max = np.zeros(n_rosters)
        top_counts = {
            0.001: np.zeros(n_rosters),
            0.01: np.zeros(n_rosters),
            0.10: np.zeros(n_rosters),
        }
        score_sum = np.zeros(n_rosters)
        score_all = np.empty((n_sims, n_rosters), dtype=np.float32)
        top_table = self._top_payout_table(n_entries, rp)
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
                # Tied block occupies ranks [exp_better+1, exp_better+1+exp_equal];
                # each co-holder receives the block's mean prize.
                lo_rank = exp_better + 1.0
                pay = rp.range_mean(lo_rank, lo_rank + exp_equal)
                # Near the top of the field the sample cannot resolve the
                # jackpot tiers; use the Beta/Binomial quadrature table, then
                # apply tie sharing as the mean over the tied rank block.
                near = n_gt < k_quad
                if near.any():
                    tie = exp_equal[near]
                    base = top_table[n_gt[near]]
                    shared = rp.range_mean(
                        np.ones_like(tie), 1.0 + tie
                    )  # block mean if the roster truly sits at the top
                    # Blend: quadrature handles rank uncertainty with no ties;
                    # when a tie block exists, the block mean at the top is the
                    # better estimate of each co-holder's share.
                    pay[near] = np.where(tie > 0.5, np.minimum(base, shared), base)
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
                    "dup_copies_in_reference": int(dup[k]),
                    "expected_contest_copies": float(expected_copies[k]),
                }
            )
        return out

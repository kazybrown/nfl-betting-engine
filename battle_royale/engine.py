"""Composition root: slate + outcome model + opponent policy in one object."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm

from .correlation import CorrelationModel
from .marginals import MarginalModel
from .opponents import OpponentPolicy
from .slate import Slate


class BattleRoyaleEngine:
    """Bundles the player-outcome model and the field draft policy for a slate."""

    def __init__(
        self,
        slate: Slate,
        marginals: MarginalModel | None = None,
        correlation: CorrelationModel | None = None,
        policy: OpponentPolicy | None = None,
        seed: int | None = None,
    ):
        self.slate = slate
        self.marginals = marginals or MarginalModel.from_slate(slate)
        self.correlation = correlation or CorrelationModel.from_slate(slate)
        self.policy = policy or OpponentPolicy(slate)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_csv(cls, path: str | Path, seed: int | None = None, **kwargs) -> BattleRoyaleEngine:
        return cls(Slate.from_csv(path), seed=seed, **kwargs)

    # ------------------------------------------------------------------

    def sample_scores(self, n_sims: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Correlated weekly scores, shape (n_sims, n_players).

        Gaussian copula over fitted Gamma marginals; each player's mean equals
        the slate projection exactly.
        """
        rng = rng or self.rng
        z = self.correlation.latent_normals(n_sims, rng)
        u = np.clip(norm.cdf(z), 1e-9, 1 - 1e-9)
        return self.marginals.ppf(u)

    def roster_scores(self, scores: np.ndarray, rosters: np.ndarray) -> np.ndarray:
        """Sum player scores per roster: (n_sims, n_rosters)."""
        return scores[:, np.asarray(rosters, dtype=np.int64)].sum(axis=2)

    def marginal_calibration_check(self, n_sims: int = 20_000, top_n: int = 24) -> list[dict]:
        """Simulated vs target mean/variance for the top-ranked players."""
        idx = np.argsort(self.slate.rank)[:top_n]
        x = self.sample_scores(n_sims)[:, idx]
        rows = []
        for k, j in enumerate(idx):
            p = self.slate.players[int(j)]
            rows.append(
                {
                    "player": p.name,
                    "position": p.position,
                    "proj": float(self.slate.proj[j]),
                    "sim_mean": float(x[:, k].mean()),
                    "target_var": float(self.marginals.variance[j]),
                    "sim_var": float(x[:, k].var()),
                }
            )
        return rows

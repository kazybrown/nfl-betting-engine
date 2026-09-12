"""Per-player weekly score distributions.

Each player's weekly Underdog score is modeled as a Gamma random variable whose
mean is anchored exactly to the slate projection and whose dispersion comes from
an empirical position-level CV^2 curve fitted to historical nflverse weekly
stats scored under Underdog rules (see :mod:`battle_royale.calibration`).

A per-player tilt from the slate's Ceiling column widens or narrows dispersion
relative to position peers with the same projection, without moving the mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.stats import gamma

from .constants import POSITIONS
from .slate import Slate

# Conservative fallback if no fitted table is available. Derived from the
# NFL-DFS-Tools historical moment table used by the transferred v2 engine
# (full-PPR; slightly wider than Underdog half-PPR reality).
_FALLBACK_CV2 = {
    "QB": [(11, 0.436), (15, 0.277), (19, 0.190), (23, 0.189), (27, 0.141)],
    "RB": [(8, 0.682), (12, 0.462), (16, 0.305), (20, 0.284), (24, 0.218)],
    "WR": [(8, 0.561), (12, 0.447), (16, 0.400), (20, 0.356), (24, 0.288)],
    "TE": [(5.7, 0.633), (9.7, 0.467), (13.7, 0.387), (17.7, 0.305)],
}

MIN_GAMMA_SHAPE = 0.12


def _load_table(path: str | Path | None) -> dict:
    """Load a fitted dispersion table, preferring packaged data."""
    if path is not None:
        return json.loads(Path(path).read_text())
    try:
        ref = resources.files("battle_royale") / "data" / "marginal_dispersion.json"
        return json.loads(ref.read_text())
    except (FileNotFoundError, ModuleNotFoundError):
        return {
            "source": "fallback: NFL-DFS-Tools full-PPR moment table",
            "positions": {
                p: {"buckets": [{"anchor": a, "cv2": c} for a, c in rows]}
                for p, rows in _FALLBACK_CV2.items()
            },
        }


@dataclass
class MarginalModel:
    """Gamma marginals with copula-ready ppf transform."""

    shape: np.ndarray
    scale: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    table_source: str

    @classmethod
    def from_slate(
        cls,
        slate: Slate,
        table_path: str | Path | None = None,
        ceiling_tilt: float = 0.45,
        cv_scale: float = 1.0,
    ) -> MarginalModel:
        """Build marginals for a slate.

        Args:
            slate: current-week slate.
            table_path: optional explicit dispersion-table JSON path.
            ceiling_tilt: exponent applied to each player's ceiling/projection
                ratio relative to the position median; 0 disables the tilt.
            cv_scale: global multiplier on CV (not CV^2) for sensitivity runs.
        """
        table = _load_table(table_path)
        cv2 = np.empty(slate.n)
        for pos_idx, pos in enumerate(POSITIONS):
            buckets = table.get("positions", {}).get(pos, {}).get("buckets") or [
                {"anchor": a, "mean_points": a, "cv2": c} for a, c in _FALLBACK_CV2[pos]
            ]
            # Prefer the realized conditional mean as the curve coordinate
            # (matches what a sharp projection estimates); fall back to the
            # legacy anchor coordinate for older tables.
            coords = np.array(
                [b.get("mean_points", b["anchor"]) for b in buckets], dtype=float
            )
            values = np.array([b["cv2"] for b in buckets], dtype=float)
            order = np.argsort(coords)
            coords, values = coords[order], values[order]
            mask = slate.pos == pos_idx
            cv2[mask] = np.interp(slate.proj[mask], coords, values)

        mean = np.maximum(slate.proj, 1e-6)
        if ceiling_tilt > 0.0:
            # Compare the slate's Ceiling to the untilted model's own p90:
            # a player whose stated ceiling exceeds what the fitted curve
            # already implies gets wider tails, and vice versa. Comparing to
            # the position-median ceiling/projection ratio instead would
            # mechanically re-tilt the curve along the projection axis
            # (ceiling/proj falls with projection because CV does).
            base_var = np.maximum(mean**2 * cv2, 1e-6)
            base_shape = np.maximum(mean**2 / base_var, MIN_GAMMA_SHAPE)
            implied_p90 = gamma.ppf(0.90, a=base_shape, scale=mean / base_shape)
            ratio = slate.ceiling / np.maximum(implied_p90, 1e-6)
            tilt = np.clip(ratio**ceiling_tilt, 0.75, 1.35)
            # Tilt acts on CV, so CV^2 picks up the square.
            cv2 = cv2 * tilt**2

        cv2 = cv2 * cv_scale**2
        variance = np.maximum(mean**2 * cv2, 1e-6)
        shape = np.maximum(mean**2 / variance, MIN_GAMMA_SHAPE)
        # Preserve the projection as the exact mean: scale = mean / shape.
        scale = mean / shape
        return cls(
            shape=shape,
            scale=scale,
            mean=mean,
            variance=shape * scale**2,
            table_source=str(table.get("source", "unknown")),
        )

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Quantile transform of uniforms (n_sims, n_players) to scores."""
        x = np.empty_like(u)
        for j in range(u.shape[1]):
            x[:, j] = gamma.ppf(u[:, j], a=self.shape[j], scale=self.scale[j])
        return x

    def quantile(self, q: float) -> np.ndarray:
        """Per-player marginal quantile (e.g. q=0.9 for ceiling-ish values)."""
        return gamma.ppf(q, a=self.shape, scale=self.scale)

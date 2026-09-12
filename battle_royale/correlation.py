"""Teammate / game-environment dependence via a Gaussian copula.

Pairwise latent correlations are assigned by relationship (same-team or
opposing-team position pair), using empirically fitted values from historical
Underdog-scored weekly data (see :mod:`battle_royale.calibration`), then the
matrix is projected to the nearest PSD correlation matrix before Cholesky.

Cross-team same-game coefficients ARE the game-environment dependence; no
additional generic game shock is layered on top (that would double-count).

An optional per-player modifier uses the slate's conditional-optimal columns
("Optimal IF QB Optimal" / "Optimal IF Opp QB Optimal") to nudge stack
strength for specific players. It is a modest multiplicative tilt, not a raw
correlation source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .constants import POSITIONS
from .slate import Slate

# Empirical fallback priors (task-2 calibration from historical DK-scored
# correlation grids, shrunk toward Battle Royale projection bands).
_FALLBACK = {
    "same_team": {
        "QB-WR": 0.34,
        "QB-TE": 0.26,
        "QB-RB": -0.02,
        "WR-WR": -0.03,
        "WR-TE": 0.00,
        "RB-WR": -0.025,
        "RB-TE": 0.00,
        "RB-RB": -0.06,
        "TE-TE": -0.04,
    },
    "opp_team": {
        "QB-QB": 0.17,
        "QB-WR": 0.05,
        "QB-TE": 0.07,
        "QB-RB": 0.035,
        "WR-WR": 0.04,
        "WR-TE": 0.03,
        "RB-WR": 0.015,
        "RB-TE": -0.025,
        "RB-RB": -0.07,
        "TE-TE": 0.00,
    },
}

RHO_CLIP = (-0.30, 0.55)
ETR_LIFT_CLIP = (0.82, 1.22)


_POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def pair_key(pos_a: str, pos_b: str) -> str:
    """Canonical relationship key in position order (QB-WR, WR-TE, ...)."""
    a, b = sorted((pos_a, pos_b), key=_POS_ORDER.__getitem__)
    return f"{a}-{b}"


def _load_table(path: str | Path | None) -> dict:
    if path is not None:
        return json.loads(Path(path).read_text())
    try:
        ref = resources.files("battle_royale") / "data" / "correlations.json"
        return json.loads(ref.read_text())
    except (FileNotFoundError, ModuleNotFoundError):
        return {"source": "fallback: task-2 shrunk historical priors", **_FALLBACK}


def _rho_from_entry(entry) -> float:
    """Table entries are either bare floats or dicts with a 'rho' field."""
    if isinstance(entry, dict):
        return float(entry["rho"])
    return float(entry)


@dataclass
class CorrelationModel:
    matrix: np.ndarray
    cholesky: np.ndarray
    table_source: str

    @classmethod
    def from_slate(
        cls,
        slate: Slate,
        table_path: str | Path | None = None,
        use_etr_lift: bool = True,
        team_env: dict[str, float] | None = None,
    ) -> CorrelationModel:
        """Build the latent correlation matrix for a slate.

        ``team_env`` optionally scales every same-game pair by a per-team
        multiplier (both teams of a game share it) — used to tilt game
        environments by Vegas totals. Multipliers are expected to be modest
        (~0.85-1.18); values are re-clipped to RHO_CLIP afterward.
        """
        table = _load_table(table_path)
        same_team = {k: _rho_from_entry(v) for k, v in table["same_team"].items()}
        opp_team = {k: _rho_from_entry(v) for k, v in table["opp_team"].items()}

        n = slate.n
        c = np.eye(n)
        pos_names = [POSITIONS[p] for p in slate.pos]
        for i in range(n):
            for j in range(i + 1, n):
                if not slate.same_game(i, j):
                    continue
                key = pair_key(pos_names[i], pos_names[j])
                if slate.same_team(i, j):
                    rho = same_team.get(key, 0.0)
                    if use_etr_lift and key in ("QB-WR", "QB-TE"):
                        skill = j if pos_names[j] != "QB" else i
                        rho *= _etr_lift(slate, skill, opposing=False)
                    elif use_etr_lift and key == "QB-RB":
                        skill = j if pos_names[j] == "RB" else i
                        rho *= _etr_lift(slate, skill, opposing=False)
                else:
                    rho = opp_team.get(key, 0.0)
                    if use_etr_lift and key in ("QB-WR", "QB-TE") and "QB" in (
                        pos_names[i],
                        pos_names[j],
                    ):
                        skill = j if pos_names[j] != "QB" else i
                        rho *= _etr_lift(slate, skill, opposing=True)
                if team_env:
                    rho *= team_env.get(str(slate.teams[i]), 1.0)
                c[i, j] = c[j, i] = float(np.clip(rho, *RHO_CLIP))

        c = _nearest_psd_correlation(c)
        return cls(
            matrix=c,
            cholesky=np.linalg.cholesky(c),
            table_source=str(table.get("source", "unknown")),
        )

    def latent_normals(self, n_sims: int, rng: np.random.Generator) -> np.ndarray:
        """Correlated standard normals of shape (n_sims, n_players)."""
        z = rng.standard_normal((n_sims, self.matrix.shape[0]))
        return z @ self.cholesky.T


def _etr_lift(slate: Slate, skill_idx: int, opposing: bool) -> float:
    """Player-specific stack tilt from conditional optimal rates.

    The ratio of P(optimal | QB optimal) to P(optimal) measures how much a
    pass-catcher's best weeks coincide with his QB's (or the opposing QB's).
    Only a damped log of that ratio is applied so noisy small rates cannot
    swing the correlation structure.
    """
    base = float(slate.optimal_rate[skill_idx])
    cond = float(
        slate.optimal_if_opp_qb[skill_idx] if opposing else slate.optimal_if_qb[skill_idx]
    )
    if base <= 0.0 or cond <= 0.0:
        return 1.0
    lift = cond / base
    return float(np.clip(1.0 + 0.10 * np.log(max(lift, 0.15)), *ETR_LIFT_CLIP))


def _nearest_psd_correlation(c: np.ndarray) -> np.ndarray:
    """Eigenvalue clipping + renormalization to a valid correlation matrix."""
    sym = (c + c.T) / 2.0
    w, v = np.linalg.eigh(sym)
    w = np.clip(w, 1e-4, None)
    out = (v * w) @ v.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 1.0)
    out += np.eye(len(out)) * 1e-9
    return out

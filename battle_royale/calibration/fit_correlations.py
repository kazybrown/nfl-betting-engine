"""Fit relationship correlations from historical same-game residuals.

Residuals are realized points minus the lagged anchor, standardized per
position. Pairs are formed within (season, week, game) for every same-team
and opposing-team position pair, correlations computed per relationship and
Fisher-shrunk toward the transferred task-2 priors (which came from an
independent DK-scored historical grid), so thin cells cannot swing the model.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_data import add_preweek_anchor, load_weekly, relevant_rows

PRIORS_SAME = {
    "QB-WR": 0.34,
    "QB-TE": 0.26,
    "QB-RB": -0.02,
    "WR-WR": -0.03,
    "WR-TE": 0.00,
    "RB-WR": -0.025,
    "RB-TE": 0.00,
    "RB-RB": -0.06,
    "TE-TE": -0.04,
}
PRIORS_OPP = {
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
}


_POS_ORDER = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}


def _pair_key(a: str, b: str) -> str:
    """Canonical relationship key in position order (matches priors tables)."""
    x, y = sorted((a, b), key=_POS_ORDER.__getitem__)
    return f"{x}-{y}"


def fisher_shrink(r: float, n: int, prior: float, n0: int = 300) -> float:
    r = float(np.clip(r, -0.995, 0.995))
    prior = float(np.clip(prior, -0.995, 0.995))
    w = max(n - 3, 0)
    z = (w * np.arctanh(r) + n0 * np.arctanh(prior)) / max(w + n0, 1)
    return float(np.tanh(z))


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["resid"] = d["points"] - d["anchor"]
    scale = d.groupby("position")["resid"].transform("std")
    d["z"] = d["resid"] / scale.clip(lower=1e-6)
    return d


def collect_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """(relation, z_a, z_b) for all same-game pairs. A game is identified by
    the unordered {team, opponent_team} pair within a week."""
    rows = []
    d = df.dropna(subset=["team", "opponent_team"])
    game_key = [
        f"{t}_{min(a, b)}_{max(a, b)}"
        for t, a, b in zip(d["time_id"], d["team"], d["opponent_team"])
    ]
    d = d.assign(_game=game_key)
    for _, g in d.groupby("_game"):
        recs = list(g.itertuples())
        for a, b in combinations(recs, 2):
            same = a.team == b.team
            key = _pair_key(a.position, b.position)
            rel = ("same" if same else "opp") + ":" + key
            rows.append((rel, a.z, b.z))
    return pd.DataFrame(rows, columns=["relation", "x", "y"])


def fit_relationships(pairs: pd.DataFrame, n0: int = 300) -> dict:
    out: dict = {"same_team": {}, "opp_team": {}}
    for scope, priors, bucket in (
        ("same", PRIORS_SAME, "same_team"),
        ("opp", PRIORS_OPP, "opp_team"),
    ):
        for key, prior in priors.items():
            d = pairs[pairs["relation"] == f"{scope}:{key}"]
            n = len(d)
            if n >= 30 and d["x"].std() > 0 and d["y"].std() > 0:
                raw = float(d[["x", "y"]].corr().iloc[0, 1])
            else:
                raw = prior
            out[bucket][key] = {
                "rho": fisher_shrink(raw, n, prior, n0=n0),
                "raw": raw,
                "n_pairs": int(n),
                "prior": prior,
            }
    return out


def main(stats_dir: str | Path, seasons: list[int], out_path: str | Path) -> dict:
    weekly = load_weekly(stats_dir, seasons)
    weekly = add_preweek_anchor(weekly)
    weekly = relevant_rows(weekly)
    pairs = collect_pairs(standardize(weekly))
    table = fit_relationships(pairs)
    table["source"] = (
        f"nflverse weekly stats {min(seasons)}-{max(seasons)} REG, Underdog scoring, "
        "standardized lagged-anchor residual pairs, Fisher shrinkage n0=300 "
        "toward task-2 DK-grid priors"
    )
    table["n_pairs_total"] = len(pairs)
    Path(out_path).write_text(json.dumps(table, indent=2))
    return table

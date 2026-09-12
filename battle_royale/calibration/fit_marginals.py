"""Fit position-level dispersion (CV^2 vs expectation) from historical data.

For each position, rows are bucketed by the preweek anchor; within a bucket
the conditional mean and variance of realized Underdog points give the
coefficient of variation used by the Gamma marginals.

Caveat recorded in the output: the anchor is a lagged EWM, which is noisier
than a good pre-kickoff projection, so conditional variance measured against
it slightly overstates the variance around a sharper projection. Buckets are
wide and the effect is modest; a global ``cv_scale`` on
:class:`battle_royale.marginals.MarginalModel` allows sensitivity checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .weekly_data import POSITIONS, add_preweek_anchor, load_weekly, relevant_rows

N_BUCKETS = {"QB": 6, "RB": 8, "WR": 8, "TE": 6}
MIN_BUCKET_ROWS = 150


def fit_dispersion(df: pd.DataFrame) -> dict:
    """Bucketed CV^2 table from anchored weekly rows."""
    out: dict = {"positions": {}}
    for pos in POSITIONS:
        d = df[df["position"] == pos]
        n_buckets = N_BUCKETS[pos]
        qs = np.quantile(d["anchor"], np.linspace(0, 1, n_buckets + 1))
        qs = np.unique(qs)
        buckets = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = d[(d["anchor"] >= lo) & (d["anchor"] <= hi)]
            if len(m) < MIN_BUCKET_ROWS:
                continue
            mean_pts = float(m["points"].mean())
            var_pts = float(m["points"].var(ddof=1))
            buckets.append(
                {
                    "anchor": float(m["anchor"].mean()),
                    "mean_points": mean_pts,
                    "cv2": var_pts / max(mean_pts, 1e-6) ** 2,
                    "n": len(m),
                }
            )
        out["positions"][pos] = {"buckets": buckets}
    return out


def main(stats_dir: str | Path, seasons: list[int], out_path: str | Path) -> dict:
    weekly = load_weekly(stats_dir, seasons)
    weekly = add_preweek_anchor(weekly)
    weekly = relevant_rows(weekly)
    table = fit_dispersion(weekly)
    table["source"] = (
        f"nflverse weekly stats {min(seasons)}-{max(seasons)} REG, Underdog scoring, "
        "lagged-EWM anchor buckets"
    )
    table["n_rows"] = len(weekly)
    Path(out_path).write_text(json.dumps(table, indent=2))
    return table

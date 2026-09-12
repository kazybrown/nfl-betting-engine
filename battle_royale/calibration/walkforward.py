"""Chronological walk-forward validation of the outcome model (leakage-safe).

For each test week, dispersion buckets and relationship correlations are
fitted on strictly earlier weeks only; forecasts for test rows use the
lagged anchor available before kickoff. Metrics:

- marginal layer: CRPS and pinball loss (vs a pooled position-ratio
  baseline), central-interval coverage;
- dependence layer: observed joint upper-tail co-exceedance per relationship
  vs the independence rate and the Gaussian-copula implied rate.

This validates the distributional machinery. It does not validate the
current-week slate projections (those are external inputs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from .fit_correlations import (
    PRIORS_OPP,
    PRIORS_SAME,
    collect_pairs,
    fit_relationships,
    standardize,
)
from .fit_marginals import fit_dispersion
from .weekly_data import add_preweek_anchor, load_weekly, relevant_rows

QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90)


def crps_from_samples(y: float, x: np.ndarray) -> float:
    x = np.sort(x)
    n = len(x)
    term1 = float(np.mean(np.abs(x - y)))
    coeff = 2 * np.arange(1, n + 1) - n - 1
    pair_mean = 2.0 * float(np.sum(coeff * x)) / (n * n)
    return term1 - 0.5 * pair_mean


def pinball(y: float, qhat: float, q: float) -> float:
    e = y - qhat
    return max(q * e, (q - 1.0) * e)


def _interp_cv2(table: dict, pos: str, anchor: float) -> float:
    buckets = table["positions"][pos]["buckets"]
    if not buckets:
        return 0.45
    xs = np.array([b["anchor"] for b in buckets])
    ys = np.array([b["cv2"] for b in buckets])
    order = np.argsort(xs)
    return float(np.interp(anchor, xs[order], ys[order]))


def _model_samples(
    table: dict, pos: str, anchor: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    cv2 = _interp_cv2(table, pos, anchor)
    var = max(anchor**2 * cv2, 1e-6)
    a = max(anchor**2 / var, 0.12)
    return gamma.rvs(a, scale=anchor / a, size=n, random_state=rng)


def walk_forward(
    stats_dir,
    seasons: list[int],
    min_train_weeks: int = 60,
    n_samples: int = 250,
    max_folds: int | None = None,
    tail_q: float = 0.90,
    seed: int = 20260912,
) -> dict:
    weekly = add_preweek_anchor(load_weekly(stats_dir, seasons))
    weekly = relevant_rows(weekly)
    tids = sorted(weekly["time_id"].unique())
    rng = np.random.default_rng(seed)
    zcut = float(norm.ppf(tail_q))

    eligible = [tid for i, tid in enumerate(tids) if i >= min_train_weeks]
    if max_folds is not None:
        # Validate the MOST RECENT weeks, not the earliest eligible ones.
        eligible = eligible[-max_folds:]
    eligible_set = set(eligible)

    folds = []
    for i, tid in enumerate(tids):
        if tid not in eligible_set:
            continue
        train = weekly[weekly["time_id"] < tid]
        test = weekly[weekly["time_id"] == tid]
        if len(test) == 0:
            continue
        assert int(train["time_id"].max()) < int(tid), "walk-forward leakage guard failed"

        disp = fit_dispersion(train)
        # Pooled baseline: empirical position ratio distribution.
        pools = {}
        for pos, g in train.groupby("position"):
            ratio = (g["points"] / g["anchor"].clip(lower=0.5)).to_numpy()
            ratio = ratio[np.isfinite(ratio)]
            pools[pos] = ratio / max(ratio.mean(), 1e-9)

        m_crps, b_crps, m_pin, b_pin = [], [], [], []
        m_c50, m_c80, b_c50, b_c80 = [], [], [], []
        for row in test.itertuples():
            y = float(row.points)
            ms = _model_samples(disp, row.position, float(row.anchor), n_samples, rng)
            pool = pools.get(row.position, np.array([1.0]))
            bs = float(row.anchor) * rng.choice(pool, size=n_samples, replace=True)
            for samples, crps_acc, pin_acc, c50, c80 in (
                (ms, m_crps, m_pin, m_c50, m_c80),
                (bs, b_crps, b_pin, b_c50, b_c80),
            ):
                qv = {q: float(np.quantile(samples, q)) for q in QUANTS}
                crps_acc.append(crps_from_samples(y, samples))
                pin_acc.append(float(np.mean([pinball(y, qv[q], q) for q in QUANTS])))
                c50.append(float(qv[0.25] <= y <= qv[0.75]))
                c80.append(float(qv[0.10] <= y <= qv[0.90]))

        # Dependence: train-fitted correlations vs test joint tails.
        corr = fit_relationships(collect_pairs(standardize(train)))
        ztrain = standardize(train)
        thresholds = ztrain.groupby("position")["z"].quantile(tail_q).to_dict()
        # Use train scale for test z to stay strictly out-of-sample.
        scale = ztrain.groupby("position")["resid"].std().to_dict()
        ztest = test.copy()
        ztest["resid"] = ztest["points"] - ztest["anchor"]
        ztest["z"] = [
            r / max(scale.get(p, 1.0), 1e-6) for r, p in zip(ztest["resid"], ztest["position"])
        ]
        tpairs = collect_pairs(ztest)
        joint = []
        for scope, priors, bucket in (
            ("same", PRIORS_SAME, "same_team"),
            ("opp", PRIORS_OPP, "opp_team"),
        ):
            for key in priors:
                d = tpairs[tpairs["relation"] == f"{scope}:{key}"]
                if len(d) == 0:
                    continue
                pos_a, pos_b = key.split("-")
                ta = thresholds.get(pos_a)
                tb = thresholds.get(pos_b)
                if ta is None or tb is None:
                    continue
                obs = float(np.mean((d["x"] > ta) & (d["y"] > tb)))
                rho = corr[bucket][key]["rho"]
                z1 = rng.standard_normal(60_000)
                z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * rng.standard_normal(60_000)
                cop = float(np.mean((z1 > zcut) & (z2 > zcut)))
                joint.append(
                    {
                        "relation": f"{scope}:{key}",
                        "n_pairs": len(d),
                        "observed": obs,
                        "independence": (1 - tail_q) ** 2,
                        "copula": cop,
                    }
                )

        folds.append(
            {
                "time_id": int(tid),
                "n_test": len(test),
                "model_crps": float(np.mean(m_crps)),
                "baseline_crps": float(np.mean(b_crps)),
                "model_pinball": float(np.mean(m_pin)),
                "baseline_pinball": float(np.mean(b_pin)),
                "model_cover_50": float(np.mean(m_c50)),
                "model_cover_80": float(np.mean(m_c80)),
                "baseline_cover_50": float(np.mean(b_c50)),
                "baseline_cover_80": float(np.mean(b_c80)),
                "joint_tail": joint,
            }
        )
        if max_folds is not None and len(folds) >= max_folds:
            break

    if not folds:
        raise ValueError("no folds; reduce min_train_weeks or add seasons")

    flat = pd.DataFrame([{k: v for k, v in f.items() if k != "joint_tail"} for f in folds])
    joint_all = pd.DataFrame([j | {"time_id": f["time_id"]} for f in folds for j in f["joint_tail"]])
    jt = {}
    if len(joint_all):
        grouped = joint_all.groupby("relation").apply(
            lambda g: pd.Series(
                {
                    "n_pairs": g["n_pairs"].sum(),
                    "observed": np.average(g["observed"], weights=g["n_pairs"]),
                    "independence": g["independence"].iloc[0],
                    "copula": np.average(g["copula"], weights=g["n_pairs"]),
                }
            ),
            include_groups=False,
        )
        jt = grouped.to_dict(orient="index")
        ind_err = float(np.mean(np.abs(grouped["observed"] - grouped["independence"])))
        cop_err = float(np.mean(np.abs(grouped["observed"] - grouped["copula"])))
    else:
        ind_err = cop_err = float("nan")

    return {
        "n_folds": len(folds),
        "marginal": {
            "model_crps": float(flat["model_crps"].mean()),
            "baseline_crps": float(flat["baseline_crps"].mean()),
            "crps_improvement_pct": float(
                100 * (flat["baseline_crps"].mean() - flat["model_crps"].mean())
                / max(flat["baseline_crps"].mean(), 1e-9)
            ),
            "model_pinball": float(flat["model_pinball"].mean()),
            "baseline_pinball": float(flat["baseline_pinball"].mean()),
            "model_cover_50": float(flat["model_cover_50"].mean()),
            "model_cover_80": float(flat["model_cover_80"].mean()),
            "baseline_cover_50": float(flat["baseline_cover_50"].mean()),
            "baseline_cover_80": float(flat["baseline_cover_80"].mean()),
        },
        "dependence": {
            "relations": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in jt.items()},
            "mean_abs_error_independence": ind_err,
            "mean_abs_error_copula": cop_err,
        },
        "folds": folds,
    }

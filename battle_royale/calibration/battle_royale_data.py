"""Calibration against real Underdog Battle Royale pick-by-pick archives.

Input: compacted weekly parquet files with columns
``draft_id, draft_entry_id, player_name, player_id, position_name,
projection_adp, source, pick_order, overall_pick_number, team_pick_number,
player_points, roster_points``.

Produces the empirical facts the simulator must reproduce:

- contest size (entries) and winning / percentile scores;
- ADP execution noise (how far real picks land from ADP);
- roster construction (FLEX position shares);
- exact-roster duplication (unique share, copy histogram);
- opponent-policy rank diagnostics: where the actually-picked player sits in
  our policy's (noise-free) preference ordering vs a pure-ADP baseline.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ROSTER_MAX, ROSTER_MIN, SEATS, TOTAL_PICKS
from ..opponents import OpponentPolicy
from ..slate import Player, Slate

POS_OK = ("QB", "RB", "WR", "TE")


def load_week(path: str | Path) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d = d[d["position_name"].isin(POS_OK)].copy()
    return d


def summarize_week(d: pd.DataFrame) -> dict:
    entries = d["draft_entry_id"].nunique()
    rooms = d["draft_id"].nunique()
    entry_scores = d.groupby("draft_entry_id")["roster_points"].first()

    # ADP execution noise among realistically-draftable players.
    core = d[(d["projection_adp"] > 0) & (d["projection_adp"] <= TOTAL_PICKS)]
    delta = (core["overall_pick_number"] - core["projection_adp"]).astype(float)
    mad_sigma = float(1.4826 * np.median(np.abs(delta - np.median(delta))))

    # Construction.
    pos_counts = (
        d.pivot_table(
            index="draft_entry_id", columns="position_name", values="player_name",
            aggfunc="count", fill_value=0,
        )
        .reindex(columns=list(POS_OK), fill_value=0)
    )
    flex = {
        "RB": float((pos_counts["RB"] == 2).mean()),
        "WR": float((pos_counts["WR"] == 3).mean()),
        "TE": float((pos_counts["TE"] == 2).mean()),
    }

    # Exact-roster duplication.
    rosters = d.groupby("draft_entry_id")["player_id"].apply(lambda s: tuple(sorted(s)))
    counts = Counter(rosters.values)
    copies = Counter(counts.values())
    unique_entries = sum(v for v in counts.values() if v == 1)

    # Room-drafted rate curve vs ADP (for latent-market validation).
    drafted_rate = (
        d.groupby("player_id")
        .agg(adp=("projection_adp", "first"), n=("draft_id", "nunique"))
        .assign(room_rate=lambda x: x["n"] / rooms)
    )

    return {
        "entries": int(entries),
        "rooms": int(rooms),
        "winning_score": float(entry_scores.max()),
        "score_quantiles": {
            str(q): float(entry_scores.quantile(q))
            for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        },
        "adp_noise_mad_sigma": mad_sigma,
        "adp_noise_mean_delta": float(delta.mean()),
        "flex_share": flex,
        "roster_min_ok": bool(
            (pos_counts["QB"] == 1).all()
            and (pos_counts["RB"] >= 1).all()
            and (pos_counts["WR"] >= 2).all()
            and (pos_counts["TE"] >= 1).all()
        ),
        "duplication": {
            "distinct_rosters": len(counts),
            "unique_entry_share": unique_entries / max(entries, 1),
            "duplicated_entry_share": 1 - unique_entries / max(entries, 1),
            "max_exact_copies": max(counts.values()) if counts else 0,
            "copy_count_histogram": {str(k): int(v) for k, v in sorted(copies.items())},
        },
        "room_rate_curve": {
            "adp_le_12": float(drafted_rate[drafted_rate["adp"] <= 12]["room_rate"].mean()),
            "adp_12_24": float(
                drafted_rate[(drafted_rate["adp"] > 12) & (drafted_rate["adp"] <= 24)][
                    "room_rate"
                ].mean()
            ),
            "adp_24_36": float(
                drafted_rate[(drafted_rate["adp"] > 24) & (drafted_rate["adp"] <= 36)][
                    "room_rate"
                ].mean()
            ),
        },
    }


def _slate_from_week(d: pd.DataFrame) -> tuple[Slate, dict]:
    """Synthetic slate from a real BR week (no teams -> stacking term inert)."""
    rooms = d["draft_id"].nunique()
    agg = (
        d.groupby("player_id")
        .agg(
            name=("player_name", "first"),
            pos=("position_name", "first"),
            adp=("projection_adp", "first"),
            n=("draft_id", "nunique"),
        )
        .reset_index()
    )
    agg["room_rate"] = agg["n"] / rooms
    agg = agg.sort_values("adp").reset_index(drop=True)
    players = []
    for i, r in agg.iterrows():
        adp_raw = r["adp"]
        adp = float(adp_raw) if pd.notna(adp_raw) and float(adp_raw) > 0 else 60.0
        players.append(
            Player(
                name=f"{r['name']} #{i}",  # disambiguate duplicate display names
                position=str(r["pos"]),
                team=f"T{i}",
                opponent="",
                rank=i + 1,
                adp=adp,
                proj=0.0,
                ceiling=0.0,
                optimal_rate=0.0,
                optimal_if_qb=0.0,
                optimal_if_opp_qb=0.0,
                top5_at_pos=0.0,
                room_drafted_rate=float(r["room_rate"]),
                player_id=str(r["player_id"]),
            )
        )
    slate = Slate(players)
    id_to_idx = {p.player_id: i for i, p in enumerate(slate.players)}
    return slate, id_to_idx


def policy_rank_diagnostics(
    d: pd.DataFrame, n_rooms: int = 800, seed: int = 20260912
) -> dict:
    """Replay real rooms: rank of the actual pick under the policy's noise-free
    preference ordering, vs a pure-ADP baseline ordering.
    """
    rng = np.random.default_rng(seed)
    slate, id_to_idx = _slate_from_week(d)
    policy = OpponentPolicy(slate)
    neutral = policy.seat_params(np.random.default_rng(0))
    neutral.market, neutral.need, neutral.stack, neutral.scroll = 1.0, 1.0, 0.5, 0.0
    neutral.rb_flex = policy.flex_target["RB"]

    draft_ids = d["draft_id"].unique()
    sample = rng.choice(draft_ids, size=min(n_rooms, len(draft_ids)), replace=False)
    dd = d[d["draft_id"].isin(sample)].sort_values(["draft_id", "overall_pick_number"])

    pol_ranks, adp_ranks = [], []
    pol_top1 = pol_top3 = adp_top1 = adp_top3 = n_evals = 0
    for _, room in dd.groupby("draft_id"):
        if len(room) != TOTAL_PICKS:
            continue
        avail = np.ones(slate.n, dtype=bool)
        counts = np.zeros((SEATS, 4), dtype=np.int8)
        rows = list(room.itertuples())
        for pick_no, row in enumerate(rows, start=1):
            pid = str(row.player_id)
            if pid not in id_to_idx:
                break
            seat = (pick_no - 1) % SEATS if ((pick_no - 1) // SEATS) % 2 == 0 else (
                SEATS - 1 - (pick_no - 1) % SEATS
            )
            left_before = 6 - int(counts[seat].sum())
            left_after = left_before - 1
            pos_ok = np.zeros(4, dtype=bool)
            for pos in range(4):
                if counts[seat, pos] >= ROSTER_MAX[pos]:
                    continue
                after = counts[seat].copy()
                after[pos] += 1
                pos_ok[pos] = int(np.maximum(ROSTER_MIN - after, 0).sum()) <= left_after
            elig = np.where(avail & pos_ok[slate.pos])[0]
            actual = id_to_idx[pid]
            if actual not in set(int(x) for x in elig):
                avail[actual] = False
                counts[seat, slate.pos[actual]] += 1
                continue
            # Policy ordering (noise-free): market prior + need bonus.
            u = -0.64 * slate.adp[elig] - 0.018 * slate.adp[elig]
            u = u + 0.0  # scroll neutral
            for k, j in enumerate(elig):
                u[k] += policy._need_bonus(counts[seat], int(slate.pos[j]), left_before, neutral)
            order_pol = elig[np.argsort(-u)]
            order_adp = elig[np.argsort(slate.adp[elig])]
            rank_pol = int(np.where(order_pol == actual)[0][0]) + 1
            rank_adp = int(np.where(order_adp == actual)[0][0]) + 1
            pol_ranks.append(rank_pol)
            adp_ranks.append(rank_adp)
            pol_top1 += rank_pol == 1
            pol_top3 += rank_pol <= 3
            adp_top1 += rank_adp == 1
            adp_top3 += rank_adp <= 3
            n_evals += 1
            avail[actual] = False
            counts[seat, slate.pos[actual]] += 1

    return {
        "n_picks_evaluated": n_evals,
        "policy": {
            "top1": pol_top1 / max(n_evals, 1),
            "top3": pol_top3 / max(n_evals, 1),
            "mean_rank": float(np.mean(pol_ranks)) if pol_ranks else None,
            "median_rank": float(np.median(pol_ranks)) if pol_ranks else None,
        },
        "adp_baseline": {
            "top1": adp_top1 / max(n_evals, 1),
            "top3": adp_top3 / max(n_evals, 1),
            "mean_rank": float(np.mean(adp_ranks)) if adp_ranks else None,
            "median_rank": float(np.median(adp_ranks)) if adp_ranks else None,
        },
    }

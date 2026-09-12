"""Weekly Battle Royale slate report: draft plans, survival cliffs, stacks.

Usage:
    python scripts/br_week_report.py --csv SLATE.csv --out reports/battle_royale \
        [--contest-size 70000] [--fast] [--guided-drafts]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import json
from pathlib import Path

import numpy as np

from battle_royale import (
    BattleRoyaleEngine,
    DraftState,
    FieldAnalytics,
    OptimizerConfig,
    PickOptimizer,
    TournamentModel,
)
from battle_royale.equity import load_payout_table
from battle_royale.field import positional_construction
from battle_royale.formats import get_format


def survival_table(engine, n_rooms: int, checkpoints: list[int], top_n: int, seed: int) -> dict:
    """P(player still available when pick k is on the clock), fresh rooms."""
    rng = np.random.default_rng(seed)
    slate = engine.slate
    top = np.argsort(slate.rank)[:top_n]
    counts = {int(j): {k: 0 for k in checkpoints} for j in top}
    for _ in range(n_rooms):
        st = DraftState(slate)
        seats = engine.policy.room_params(rng)
        times = engine.policy.latent_market(rng)
        pick_of = np.full(slate.n, slate.fmt.total_picks + 1, dtype=int)
        while not st.complete:
            pick = st.next_pick
            chosen = engine.policy.choose(st, seats, times, rng)
            pick_of[chosen] = pick
        for j in top:
            for k in checkpoints:
                if pick_of[j] >= k:
                    counts[int(j)][k] += 1
    return {
        slate.players[j].name: {str(k): counts[int(j)][k] / n_rooms for k in checkpoints}
        for j in top
    }


def field_score_distribution(engine, field, n_sims: int, seed: int) -> dict:
    """Per-outcome-sim quantiles of field entry scores (averaged over sims).

    Comparable to the within-week cross-entry score distributions observed in
    real Battle Royale archives (2023-2024: median 72-99, p99 105-156, winner
    125-185 at 28-68k entries). The sample max understates the true contest
    winner when the sampled field is smaller than the contest.
    """
    rng = np.random.default_rng(seed)
    qs = {"0.5": [], "0.9": [], "0.99": [], "0.999": []}
    mx = []
    done = 0
    while done < n_sims:
        m = min(200, n_sims - done)
        x = engine.sample_scores(m, rng)
        fs = x[:, field].sum(axis=2)
        for q in qs:
            qs[q].append(np.quantile(fs, float(q), axis=1))
        mx.append(fs.max(axis=1))
        done += m
    out = {f"p{q}": float(np.mean(np.concatenate(v))) for q, v in qs.items()}
    mx = np.concatenate(mx)
    out["sample_max_mean"] = float(mx.mean())
    out["sample_max_p10"] = float(np.quantile(mx, 0.10))
    out["sample_max_p90"] = float(np.quantile(mx, 0.90))
    out["sample_entries"] = len(field)
    return out


def stack_ownership(analytics: FieldAnalytics, top_qbs: int = 8) -> list[dict]:
    slate = analytics.slate
    qbs = [i for i in np.argsort(slate.rank) if slate.players[int(i)].position == "QB"]
    rows = []
    for qb in qbs[:top_qbs]:
        qb = int(qb)
        team = slate.players[qb].team
        mates = [
            int(i)
            for i in np.argsort(slate.rank)
            if slate.players[int(i)].team == team
            and slate.players[int(i)].position in ("WR", "TE")
        ][:3]
        for m in mates:
            rows.append(analytics.combo_metrics([qb, m]))
    return rows


def guided_draft(optimizer, seat: int, seed: int) -> dict:
    """One model-guided draft from a seat: our picks via the optimizer,
    opponents via the field policy."""
    engine = optimizer.engine
    rng = np.random.default_rng(seed)
    st = DraftState(engine.slate)
    seats = engine.policy.room_params(rng)
    times = engine.policy.latent_market(rng)
    decisions = []
    while not st.complete:
        if st.seat_on_clock() == seat:
            rec = optimizer.recommend(st)
            best = rec["options"][0]
            for name in best["players"]:
                st.apply_pick(engine.slate.name_to_idx[name])
            decisions.append(
                {
                    "pick": rec["pick"],
                    "mode": rec["mode"],
                    "selection": best["players"],
                    "expected_payout": best["expected_payout"],
                    "runner_up": rec["options"][1]["players"] if len(rec["options"]) > 1 else None,
                }
            )
        else:
            engine.policy.choose(st, seats, times, rng)
    roster = engine.slate.names(st.rosters[seat])
    return {"seat": seat, "picks": engine.slate.fmt.picks_of_seat(seat),
            "roster": roster, "decisions": decisions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/battle_royale")
    ap.add_argument("--format", default="battle_royale", dest="fmt",
                    help="contest format key from battle_royale/data/formats.json")
    ap.add_argument("--contest-size", type=int, default=None,
                    help="entries in the contest (default: the format's field size)")
    ap.add_argument("--payouts", default=None,
                    help="prize-table JSON path (default: packaged real table if present)")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--guided-drafts", action="store_true")
    ap.add_argument("--survival-rooms", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fmt = get_format(args.fmt)
    contest_size = args.contest_size or fmt.default_contest_size
    engine = BattleRoyaleEngine.from_csv(args.csv, seed=args.seed, fmt=fmt)
    config = OptimizerConfig.fast() if args.fast else OptimizerConfig()
    config.seed = args.seed
    curve, pay_meta = load_payout_table(args.payouts, filename=fmt.payouts_file)
    optimizer = PickOptimizer(
        engine, TournamentModel(contest_size=contest_size, curve=curve), config
    )

    report: dict = {
        "format": fmt.key,
        "slate_players": engine.slate.n,
        "contest_size": contest_size,
        "payouts": pay_meta.get("contest", pay_meta["source"]),
        "marginal_table": engine.marginals.table_source,
        "correlation_table": engine.correlation.table_source,
    }

    analytics = optimizer.analytics
    report["field_calibration"] = analytics.calibration_vs_slate()
    report["field_duplication"] = analytics.duplication_summary()
    report["field_construction"] = positional_construction(engine.slate, optimizer.field)

    report["field_score_distribution"] = field_score_distribution(
        engine, optimizer.field, n_sims=1200, seed=args.seed + 3
    )

    checkpoints = [7, 8, 13, 18, 19, 24, 25, 30, 31]
    report["survival"] = survival_table(
        engine, args.survival_rooms, checkpoints, top_n=40, seed=args.seed + 7
    )
    report["stack_ownership"] = stack_ownership(analytics)

    print("computing seat-1 opening recommendation...")
    report["first_pick_board"] = optimizer.recommend(DraftState(engine.slate))["options"][:10]

    if args.guided_drafts:
        report["guided_drafts"] = []
        for seat in range(engine.slate.fmt.seats):
            print(f"guided draft, seat {seat}...")
            report["guided_drafts"].append(guided_draft(optimizer, seat, args.seed + 100 + seat))

    (out / "week_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'week_report.json'}")


if __name__ == "__main__":
    main()

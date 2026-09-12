"""Pre-draft plan generator: per-seat ranked queues for every own pick slot.

Live Battle Royale drafts run on a fast clock, and the drafter can see their
own board but has no time to transcribe it. So all optimization happens
BEFORE the draft: for each seat and each of that seat's six pick slots, the
full optimizer is run on sampled boards (opponent picks unknown, marginalized
by simulation, conditioned on our modal earlier picks) and its choices are
aggregated into a ranked target queue — "take the first name still on your
board". Pairwise stack synergies and roster legality ship with the plan so a
client can re-rank queues as the user's actual picks diverge from the modal
path, with no server round trip.

Usage:
    python scripts/br_plan.py --csv SLATE.csv --out plan.json \
        [--seats 1 2 3 4 5 6] [--boards 6] [--avail-boards 300]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import json
import time
from collections import Counter, defaultdict

import numpy as np

from battle_royale import (
    BattleRoyaleEngine,
    DraftState,
    OptimizerConfig,
    PickOptimizer,
    TournamentModel,
)
from battle_royale.cache import DEFAULT_CACHE_DIR
from battle_royale.constants import picks_of_seat


def sample_board(engine, our_picks: dict[int, int], upto_pick: int, seed: int) -> DraftState:
    """Simulate a room up to (not including) ``upto_pick``: our known picks
    pinned at their slots, opponents never taking them."""
    s = engine.slate
    rng = np.random.default_rng(seed)
    st = DraftState(s)
    seats = engine.policy.room_params(rng)
    times = engine.policy.latent_market(rng)
    if our_picks:
        times[list(our_picks.values())] = 999.0
    while st.next_pick < upto_pick:
        p = st.next_pick
        if p in our_picks:
            st.apply_pick(our_picks[p])
        else:
            engine.policy.choose(st, seats, times, rng)
    return st


def build_seat_plan(opt: PickOptimizer, seat0: int, n_boards: int, avail_boards: int,
                    base_seed: int) -> dict:
    engine = opt.engine
    s = opt.slate
    own_slots = picks_of_seat(seat0)
    our_picks: dict[int, int] = {}
    slots_out = []

    # Availability per own slot from cheap board sims (no optimizer).
    avail = {}
    for slot in own_slots:
        cnt = np.zeros(s.n)
        for k in range(avail_boards):
            cnt += sample_board(engine, our_picks, slot, base_seed + 50_000 + slot * 1000 + k).avail
        avail[slot] = cnt / avail_boards
        # our_picks not yet decided for this pass; modal path fills in below,
        # so early-slot availability ignores our own later picks (harmless).

    handled: set[int] = set()
    for slot in own_slots:
        if slot in handled:
            continue
        votes: Counter = Counter()
        eq_sum: dict[int, float] = defaultdict(float)
        eq_n: dict[int, int] = defaultdict(int)
        pair_partner: dict[int, Counter] = defaultdict(Counter)
        mode = "single"
        for k in range(n_boards):
            st = sample_board(engine, our_picks, slot, base_seed + slot * 1000 + k)
            rec = opt.recommend(st)
            mode = rec["mode"]
            for rank_i, o in enumerate(rec["options"][:6]):
                ids = [s.name_to_idx[nm] for nm in o["players"]]
                w = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18][rank_i]
                for i in ids:
                    votes[i] += w
                    eq_sum[i] += o["expected_payout"]
                    eq_n[i] += 1
                if len(ids) == 2:
                    pair_partner[ids[0]][ids[1]] += 1
                    pair_partner[ids[1]][ids[0]] += 1
        queue = sorted(votes, key=lambda i: (-votes[i], -eq_sum[i] / max(eq_n[i], 1)))[:16]
        slot_avail = avail[slot]
        entry = {
            "pick": slot,
            "mode": mode,
            "queue": [
                {
                    "id": int(i),
                    "score": round(votes[i] / n_boards, 3),
                    "eq": round(eq_sum[i] / max(eq_n[i], 1), 2),
                    "avail": round(float(slot_avail[i]), 3),
                    "pairs_with": [
                        int(j) for j, _ in pair_partner[i].most_common(2)
                    ] if pair_partner.get(i) else [],
                }
                for i in queue
            ],
        }
        # Modal path: commit the top choice (both names for a pair) so later
        # slots condition on a realistic own roster.
        if mode == "pair":
            top_pair = None
            best = -1.0
            for a in queue[:6]:
                for b, _ in pair_partner[a].most_common(1):
                    v = votes[a] + votes[b]
                    if v > best:
                        best, top_pair = v, (a, b)
            nxt = slot + 1
            if top_pair:
                our_picks[slot], our_picks[nxt] = int(top_pair[0]), int(top_pair[1])
            entry["modal"] = [int(x) for x in (top_pair or queue[:2])]
            entry["covers_picks"] = [slot, nxt]
            handled.add(nxt)
        else:
            our_picks[slot] = int(queue[0])
            entry["modal"] = [int(queue[0])]
            entry["covers_picks"] = [slot]
        slots_out.append(entry)

    return {"seat": seat0 + 1, "own_picks": own_slots, "slots": slots_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/battle_royale/draft_plan.json")
    ap.add_argument("--seats", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--boards", type=int, default=6)
    ap.add_argument("--avail-boards", type=int, default=250)
    ap.add_argument("--contest-size", type=int, default=70_000)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()

    engine = BattleRoyaleEngine.from_csv(args.csv, seed=args.seed)
    # Offline generation: fast rollout counts, but full-size outcome sims and
    # fields — the jackpot region is too noisy under the live-draft preset,
    # and the heavy artifacts are disk-cached anyway.
    cfg = OptimizerConfig.fast()
    cfg.n_outcome_sims = 2500
    cfg.eval_field_entries = 3600
    cfg.dup_field_entries = 24_000
    cfg.n_rollouts = 60
    cfg.seed = args.seed
    cfg.cache_dir = str(DEFAULT_CACHE_DIR)
    opt = PickOptimizer(engine, TournamentModel(contest_size=args.contest_size), cfg)
    opt.warm()
    s = engine.slate

    # Sparse same-game synergy pairs (score deltas in queue-score units) so a
    # client can re-rank when the user's roster diverges from the modal path.
    cov = opt.cov
    synergy = []
    top = set(int(i) for i in np.argsort(s.rank)[:110])
    for i in range(s.n):
        for j in range(i + 1, s.n):
            if i not in top or j not in top or not s.same_game(i, j):
                continue
            c = float(cov[i, j])
            if abs(c) >= 1.5:
                synergy.append([i, j, round(0.35 * c / 10.0, 3)])

    plan = {
        "generated_for": "Underdog Battle Royale",
        "contest_size": args.contest_size,
        "players": [
            {
                "id": i,
                "name": p.name,
                "pos": p.position,
                "team": p.team,
                "opp": p.opponent,
                "adp": p.adp,
                "own": round(p.room_drafted_rate, 3),
                "value": round(float(opt.values[i]), 2),
            }
            for i, p in enumerate(s.players)
        ],
        "synergy": synergy,
        "seats": {},
    }
    for lobby_seat in args.seats:
        t0 = time.time()
        plan["seats"][str(lobby_seat)] = build_seat_plan(
            opt, lobby_seat - 1, args.boards, args.avail_boards, args.seed
        )
        print(f"seat {lobby_seat} planned in {time.time() - t0:.0f}s", flush=True)

    _Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _Path(args.out).write_text(json.dumps(plan))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

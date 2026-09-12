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
from battle_royale.equity import load_payout_table
from battle_royale.formats import get_format


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


def _plan_path(opt: PickOptimizer, own_slots: list[int], our_picks_init: dict[int, int],
               avail: dict, n_boards: int, base_seed: int) -> list[dict]:
    """Plan every own slot not already committed in ``our_picks_init``.

    Committed slots (a branch prefix) condition the boards but produce no
    entries, so a branch continuation contains exactly the remaining groups.
    """
    engine = opt.engine
    s = opt.slate
    our_picks = dict(our_picks_init)
    slots_out: list[dict] = []
    handled: set[int] = set(our_picks)
    for slot in own_slots:
        if slot in handled:
            continue
        votes: Counter = Counter()
        eq_sum: dict[int, float] = defaultdict(float)
        eq_n: dict[int, int] = defaultdict(int)
        pair_partner: dict[int, Counter] = defaultdict(Counter)
        pair_votes: Counter = Counter()
        pair_eq: dict[tuple, float] = defaultdict(float)
        pair_n: dict[tuple, int] = defaultdict(int)
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
                    pk = tuple(sorted(ids))
                    pair_votes[pk] += w
                    pair_eq[pk] += o["expected_payout"]
                    pair_n[pk] += 1
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
            # Back-to-back turn picks are ONE decision: rank the actual joint
            # options the optimizer scored, not the two names independently.
            top_pairs = sorted(pair_votes, key=lambda t: -pair_votes[t])[:4]
            entry["pair_options"] = [
                {
                    "ids": [int(a) for a in pk],
                    "score": round(pair_votes[pk] / n_boards, 3),
                    "eq": round(pair_eq[pk] / max(pair_n[pk], 1), 2),
                }
                for pk in top_pairs
            ]
            top_pair = tuple(top_pairs[0]) if top_pairs else tuple(queue[:2])
            nxt = slot + 1
            our_picks[slot], our_picks[nxt] = int(top_pair[0]), int(top_pair[1])
            entry["modal"] = [int(x) for x in top_pair]
            entry["covers_picks"] = [slot, nxt]
            handled.add(nxt)
        else:
            our_picks[slot] = int(queue[0])
            entry["modal"] = [int(queue[0])]
            entry["covers_picks"] = [slot]
        slots_out.append(entry)

    return slots_out


def build_seat_plan(opt: PickOptimizer, seat0: int, n_boards: int, avail_boards: int,
                    base_seed: int, alt_k: int = 2, alt_boards: int = 6) -> dict:
    engine = opt.engine
    s = opt.slate
    own_slots = s.fmt.picks_of_seat(seat0)

    # Availability per own slot from cheap board sims (no optimizer); shared
    # by every branch — early-slot availability ignores our own later picks
    # (harmless), so branch prefixes don't change it materially.
    avail = {}
    for slot in own_slots:
        cnt = np.zeros(s.n)
        for k in range(avail_boards):
            cnt += sample_board(engine, {}, slot, base_seed + 50_000 + slot * 1000 + k).avail
        avail[slot] = cnt / avail_boards

    slots_out = _plan_path(opt, own_slots, {}, avail, n_boards, base_seed)
    plan = {"seat": seat0 + 1, "own_picks": own_slots, "slots": slots_out}

    # Branch-conditional continuations: the equity votes of later slots are
    # conditioned on the FIRST group's actual selection, not just the modal
    # one, so the client stays on a fresh path when the user opens
    # differently. Keys are the sorted first-group pick ids joined by "-".
    first = slots_out[0] if slots_out else None
    if first and alt_k > 0:
        covers = first["covers_picks"]
        if first.get("pair_options"):
            cands = [tuple(po["ids"]) for po in first["pair_options"]]
        else:
            cands = [(q["id"],) for q in first["queue"]]
        modal = tuple(sorted(first["modal"]))
        branches = [modal] + [c for c in cands if tuple(sorted(c)) != modal][:alt_k]
        alt_paths = {}
        for pi, key in enumerate(branches):
            if pi == 0:
                cont = slots_out[1:]  # the modal continuation is the default path
            else:
                picks = {covers[j]: int(key[j]) for j in range(len(covers))}
                cont = _plan_path(opt, own_slots, picks, avail, alt_boards,
                                  base_seed + 7919 * (pi + 1))
            alt_paths["-".join(str(x) for x in sorted(key))] = cont
        plan["alt_paths"] = alt_paths

    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="reports/battle_royale/draft_plan.json")
    ap.add_argument("--format", default="battle_royale", dest="fmt",
                    help="contest format key from battle_royale/data/formats.json")
    ap.add_argument("--seats", nargs="+", type=int, default=None,
                    help="seats to plan (default: all)")
    ap.add_argument("--boards", type=int, default=10)
    ap.add_argument("--alt-paths", type=int, default=2,
                    help="branch continuations for top non-modal opening picks")
    ap.add_argument("--alt-boards", type=int, default=6)
    ap.add_argument("--avail-boards", type=int, default=250)
    ap.add_argument("--contest-size", type=int, default=None,
                    help="entries in the contest (default: the format's field size)")
    ap.add_argument("--payouts", default=None,
                    help="prize-table JSON path (default: packaged real table if present)")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--season", type=int, default=None, help="fetch Vegas lines + player status")
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    from battle_royale.external import (
        load_game_lines,
        load_player_status,
        sit_probabilities,
        slate_status,
    )
    from battle_royale.slate import Slate

    fmt = get_format(args.fmt)
    seats = args.seats or list(range(1, fmt.seats + 1))
    contest_size = args.contest_size or fmt.default_contest_size
    lines, statuses = {}, {}
    slate_obj = Slate.from_csv(args.csv, fmt=fmt)
    if args.season and args.week:
        lines = load_game_lines(args.season, args.week)
        statuses = slate_status(slate_obj, load_player_status(args.season, args.week))
        print(f"lines for {sum(1 for t in set(slate_obj.teams) if t in lines)} slate teams; "
              f"{len(statuses)} players with status flags")
        # Out/rostered-off players should not be targeted by the simulated
        # field: their room-drafted rate predates the news.
        for i, code in statuses.items():
            if code in ("O", "IR"):
                slate_obj.room_drafted_rate[i] = min(slate_obj.room_drafted_rate[i], 0.01)

    sit = sit_probabilities(slate_obj, statuses) if statuses else None
    engine = BattleRoyaleEngine(slate_obj, seed=args.seed, game_lines=lines, sit_prob=sit)
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
    curve, pay_meta = load_payout_table(args.payouts, filename=fmt.payouts_file)
    print(f"format: {fmt.name} ({fmt.seats} seats x {fmt.rounds} rounds); "
          f"payout curve: {pay_meta.get('contest', pay_meta['source'])}")
    opt = PickOptimizer(
        engine, TournamentModel(contest_size=contest_size, curve=curve), cfg
    )
    opt.warm()
    s = engine.slate

    # Sparse same-game synergy pairs (score deltas in queue-score units) so a
    # client can re-rank when the user's roster diverges from the modal path.
    cov = opt.cov
    synergy = []
    top = set(int(i) for i in np.argsort(s.rank)[:130])
    for i in range(s.n):
        for j in range(i + 1, s.n):
            if i not in top or j not in top or not s.same_game(i, j):
                continue
            c = float(cov[i, j])
            # Keep weaker (especially negative) pairs too: same-team slot
            # competition and RB/RB overlap matter to the client re-rank.
            if abs(c) >= 0.8:
                synergy.append([i, j, round(0.35 * c / 10.0, 3)])

    plan = {
        "generated_for": f"Underdog {fmt.name}",
        "format": {
            "key": fmt.key,
            "name": fmt.name,
            "seats": fmt.seats,
            "rounds": fmt.rounds,
            "pos_min": dict(zip(("QB", "RB", "WR", "TE"), fmt.roster_min)),
            "pos_max": dict(zip(("QB", "RB", "WR", "TE"), fmt.roster_max)),
        },
        "contest_size": contest_size,
        "payouts": pay_meta.get("contest", pay_meta["source"]),
        "players": [
            {
                "id": i,
                "name": p.name,
                "pos": p.position,
                "team": p.team,
                "opp": p.opponent,
                "rank": p.rank,
                "adp": p.adp,
                "own": round(p.room_drafted_rate, 3),
                "value": round(float(opt.values[i]), 2),
                **({"ou": lines[p.team]["total"], "itt": lines[p.team]["implied"]}
                   if p.team in lines else {}),
                **({"inj": statuses[i]} if i in statuses else {}),
            }
            for i, p in enumerate(s.players)
        ],
        "synergy": synergy,
        "seats": {},
    }
    for lobby_seat in seats:
        t0 = time.time()
        plan["seats"][str(lobby_seat)] = build_seat_plan(
            opt, lobby_seat - 1, args.boards, args.avail_boards, args.seed,
            alt_k=args.alt_paths, alt_boards=args.alt_boards,
        )
        print(f"seat {lobby_seat} planned in {time.time() - t0:.0f}s", flush=True)

    _Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _Path(args.out).write_text(json.dumps(plan))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

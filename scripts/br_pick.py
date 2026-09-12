"""Live-draft pick recommendation, built for speed.

One-shot: give it the slate CSV, your seat (1-6 as shown in the draft lobby,
or 0-5 zero-based via --seat0) and the picks made so far, get the ranked
next-pick board in a few seconds. Slate-level artifacts (reference field,
duplication index, outcome sims) are cached on disk per slate, so only the
first call after a new CSV pays the build cost — run --warm once when a new
rankings file arrives.

Examples:
    # once per new slate CSV (also run with --full for the deep preset)
    python scripts/br_pick.py --csv slate.csv --warm

    # during the draft (picks in order, partial names fine)
    python scripts/br_pick.py --csv slate.csv --seat 3 \
        --picks "gibbs, st. brown, chase, robinson"

    # a tough decision worth a slower, deeper look
    python scripts/br_pick.py --csv slate.csv --seat 3 --picks "..." --full
"""

from __future__ import annotations

import sys
import time
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse

from battle_royale import (
    BattleRoyaleEngine,
    DraftState,
    OptimizerConfig,
    PickOptimizer,
    TournamentModel,
)
from battle_royale.assistant import format_options, match_player
from battle_royale.cache import DEFAULT_CACHE_DIR
from battle_royale.formats import get_format


def build_optimizer(
    csv: str, contest_size: int | None, full: bool, seed: int,
    season: int | None = None, week: int | None = None,
    payouts: str | None = None, fmt_key: str = "battle_royale",
) -> PickOptimizer:
    from battle_royale.equity import load_payout_table
    from battle_royale.slate import Slate

    fmt = get_format(fmt_key)
    contest_size = contest_size or fmt.default_contest_size
    slate = Slate.from_csv(csv, fmt=fmt)
    game_lines, sit = {}, None
    if season and week:
        from battle_royale.external import (
            load_game_lines,
            load_player_status,
            sit_probabilities,
            slate_status,
        )

        game_lines = load_game_lines(season, week)
        statuses = slate_status(slate, load_player_status(season, week))
        if statuses:
            sit = sit_probabilities(slate, statuses)
    engine = BattleRoyaleEngine(slate, seed=seed, game_lines=game_lines, sit_prob=sit)
    config = OptimizerConfig() if full else OptimizerConfig.fast()
    config.seed = seed
    config.cache_dir = str(DEFAULT_CACHE_DIR)
    curve, _ = load_payout_table(payouts, filename=fmt.payouts_file)
    return PickOptimizer(
        engine, TournamentModel(contest_size=contest_size, curve=curve), config
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Battle Royale next-pick recommendation")
    ap.add_argument("--csv", required=True, help="current rankings/simulations slate CSV")
    ap.add_argument("--seat", type=int, default=None, help="your draft position, 1-6")
    ap.add_argument("--seat0", type=int, default=None, help="your seat zero-based, 0-5")
    ap.add_argument("--picks", default="", help="comma-separated picks so far, in order")
    ap.add_argument("--full", action="store_true", help="deep preset (slower, steadier)")
    ap.add_argument("--warm", action="store_true", help="just build+cache slate artifacts")
    ap.add_argument("--top", type=int, default=8, help="options to display")
    ap.add_argument("--format", default="battle_royale", dest="fmt",
                    help="contest format key from battle_royale/data/formats.json")
    ap.add_argument("--contest-size", type=int, default=None,
                    help="entries in the contest (default: the format's field size)")
    ap.add_argument("--payouts", default=None,
                    help="prize-table JSON path (default: packaged real table if present)")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--season", type=int, default=None, help="fetch Vegas lines for game-env tilt")
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    if args.warm:
        for full in (False, True) if args.full else (False,):
            opt = build_optimizer(args.csv, args.contest_size, full, args.seed,
                                  args.season, args.week, args.payouts, args.fmt)
            opt.warm()
            print(f"warmed {'full' if full else 'fast'} preset in {time.time() - t0:.1f}s")
        return 0

    n_seats = get_format(args.fmt).seats
    seat = args.seat0 if args.seat0 is not None else (args.seat - 1 if args.seat else None)
    if seat is None or not 0 <= seat < n_seats:
        ap.error(f"give --seat 1-{n_seats} (lobby position) or --seat0 0-{n_seats - 1}")

    opt = build_optimizer(args.csv, args.contest_size, args.full, args.seed,
                          args.season, args.week, args.payouts, args.fmt)
    state = DraftState(opt.slate)
    recorded = []
    for name in [x for x in args.picks.split(",") if x.strip()]:
        try:
            idx = match_player(opt.slate, name)
            state.apply_pick(idx)
            recorded.append(opt.slate.players[idx].name)
        except (KeyError, ValueError) as e:
            print(f"error recording pick {name.strip()!r}: {e}")
            return 1
    if recorded:
        print("picks recorded:", ", ".join(recorded))

    if state.complete:
        print("draft complete — roster:", ", ".join(opt.slate.names(state.rosters[seat])))
        return 0
    on_clock = state.seat_on_clock()
    if on_clock != seat:
        print(
            f"note: {len([x for x in args.picks.split(',') if x.strip()])} picks recorded; "
            f"seat {on_clock + 1} is on the clock (you are seat {seat + 1}). "
            f"Showing their board — add the missing picks if this is wrong."
        )
    rec = opt.recommend(state)
    mine = ", ".join(opt.slate.names(state.rosters[seat])) or "(empty)"
    print(f"your roster: {mine}")
    print(format_options(rec, limit=args.top))
    print(f"[{time.time() - t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

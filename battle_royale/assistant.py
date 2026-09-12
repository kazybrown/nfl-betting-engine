"""Live draft assistant CLI.

Non-interactive (one recommendation for the current board):

    python -m battle_royale.assistant --csv slate.csv --seat 4 \
        --history "Jahmyr Gibbs, Bijan Robinson, Ja'Marr Chase"

Interactive (drive a whole draft):

    python -m battle_royale.assistant --csv slate.csv --seat 4 --interactive

Interactive commands:
    p <name>   record the pick for whichever seat is on the clock
    rec        (re)compute recommendations for the current pick
    board      top remaining players by ADP
    roster     show all rosters
    undo       revert the last recorded pick
    q          quit
"""

from __future__ import annotations

import argparse
import sys

from .draft import DraftState
from .engine import BattleRoyaleEngine
from .equity import TournamentModel
from .optimizer import OptimizerConfig, PickOptimizer
from .slate import Slate


def match_player(slate: Slate, text: str) -> int:
    """Resolve a (possibly partial, case-insensitive) name to a player index.

    Tiers: exact full name, then exact last-name token, then any-word prefix,
    then substring. Within a tier, ties break to the better (lower) ADP —
    during a draft, "chase" means Ja'Marr Chase, not Chase Brown. Raises only
    when nothing matches at all.
    """
    t = text.strip().lower()
    if not t:
        raise KeyError("empty player name")

    def _tokens(name: str) -> list[str]:
        return name.lower().replace(".", " ").replace("'", "").split()

    q = t.replace(".", " ").replace("'", "")
    tiers: list[list[int]] = [[], [], [], []]
    for i, p in enumerate(slate.players):
        name = p.name.lower()
        toks = _tokens(p.name)
        if name == t or " ".join(toks) == q:
            tiers[0].append(i)
        elif toks and toks[-1] == q:
            tiers[1].append(i)
        elif any(tok.startswith(q) for tok in toks) or " ".join(toks).startswith(q):
            tiers[2].append(i)
        elif q in " ".join(toks) or t in name:
            tiers[3].append(i)
    for tier in tiers:
        if tier:
            return min(tier, key=lambda i: slate.adp[i])
    raise KeyError(f"no player matches {text!r}")


def format_options(rec: dict, limit: int = 10) -> str:
    lines = []
    mode = rec["mode"]
    header = (
        f"pick {rec['pick']} (drafter {rec['seat'] + 1}/6, "
        f"next own pick: {rec['next_own_pick']})"
        f" — {'JOINT PAIR' if mode == 'pair' else 'single pick'}"
    )
    lines.append(header)
    lines.append(
        f"{'action':<42}{'pos':<10}{'equity':>8}{'win%':>8}{'top1%':>8}"
        f"{'surv':>7}{'own%':>7}  notes"
    )
    for e in rec["options"][:limit]:
        action = " + ".join(e["players"])
        pos = "/".join(e["positions"])
        surv = e.get("survival_to_next_pick")
        surv_s = f"{surv * 100:.0f}%" if surv is not None else "-"
        own = max(e["room_drafted_rate"]) * 100
        notes = "; ".join(n for n in e["stack_notes"] if n)
        lines.append(
            f"{action:<42}{pos:<10}{e['expected_payout']:>8.2f}"
            f"{e['win_rate_vs_sample'] * 100:>7.2f}%"
            f"{e['p_top_1pct'] * 100:>7.2f}%"
            f"{surv_s:>7}{own:>6.0f}%  {notes}"
        )
    return "\n".join(lines)


def _print_board(state: DraftState, limit: int = 15) -> None:
    s = state.slate
    import numpy as np

    avail = np.where(state.avail)[0]
    order = avail[np.argsort(s.adp[avail])][:limit]
    for j in order:
        p = s.players[int(j)]
        print(
            f"  {p.name:<24}{p.position:<4}{p.team:<5}adp {p.adp:>5.1f}  "
            f"proj {p.proj:>5.1f}  own {p.room_drafted_rate * 100:>5.1f}%"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Battle Royale live draft assistant")
    ap.add_argument("--csv", required=True, help="rankings/simulations slate CSV")
    ap.add_argument("--seat", type=int, required=True, help="your seat, 0-5 (pick order - 1)")
    ap.add_argument("--history", default="", help="comma-separated picks so far, in order")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--fast", action="store_true", help="lighter sims for live use")
    ap.add_argument("--contest-size", type=int, default=70_000)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args(argv)
    if not 0 <= args.seat <= 5:
        ap.error("--seat must be 0-5 (your pick order minus one)")

    engine = BattleRoyaleEngine.from_csv(args.csv, seed=args.seed)
    config = OptimizerConfig.fast() if args.fast else OptimizerConfig()
    config.seed = args.seed
    optimizer = PickOptimizer(
        engine, TournamentModel(contest_size=args.contest_size), config
    )

    state = DraftState(engine.slate)
    history_idxs: list[int] = []
    if args.history.strip():
        for name in args.history.split(","):
            try:
                idx = match_player(engine.slate, name)
                state.apply_pick(idx)
            except (KeyError, ValueError) as e:
                ap.error(f"--history: {e}")
            history_idxs.append(idx)

    if not args.interactive:
        if state.complete:
            print("draft is complete")
            return 0
        if state.seat_on_clock() != args.seat:
            print(
                f"note: seat {state.seat_on_clock()} is on the clock at pick "
                f"{state.next_pick}; recommendations are for that seat"
            )
        print(format_options(optimizer.recommend(state)))
        return 0

    print("interactive mode — 'p <name>' to record picks, 'rec', 'board', 'roster', 'undo', 'q'")
    # Seed the undo stack with --history picks (in original pick order) so
    # undo after startup replays them instead of discarding them.
    undo_stack: list[int] = list(history_idxs)
    while not state.complete:
        on_clock = state.seat_on_clock()
        marker = " (YOU)" if on_clock == args.seat else ""
        try:
            cmd = input(f"pick {state.next_pick} / seat {on_clock}{marker} > ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "board":
            _print_board(state)
        elif cmd == "roster":
            for seat in range(6):
                names = ", ".join(engine.slate.players[i].name for i in state.rosters[seat])
                you = " (YOU)" if seat == args.seat else ""
                print(f"  seat {seat}{you}: {names}")
        elif cmd == "undo":
            if not undo_stack:
                print("nothing to undo")
                continue
            picks = undo_stack[:-1]
            undo_stack = []
            state = DraftState(engine.slate)
            for idx in picks:
                state.apply_pick(idx)
                undo_stack.append(idx)
            print("undone")
        elif cmd == "rec":
            print(format_options(optimizer.recommend(state)))
        elif cmd.startswith("p "):
            recorded = False
            try:
                idx = match_player(engine.slate, cmd[2:])
                state.apply_pick(idx)
                undo_stack.append(idx)
                recorded = True
                p = engine.slate.players[idx]
                print(f"  recorded: {p.name} ({p.position}) to seat {on_clock}")
            except (KeyError, ValueError) as e:
                print(f"  error: {e}")
            if recorded and not state.complete and state.seat_on_clock() == args.seat:
                print("\nYOU are on the clock — computing recommendations...\n")
                print(format_options(optimizer.recommend(state)))
        else:
            print("commands: p <name> | rec | board | roster | undo | q")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

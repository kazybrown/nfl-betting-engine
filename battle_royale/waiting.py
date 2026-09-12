"""Pick survival and cost of waiting (transferred task-4 layer).

Survival is always computed through the exact seats picking between now and
our next selection, conditioned on the actual room state — not from a global
ADP curve. Cost-of-waiting compares the take-now branch against taking the
best alternative now and returning for the candidate, using common random
numbers (shared seat tendencies and market clocks across branches) to cut
Monte Carlo comparison noise.

Back-to-back turn picks (6→7, 18→19, 30→31) have zero intervening opponents;
the real waiting risk sits after picks 7 and 19 where ten opponent picks
occur before the seat returns. Those pairs should be optimized jointly (see
:meth:`battle_royale.optimizer.PickOptimizer.recommend`).
"""

from __future__ import annotations

import numpy as np

from .draft import DraftState
from .opponents import OpponentPolicy


def survival_to_next_pick(
    policy: OpponentPolicy,
    state: DraftState,
    candidates: list[int],
    taken_now: int,
    rng: np.random.Generator,
    n_sims: int = 400,
) -> dict[int, float]:
    """P(candidate still available at our next pick | we take ``taken_now`` now).

    Returns survival for every candidate index in ``candidates`` (the one equal
    to ``taken_now`` reports 1.0 by convention — it is already on our roster).
    """
    seat = state.seat_on_clock()
    nxt = state.next_own_pick(seat)
    if nxt is None:
        return {c: 0.0 for c in candidates}
    survived = {c: 0 for c in candidates}
    for _ in range(n_sims):
        st = state.clone()
        st.apply_pick(taken_now)
        seats = policy.room_params(rng)
        times = policy.latent_market(rng)
        while st.next_pick < nxt:
            policy.choose(st, seats, times, rng)
        for c in candidates:
            if c == taken_now or st.avail[c]:
                survived[c] += 1
    return {c: survived[c] / n_sims for c in candidates}


def cost_of_waiting(
    policy: OpponentPolicy,
    state: DraftState,
    candidate: int,
    values: np.ndarray,
    rng: np.random.Generator,
    alternative: int | None = None,
    n_sims: int = 800,
) -> dict:
    """Take-now vs wait comparison for one candidate in the current state.

    Branch A takes ``candidate`` now and the best-value legal player at the
    next own pick. Branch B takes ``alternative`` (default: best-value legal
    other than the candidate) now, then the candidate at the next pick if he
    survived, otherwise the best-value legal replacement.
    """
    slate = policy.slate
    seat = state.seat_on_clock()
    p = state.next_pick
    nxt = state.next_own_pick(seat)
    if nxt is None:
        raise ValueError("no later pick for this seat")
    if not state.eligible_mask(seat)[candidate]:
        raise ValueError("candidate is unavailable or illegal now")

    if alternative is None:
        mask = state.eligible_mask(seat).copy()
        mask[candidate] = False
        legal = np.where(mask)[0]
        if len(legal) == 0:
            raise ValueError("no legal alternative")
        alternative = int(legal[np.argmax(values[legal])])

    def _best_legal(st: DraftState, force: int | None = None) -> int | None:
        mask = st.eligible_mask(seat)
        if force is not None and mask[force]:
            return force
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return None
        return int(legal[np.argmax(values[legal])])

    take_tot, wait_tot = [], []
    survive = 0
    for _ in range(n_sims):
        seats = policy.room_params(rng)
        times = policy.latent_market(rng)
        st1 = state.clone()
        st1.apply_pick(candidate)
        st2 = state.clone()
        st2.apply_pick(alternative)
        while st1.next_pick < nxt:
            policy.choose(st1, seats, times, rng)
            policy.choose(st2, seats, times, rng)
        j1 = _best_legal(st1)
        j2 = _best_legal(st2, force=candidate)
        if j2 == candidate:
            survive += 1
        v1 = values[candidate] + (values[j1] if j1 is not None else 0.0)
        v2 = values[alternative] + (values[j2] if j2 is not None else 0.0)
        take_tot.append(v1)
        wait_tot.append(v2)

    take = np.asarray(take_tot)
    wait = np.asarray(wait_tot)
    return {
        "current_pick": p,
        "next_pick": nxt,
        "intervening_picks": nxt - p - 1,
        "candidate": slate.players[candidate].name,
        "alternative_now": slate.players[alternative].name,
        "p_candidate_survives_and_legal": survive / n_sims,
        "take_now_pair_value": float(take.mean()),
        "wait_pair_value": float(wait.mean()),
        "cost_of_waiting": float(take.mean() - wait.mean()),
        "prob_take_now_better": float(np.mean(take > wait)),
        "sims": n_sims,
    }

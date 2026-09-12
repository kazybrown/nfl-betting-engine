"""Roster-state-aware opponent draft policy (transferred task-3 layer).

Opponent picks are driven by market/draft information only: a latent
per-drafter market clock built from ADP and room-drafted rate, plus the
seat's actual roster needs, FLEX construction tendency and stacking taste.
Slate projections and optimal rates deliberately do NOT enter opponent
utility — the field drafts the market, not our valuations.

Calibration on the supplied slate export reproduced room-drafted rates with
~1.7% MAE (corr 0.998) and the observed ~84/16 RB/WR FLEX split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import RB, TE, WR
from .draft import DraftState
from .slate import Slate


@dataclass
class SeatParams:
    market: float
    need: float
    stack: float
    scroll: float
    rb_flex: float


class OpponentPolicy:
    """Field draft policy.

    ``adp_sigma`` (latent market clock spread), ``choice_noise`` (per-pick
    Gumbel utility noise) and ``stack_scale`` control field concentration and
    stacking behavior. Defaults are calibrated end-to-end against real Battle
    Royale archives (2023-2024): at 45k entries the simulated field reproduces
    ~79% unique exact rosters (real: 75-85%), top same-team QB-stack pair
    affinity ~5x independence (real: 4.5-7.5x), and room-drafted rates within
    1.5% MAE of the slate's Own%. The transferred defaults (sigma 4.2, noise
    0.20, no stack scaling) produced an unrealistically diverse field (~96%
    unique, ~1.3x stack affinity).
    """

    def __init__(
        self,
        slate: Slate,
        adp_sigma: float = 2.4,
        choice_noise: float = 0.12,
        stack_scale: float = 3.5,
    ):
        self.slate = slate
        self.adp_sigma = adp_sigma
        self.choice_noise = choice_noise
        self.stack_scale = stack_scale
        # Infer the field's FLEX construction from room-drafted rates: excess
        # of expected per-roster position counts above mandatory minima.
        fmt = slate.fmt
        per_roster = np.array(
            [slate.room_drafted_rate[slate.pos == k].sum() for k in range(4)]
        ) / float(fmt.seats)
        rb_extra = max(per_roster[RB] - float(fmt.roster_min[RB]), 0.0)
        wr_extra = max(per_roster[WR] - float(fmt.roster_min[WR]), 0.0)
        denom = max(rb_extra + wr_extra, 1e-9)
        self.flex_target = {"RB": rb_extra / denom, "WR": wr_extra / denom, "TE": 0.0}
        self.per_roster_target = per_roster

    # ------------------------------------------------------------------
    # Latent draws
    # ------------------------------------------------------------------

    def seat_params(self, rng: np.random.Generator) -> SeatParams:
        return SeatParams(
            market=float(np.clip(rng.normal(1.0, 0.08), 0.78, 1.22)),
            need=float(np.clip(rng.normal(1.0, 0.12), 0.70, 1.35)),
            stack=float(np.clip(rng.lognormal(mean=-0.30, sigma=0.55), 0.15, 2.0)),
            scroll=float(np.clip(rng.normal(0.0, 0.20), -0.40, 0.55)),
            rb_flex=float(np.clip(rng.normal(self.flex_target["RB"], 0.055), 0.62, 0.96)),
        )

    def room_params(self, rng: np.random.Generator) -> list[SeatParams]:
        return [self.seat_params(rng) for _ in range(self.slate.fmt.seats)]

    def latent_market(self, rng: np.random.Generator) -> np.ndarray:
        """Per-room latent pick times: Normal(ADP, sigma) mixture with an
        undrafted tail governed by room-drafted rate."""
        s = self.slate
        targeted = rng.random(s.n) < s.room_drafted_rate
        times = np.empty(s.n)
        times[targeted] = rng.normal(s.adp[targeted], self.adp_sigma)
        n_untargeted = int((~targeted).sum())
        times[~targeted] = (
            s.fmt.total_picks + rng.exponential(19.0, n_untargeted) + 0.12 * s.adp[~targeted]
        )
        return np.maximum(times, 0.25)

    # ------------------------------------------------------------------
    # Pick utility
    # ------------------------------------------------------------------

    def _need_bonus(
        self, counts: np.ndarray, pos: int, left_before: int, seat: SeatParams
    ) -> float:
        rmin = self.slate.fmt.min_arr
        missing = np.maximum(rmin - counts, 0)
        mandatory = int(missing.sum())
        slack = left_before - mandatory
        b = 0.0
        if missing[pos] > 0:
            urgency = 1.0 + 1.20 / (max(slack, 0) + 1.0)
            b += seat.need * (1.25 + 1.35 * urgency)
        if pos == WR and counts[WR] < rmin[WR]:
            b += 0.45 * seat.need
        if pos == RB and counts[RB] < rmin[RB]:
            b += 0.35 * seat.need
        if mandatory == 0:
            prb = min(max(seat.rb_flex, 0.01), 0.99)
            if pos == RB:
                b += 0.24 * math.log(prb / (1.0 - prb))
            elif pos == TE:
                b -= 0.85
        if pos == TE and counts[TE] >= 1:
            b -= 1.10
        return b

    def _stack_bonus(self, cand: int, roster: list[int], seat: SeatParams) -> float:
        if not roster:
            return 0.0
        s = self.slate
        p = s.players[cand]
        b = 0.0
        for ridx in roster:
            q = s.players[int(ridx)]
            if p.team == q.team:
                if (p.position == "QB" and q.position in ("WR", "TE")) or (
                    q.position == "QB" and p.position in ("WR", "TE")
                ):
                    b += 0.72 * self.stack_scale * seat.stack
            elif p.team == q.opponent or q.team == p.opponent:
                if "QB" in (p.position, q.position):
                    b += 0.10 * self.stack_scale * seat.stack
        return min(b, 1.65 * self.stack_scale)

    def choose(
        self,
        state: DraftState,
        seat_params: list[SeatParams],
        times: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """Pick for the seat on the clock; mutates ``state``."""
        s = self.slate
        pick = state.next_pick
        seat_idx = state.seat_on_clock()
        seat = seat_params[seat_idx]
        counts = state.counts[seat_idx]
        left_before = state.picks_left(seat_idx)

        elig = np.where(state.eligible_mask(seat_idx))[0]
        if len(elig) == 0:
            raise RuntimeError(f"no legal candidate at pick {pick}")

        missing = np.maximum(s.fmt.min_arr - counts, 0)
        pool_n = 22 if int(missing.sum()) >= left_before - 1 else 18
        order = np.argsort(times[elig] + 0.008 * s.rank[elig])
        pool = elig[order[: min(pool_n, len(order))]]

        u = -0.64 * seat.market * times[pool] - 0.018 * s.adp[pool]
        u = u + seat.scroll * 0.025 * (pick - s.adp[pool])
        for k, j in enumerate(pool):
            u[k] += self._need_bonus(counts, int(s.pos[j]), left_before, seat)
            u[k] += self._stack_bonus(int(j), state.rosters[seat_idx], seat)
        u = u + rng.gumbel(0.0, self.choice_noise, size=len(pool))

        chosen = int(pool[int(np.argmax(u))])
        state.apply_pick(chosen)
        return chosen

    # ------------------------------------------------------------------
    # Room simulation
    # ------------------------------------------------------------------

    def simulate_room(
        self,
        rng: np.random.Generator,
        state: DraftState | None = None,
        user_seat: int | None = None,
        user_policy=None,
    ) -> DraftState:
        """Draft a room to completion.

        Args:
            rng: random generator.
            state: optional partial state to continue from (cloned, not mutated).
            user_seat: seat controlled by ``user_policy`` instead of the field
                policy (None = all six seats field-controlled).
            user_policy: callable ``(state, rng) -> player_idx`` for user picks;
                the returned index is applied to the state.
        """
        st = state.clone() if state is not None else DraftState(self.slate)
        seat_params = self.room_params(rng)
        times = self.latent_market(rng)
        while not st.complete:
            if user_seat is not None and st.seat_on_clock() == user_seat:
                st.apply_pick(int(user_policy(st, rng)))
            else:
                self.choose(st, seat_params, times, rng)
        return st

"""Integrated pick optimizer: the decision layer for live drafts.

For each candidate at the current pick, the optimizer:

1. rolls the room forward many times (opponents via the calibrated field
   policy, our own remaining picks via a stack-aware greedy completion),
2. scores every completed roster's tournament equity against a cached
   simulated reference field under correlated player outcomes, with
   exact-duplicate payout sharing at the top of the field, and
3. reports survival to our next pick so "take now vs wait" is explicit.

Back-to-back turn picks (e.g. 6+7, 18+19, 30+31) are optimized jointly as a
pair, never one at a time.

Priority ordering encoded here (from historical winner review): tournament
equity first, current value/ceiling second, scarcity/cost-of-waiting third,
correlation fourth, and combinatorial leverage last — uniqueness is never
manufactured for its own sake; it enters only through duplication-adjusted
payout sharing and is regularized by projection cost automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import cache as _cache
from .constants import POSITIONS
from .draft import DraftState
from .engine import BattleRoyaleEngine
from .equity import TournamentModel
from .field import FieldAnalytics, RosterIndex, generate_field
from .slate import Slate


@dataclass
class OptimizerConfig:
    n_candidates: int = 14
    per_position_floor: int = 2
    n_rollouts: int = 120
    n_outcome_sims: int = 1500
    eval_field_entries: int = 3600
    # Exact-duplicate counts come from a larger field than the scoring field:
    # at contest scale, one sampled copy in a small field converts to dozens
    # of expected contest copies, quantizing the sharing penalty. A 24k-entry
    # duplication reference (built once per slate) resolves ~3 contest copies
    # per sampled copy at a 70k contest.
    dup_field_entries: int = 24_000
    n_survival_sims: int = 300
    ceiling_lambda: float = 0.5
    cov_weight: float = 0.35
    completion_temp: float = 0.30
    pair_top_k: int = 8
    equity_chunk: int = 250
    seed: int = 20260912
    # Directory for on-disk caching of the reference field, duplication index
    # and outcome-sim matrix (None disables). Keys include content hashes of
    # the slate, policy parameters and fitted tables, so a new rankings CSV or
    # recalibration invalidates automatically.
    cache_dir: str | None = None

    @classmethod
    def fast(cls) -> OptimizerConfig:
        """Live-draft preset (a few seconds per pick on modest hardware)."""
        return cls(
            n_candidates=10,
            n_rollouts=48,
            n_outcome_sims=600,
            eval_field_entries=1800,
            dup_field_entries=9_000,
            n_survival_sims=150,
            pair_top_k=6,
        )


class PickOptimizer:
    def __init__(
        self,
        engine: BattleRoyaleEngine,
        tournament: TournamentModel | None = None,
        config: OptimizerConfig | None = None,
    ):
        self.engine = engine
        self.slate: Slate = engine.slate
        self.tournament = tournament or TournamentModel()
        self.config = config or OptimizerConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._field: np.ndarray | None = None
        self._analytics: FieldAnalytics | None = None
        self._dup_index: RosterIndex | FieldAnalytics | None = None
        self._values: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._scores_cache: np.ndarray | None = None
        s = self.slate
        self._slate_hash = _cache.array_hash(
            s.proj, s.adp, s.room_drafted_rate, s.ceiling, s.pos.astype(np.int8)
        ) + _cache.text_hash([p.name for p in s.players])
        p = self.engine.policy
        self._policy_hash = _cache.text_hash(p.adp_sigma, p.choice_noise, p.stack_scale)

    def _cached(self, key: str, build):
        """Build-or-load a slate-level artifact through the disk cache."""
        if self.config.cache_dir is None:
            return build()
        hit = _cache.load(self.config.cache_dir, key)
        if hit is not None:
            return hit
        obj = build()
        _cache.save(self.config.cache_dir, key, obj)
        return obj

    def warm(self) -> None:
        """Precompute (and cache) everything a live draft needs."""
        _ = self.values, self.cov, self.field, self.dup_index
        self._outcome_scores()

    # ------------------------------------------------------------------
    # Cached slate-level artifacts
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Static ceiling-weighted player value: proj + lambda * (p90 - proj)."""
        if self._values is None:
            q90 = self.engine.marginals.quantile(0.90)
            self._values = self.slate.proj + self.config.ceiling_lambda * (
                q90 - self.slate.proj
            )
        return self._values

    @property
    def cov(self) -> np.ndarray:
        if self._cov is None:
            sd = np.sqrt(self.engine.marginals.variance)
            self._cov = self.engine.correlation.matrix * np.outer(sd, sd)
        return self._cov

    @property
    def field(self) -> np.ndarray:
        if self._field is None:
            key = (
                f"field_{self._slate_hash}_{self._policy_hash}"
                f"_{self.config.eval_field_entries}_{self.config.seed}"
            )
            self._field = self._cached(
                key,
                lambda: generate_field(
                    self.engine.policy,
                    self.config.eval_field_entries,
                    np.random.default_rng(self.config.seed + 1),
                ),
            )
        return self._field

    @property
    def analytics(self) -> FieldAnalytics:
        if self._analytics is None:
            self._analytics = FieldAnalytics(self.slate, self.field)
        return self._analytics

    def _outcome_scores(self) -> np.ndarray:
        """Outcome sims shared by every recommend() call (fixed seed, cached)."""
        if self._scores_cache is None:
            m, c = self.engine.marginals, self.engine.correlation
            key = (
                f"scores_{_cache.array_hash(m.shape, m.scale, c.cholesky)}"
                f"_{self.config.n_outcome_sims}_{self.config.seed}"
            )
            self._scores_cache = self._cached(
                key,
                lambda: self.engine.sample_scores(
                    self.config.n_outcome_sims, np.random.default_rng(self.config.seed + 999)
                ),
            )
        return self._scores_cache

    @property
    def dup_index(self) -> RosterIndex | FieldAnalytics:
        """Duplication reference: a larger field, built once, copy counts only."""
        if self._dup_index is None:
            n = self.config.dup_field_entries
            if n <= self.config.eval_field_entries:
                self._dup_index = self.analytics
            else:
                key = (
                    f"dup_{self._slate_hash}_{self._policy_hash}"
                    f"_{n}_{self.config.seed}"
                )
                self._dup_index = self._cached(
                    key,
                    lambda: RosterIndex(
                        generate_field(
                            self.engine.policy, n, np.random.default_rng(self.config.seed + 2)
                        )
                    ),
                )
        return self._dup_index

    # ------------------------------------------------------------------
    # Completion policy for our own future picks inside rollouts
    # ------------------------------------------------------------------

    def _completion_pick(
        self, state: DraftState, seat: int, rng: np.random.Generator
    ) -> int:
        mask = state.eligible_mask(seat)
        legal = np.where(mask)[0]
        if len(legal) == 0:
            raise RuntimeError("no legal completion pick")
        u = self.values[legal].astype(float).copy()
        roster = state.rosters[seat]
        if roster and self.config.cov_weight > 0:
            u += self.config.cov_weight * self.cov[np.ix_(legal, roster)].sum(axis=1)
        if self.config.completion_temp > 0:
            u += rng.gumbel(0.0, self.config.completion_temp, size=len(legal))
        return int(legal[int(np.argmax(u))])

    def _rollout(
        self,
        state: DraftState,
        user_seat: int,
        room_params,
        times: np.ndarray,
        rng: np.random.Generator,
    ) -> list[int]:
        """Complete the room; returns the user's final six-player roster."""
        st = state.clone()
        while not st.complete:
            if st.seat_on_clock() == user_seat:
                st.apply_pick(self._completion_pick(st, user_seat, rng))
            else:
                self.engine.policy.choose(st, room_params, times, rng)
        return st.rosters[user_seat]

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _candidates(self, state: DraftState, seat: int, k: int) -> list[int]:
        legal = np.where(state.eligible_mask(seat))[0]
        if len(legal) == 0:
            return []
        chosen: list[int] = []
        # Per-position value floors, interleaved by depth (best of each
        # position first) so a small k still compares across positions
        # instead of truncating the later-listed positions away.
        floors: list[list[int]] = []
        for pos_idx in range(len(POSITIONS)):
            at_pos = legal[self.slate.pos[legal] == pos_idx]
            top = at_pos[np.argsort(-self.values[at_pos])][: self.config.per_position_floor]
            floors.append([int(x) for x in top])
        for depth in range(self.config.per_position_floor):
            for pos_floors in floors:
                if depth < len(pos_floors):
                    chosen.append(pos_floors[depth])
        # Market-adjacent players the room is about to consider.
        near_market = legal[np.argsort(self.slate.adp[legal])][:3]
        chosen.extend(int(x) for x in near_market)
        # Fill by overall value.
        for j in legal[np.argsort(-self.values[legal])]:
            if len(set(chosen)) >= k:
                break
            chosen.append(int(j))
        out = list(dict.fromkeys(chosen))[:k]
        return out

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend(self, state: DraftState) -> dict:
        """Rank actions for the seat currently on the clock.

        Returns a dict with ``mode`` ("single" or "pair"), the on-clock pick
        number and seat, and ``options`` sorted by expected tournament payout.
        """
        cfg = self.config
        seat = state.seat_on_clock()
        pick = state.next_pick
        next_own = state.next_own_pick(seat)
        pair_mode = next_own == pick + 1

        actions: list[tuple[int, ...]] = []
        if pair_mode:
            actions = self._pair_actions(state, seat)
            if not actions:
                pair_mode = False  # degenerate endgame: no legal ordered pair
        if not pair_mode:
            actions = [(c,) for c in self._candidates(state, seat, cfg.n_candidates)]
        if not actions:
            raise ValueError("no legal candidates")

        latent_rng = np.random.default_rng(cfg.seed + 100 + pick)
        # Common random numbers: one latent room per rollout, shared by every
        # action, and one fresh utility-noise stream PER (rollout) reused for
        # every action, so equity differences come from the action, not the
        # draw. (Streams diverge once actions change pool sizes, but early
        # picks stay matched.)
        latents = []
        for _ in range(cfg.n_rollouts):
            latents.append(
                (
                    self.engine.policy.room_params(latent_rng),
                    self.engine.policy.latent_market(latent_rng),
                )
            )

        completed = np.empty((len(actions), cfg.n_rollouts, 6), dtype=np.int16)
        for a_idx, action in enumerate(actions):
            base = state.clone()
            for c in action:
                base.apply_pick(int(c))
            for r_idx, (params, times) in enumerate(latents):
                noise_rng = np.random.default_rng(
                    (cfg.seed, state.next_pick, r_idx)
                )
                roster = self._rollout(base, seat, params, times, noise_rng)
                completed[a_idx, r_idx] = roster

        # Deduplicate rosters before the (expensive) equity evaluation.
        flat = completed.reshape(-1, 6)
        keys = np.sort(flat, axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        dup_ref = self.dup_index
        dup_counts = dup_ref.roster_copies(uniq)
        scores = self._outcome_scores()
        metrics = self.tournament.evaluate_rosters(
            scores,
            self.field,
            uniq,
            dup_counts,
            dup_reference_entries=dup_ref.n_entries,
            chunk=cfg.equity_chunk,
        )
        payouts = np.array([m["expected_payout"] for m in metrics])[inverse].reshape(
            len(actions), cfg.n_rollouts
        )
        win = np.array([m["win_rate_vs_sample"] for m in metrics])[inverse].reshape(
            len(actions), cfg.n_rollouts
        )
        top1 = np.array([m["p_top_1pct"] for m in metrics])[inverse].reshape(
            len(actions), cfg.n_rollouts
        )
        mean_scores = np.array([m["mean_score"] for m in metrics])[inverse].reshape(
            len(actions), cfg.n_rollouts
        )

        survival = self._survival(state, seat, actions) if not pair_mode else None

        options = []
        for a_idx, action in enumerate(actions):
            names = self.slate.names(action)
            entry = {
                "players": names,
                "positions": [self.slate.players[int(c)].position for c in action],
                "value": float(np.sum(self.values[list(action)])),
                "expected_payout": float(payouts[a_idx].mean()),
                "win_rate_vs_sample": float(win[a_idx].mean()),
                "p_top_1pct": float(top1[a_idx].mean()),
                "mean_completed_score": float(mean_scores[a_idx].mean()),
                "adp_delta": [float(self.slate.adp[int(c)] - pick) for c in action],
                "room_drafted_rate": [
                    float(self.slate.room_drafted_rate[int(c)]) for c in action
                ],
                "stack_notes": [self._stack_note(state, seat, int(c)) for c in action],
                "example_completion": self.slate.names(
                    completed[a_idx, int(np.argmax(payouts[a_idx]))]
                ),
            }
            if survival is not None:
                c = int(action[0])
                entry["survival_to_next_pick"] = survival.get(c)
            options.append(entry)

        options.sort(key=lambda e: -e["expected_payout"])
        return {
            "pick": pick,
            "seat": seat,
            "mode": "pair" if pair_mode else "single",
            "next_own_pick": next_own,
            "options": options,
        }

    def recommend_from_history(self, picks: list[str | int]) -> dict:
        return self.recommend(DraftState.from_history(self.slate, picks))

    # ------------------------------------------------------------------

    def _pair_actions(self, state: DraftState, seat: int) -> list[tuple[int, int]]:
        cands = self._candidates(state, seat, self.config.pair_top_k)
        pairs: list[tuple[int, int]] = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                st = state.clone()
                try:
                    st.apply_pick(a)
                    st.apply_pick(b)
                except ValueError:
                    continue
                pairs.append((a, b))
        return pairs

    def _survival(
        self, state: DraftState, seat: int, actions: list[tuple[int, ...]]
    ) -> dict[int, float]:
        """Survival of each candidate to our next pick, via the two most
        plausible alternative take-now actions (each candidate's survival is
        measured on runs where it was not the player taken)."""
        from .waiting import survival_to_next_pick

        cands = [int(a[0]) for a in actions]
        if state.next_own_pick(seat) is None:
            return {c: 0.0 for c in cands}
        by_value = sorted(cands, key=lambda c: -self.values[c])
        rng = np.random.default_rng(self.config.seed + 555 + state.next_pick)
        n = max(self.config.n_survival_sims // 2, 50)
        out: dict[int, float] = {}
        s1 = survival_to_next_pick(self.engine.policy, state, cands, by_value[0], rng, n)
        out.update({c: v for c, v in s1.items() if c != by_value[0]})
        if len(by_value) > 1:
            s2 = survival_to_next_pick(
                self.engine.policy, state, [by_value[0]], by_value[1], rng, n
            )
            out[by_value[0]] = s2[by_value[0]]
        else:
            out[by_value[0]] = 1.0
        return out

    def _stack_note(self, state: DraftState, seat: int, cand: int) -> str:
        notes = []
        p = self.slate.players[cand]
        for r in state.rosters[seat]:
            q = self.slate.players[int(r)]
            if p.team == q.team:
                if "QB" in (p.position, q.position):
                    notes.append(f"stacks with {q.name}")
                else:
                    notes.append(f"teammate of {q.name}")
            elif p.team == q.opponent or q.team == p.opponent:
                if "QB" in (p.position, q.position):
                    notes.append(f"bring-back vs {q.name}")
        return "; ".join(notes)

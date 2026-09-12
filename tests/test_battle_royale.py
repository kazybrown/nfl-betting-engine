"""Tests for the battle_royale package (synthetic 8-team slate)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from battle_royale import (
    BattleRoyaleEngine,
    DraftState,
    FieldAnalytics,
    OptimizerConfig,
    PayoutCurve,
    PickOptimizer,
    Slate,
    TournamentModel,
)
from battle_royale.assistant import match_player
from battle_royale.constants import picks_of_seat, seat_of_pick
from battle_royale.field import generate_field, positional_construction
from battle_royale.opponents import OpponentPolicy
from battle_royale.slate import Player
from battle_royale.waiting import cost_of_waiting, survival_to_next_pick


def synthetic_slate() -> Slate:
    """8 teams (4 games), 8 players each: 1 QB, 2 RB, 3 WR, 2 TE."""
    teams = [f"T{k}" for k in range(8)]
    opp = {}
    for a in range(0, 8, 2):
        opp[teams[a]] = teams[a + 1]
        opp[teams[a + 1]] = teams[a]
    base_proj = {"QB": 19.0, "RB": 14.0, "WR": 11.0, "TE": 8.0}
    players = []
    rank = 0
    rng = np.random.default_rng(7)
    for depth in range(3):
        for t in teams:
            spots = []
            if depth == 0:
                spots = [("QB", 0), ("RB", 0), ("WR", 0)]
            elif depth == 1:
                spots = [("RB", 1), ("WR", 1), ("TE", 0)]
            else:
                spots = [("WR", 2), ("TE", 1)]
            for pos, d in spots:
                rank += 1
                proj = base_proj[pos] * (1.0 - 0.22 * d) - 0.05 * rank
                adp = min(rank + rng.normal(0, 0.5), 50.0)
                players.append(
                    Player(
                        name=f"{t} {pos}{d + 1}",
                        position=pos,
                        team=t,
                        opponent=opp[t],
                        rank=rank,
                        adp=float(max(adp, 1.0)),
                        proj=float(max(proj, 3.0)),
                        ceiling=float(max(proj, 3.0)) * 1.8,
                        optimal_rate=max(0.30 - 0.004 * rank, 0.01),
                        optimal_if_qb=max(0.45 - 0.005 * rank, 0.01) if pos != "QB" else 0.99,
                        optimal_if_opp_qb=max(0.32 - 0.004 * rank, 0.01) if pos != "QB" else 0.0,
                        top5_at_pos=0.2,
                        room_drafted_rate=float(np.clip(1.25 - 0.028 * rank, 0.02, 1.0)),
                    )
                )
    return Slate(players)


@pytest.fixture(scope="module")
def slate() -> Slate:
    return synthetic_slate()


@pytest.fixture(scope="module")
def engine(slate) -> BattleRoyaleEngine:
    return BattleRoyaleEngine(slate, seed=11)


# ----------------------------------------------------------------------
# constants / snake math
# ----------------------------------------------------------------------


def test_snake_order():
    order = [seat_of_pick(p) for p in range(1, 13)]
    assert order == [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]
    assert picks_of_seat(5) == [6, 7, 18, 19, 30, 31]
    assert picks_of_seat(0) == [1, 12, 13, 24, 25, 36]


# ----------------------------------------------------------------------
# draft state
# ----------------------------------------------------------------------


def test_draft_legality(slate):
    st = DraftState(slate)
    qbs = [i for i, p in enumerate(slate.players) if p.position == "QB"]
    st.apply_pick(qbs[0])  # seat 0 takes a QB at pick 1
    for i in range(1, 6):
        rbs = [j for j, p in enumerate(slate.players) if p.position == "RB" and st.avail[j]]
        st.apply_pick(rbs[0])
    # Seat 5 on the clock at pick 7 after picking at 6; seat 0 cannot act.
    assert st.seat_on_clock() == 5
    # Back to seat 0 at pick 12: a second QB must be illegal.
    while st.seat_on_clock() != 0:
        legal = np.where(st.eligible_mask(st.seat_on_clock()))[0]
        st.apply_pick(int(legal[0]))
    assert st.next_pick == 12
    with pytest.raises(ValueError):
        st.apply_pick(qbs[1])


def test_forced_positions_at_end(slate):
    """A seat that drafts RB/RB/WR/WR first must fill QB and TE last."""
    st = DraftState(slate)
    rng = np.random.default_rng(3)
    policy = OpponentPolicy(slate)
    seats = policy.room_params(rng)
    times = policy.latent_market(rng)
    user_seat = 0
    plan = ["RB", "RB", "WR", "WR"]
    while not st.complete:
        seat = st.seat_on_clock()
        if seat == user_seat and plan:
            want = plan.pop(0)
            legal = np.where(st.eligible_mask(seat))[0]
            pick = next(int(j) for j in legal if slate.players[int(j)].position == want)
            st.apply_pick(pick)
        elif seat == user_seat:
            elig = st.position_eligibility(seat)
            assert not elig[1] and not elig[2]  # RB and WR maxed / illegal
            legal = np.where(st.eligible_mask(seat))[0]
            st.apply_pick(int(legal[0]))
        else:
            policy.choose(st, seats, times, rng)
    positions = sorted(slate.players[i].position for i in st.rosters[user_seat])
    assert positions.count("QB") == 1 and positions.count("TE") == 1


def test_from_history_roundtrip(slate):
    names = []
    st = DraftState(slate)
    rng = np.random.default_rng(5)
    for _ in range(9):
        legal = np.where(st.eligible_mask(st.seat_on_clock()))[0]
        j = int(rng.choice(legal))
        names.append(slate.players[j].name)
        st.apply_pick(j)
    st2 = DraftState.from_history(slate, names)
    assert st2.next_pick == 10
    assert all(st2.avail == st.avail)


# ----------------------------------------------------------------------
# outcome model
# ----------------------------------------------------------------------


def test_marginal_means_match_projection(engine):
    x = engine.sample_scores(20_000)
    err = np.abs(x.mean(0) - engine.slate.proj) / engine.slate.proj
    assert float(err.mean()) < 0.03


def test_correlation_matrix_valid(engine):
    c = engine.correlation.matrix
    assert np.allclose(c, c.T)
    assert np.allclose(np.diag(c), 1.0)
    w = np.linalg.eigvalsh(c)
    assert w.min() > 0


def test_stack_correlation_realized(engine):
    s = engine.slate
    qb = s.name_to_idx["T0 QB1"]
    wr = s.name_to_idx["T0 WR1"]
    other = s.name_to_idx["T7 WR1"]  # different game entirely
    x = engine.sample_scores(30_000)
    r_stack = np.corrcoef(x[:, qb], x[:, wr])[0, 1]
    r_other = np.corrcoef(x[:, qb], x[:, other])[0, 1]
    assert r_stack > 0.15
    assert abs(r_other) < 0.05


# ----------------------------------------------------------------------
# opponents / field
# ----------------------------------------------------------------------


def test_contest_format_structure():
    from battle_royale.formats import ContestFormat, get_format

    fast = ContestFormat(key="test4", name="Test 4-round", rounds=4,
                         roster_min=(1, 1, 1, 0), roster_max=(1, 2, 2, 1))
    assert fast.total_picks == 24 and fast.roster_size == 4
    assert fast.picks_of_seat(0) == [1, 12, 13, 24]
    assert fast.picks_of_seat(5) == [6, 7, 18, 19]
    assert fast.seat_of_pick(6) == 5 and fast.seat_of_pick(7) == 5
    # 6 rounds cannot fit inside max 1+2+2+0=5 roster slots.
    with pytest.raises(ValueError):
        ContestFormat(key="bad", name="bad", rounds=6,
                      roster_min=(1, 1, 1, 0), roster_max=(1, 2, 2, 0))
    assert get_format("battle_royale").total_picks == 36
    with pytest.raises(KeyError):
        get_format("not_a_format")


def test_alt_format_room_and_field():
    from battle_royale.field import generate_field
    from battle_royale.formats import ContestFormat

    fast = ContestFormat(key="test4", name="Test 4-round", rounds=4,
                         roster_min=(1, 1, 1, 0), roster_max=(1, 2, 2, 1),
                         default_contest_size=10_000)
    s = synthetic_slate()
    alt = Slate(s.players, fmt=fast)
    policy = OpponentPolicy(alt)
    rng = np.random.default_rng(11)
    st = policy.simulate_room(rng)
    assert st.complete and st.next_pick == 25
    for seat in range(6):
        pos = [alt.players[i].position for i in st.rosters[seat]]
        assert len(pos) == 4
        assert pos.count("QB") == 1
        assert 1 <= pos.count("RB") <= 2
        assert 1 <= pos.count("WR") <= 2
        assert pos.count("TE") <= 1
    field = generate_field(policy, 60, rng)
    assert field.shape == (60, 4)


def test_room_is_legal(engine):
    rng = np.random.default_rng(1)
    st = engine.policy.simulate_room(rng)
    assert st.complete
    for seat in range(6):
        pos = [engine.slate.players[i].position for i in st.rosters[seat]]
        assert pos.count("QB") == 1
        assert 1 <= pos.count("RB") <= 2
        assert 2 <= pos.count("WR") <= 3
        assert 1 <= pos.count("TE") <= 2


def test_field_ownership_tracks_input(engine):
    rng = np.random.default_rng(2)
    field = generate_field(engine.policy, 1200, rng)
    fa = FieldAnalytics(engine.slate, field)
    corr = np.corrcoef(fa.room_drafted_rate, engine.slate.room_drafted_rate)[0, 1]
    assert corr > 0.9
    cons = positional_construction(engine.slate, field)
    assert cons["flex_share"]["RB"] > cons["flex_share"]["TE"]


def test_field_analytics_counts(slate):
    field = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 6]], dtype=np.int16)
    fa = FieldAnalytics(slate, field)
    assert fa.combo_count([0, 1]) == 3
    assert fa.combo_count([5, 6]) == 0
    assert fa.roster_copies(np.array([[0, 1, 2, 3, 4, 5]]))[0] == 2
    summary = fa.duplication_summary()
    assert summary["max_exact_copies"] == 2
    assert summary["unique_entry_share"] == pytest.approx(1 / 3)


# ----------------------------------------------------------------------
# equity
# ----------------------------------------------------------------------


def test_payout_curve_ranks():
    from battle_royale.equity import _RankPayout

    curve = PayoutCurve([(1, 1000.0), (0.001, 100.0), (0.10, 2.0)])
    rp = _RankPayout(curve, 10_000)
    # Winner tier is absolute rank 1; fraction tiers scale with contest size.
    assert list(rp.payout(np.array([1.0, 5.0, 500.0, 5000.0]))) == [1000.0, 100.0, 2.0, 0.0]
    # Range mean over a tied block spanning tiers pools the prizes.
    block = rp.range_mean(np.array([1.0]), np.array([3.0]))
    assert 100.0 < block[0] < 1000.0


def test_payout_curve_from_prize_table(tmp_path):
    from battle_royale.equity import _RankPayout, load_payout_table

    tiers = [
        {"rank_from": 1, "rank_to": 1, "prize_usd": 25_000.0},
        {"rank_from": 2, "rank_to": 2, "prize_usd": 5_000.0},
        {"rank_from": 3, "rank_to": 10, "prize_usd": 500.0},
        {"rank_from": 11, "rank_to": 1000, "prize_usd": 25.0},
    ]
    curve = PayoutCurve.from_prize_table(tiers, entry_fee=10.0)
    rp = _RankPayout(curve, 70_000)
    # Dollar prizes become entry-fee multiples at the right absolute ranks.
    assert list(rp.payout(np.array([1.0, 2.0, 3.0, 10.0, 11.0, 1000.0, 1001.0]))) == [
        2500.0, 500.0, 50.0, 50.0, 2.5, 2.5, 0.0
    ]

    with pytest.raises(ValueError):
        PayoutCurve.from_prize_table(tiers[1:], entry_fee=10.0)  # doesn't start at 1
    with pytest.raises(ValueError):
        PayoutCurve.from_prize_table([tiers[0], tiers[2]], entry_fee=10.0)  # gap at rank 2

    # Loader round-trips a table file; metadata carries provenance for
    # display; a typo'd explicit path fails loudly instead of silently
    # falling back to the placeholder curve.
    with pytest.raises(FileNotFoundError):
        load_payout_table(tmp_path / "nope.json")
    f = tmp_path / "payouts.json"
    f.write_text(json.dumps({"contest": "unit test", "entry_fee_usd": 10.0, "tiers": tiers}))
    loaded, meta = load_payout_table(f)
    assert loaded.points == curve.points
    assert meta["contest"] == "unit test" and meta["n_tiers"] == 4


def test_packaged_br_payout_table():
    from battle_royale.equity import _RankPayout, load_payout_table

    curve, meta = load_payout_table()  # packaged table must exist and load
    assert meta["source"] != "placeholder"
    assert meta["entry_fee_usd"] == 7.0 and meta["verified_through_rank"] == 4
    rp = _RankPayout(curve, meta["max_entries"])
    # Sourced facts: $50k/$25k/$15k/$10k top four, ~$9.94 min cash through
    # rank 9,500, nothing past the last paid place.
    assert list(rp.payout(np.array([1.0, 2.0, 3.0, 4.0]))) == pytest.approx(
        [50000 / 7, 25000 / 7, 15000 / 7, 10000 / 7]
    )
    assert rp.payout(np.array([9500.0]))[0] == pytest.approx(10 / 7)
    assert rp.payout(np.array([9501.0]))[0] == 0.0
    # Interpolated middle must decline monotonically.
    mids = rp.payout(np.arange(5.0, 9500.0))
    assert (np.diff(mids) <= 1e-9).all()


def test_better_roster_higher_equity(engine):
    s = engine.slate
    rng = np.random.default_rng(4)
    field = generate_field(engine.policy, 600, rng)
    top = np.argsort(s.rank)
    strong = _legal_roster(s, list(top))
    weak = _legal_roster(s, list(top[::-1]))
    scores = engine.sample_scores(800)
    tm = TournamentModel(contest_size=50_000)
    m = tm.evaluate_rosters(scores, field, np.array([strong, weak]))
    assert m[0]["expected_payout"] > m[1]["expected_payout"]
    assert m[0]["mean_score"] > m[1]["mean_score"]


def test_duplication_shares_payout(engine):
    s = engine.slate
    rng = np.random.default_rng(6)
    field = generate_field(engine.policy, 600, rng)
    top = np.argsort(s.rank)
    roster = np.array(_legal_roster(s, list(top)))
    dup_field = field.copy()
    dup_field[:40] = roster  # heavy duplication of our exact roster
    scores = engine.sample_scores(600)
    tm = TournamentModel(contest_size=50_000)
    unique = tm.evaluate_rosters(scores, field, roster[None, :])[0]
    duped = tm.evaluate_rosters(
        scores, dup_field, roster[None, :], roster_dup_counts=np.array([40])
    )[0]
    assert duped["expected_payout"] < unique["expected_payout"]


def _legal_roster(slate, ordered):
    from battle_royale.constants import ROSTER_MAX, ROSTER_MIN

    counts = np.zeros(4, dtype=int)
    picked = []
    for j in ordered:
        if len(picked) == 6:
            break
        pos = int(slate.pos[j])
        if counts[pos] >= ROSTER_MAX[pos]:
            continue
        after = counts.copy()
        after[pos] += 1
        if int(np.maximum(ROSTER_MIN - after, 0).sum()) <= 5 - len(picked):
            picked.append(int(j))
            counts = after
    return picked


# ----------------------------------------------------------------------
# waiting
# ----------------------------------------------------------------------


def test_survival_and_cost(engine):
    s = engine.slate
    rng = np.random.default_rng(8)
    st = DraftState(s)
    policy = engine.policy
    seats = policy.room_params(rng)
    times = policy.latent_market(rng)
    for _ in range(3):
        policy.choose(st, seats, times, rng)
    seat = st.seat_on_clock()
    legal = np.where(st.eligible_mask(seat))[0]
    values = s.proj
    cands = [int(j) for j in legal[np.argsort(-values[legal])][:4]]
    surv = survival_to_next_pick(policy, st, cands, cands[0], rng, n_sims=60)
    assert surv[cands[0]] == 1.0
    assert all(0.0 <= v <= 1.0 for v in surv.values())
    result = cost_of_waiting(policy, st, cands[1], values, rng, n_sims=60)
    assert result["intervening_picks"] >= 0
    assert 0.0 <= result["p_candidate_survives_and_legal"] <= 1.0


# ----------------------------------------------------------------------
# optimizer
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def optimizer(engine):
    cfg = OptimizerConfig(
        n_candidates=6,
        n_rollouts=12,
        n_outcome_sims=200,
        eval_field_entries=300,
        dup_field_entries=300,
        n_survival_sims=40,
        pair_top_k=4,
        seed=13,
    )
    return PickOptimizer(engine, TournamentModel(contest_size=50_000), cfg)


def test_recommend_single(optimizer, engine):
    rec = optimizer.recommend(DraftState(engine.slate))
    assert rec["mode"] == "single"
    assert rec["options"]
    pays = [o["expected_payout"] for o in rec["options"]]
    assert pays == sorted(pays, reverse=True)
    assert all(len(o["players"]) == 1 for o in rec["options"])
    assert "survival_to_next_pick" in rec["options"][0]


def test_recommend_pair_at_turn(optimizer, engine):
    st = DraftState(engine.slate)
    rng = np.random.default_rng(9)
    seats = engine.policy.room_params(rng)
    times = engine.policy.latent_market(rng)
    for _ in range(5):
        engine.policy.choose(st, seats, times, rng)
    assert st.next_pick == 6 and st.seat_on_clock() == 5
    rec = optimizer.recommend(st)
    assert rec["mode"] == "pair"
    assert all(len(o["players"]) == 2 for o in rec["options"])


# ----------------------------------------------------------------------
# assistant helpers
# ----------------------------------------------------------------------


def test_match_player(slate):
    assert slate.players[match_player(slate, "t0 qb1")].name == "T0 QB1"
    assert slate.players[match_player(slate, "T3 TE2")].name == "T3 TE2"
    with pytest.raises(KeyError):
        match_player(slate, "nobody at all")
    # Ambiguous partials resolve to the best (lowest) ADP, draft-style.
    ambiguous = match_player(slate, "QB1")
    assert slate.players[ambiguous].position == "QB"
    assert slate.adp[ambiguous] == min(
        slate.adp[i] for i, p in enumerate(slate.players) if "QB1" in p.name
    )

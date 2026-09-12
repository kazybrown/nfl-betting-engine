# Battle Royale draft engine

A tournament-equity model for Underdog's NFL **Battle Royale** format: six-person
snake drafts, six rounds, rosters of exactly QB / RB / WR / WR / TE / FLEX
(FLEX = RB/WR/TE), every entry pooled into one large weekly tournament under
Underdog half-PPR scoring.

Successor to the transferred "hurry_up" prototype series (v1/v2 + tasks 2–6),
unified into one package, refitted on real data, and extended with the layer
those prototypes lacked: an integrated pick optimizer.

## Decision philosophy

Priority ordering (from historical winner review — see the transfer notes):

1. tournament win probability / payout equity,
2. current value and ceiling,
3. cost of waiting / scarcity through the actual seats picking before our return,
4. useful QB–pass-catcher or game-environment correlation,
5. combinatorial leverage — **regularized by projection cost**, never manufactured.

Uniqueness enters the objective only through duplicate payout sharing at the top
of the field; the optimizer will not scroll past a clearly superior player to be
different.

## Layers

| module | what it does |
|---|---|
| `slate` | current-week rankings/simulations CSV (ETR/Underdog export format) |
| `marginals` | Gamma weekly-score marginals, mean = slate projection exactly; dispersion from fitted CV² curves + per-player ceiling tilt |
| `correlation` | Gaussian copula; same-team & opposing-team relationship correlations, PSD-projected |
| `opponents` | roster-state-aware field draft policy (market clock from ADP + room-drafted rate; need/FLEX/stacking terms; no projections — the field drafts the market) |
| `field` | simulated reference field; exact pair/triple/roster ownership and duplication |
| `waiting` | survival through the exact intervening seats; take-now vs wait branch comparison with common random numbers |
| `equity` | expected tournament payout vs the field: rank scaled to contest size, Beta-posterior tail quadrature for the jackpot region, exact-duplicate payout sharing |
| `optimizer` | per-pick recommendations via rollouts + equity; **joint two-pick optimization at the turns** (6+7, 18+19, 30+31) |
| `assistant` | live draft CLI (interactive or one-shot from pick history) |
| `calibration` | fitting + validation from nflverse weekly stats and real Battle Royale pick-by-pick archives |

## Calibration provenance

Fitted tables live in `battle_royale/data/` and are loaded automatically:

- `marginal_dispersion.json` — CV² vs expectation by position, fitted on
  2018–2025 nflverse weekly stats scored under exact Underdog rules, bucketed
  by a strictly lagged EWM anchor (no same-week leakage).
- `correlations.json` — relationship correlations from standardized same-game
  residual pairs (e.g. same-team QB–WR **+0.32**, n=7,337 pairs; QB–TE +0.27;
  opposing QB–QB +0.17), Fisher-shrunk toward the independent DK-grid priors
  from the transfer package. The two sources agree closely, which is a genuine
  out-of-family confirmation.

Opponent-policy concentration (`adp_sigma`, `choice_noise`) is calibrated so
that simulated exact-roster duplication matches real Battle Royale fields
(2023–2024 archives: 75–85% of entries on unique rosters at 28–68k entries),
not the transferred defaults, which produced an unrealistically diverse field.

## Validation status

Run `scripts/br_validate.py`; results in `reports/battle_royale/`.

- **Walk-forward, leakage-safe (40 folds through 2023–2025)**: central-interval
  coverage 0.500 @ 50% and 0.776 @ 80%; CRPS beats the pooled-position baseline
  (+0.8%); Gaussian-copula joint-tail error roughly half of independence
  (0.0022 vs 0.0040).
- **Real Battle Royale archives (9 weeks, 2023–2024, ~460k entries)**: contest
  sizes 28–68k; winning scores 125–185 (median ≈ 150); real ADP execution noise
  MAD-σ ≈ 3.1–3.6; FLEX construction 55–78% RB; exact-roster unique share
  0.75–0.85. The policy's next-pick top-3 hit rate matches a pure-ADP ordering
  (~0.63) — the roster-state terms matter for legality and late rounds, not for
  predicting early picks, and the archives confirm ADP dominates early.

What is **not** claimed: historical ROI/win-rate lift. That requires replaying
archived weeks against archived pre-kickoff projections, which are not
available. All validation here is distributional and structural.

## Usage

```bash
# one-shot recommendation mid-draft (you are seat 4, three picks made)
PYTHONPATH=. python -m battle_royale.assistant \
  --csv slate.csv --seat 4 --history "Jahmyr Gibbs, Ja'Marr Chase, Bijan Robinson"

# interactive live draft
PYTHONPATH=. python -m battle_royale.assistant --csv slate.csv --seat 4 --interactive --fast

# weekly report: field diagnostics, survival cliffs, stack ownership, guided drafts
PYTHONPATH=. python scripts/br_week_report.py --csv slate.csv --guided-drafts

# refit calibration tables (after downloading nflverse stats_player_week_*.parquet)
PYTHONPATH=. python scripts/br_fit_calibration.py --stats-dir DATA --seasons 2018 2025

# validation suite (nflverse walk-forward + Battle Royale archives)
PYTHONPATH=. python scripts/br_validate.py --stats-dir DATA --br-dir DATA/br
```

Historical inputs (not committed):

- nflverse weekly stats: `https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet`
- Underdog Battle Royale pick-by-pick archives (public GCS bucket `underdog-inc`,
  folders `underblog/NFL Battle Royale 2023/` and `underblog/2024_nfl_battle_royale/`);
  compact with `draft_id, draft_entry_id, player_name, player_id, position_name,
  projection_adp, source, pick_order, overall_pick_number, team_pick_number,
  player_points, roster_points` into `br_{season}_w{week}.parquet`.

## Live-draft checklist (what the optimizer evaluates every pick)

current value & ceiling → empirical marginal → teammate/opponent dependence →
legal construction → each intervening opponent's roster needs → candidate
survival to our next pick → tier opportunity cost → pair/triple combinatorial
ownership → exact-roster duplication. Consecutive turn picks are solved as a
pair, never independently.

# Battle Royale — Week 2, 2026 draft strategy

Model outputs from `scripts/br_week_report.py` on the Week 2 rankings/simulations
export (150 players, 12 games). Contest modeled at 70,000 entries (real 2024
Week 2 had 67,986). Full machine-readable detail: `week_report.json`.

## The one-paragraph version

Open with **Amon-Ra St. Brown or Jahmyr Gibbs** from any early seat (the model
slightly prefers St. Brown for completion upside; both are ~co-equal within
noise). The single biggest market inefficiency on the slate is **Ashton Jeanty
(rank 18, ADP 25.1, drafted in only 44% of rooms)** — he survives to pick 19
essentially always and to pick 25 in 92% of rooms, so he anchors rounds 4–5 in
almost every model-guided build. **Sam LaPorta (rank 23, ADP 28.4, 63% own)**
is the same trade at TE. Take QB late: Goff/Lawrence-tier QBs survive the
first turn 100% of the time, and five of the six model-guided drafts wait on
QB until round 4+. Stack when the cost is small (Goff + St. Brown/LaPorta
completions appear constantly), and prefer the cheap leverage the market
ignores over forced contrarianism — the equity objective already prices
duplication sharing.

## Survival cliffs (P(still available) when pick N is on the clock)

| player | pick 7 | 13 | 18 | 19 | 25 | 31 |
|---|---|---|---|---|---|---|
| Jonathan Taylor | 25% | 2% | — | — | — | — |
| De'Von Achane | 43% | 4% | — | — | — | — |
| Derrick Henry | 72% | 16% | 1% | — | — | — |
| Saquon Barkley | 83% | 23% | 2% | — | — | — |
| Chris Olave | 92% | 2% | — | — | — | — |
| Trey McBride | 100% | 27% | 1% | — | — | — |
| Omarion Hampton | 95% | 52% | 8% | 3% | — | — |
| Colston Loveland | 100% | 43% | 2% | 1% | — | — |
| Nico Collins | 100% | 73% | 7% | 4% | — | — |
| Tee Higgins | 100% | 96% | 47% | 30% | 4% | 4% |
| Joe Burrow | 100% | 99% | 66% | 54% | 7% | 7% |
| Zay Flowers | 100% | 100% | 91% | 84% | 15% | 14% |
| Tyler Warren | 100% | 100% | 98% | 93% | 15% | 4% |
| Ladd McConkey | 100% | 100% | 93% | 85% | 8% | 7% |
| **Ashton Jeanty** | 100% | 100% | 100% | 100% | **92%** | 60% |
| **Sam LaPorta** | 100% | 100% | 100% | 100% | **80%** | 38% |
| James Cook | 100% | 100% | 100% | 100% | 92% | 26% |
| Jalen Hurts | 100% | 100% | 100% | 100% | 38% | 17% |

Tactical reads:

- **The turn cliffs are brutal.** Ten opponents pick between 7→18 and 19→30.
  Anything with meaningful ADP pressure (Olave 92%→2%, McBride 100%→27%)
  cannot be waited on across a turn. Solve turn pairs jointly — the engine
  does this automatically (pair mode at picks 6+7, 18+19, 30+31).
- **Burrow is a turn decision**: 54% to survive to 19, 7% to 25. If you want
  the Bengals stack from a turn seat, pick 18/19 is the moment.
- **Jeanty and LaPorta are free rounds-4/5 anchors**; James Cook similar.
  Zay Flowers at 84% to pick 19 is the WR version.

## First-pick board (expected tournament payout, entry-fee multiples)

| pick 1 candidate | equity | P(top 1%) | completed-roster mean |
|---|---|---|---|
| Amon-Ra St. Brown | 1.58 | 2.5% | 85.3 |
| Jahmyr Gibbs | 1.49 | 2.1% | 87.2 |
| Ja'Marr Chase | 1.12 | 1.8% | 85.7 |
| Trey McBride | 1.11 | 1.2% | 75.0 |
| Jared Goff | 0.97 | 1.4% | 82.5 |
| Bijan Robinson | 0.87 | 1.4% | 84.8 |

St. Brown vs Gibbs is within noise; both are correct. Everything that
survives to pick 12 (QBs, TEs) is a value leak at 1.01 — the survival
discount is priced in.

## Model-guided drafts (one per seat; opponents = calibrated field policy)

| seat (picks) | roster |
|---|---|
| 0 (1,12,13,24,25,36) | Amon-Ra St. Brown, Omarion Hampton, Ashton Jeanty, Sam LaPorta, Caleb Williams, Luther Burden |
| 1 (2,11,14,23,26,35) | Amon-Ra St. Brown, Chris Olave, Ashton Jeanty, Sam LaPorta, Breece Hall, Jared Goff |
| 2 (3,10,15,22,27,34) | Jahmyr Gibbs, Joe Burrow, Zay Flowers, Tee Higgins, Ashton Jeanty, Michael Mayer |
| 3 (4,9,16,21,28,33) | Amon-Ra St. Brown, Jonathan Taylor, Sam LaPorta, Ashton Jeanty, Trevor Lawrence, Parker Washington |
| 4 (5,8,17,20,29,32) | Amon-Ra St. Brown, Jonathan Taylor, Sam LaPorta, Jared Goff, Breece Hall, Garrett Wilson |
| 5 (6,7,18,19,30,31) | Derrick Henry, Chris Olave, Ashton Jeanty, Sam LaPorta, Jared Goff, Jameson Williams |

Recurring structure: elite WR/RB early → Jeanty + LaPorta as the value core →
QB in rounds 4–6 (Goff/Lawrence/Caleb tier), often completing a Detroit or
Cincinnati game stack. Seat 2 shows the turn-oriented Bengals build (Burrow at
the wheel + Flowers bring-back + Higgins).

## Simulated field (what you're playing against)

- Room-drafted rates match the slate's Own% at 1.7% MAE (corr 0.998).
- FLEX construction: 81% RB / 18% WR / 2% TE.
- Stack ownership runs 1.8–2.5× independence for the chalk QB stacks
  (Hurts+Smith 5.9% of entries, Chase+Burrow 5.8%, Flowers+Lamar 5.7%) —
  calibrated against real 2023–2024 fields.
- Entry-score distribution: median 84, p99 134, sampled-field max ≈159
  (extrapolates to ≈165–175 at 70k entries; real winning scores at this size
  ran 125–185, median ≈150).

## Caveats

- Equity numbers use a generic top-heavy payout curve
  (`battle_royale/equity.py:DEFAULT_CURVE_POINTS`); swap in the actual Week 2
  structure for exact dollar EVs. Rankings are insensitive to reasonable
  curve choices; absolute equity multiples are not.
- The model anchors means to the slate projections; it does not re-project
  players. Garbage in, garbage out.
- No historical ROI claim is made — validation is distributional (walk-forward
  coverage/CRPS, copula joint tails) and structural (real-field duplication,
  stack affinity, ADP noise, construction).

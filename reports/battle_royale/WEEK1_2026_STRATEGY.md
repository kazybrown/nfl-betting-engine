# Battle Royale — Week 1, 2026 draft strategy

Model outputs from `scripts/br_week_report.py` on the Week 1 rankings/simulations
export (150 players, 12 games). Contest modeled at 70,000 entries (the
comparable real 2024 week had 67,986). Full machine-readable detail: `week_report.json`.

## The one-paragraph version

Open with **Jahmyr Gibbs or Amon-Ra St. Brown** from any early seat — they are
co-1A on tournament equity and clear of everything else. The single biggest
market inefficiency on the slate is **Ashton Jeanty (rank 18, ADP 25.1,
drafted in only 44% of rooms)** — he survives to pick 19 essentially always
and to pick 25 in 92% of rooms, so he anchors rounds 4–5 in most model-guided
builds; **Sam LaPorta (rank 23, ADP 28.4, 63% own)** is the same trade at TE.
Take QB late and take him as your leverage: the Goff/Lawrence tier survives
the first turn 100% of the time, every model draft waits on QB, and under the
winner-heavy payout structure the engine repeatedly lands on low-owned QB
stacks (Lawrence 21% own, Herbert 60%, even Shough 9%) attached to a chalk
skill core — which is exactly the shape of real historical Battle Royale
winners (strong RB/TE cores, QB as the concentrated leverage). Do not chase
uniqueness at RB/WR for its own sake: the equity objective already prices
duplication sharing, and it keeps choosing the chalk core anyway.

A note on the equity mechanics: under a jackpot-heavy payout curve, what
separates candidates at the very top is not raw ceiling (per-roster score
distributions are nearly identical across the top archetypes) but how often
your ceiling arrives when the field's does NOT. Fully-chalk builds spike
exactly when the field's thousands of near-copies spike, so they rarely clear
the field max; a differentiated stack spikes idiosyncratically. That — not a
uniqueness bonus — is why the model attaches leverage QBs to chalk cores.

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
| Jahmyr Gibbs | 10.9 | 2.3% | 88.0 |
| Amon-Ra St. Brown | 10.3 | 2.4% | 85.0 |
| Trey McBride | 6.5 | 1.1% | 81.0 |
| Trevor Lawrence | 5.8 | 1.4% | 77.6 |
| Jared Goff | 5.7 | 1.4% | 82.7 |
| Colston Loveland | 4.9 | 1.4% | 77.0 |
| Bijan Robinson | 4.3 | 1.5% | 85.2 |
| Ja'Marr Chase | 4.0 | 1.9% | 87.2 |

Gibbs vs St. Brown is within noise; both are correct and clearly ahead.
Absolute equity multiples depend on the (generic) payout curve; the ranking
is the decision-relevant output. Chase's placement is the differentiation
effect described above — his completions mirror the field's most common
cores, so his top-1% rate is elite but his beat-the-whole-field rate is not.
Taking a pick-12-surviving QB at pick 1 is still a value leak in practice:
the same QB is available at your next pick, so the comparison you actually
face at pick 1 is "Gibbs plus Goff at 12" versus "Goff plus the RB leftovers
at 12" — the guided drafts below show the engine never spends pick 1 on a QB
when it controls the whole draft.

## Model-guided drafts (one per seat; opponents = calibrated field policy)

| seat (picks) | roster |
|---|---|
| 0 (1,12,13,24,25,36) | Jahmyr Gibbs, Joe Burrow, Ashton Jeanty, Tetairoa McMillan, Emeka Egbuka, Juwan Johnson |
| 1 (2,11,14,23,26,35) | Amon-Ra St. Brown, Chris Olave, Ashton Jeanty, Tyler Shough, David Montgomery, Michael Mayer |
| 2 (3,10,15,22,27,34) | Jahmyr Gibbs, Jared Goff, Travis Etienne Jr., Garrett Wilson, Jameson Williams, Juwan Johnson |
| 3 (4,9,16,21,28,33) | Amon-Ra St. Brown, Jonathan Taylor, Sam LaPorta, Trevor Lawrence, Breece Hall, Parker Washington |
| 4 (5,8,17,20,29,32) | Chris Olave, Nico Collins, Ashton Jeanty, Breece Hall, Juwan Johnson, Tyler Shough |
| 5 (6,7,18,19,30,31) | Chris Olave, Derrick Henry, Omarion Hampton, Justin Herbert, Tetairoa McMillan, Michael Mayer |

Recurring structure: elite RB/WR chalk early → Jeanty (and often LaPorta) as
the value core → a low-owned QB in rounds 4–6, frequently completing a game
environment (Gibbs+Burrow bring-back; Goff+Jamo Detroit stack; Olave+Shough+
Juwan Johnson is the unowned Saints side of the DET–NO game; Lawrence+Parker
Washington the Jacksonville stack). Note the DET–NO game appears in some form
in five of six builds — the model wants that environment, from either side.

## Simulated field (what you're playing against)

- Room-drafted rates match the slate's Own% at 1.7% MAE (corr 0.998).
- FLEX construction: 81% RB / 18% WR / 2% TE.
- Stack ownership runs 1.8–2.5× independence for the chalk QB stacks
  (Hurts+Smith 5.9% of entries, Chase+Burrow 5.8%, Flowers+Lamar 5.7%) —
  calibrated against real 2023–2024 fields.
- Entry-score distribution: median 84, p99 132, sampled-field max ≈156
  (extrapolates to ≈165–175 at 70k entries; real winning scores at this size
  ran 125–185, median ≈150).

## Caveats

- Equity numbers use a generic top-heavy payout curve
  (`battle_royale/equity.py:DEFAULT_CURVE_POINTS`); swap in the actual weekly contest
  structure for exact dollar EVs. Rankings are insensitive to reasonable
  curve choices; absolute equity multiples are not.
- The model anchors means to the slate projections; it does not re-project
  players. Garbage in, garbage out.
- No historical ROI claim is made — validation is distributional (walk-forward
  coverage/CRPS, copula joint tails) and structural (real-field duplication,
  stack affinity, ADP noise, construction).

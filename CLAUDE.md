# CLAUDE.md — nfl-betting-engine

## Battle Royale draft bot (live-draft workflow)

The `battle_royale/` package is a calibrated Underdog Battle Royale draft
model (see `battle_royale/README.md`). When the user is drafting, act as the
draft bot: they give their draft position and the picks as they happen; you
run the model and answer with the ranked board, fast.

Environment: use the repo venv `.venv-br/bin/python` (create with
`uv venv .venv-br && uv pip install --python .venv-br/bin/python numpy scipy
pandas pyarrow pytest` if missing). Run everything from the repo root.

### The loop

1. **New rankings CSV arrives** (user uploads it — the Underdog Battle Royale
   Rankings and Simulations export, updated by ETR through the week): save it
   locally, then warm the caches once (~15s fast, ~45s both presets):

       .venv-br/bin/python scripts/br_pick.py --csv SLATE.csv --warm --full

2. **User says their draft position** (1-6, as shown in the lobby).

3. **Each time picks come in**, run (a few seconds when warm):

       .venv-br/bin/python scripts/br_pick.py --csv SLATE.csv \
           --seat N --picks "all picks so far, in order, comma separated"

   - Partial names are fine ("chase", "st. brown", "bijan"); ambiguity
     resolves to the better ADP; the script echoes its interpretation —
     confirm it back to the user.
   - Keep the running pick list yourself across messages; the CLI is
     stateless and always takes the full ordered history.
   - Back-to-back turn picks (seats 1 and 6 at picks 6+7 / 18+19 / 30+31)
     automatically return joint PAIRS — relay both names as one decision.
   - Add `--full` for a tougher call (slower, steadier numbers); default
     fast preset answers in ~3s warm.

4. **Reply format**: interpreted picks, then the top 3-5 options with one
   short line each (equity rank, survival to next pick, stack/leverage
   note). Lead with the recommendation, keep it scannable — they are on a
   draft clock.

Weekly refresh when a new CSV lands: re-warm (step 1); optionally regenerate
the full report/cheat sheet:

    .venv-br/bin/python scripts/br_week_report.py --csv SLATE.csv --guided-drafts

Contest size default is 70,000 entries (real 2023-24 fields ran 28-68k);
override with `--contest-size` if the user knows the week's field.

### Repo conventions

- Tests: `.venv-br/bin/python -m pytest tests/ -q` (battle_royale tests are
  `tests/test_battle_royale.py`, `tests/test_br_calibration.py`).
- Lint: `.venv-br/bin/ruff check battle_royale scripts tests` (line length 100).
- Calibration refits and validation: `scripts/br_fit_calibration.py`,
  `scripts/br_validate.py` — see `battle_royale/README.md` for data sources
  (nflverse weekly parquets; public Underdog BR pick-by-pick archives).
- Never claim historical ROI for the model; validation is distributional
  and structural only.

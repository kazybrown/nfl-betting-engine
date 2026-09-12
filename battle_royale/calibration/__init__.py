"""Calibration and validation from historical data.

- :mod:`.weekly_data` — nflverse weekly stats under exact Underdog scoring,
  with a strictly-lagged preweek strength anchor (no leakage).
- :mod:`.fit_marginals` — position-level dispersion (CV^2) curve fitting.
- :mod:`.fit_correlations` — relationship correlation fitting with shrinkage.
- :mod:`.walkforward` — chronological walk-forward validation of the marginal
  and dependence layers (CRPS / pinball / coverage / joint tails).
- :mod:`.battle_royale_data` — calibration against real Underdog Battle
  Royale pick-by-pick archives (ADP noise, construction, duplication,
  winning scores, opponent-policy rank diagnostics).
"""

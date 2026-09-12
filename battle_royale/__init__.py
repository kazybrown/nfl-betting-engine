"""Underdog Battle Royale draft engine.

A tournament-equity draft model for Underdog's Battle Royale format:
six-person snake drafts, six rounds, rosters of QB/RB/WR/WR/TE/FLEX,
all entries pooled into one large weekly tournament.

Layers (each usable on its own):

- :mod:`battle_royale.slate` — current-week player slate (rankings CSV).
- :mod:`battle_royale.marginals` — per-player weekly score distributions,
  mean-anchored to slate projections with empirically fitted dispersion.
- :mod:`battle_royale.correlation` — teammate/game dependence via a
  Gaussian copula with empirically fitted relationship correlations.
- :mod:`battle_royale.opponents` — roster-state-aware opponent draft policy.
- :mod:`battle_royale.field` — simulated contest field, combinatorial
  ownership and exact-roster duplication analytics.
- :mod:`battle_royale.waiting` — pick survival and cost-of-waiting.
- :mod:`battle_royale.equity` — tournament payout equity for rosters.
- :mod:`battle_royale.optimizer` — integrated pick recommendations,
  including joint optimization of back-to-back turn picks.
- :mod:`battle_royale.assistant` — live draft assistant CLI.
- :mod:`battle_royale.calibration` — data fitting and walk-forward
  validation from nflverse weekly stats under exact Underdog scoring.
"""

from .constants import POSITIONS, ROSTER_MAX, ROSTER_MIN, ROSTER_SIZE, SEATS
from .correlation import CorrelationModel
from .draft import DraftState
from .engine import BattleRoyaleEngine
from .equity import PayoutCurve, TournamentModel, load_payout_table
from .field import FieldAnalytics
from .formats import BATTLE_ROYALE, ContestFormat, get_format, load_formats
from .marginals import MarginalModel
from .opponents import OpponentPolicy
from .optimizer import OptimizerConfig, PickOptimizer
from .slate import Slate

__all__ = [
    "BATTLE_ROYALE",
    "POSITIONS",
    "ROSTER_MAX",
    "ROSTER_MIN",
    "ROSTER_SIZE",
    "SEATS",
    "BattleRoyaleEngine",
    "ContestFormat",
    "CorrelationModel",
    "DraftState",
    "FieldAnalytics",
    "MarginalModel",
    "OpponentPolicy",
    "OptimizerConfig",
    "PayoutCurve",
    "PickOptimizer",
    "Slate",
    "TournamentModel",
    "get_format",
    "load_formats",
    "load_payout_table",
]

"""Tests for calibration utilities: scoring, anchors, leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battle_royale.calibration.fit_correlations import fisher_shrink
from battle_royale.calibration.weekly_data import add_preweek_anchor, underdog_points


def test_underdog_points_hand_computed():
    df = pd.DataFrame(
        [
            {
                "passing_yards": 300,
                "passing_tds": 2,
                "passing_interceptions": 1,
                "rushing_yards": 20,
                "rushing_tds": 0,
                "receiving_yards": 0,
                "receiving_tds": 0,
                "receptions": 0,
                "passing_2pt_conversions": 1,
                "rushing_2pt_conversions": 0,
                "receiving_2pt_conversions": 0,
                "rushing_fumbles_lost": 1,
                "receiving_fumbles_lost": 0,
                "sack_fumbles_lost": 0,
            }
        ]
    )
    # 12 + 8 - 1 + 2 + 0 + 2 - 2 = 21
    assert underdog_points(df).iloc[0] == pytest.approx(21.0)


def test_underdog_points_receiver():
    df = pd.DataFrame(
        [{"receiving_yards": 110, "receiving_tds": 1, "receptions": 8, "rushing_yards": 5}]
    )
    # 11 + 6 + 4 + 0.5 = 21.5
    assert underdog_points(df).iloc[0] == pytest.approx(21.5)


def test_anchor_is_strictly_lagged():
    rows = []
    for week in range(1, 9):
        rows.append(
            {
                "player_id": "p1",
                "player_name": "P One",
                "position": "WR",
                "season": 2024,
                "week": week,
                "team": "A",
                "opponent_team": "B",
                "points": 10.0 + week,
                "time_id": 202400 + week,
            }
        )
    # A second player so the position prior exists for p1's first row.
    for week in range(1, 9):
        rows.append(
            {
                "player_id": "p2",
                "player_name": "P Two",
                "position": "WR",
                "season": 2024,
                "week": week,
                "team": "B",
                "opponent_team": "A",
                "points": 8.0,
                "time_id": 202400 + week,
            }
        )
    df = pd.DataFrame(rows)
    base = add_preweek_anchor(df)
    # Shock the final week's outcome: no earlier anchor may change.
    df2 = df.copy()
    df2.loc[(df2["player_id"] == "p1") & (df2["week"] == 8), "points"] = 99.0
    shocked = add_preweek_anchor(df2)
    b = base[(base["player_id"] == "p1")].sort_values("week")["anchor"].to_numpy()
    s = shocked[(shocked["player_id"] == "p1")].sort_values("week")["anchor"].to_numpy()
    assert np.allclose(b, s)  # week-8 outcome never leaks into any anchor


def test_fisher_shrink_behavior():
    # Tiny samples collapse to the prior; huge samples keep the estimate.
    assert fisher_shrink(0.9, 5, 0.1) == pytest.approx(0.1, abs=0.02)
    assert fisher_shrink(0.5, 100_000, 0.0) == pytest.approx(0.5, abs=0.01)

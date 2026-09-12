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


def test_lines_from_frame_implied_totals():
    from battle_royale.external import lines_from_frame

    games = pd.DataFrame(
        [
            {"season": 2026, "week": 1, "home_team": "DET", "away_team": "NO",
             "spread_line": 7.0, "total_line": 49.5},
            {"season": 2026, "week": 1, "home_team": "CAR", "away_team": "CHI",
             "spread_line": -3.0, "total_line": 47.5},
            {"season": 2026, "week": 2, "home_team": "LA", "away_team": "WSH",
             "spread_line": 1.0, "total_line": 40.0},
        ]
    )
    lines = lines_from_frame(games, 2026, 1)
    assert lines["DET"]["implied"] == pytest.approx(28.2, abs=0.1)
    assert lines["NO"]["implied"] == pytest.approx(21.2, abs=0.1)
    assert lines["CHI"]["implied"] > lines["CAR"]["implied"]  # road favorite
    assert "LAR" not in lines  # week 2 filtered out
    lines2 = lines_from_frame(games, 2026, 2)
    assert "LAR" in lines2 and "WAS" in lines2  # aliases normalized


def test_status_from_frames_precedence():
    from battle_royale.external import status_from_frames

    inj = pd.DataFrame(
        [
            {"week": 1, "full_name": "Some Guy", "team": "DET", "report_status": "Questionable"},
            {"week": 1, "full_name": "Hurt Man", "team": "NO", "report_status": "Out"},
            {"week": 2, "full_name": "Future Case", "team": "NO", "report_status": "Out"},
        ]
    )
    ros = pd.DataFrame(
        [
            {"week": 1, "full_name": "Some Guy", "team": "DET", "status": "ACT"},
            {"week": 1, "full_name": "Stashed Vet", "team": "CHI", "status": "RES"},
        ]
    )
    status = status_from_frames(inj, ros, week=1)
    assert status[("some guy", "DET")] == "Q"
    assert status[("hurt man", "NO")] == "O"
    assert status[("stashed vet", "CHI")] == "IR"
    assert ("future case", "NO") not in status


def test_game_env_scales_correlations():
    from battle_royale.correlation import CorrelationModel
    from tests.test_battle_royale import synthetic_slate

    slate = synthetic_slate()
    base = CorrelationModel.from_slate(slate)
    env = {t: (1.15 if t in ("T0", "T1") else 0.9) for t in set(slate.teams)}
    tilted = CorrelationModel.from_slate(slate, team_env=env)
    qb = slate.name_to_idx["T0 QB1"]
    wr = slate.name_to_idx["T0 WR1"]
    lo_qb = slate.name_to_idx["T4 QB1"]
    lo_wr = slate.name_to_idx["T4 WR1"]
    assert tilted.matrix[qb, wr] > base.matrix[qb, wr]
    assert tilted.matrix[lo_qb, lo_wr] < base.matrix[lo_qb, lo_wr]

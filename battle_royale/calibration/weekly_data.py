"""Historical weekly player data under exact Underdog scoring.

Loads nflverse ``stats_player_week_{season}.parquet`` files, recomputes
Underdog fantasy points from raw stat columns, and builds a strictly
preweek player-strength anchor (lagged EWM with position fallback) so that
no row's own outcome can influence its anchor — the leakage guard for all
downstream fitting and validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")

# Anchor floors: rows below these are not Battle-Royale-relevant and only add
# noise to dispersion/correlation fits aimed at draftable players.
ANCHOR_FLOORS = {"QB": 10.0, "RB": 7.0, "WR": 7.0, "TE": 5.0}


def underdog_points(df: pd.DataFrame) -> pd.Series:
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    interceptions = col("passing_interceptions")
    if "passing_interceptions" not in df.columns:
        interceptions = col("interceptions")
    fumbles_lost = (
        col("rushing_fumbles_lost") + col("receiving_fumbles_lost") + col("sack_fumbles_lost")
    )
    if (
        "rushing_fumbles_lost" not in df.columns
        and "fumbles_lost" in df.columns
    ):
        fumbles_lost = col("fumbles_lost")
    return (
        0.04 * col("passing_yards")
        + 4.0 * col("passing_tds")
        - 1.0 * interceptions
        + 0.10 * col("rushing_yards")
        + 6.0 * col("rushing_tds")
        + 0.10 * col("receiving_yards")
        + 6.0 * col("receiving_tds")
        + 0.50 * col("receptions")
        + 2.0
        * (
            col("passing_2pt_conversions")
            + col("rushing_2pt_conversions")
            + col("receiving_2pt_conversions")
        )
        - 2.0 * fumbles_lost
    )


def load_weekly(stats_dir: str | Path, seasons: list[int]) -> pd.DataFrame:
    """Load and normalize weekly rows for QB/RB/WR/TE, regular season only."""
    frames = []
    for season in seasons:
        path = Path(stats_dir) / f"stats_player_week_{season}.parquet"
        d = pd.read_parquet(path)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["position"].isin(POSITIONS)].copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper().isin(["REG"])].copy()
    df["points"] = underdog_points(df)
    keep = [
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "team",
        "opponent_team",
        "points",
    ]
    df = df[keep].rename(columns={"player_display_name": "player_name"})
    df["time_id"] = df["season"].astype(int) * 100 + df["week"].astype(int)
    df = (
        df.sort_values(["time_id", "player_id"])
        .drop_duplicates(["time_id", "player_id"], keep="last")
        .reset_index(drop=True)
    )
    return df


def add_preweek_anchor(
    df: pd.DataFrame, ewm_alpha: float = 0.35, min_player_games: int = 2
) -> pd.DataFrame:
    """Strictly lagged expectation anchor per row (no same-week leakage)."""
    out = df.sort_values(["time_id", "player_id"]).reset_index(drop=True)

    def _lagged_ewm(s: pd.Series) -> pd.Series:
        return s.shift(1).ewm(alpha=ewm_alpha, adjust=False, min_periods=1).mean()

    out["_player_prior"] = out.groupby("player_id", group_keys=False)["points"].apply(_lagged_ewm)
    out["_pos_prior"] = (
        out.groupby("position", group_keys=False)["points"]
        .apply(lambda s: s.expanding(min_periods=1).mean().shift(1))
    )
    out["_prior_games"] = out.groupby("player_id").cumcount()
    use_player = (out["_prior_games"] >= min_player_games) & out["_player_prior"].notna()
    out["anchor"] = np.where(use_player, out["_player_prior"], out["_pos_prior"])
    out = out[out["anchor"].notna()].copy()
    out["anchor"] = out["anchor"].clip(lower=0.5)
    return out.drop(columns=["_player_prior", "_pos_prior", "_prior_games"])


def relevant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows above position-specific anchor floors (draftable-caliber weeks)."""
    floor = df["position"].map(ANCHOR_FLOORS)
    return df[df["anchor"] >= floor].copy()

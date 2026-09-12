"""External weekly context: Vegas lines and player availability status.

Two feeds, both from nflverse releases (no auth):

- **Game lines** (`schedules/games.csv`): spread and total per game. Implied
  team totals tilt the game-environment correlations (a 50.5-total game
  carries more shared ceiling than a 38.5 one) and annotate the draft board.
- **Player status** (`injuries/injuries_{season}.parquet` +
  `weekly_rosters/roster_weekly_{season}.parquet`): late scratches and
  designations, so a stale rankings CSV cannot sell an Out player.

Loaders are split from parsers so parsing is unit-testable without network.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import DEFAULT_CACHE_DIR
from .slate import Slate

SCHEDULES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
INJURIES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/"
    "injuries_{season}.parquet"
)
ROSTERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
    "roster_weekly_{season}.parquet"
)

# games.csv team codes -> Underdog slate codes.
TEAM_ALIASES = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX"}

# Correlation tilt per point of game total away from the slate median,
# bounded so Vegas can shade the fitted correlations, never rewrite them.
GAME_ENV_SLOPE = 0.04
GAME_ENV_CLIP = (0.85, 1.18)


def _fetch(url: str, dest: Path, max_age_hours: float = 6.0) -> Path | None:
    """Download to cache unless a fresh copy exists; None on failure."""
    import time

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_hours * 3600:
        return dest
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "120", "-o", str(dest), url], capture_output=True
    )
    if r.returncode == 0 and dest.exists() and dest.stat().st_size > 1000:
        return dest
    return dest if dest.exists() and dest.stat().st_size > 1000 else None


# ---------------------------------------------------------------------------
# Vegas lines
# ---------------------------------------------------------------------------


def lines_from_frame(games: pd.DataFrame, season: int, week: int) -> dict[str, dict]:
    """Per-team game lines: {team: {opp, total, spread, implied}}.

    ``spread_line`` in games.csv is the home team's expected margin (positive
    = home favored), so implied_home = (total + spread) / 2.
    """
    w = games[(games["season"] == season) & (games["week"] == week)]
    out: dict[str, dict] = {}
    for row in w.itertuples():
        total = getattr(row, "total_line", None)
        spread = getattr(row, "spread_line", None)
        if total is None or pd.isna(total):
            continue
        spread = 0.0 if spread is None or pd.isna(spread) else float(spread)
        home = TEAM_ALIASES.get(str(row.home_team), str(row.home_team))
        away = TEAM_ALIASES.get(str(row.away_team), str(row.away_team))
        implied_home = (float(total) + spread) / 2.0
        implied_away = (float(total) - spread) / 2.0
        out[home] = {"opp": away, "total": float(total), "spread": spread,
                     "implied": round(implied_home, 1)}
        out[away] = {"opp": home, "total": float(total), "spread": -spread,
                     "implied": round(implied_away, 1)}
    return out


def load_game_lines(
    season: int, week: int, cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> dict[str, dict]:
    """Fetch and parse game lines; {} when the feed is unreachable."""
    path = _fetch(SCHEDULES_URL, Path(cache_dir) / "games.csv")
    if path is None:
        return {}
    return lines_from_frame(pd.read_csv(path, low_memory=False), season, week)


def game_total_multipliers(slate: Slate, lines: dict[str, dict]) -> dict[str, float]:
    """Per-team same-game correlation multiplier from the game total.

    Reference point is the median total among the slate's games so the tilt
    is relative to this week's scoring environment.
    """
    totals = [lines[t]["total"] for t in set(slate.teams) if t in lines]
    if not totals:
        return {}
    ref = float(np.median(totals))
    out = {}
    for t in set(slate.teams):
        if t in lines:
            mult = 1.0 + GAME_ENV_SLOPE * (lines[t]["total"] - ref)
            out[t] = float(np.clip(mult, *GAME_ENV_CLIP))
    return out


# ---------------------------------------------------------------------------
# Player status
# ---------------------------------------------------------------------------

_STATUS_RANK = {"O": 3, "IR": 3, "D": 2, "Q": 1}


def _norm_name(name: str) -> str:
    s = re.sub(r"[.'\-]", "", str(name).lower())
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s.strip())
    return re.sub(r"\s+", " ", s)


def status_from_frames(
    injuries: pd.DataFrame | None, rosters: pd.DataFrame | None, week: int
) -> dict[tuple[str, str], str]:
    """{(normalized name, team): code} with code in O/D/Q/IR.

    Injury report designations (Out/Doubtful/Questionable) take precedence;
    roster status RES/PUP/SUS/INA/NON maps to IR-style unavailability.
    """
    out: dict[tuple[str, str], str] = {}

    def put(name, team, code):
        key = (_norm_name(name), str(team))
        if _STATUS_RANK.get(code, 0) >= _STATUS_RANK.get(out.get(key, ""), 0):
            out[key] = code

    if rosters is not None and len(rosters):
        r = rosters[rosters["week"] == week] if "week" in rosters.columns else rosters
        bad = r[r["status"].isin(["RES", "PUP", "SUS", "INA", "NON", "EXE"])]
        for row in bad.itertuples():
            put(row.full_name, row.team, "IR")

    if injuries is not None and len(injuries):
        i = injuries[injuries["week"] == week] if "week" in injuries.columns else injuries
        codes = {"Out": "O", "Doubtful": "D", "Questionable": "Q"}
        for row in i.itertuples():
            code = codes.get(str(getattr(row, "report_status", "")), None)
            if code:
                put(row.full_name, row.team, code)
    return out


def load_player_status(
    season: int, week: int, cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> dict[tuple[str, str], str]:
    inj_path = _fetch(
        INJURIES_URL.format(season=season), Path(cache_dir) / f"injuries_{season}.parquet"
    )
    ros_path = _fetch(
        ROSTERS_URL.format(season=season), Path(cache_dir) / f"rosters_{season}.parquet"
    )
    injuries = pd.read_parquet(inj_path) if inj_path else None
    rosters = pd.read_parquet(ros_path) if ros_path else None
    return status_from_frames(injuries, rosters, week)


def slate_status(slate: Slate, status: dict[tuple[str, str], str]) -> dict[int, str]:
    """Map status codes onto slate player indices (name+team, name fallback)."""
    by_name: dict[str, list[tuple[str, str]]] = {}
    for (name, team), code in status.items():
        by_name.setdefault(name, []).append((team, code))
    out: dict[int, str] = {}
    for i, p in enumerate(slate.players):
        n = _norm_name(p.name)
        hits = by_name.get(n, [])
        exact = [c for t, c in hits if t == p.team]
        if exact:
            out[i] = exact[0]
        elif len(hits) == 1:
            out[i] = hits[0][1]
    return out


# Approximate historical play/sit rates by designation: Questionable players
# play ~75-80% of the time, Doubtful ~10-15%; Out/inactive is near-certain.
SIT_PROB = {"Q": 0.22, "D": 0.85, "O": 0.97, "IR": 0.97}


def sit_probabilities(slate: Slate, statuses: dict[int, str]) -> np.ndarray:
    """Per-player inactive probability vector for the engine's sit mixture."""
    out = np.zeros(slate.n)
    for i, code in statuses.items():
        out[i] = SIT_PROB.get(code, 0.0)
    return out

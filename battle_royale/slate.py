"""Current-week player slate loaded from a Battle Royale rankings/simulations CSV.

Expected columns (the Underdog Battle Royale export format supplied weekly):
Player, Position, Team, Opponent, Rank, ADP, Proj, Ceiling, Pos Rank,
Optimal Rate, Optimal IF QB Optimal, Optimal IF Opp QB Optimal,
Top 5 At Pos Rate, Own %, id.

``Own %`` is interpreted as the room-drafted rate: the share of six-person
rooms in which the player is selected (a player can appear at most once per
room, so this is per-room, not per-entry, ownership).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import POS_INDEX, POSITIONS


def _pct(x: str | float | None) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    return float(s.rstrip("%")) / 100.0


@dataclass(frozen=True)
class Player:
    name: str
    position: str
    team: str
    opponent: str
    rank: int
    adp: float
    proj: float
    ceiling: float
    optimal_rate: float
    optimal_if_qb: float
    optimal_if_opp_qb: float
    top5_at_pos: float
    room_drafted_rate: float
    player_id: str = ""


@dataclass
class Slate:
    """Array-backed slate; index order is the CSV row order."""

    players: list[Player]
    name_to_idx: dict[str, int] = field(init=False)
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.players)
        self.name_to_idx = {p.name: i for i, p in enumerate(self.players)}
        if len(self.name_to_idx) != self.n:
            seen: set[str] = set()
            dupes = [p.name for p in self.players if p.name in seen or seen.add(p.name)]
            raise ValueError(f"duplicate player names in slate: {dupes}")
        self.rank = np.array([p.rank for p in self.players], dtype=float)
        self.adp = np.array([p.adp for p in self.players], dtype=float)
        self.proj = np.array([p.proj for p in self.players], dtype=float)
        self.ceiling = np.array([p.ceiling for p in self.players], dtype=float)
        self.optimal_rate = np.array([p.optimal_rate for p in self.players], dtype=float)
        self.optimal_if_qb = np.array([p.optimal_if_qb for p in self.players], dtype=float)
        self.optimal_if_opp_qb = np.array(
            [p.optimal_if_opp_qb for p in self.players], dtype=float
        )
        self.room_drafted_rate = np.array(
            [p.room_drafted_rate for p in self.players], dtype=float
        )
        self.pos = np.array([POS_INDEX[p.position] for p in self.players], dtype=np.int8)
        self.teams = np.array([p.team for p in self.players], dtype=object)
        self.opps = np.array([p.opponent for p in self.players], dtype=object)

    REQUIRED_COLUMNS = ("Player", "Position", "Team", "Opponent", "Rank", "ADP",
                        "Proj", "Ceiling", "Own %")

    @classmethod
    def from_csv(cls, path: str | Path) -> Slate:
        players: list[Player] = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            missing = [c for c in cls.REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                # Own % drives the entire opponent model; silently defaulting
                # any of these to zero would corrupt the field simulation.
                raise ValueError(f"slate CSV {path} is missing columns: {missing}")
            for row in reader:
                position = row["Position"].strip().upper()
                if position not in POSITIONS:
                    continue
                players.append(
                    Player(
                        name=row["Player"].strip(),
                        position=position,
                        team=row["Team"].strip(),
                        opponent=row["Opponent"].strip(),
                        rank=int(row["Rank"]),
                        adp=float(row["ADP"]),
                        proj=float(row["Proj"]),
                        ceiling=float(row["Ceiling"]),
                        optimal_rate=_pct(row.get("Optimal Rate")),
                        optimal_if_qb=_pct(row.get("Optimal IF QB Optimal")),
                        optimal_if_opp_qb=_pct(row.get("Optimal IF Opp QB Optimal")),
                        top5_at_pos=_pct(row.get("Top 5 At Pos Rate")),
                        room_drafted_rate=_pct(row.get("Own %")),
                        player_id=(row.get("id") or "").strip(),
                    )
                )
        if not players:
            raise ValueError(f"no usable player rows found in {path}")
        return cls(players)

    def idx(self, player: str | int) -> int:
        """Resolve a player name or index to an index."""
        if isinstance(player, str):
            try:
                return self.name_to_idx[player]
            except KeyError:
                raise KeyError(f"unknown player: {player!r}") from None
        return int(player)

    def names(self, idxs) -> list[str]:
        return [self.players[int(i)].name for i in idxs]

    def same_team(self, i: int, j: int) -> bool:
        return self.players[i].team == self.players[j].team

    def same_game(self, i: int, j: int) -> bool:
        a, b = self.players[i], self.players[j]
        return a.team == b.team or a.team == b.opponent or b.team == a.opponent

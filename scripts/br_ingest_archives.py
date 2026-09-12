"""Download Underdog Battle Royale pick-by-pick archives and compact to parquet.

The archives are public CSVs (~70-185MB each) on the ``underdog-inc`` GCS
bucket; this keeps only the columns the calibration layer needs (~3-7MB/week).

Usage:
    python scripts/br_ingest_archives.py --out DATA/br [--weeks 2024:1 2024:2 2023:9]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import subprocess
from pathlib import Path

import pandas as pd

B23 = "https://storage.googleapis.com/underdog-inc/underblog/NFL%20Battle%20Royale%202023"
B24 = "https://storage.googleapis.com/underdog-inc/underblog/2024_nfl_battle_royale"

# 2024 files are keyed by slate date.
DATES_2024 = {
    1: "2024-09-07", 2: "2024-09-15", 3: "2024-09-22", 4: "2024-09-29",
    5: "2024-10-06", 6: "2024-10-13", 7: "2024-10-20", 8: "2024-10-27",
    9: "2024-11-03", 10: "2024-11-10", 11: "2024-11-17", 12: "2024-11-24",
    13: "2024-12-01", 14: "2024-12-08", 15: "2024-12-15", 16: "2024-12-22",
    17: "2024-12-29", 18: "2025-01-05",
}

KEEP = [
    "draft_id", "draft_entry_id", "player_name", "player_id", "position_name",
    "projection_adp", "source", "pick_order", "overall_pick_number",
    "team_pick_number", "player_points", "roster_points",
]


def url_for(season: int, week: int) -> str:
    if season == 2023:
        return f"{B23}/Battle_Royale_NFL_2023_Week_{week}.csv"
    if season == 2024:
        return f"{B24}/Battle_Royale_NFL_{DATES_2024[week]}_Battle%20Royale%20-%20Week%20{week}.csv"
    raise ValueError(f"no known archive for season {season}")


def ingest(season: int, week: int, out_dir: Path) -> None:
    out_pq = out_dir / f"br_{season}_w{week:02d}.parquet"
    if out_pq.exists():
        print(f"[skip] {out_pq.name}")
        return
    tmp = out_dir / f"_dl_{season}_{week}.csv"
    url = url_for(season, week)
    print(f"[dl] {season} week {week} ...", flush=True)
    subprocess.run(
        ["curl", "-sL", "--max-time", "1800", "-o", str(tmp), url], check=True
    )
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink()
        raise RuntimeError(f"download too small for {url}")
    chunks = []
    for ch in pd.read_csv(tmp, chunksize=500_000, low_memory=False):
        chunks.append(ch[[c for c in KEEP if c in ch.columns]])
    d = pd.concat(chunks, ignore_index=True)
    for c in ("projection_adp", "player_points", "roster_points"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    for c in ("pick_order", "overall_pick_number", "team_pick_number"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("Int32")
    d.to_parquet(out_pq, index=False)
    tmp.unlink()
    print(f"[ok] {out_pq.name}: {len(d)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/br")
    ap.add_argument(
        "--weeks",
        nargs="+",
        default=["2023:1", "2023:9", "2023:18", "2024:1", "2024:2", "2024:5",
                 "2024:9", "2024:14", "2024:18"],
        help="season:week entries",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.weeks:
        season, week = (int(x) for x in spec.split(":"))
        ingest(season, week, out_dir)


if __name__ == "__main__":
    main()

"""Fit Battle Royale marginal-dispersion and correlation tables.

Usage:
    python scripts/br_fit_calibration.py --stats-dir DATA --seasons 2018 2025 \
        [--out battle_royale/data]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import json
from pathlib import Path

from battle_royale.calibration import fit_correlations, fit_marginals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--seasons", nargs=2, type=int, default=[2018, 2025])
    ap.add_argument("--out", default="battle_royale/data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seasons = list(range(args.seasons[0], args.seasons[1] + 1))

    disp = fit_marginals.main(args.stats_dir, seasons, out / "marginal_dispersion.json")
    print("dispersion buckets:")
    for pos, block in disp["positions"].items():
        cells = ", ".join(
            f"{b['anchor']:.1f}->cv2 {b['cv2']:.3f} (n={b['n']})" for b in block["buckets"]
        )
        print(f"  {pos}: {cells}")

    corr = fit_correlations.main(args.stats_dir, seasons, out / "correlations.json")
    print(json.dumps({k: corr[k] for k in ("same_team", "opp_team")}, indent=2))


if __name__ == "__main__":
    main()

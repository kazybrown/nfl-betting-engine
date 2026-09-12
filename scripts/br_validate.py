"""Validation runs: walk-forward (nflverse) + real Battle Royale archives.

Usage:
    python scripts/br_validate.py --stats-dir DATA --br-dir DATA/br \
        --out reports/battle_royale [--max-folds 40] [--skip-walkforward]
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root

import argparse
import json
from pathlib import Path

from battle_royale.calibration import battle_royale_data as brd
from battle_royale.calibration.walkforward import walk_forward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-dir", required=True)
    ap.add_argument("--br-dir", default=None)
    ap.add_argument("--out", default="reports/battle_royale")
    ap.add_argument("--seasons", nargs=2, type=int, default=[2018, 2025])
    ap.add_argument("--max-folds", type=int, default=40)
    ap.add_argument("--min-train-weeks", type=int, default=60)
    ap.add_argument("--skip-walkforward", action="store_true")
    ap.add_argument("--replay-rooms", type=int, default=600)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_walkforward:
        seasons = list(range(args.seasons[0], args.seasons[1] + 1))
        wf = walk_forward(
            args.stats_dir,
            seasons,
            min_train_weeks=args.min_train_weeks,
            max_folds=args.max_folds,
        )
        (out / "walkforward.json").write_text(json.dumps(wf, indent=2))
        m = wf["marginal"]
        print(
            f"walk-forward: {wf['n_folds']} folds | CRPS {m['model_crps']:.3f} "
            f"vs baseline {m['baseline_crps']:.3f} "
            f"({m['crps_improvement_pct']:+.2f}%) | cover50 {m['model_cover_50']:.3f} "
            f"cover80 {m['model_cover_80']:.3f}"
        )
        dep = wf["dependence"]
        print(
            f"dependence: |err| copula {dep['mean_abs_error_copula']:.4f} "
            f"vs independence {dep['mean_abs_error_independence']:.4f}"
        )

    if args.br_dir:
        results = {}
        for path in sorted(Path(args.br_dir).glob("br_*.parquet")):
            tag = path.stem.replace("br_", "")
            d = brd.load_week(path)
            summary = brd.summarize_week(d)
            summary["policy_rank_diagnostics"] = brd.policy_rank_diagnostics(
                d, n_rooms=args.replay_rooms
            )
            results[tag] = summary
            dup = summary["duplication"]
            pol = summary["policy_rank_diagnostics"]
            print(
                f"{tag}: entries={summary['entries']} win={summary['winning_score']:.1f} "
                f"unique={dup['unique_entry_share']:.4f} "
                f"flexRB={summary['flex_share']['RB']:.3f} "
                f"adp_sigma={summary['adp_noise_mad_sigma']:.2f} "
                f"policy_top3={pol['policy']['top3']:.3f} "
                f"adp_top3={pol['adp_baseline']['top3']:.3f}"
            )
        (out / "battle_royale_archives.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

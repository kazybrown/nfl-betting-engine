"""Build the Draft Companion HTML from a generated plan JSON.

Usage:
    python scripts/br_plan.py --csv SLATE.csv --out plan.json
    python scripts/br_build_companion.py --plan plan.json --out companion.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "battle_royale" / "companion_template.html"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="reports/battle_royale/draft_plan.json")
    ap.add_argument("--out", default="reports/battle_royale/draft_companion.html")
    ap.add_argument("--entries", default="reports/battle_royale/entries_2026.json",
                    help="entry log; matching rosters ride inside the page so the "
                         "self-dup guard survives phone storage eviction")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    fmt_key = plan.get("format", {}).get("key", "battle_royale")
    fmt_name = plan.get("format", {}).get("name", "Battle Royale")
    entries_path = Path(args.entries)
    if entries_path.exists() and args.season and args.week:
        known = {p["name"] for p in plan["players"]}
        prior = [
            e["roster"] for e in json.loads(entries_path.read_text())
            if e.get("season") == args.season and e.get("week") == args.week
            and e.get("contest") == fmt_key and set(e["roster"]) <= known
        ]
        if prior:
            plan["prior_entries"] = prior
            print(f"baked {len(prior)} logged entries into the page")
    html = (
        TEMPLATE.read_text()
        .replace("__PLAN_JSON__", json.dumps(plan, separators=(",", ":")))
        .replace("__FMT_NAME__", fmt_name)
    )
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()

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
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    fmt_name = plan.get("format", {}).get("name", "Battle Royale")
    html = (
        TEMPLATE.read_text()
        .replace("__PLAN_JSON__", json.dumps(plan, separators=(",", ":")))
        .replace("__FMT_NAME__", fmt_name)
    )
    Path(args.out).write_text(html)
    print(f"wrote {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()

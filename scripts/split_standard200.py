#!/usr/bin/env python3
"""
Split prompts/standard_200.jsonl into three benchmark slices:

  prompts/track_estimathon.jsonl   -- numerical tasks only (regression/pairwise/interval)
  prompts/track_longebench.jsonl   -- LongeBench tasks only (excludes EP-04, RNA-EB0X)
  prompts/track_ep04.jsonl         -- EP-04 / RNA-EB0X tasks only

Run from inside longivity_hack/:
    python ../scripts/split_standard200.py
"""
from __future__ import annotations

import json
from pathlib import Path

_ESTIMATHON_FORMATS = {"regression", "pairwise", "interval"}
_RNA_IDS = {"EP-04", "RNA-EB0X"}

INPUT = Path("prompts/standard_200.jsonl")
OUT_ESTIMATHON = Path("prompts/track_estimathon.jsonl")
OUT_LONGEBENCH = Path("prompts/track_longebench.jsonl")
OUT_EP04 = Path("prompts/track_ep04.jsonl")


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}\nRun scripts/build_standard200.py first.")

    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]

    estimathon = [r for r in rows if r.get("format") in _ESTIMATHON_FORMATS]
    longebench  = [r for r in rows if r.get("lb_id") not in _RNA_IDS]
    ep04        = [r for r in rows if r.get("lb_id") in _RNA_IDS]

    OUT_ESTIMATHON.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {path}  ({len(items)} tasks)")

    print(f"Loaded {len(rows)} tasks from {INPUT}\n")
    _write(OUT_ESTIMATHON, estimathon)
    _write(OUT_LONGEBENCH, longebench)
    _write(OUT_EP04, ep04)
    print("\nDone. Use these files with cli.py run --tasks <path>")


if __name__ == "__main__":
    main()

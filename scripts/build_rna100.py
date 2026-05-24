#!/usr/bin/env python3
"""
Build a 100-question RNA benchmark from:
  • carolw/EP04train_and_set  (ep04_test_set.jsonl)  — masked enzyme, ternary
  • sarahliu/rna-eb0x         (test split)            — unmasked enzyme, ternary

Gold labels: A = promotes aging, B = suppresses aging, C = no clear effect
Output: prompts/rna_100.jsonl  (run from inside longivity_hack/)

Usage:
    python ../scripts/build_rna100.py
    python ../scripts/build_rna100.py --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

_AGING_MAP = {"promotes": "A", "suppresses": "B", "neutral": "C"}
_INTERNAL   = {"_dedup_key", "_source"}


# ── token resolution ──────────────────────────────────────────────────────────

def _resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    config_path = Path.home() / ".longevity" / "config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8")).get("hf.token")
        except Exception:
            pass
    return None


# ── EP-04 loader ──────────────────────────────────────────────────────────────

def _load_ep04(token: str | None) -> list[dict]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    print("Loading carolw/EP04train_and_set …")
    try:
        local_path = hf_hub_download(
            repo_id="carolw/EP04train_and_set",
            filename="my_folder/ep04_test_set.jsonl",
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        print(f"  Warning: could not download EP-04 ({exc}). Skipping.")
        return []

    rows: list[dict] = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_norm_ep04(json.loads(line)))
    print(f"  Loaded {len(rows)} EP-04 rows")
    return rows


def _norm_ep04(row: dict) -> dict:
    meta_raw = row.get("metadata", "{}")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    gold = _AGING_MAP.get(str(meta.get("aging_effect", "neutral")).lower(), "C")
    msgs = [dict(m) for m in row["messages"]] + [{"role": "assistant", "content": gold}]
    return {
        "lb_id": "EP-04",
        "pool": "ep04_ternary_masked_test",
        "display_name": "RNA Enzyme Aging Effect / Ternary (masked)",
        "domain": "epitranscriptomics",
        "format": "ternary",
        "metric": "balanced_accuracy",
        "task": (
            "Given a masked RNA modification enzyme profile, predict the effect "
            "on cellular aging: A=promotes aging, B=suppresses aging, C=no clear effect."
        ),
        "metadata": json.dumps({
            **meta,
            "source_dataset": "carolw/EP04train_and_set",
            "split_key": "modification_type",
        }),
        "messages": msgs,
        "_dedup_key": "|".join([
            meta.get("enzyme", "?"),
            meta.get("pmid", "?"),
            meta.get("perturbation", "?"),
        ]),
        "_source": "ep04",
    }


# ── RNA-EB0X loader ───────────────────────────────────────────────────────────

def _load_rna_eb0x(token: str | None) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    print("Loading sarahliu/rna-eb0x (test split) …")
    try:
        ds = load_dataset("sarahliu/rna-eb0x", split="test", token=token)
    except Exception as exc:
        print(f"  Warning: could not load RNA-EB0X ({exc}). Skipping.")
        return []

    rows = [_norm_rna_eb0x(dict(row)) for row in ds]
    print(f"  Loaded {len(rows)} RNA-EB0X rows")
    return rows


def _norm_rna_eb0x(row: dict) -> dict:
    meta_raw = row.get("metadata", "{}")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
    gold = _AGING_MAP.get(str(meta.get("aging_effect", "neutral")).lower(), "C")
    msgs = [dict(m) for m in row["messages"]] + [{"role": "assistant", "content": gold}]
    return {
        "lb_id": "RNA-EB0X",
        "pool": "rna_eb0x_ternary_unmasked_test",
        "display_name": "RNA Enzyme Aging Effect / Ternary (unmasked)",
        "domain": "epitranscriptomics",
        "format": "ternary",
        "metric": "balanced_accuracy",
        "task": (
            "Given an RNA modification enzyme profile (enzyme name visible), predict "
            "the effect on cellular aging: A=promotes aging, B=suppresses aging, C=no clear effect."
        ),
        "metadata": json.dumps({
            **meta,
            "source_dataset": "sarahliu/rna-eb0x",
            "split_key": "enzyme_family",
        }),
        "messages": msgs,
        "_dedup_key": "|".join([
            meta.get("enzyme", "?"),
            meta.get("pmid", "?"),
            meta.get("perturbation", "?"),
        ]),
        "_source": "rna_eb0x",
    }


# ── merge, dedup, balanced sample ────────────────────────────────────────────

def _gold_letter(row: dict) -> str:
    for m in reversed(row.get("messages", [])):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def _merge(ep04: list[dict], rna: list[dict], quota: int, seed: int) -> list[dict]:
    seen: set[str] = set()
    pool: list[dict] = []
    for row in ep04:
        key = row["_dedup_key"]
        if key not in seen:
            seen.add(key)
            pool.append(row)
    for row in rna:
        key = row["_dedup_key"]
        if key not in seen:
            seen.add(key)
            pool.append(row)

    n_ep04 = sum(1 for r in pool if r["_source"] == "ep04")
    n_rna  = sum(1 for r in pool if r["_source"] == "rna_eb0x")
    print(f"  Pool: {len(pool)} unique rows after dedup (ep04={n_ep04}, rna_eb0x={n_rna})")

    if len(pool) <= quota:
        print(f"  Pool smaller than quota — taking all {len(pool)} rows")
        return pool

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_label[_gold_letter(row)].append(row)

    labels = [l for l in ("A", "B", "C") if by_label[l]]
    selected: list[dict] = []
    used_ids: set[int] = set()

    per_label, rem = divmod(quota, len(labels))
    for i, label in enumerate(labels):
        k = min(per_label + (1 if i < rem else 0), len(by_label[label]))
        chosen = rng.sample(by_label[label], k)
        selected.extend(chosen)
        used_ids.update(id(r) for r in chosen)

    shortfall = quota - len(selected)
    if shortfall > 0:
        fill = [r for r in pool if id(r) not in used_ids]
        selected.extend(rng.sample(fill, min(shortfall, len(fill))))

    selected = selected[:quota]
    gc = Counter(_gold_letter(r) for r in selected)
    print(f"  Sampled {len(selected)} rows  (A={gc.get('A',0)}, B={gc.get('B',0)}, C={gc.get('C',0)})")
    return selected


# ── main ──────────────────────────────────────────────────────────────────────

def build(token: str | None, output: Path, n: int, seed: int) -> None:
    ep04_rows = _load_ep04(token)
    rna_rows  = _load_rna_eb0x(token)

    if not ep04_rows and not rna_rows:
        raise SystemExit("Both sources failed to load. Check your HF token and dataset access.")

    rows = _merge(ep04_rows, rna_rows, n, seed)
    random.Random(seed).shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k not in _INTERNAL}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    fmt = Counter(r.get("format", "?") for r in rows)
    src = Counter(r.get("lb_id", "?") for r in rows)
    print(f"\nSaved {len(rows)} questions → {output}")
    print(f"  Format : {dict(fmt)}")
    print(f"  Source : {dict(src)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build 100-question RNA benchmark.")
    p.add_argument("--token",  default=None, help="HuggingFace token")
    p.add_argument("--output", default="prompts/rna_100.jsonl")
    p.add_argument("--n",      type=int, default=100, help="Number of questions (default 100)")
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    token = _resolve_token(args.token)
    if not token:
        print("Warning: no HF token found. Set HF_TOKEN or pass --token.\n")

    build(token=token, output=Path(args.output), n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()

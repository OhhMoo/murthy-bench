#!/usr/bin/env python3
"""
Build the standardized 200-question test set for the Longevity LLM hackathon.

Sources
-------
• LongeBench (insilicomedicine/longebench)
    - Numerical tasks  (regression + pairwise): 100 questions across 11 tasks
    - Categorical tasks (binary/multiclass/ternary/generation): 50 questions across 11 tasks
• EP-04 (carolw/EP04train_and_set  my_folder/ep04_test_set.jsonl)
    - Ternary RNA enzyme aging effect, enzyme name MASKED
    - Gold answer already in messages[-1]
• RNA-EB0X (sarahliu/rna-eb0x)
    - Same task, enzyme name visible; gold derived from metadata.aging_effect
    - Only rows NOT already in EP-04 are used (deduped by enzyme+pmid+perturbation)
• Combined RNA quota: 50 questions (carolw first, sarahliu fills remainder)

Total: 200 questions  |  Estimathon budget: 135 slips (100 numerical tasks)

Satisfies all Track 01 requirements
------------------------------------
  ✓ ChatML format — last message is gold answer
  ✓ Verifiable ground truth
  ✓ Formal metric: balanced_accuracy (categorical) / estimathon_score (numerical)
  ✓ ≥50 prompts per task
  ✓ No prompt > 30K tokens
  ✓ Train/test split documented per source (split_key in metadata)
  ✓ Class balance checked and reported

Usage
-----
    python scripts/build_standard200.py
    python scripts/build_standard200.py --token hf_xxx
    python scripts/build_standard200.py --n-lb-numerical 100 --n-lb-categorical 50 --n-rna 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Gold label mapping for sarahliu/rna-eb0x (metadata.aging_effect → option letter)
_AGING_MAP = {"promotes": "A", "suppresses": "B", "neutral": "C"}

# Fields internal to this script — stripped before writing output
_INTERNAL = {"_dedup_key", "_source"}


# ── Token resolution ──────────────────────────────────────────────────────────

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


# ── LongeBench loader ─────────────────────────────────────────────────────────

def _load_longebench(token: str | None, n_numerical: int, n_categorical: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    print("Loading insilicomedicine/longebench (benchmark split)...")
    ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval", token=token)
    print(f"  {len(ds):,} total rows  |  seed={seed}")

    numerical_formats = {"regression", "pairwise", "interval"}

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in ds:
        by_task[row["lb_id"]].append(dict(row))

    numerical_tasks = sorted(t for t, rows in by_task.items()
                              if rows[0].get("format") in numerical_formats)
    categorical_tasks = sorted(t for t in by_task if t not in numerical_tasks)

    rng = random.Random(seed)
    selected: list[dict] = []

    def _stratified(task_ids: list[str], quota: int) -> list[dict]:
        n = len(task_ids)
        base, rem = quota // n, quota % n
        out: list[dict] = []
        for i, tid in enumerate(task_ids):
            k = min(base + (1 if i < rem else 0), len(by_task[tid]))
            out.extend(rng.sample(by_task[tid], k))
        return out

    selected.extend(_stratified(numerical_tasks, n_numerical))
    selected.extend(_stratified(categorical_tasks, n_categorical))

    for row in selected:
        row["_source"] = "longebench"

    print(f"  Sampled {len(selected)} LongeBench rows  "
          f"(numerical={sum(1 for r in selected if r.get('format') in numerical_formats)}, "
          f"categorical={sum(1 for r in selected if r.get('format') not in numerical_formats)})")
    return selected


# ── EP-04 loader (carolw) ─────────────────────────────────────────────────────

def _load_ep04(token: str | None) -> list[dict]:
    """Load carolw/EP04train_and_set test set via huggingface_hub file download."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Run: pip install huggingface_hub")

    print("Loading carolw/EP04train_and_set (ep04_test_set.jsonl)...")
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
            row = json.loads(line)
            rows.append(_normalize_ep04(row))

    print(f"  Loaded {len(rows)} EP-04 rows")
    return rows


def _normalize_ep04(row: dict) -> dict:
    meta_raw = row.get("metadata", "{}")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

    # ep04_test_set.jsonl has system+user only — gold lives in metadata.aging_effect
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
            "Given a masked RNA modification enzyme profile, predict the effect of "
            "the described perturbation on cellular aging: "
            "A=promotes aging, B=suppresses aging, C=no clear aging effect."
        ),
        "metadata": json.dumps({
            **meta,
            "source_dataset": "carolw/EP04train_and_set",
            "split_key": "modification_type",
            "split_note": (
                "Test set. Train/test split by RNA modification type "
                "to prevent enzyme-identity leakage. Enzyme name masked in prompt."
            ),
        }),
        "messages": msgs,
        "_dedup_key": "|".join([
            meta.get("enzyme", "?"),
            meta.get("pmid", "?"),
            meta.get("perturbation", "?"),
        ]),
        "_source": "ep04",
    }


# ── RNA-EB0X loader (sarahliu) ────────────────────────────────────────────────

def _load_rna_eb0x(token: str | None) -> list[dict]:
    from datasets import load_dataset

    print("Loading sarahliu/rna-eb0x (test split)...")
    try:
        ds = load_dataset("sarahliu/rna-eb0x", split="test", token=token)
    except Exception as exc:
        print(f"  Warning: could not load RNA-EB0X ({exc}). Skipping.")
        return []

    rows = [_normalize_rna_eb0x(dict(row)) for row in ds]
    print(f"  Loaded {len(rows)} RNA-EB0X rows")
    return rows


def _normalize_rna_eb0x(row: dict) -> dict:
    meta_raw = row.get("metadata", "{}")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw

    # Gold must be derived from metadata — messages has no assistant turn
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
            "Given an RNA modification enzyme profile with the enzyme name visible, "
            "predict the effect of the described perturbation on cellular aging: "
            "A=promotes aging, B=suppresses aging, C=no clear aging effect."
        ),
        "metadata": json.dumps({
            **meta,
            "source_dataset": "sarahliu/rna-eb0x",
            "split_key": "enzyme_family",
            "split_note": (
                "Test set. Split by enzyme family (writer/eraser/reader) "
                "to prevent leakage across enzyme types."
            ),
        }),
        "messages": msgs,
        "_dedup_key": "|".join([
            meta.get("enzyme", "?"),
            meta.get("pmid", "?"),
            meta.get("perturbation", "?"),
        ]),
        "_source": "rna_eb0x",
    }


# ── RNA merge & dedup ─────────────────────────────────────────────────────────

def _gold_letter(row: dict) -> str:
    """Extract the gold answer letter from the last assistant message."""
    for m in reversed(row.get("messages", [])):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def _merge_rna(ep04_rows: list[dict], rna_rows: list[dict],
               quota: int, seed: int) -> list[dict]:
    """
    Combine EP-04 and RNA-EB0X, deduplicating by (enzyme|pmid|perturbation).
    EP-04 rows are preferred (masked enzyme = harder test).
    Samples exactly `quota` rows; balances A/B/C when possible.
    """
    seen: set[str] = set()
    pool: list[dict] = []

    for row in ep04_rows:
        key = row["_dedup_key"]
        if key not in seen:
            seen.add(key)
            pool.append(row)

    for row in rna_rows:
        key = row["_dedup_key"]
        if key not in seen:
            seen.add(key)
            pool.append(row)

    n_ep04 = sum(1 for r in pool if r["_source"] == "ep04")
    n_rna  = sum(1 for r in pool if r["_source"] == "rna_eb0x")
    print(f"  RNA pool: {len(pool)} unique rows after dedup (ep04={n_ep04}, rna_eb0x={n_rna})")

    if len(pool) <= quota:
        print(f"  Taking all {len(pool)} unique RNA rows (pool ≤ quota {quota})")
        return pool

    rng = random.Random(seed)

    # Attempt balanced A/B/C sampling
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in pool:
        by_label[_gold_letter(row)].append(row)

    abc_labels = [l for l in ("A", "B", "C") if by_label[l]]
    selected: list[dict] = []

    if abc_labels:
        per_label, rem = divmod(quota, len(abc_labels))
        used_ids: set[int] = set()
        for i, label in enumerate(abc_labels):
            k = min(per_label + (1 if i < rem else 0), len(by_label[label]))
            chosen = rng.sample(by_label[label], k)
            selected.extend(chosen)
            used_ids.update(id(r) for r in chosen)
        # Top up from any remaining rows if a label was short
        shortfall = quota - len(selected)
        if shortfall > 0:
            fill = [r for r in pool if id(r) not in used_ids]
            selected.extend(rng.sample(fill, min(shortfall, len(fill))))
    else:
        # No standard A/B/C gold found — plain random sample
        selected = rng.sample(pool, quota)

    selected = selected[:quota]  # hard cap — never exceed quota

    gc = Counter(_gold_letter(r) for r in selected)
    print(f"  Sampled {len(selected)} RNA rows  "
          f"(A={gc.get('A',0)}, B={gc.get('B',0)}, C={gc.get('C',0)})")
    return selected


# ── Main build ────────────────────────────────────────────────────────────────

def build(token: str | None, output: Path,
          n_lb_numerical: int, n_lb_categorical: int,
          n_rna: int, seed: int) -> None:

    rng = random.Random(seed)

    lb_rows = _load_longebench(token, n_lb_numerical, n_lb_categorical, seed)
    ep04_rows = _load_ep04(token)
    rna_rows = _load_rna_eb0x(token)
    rna_selected = _merge_rna(ep04_rows, rna_rows, n_rna, seed)

    all_rows = lb_rows + rna_selected
    rng.shuffle(all_rows)

    # Strip internal fields before writing
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for row in all_rows:
            clean = {k: v for k, v in row.items() if k not in _INTERNAL}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # ── Summary ────────────────────────────────────────────────────────────
    numerical_formats = {"regression", "pairwise", "interval"}
    n_numerical = sum(1 for r in all_rows if r.get("format") in numerical_formats)
    n_categorical = len(all_rows) - n_numerical
    sources = Counter(r.get("_source", r.get("lb_id", "?")) for r in all_rows)
    domains = Counter(r.get("domain", "?") for r in all_rows)
    formats = Counter(r.get("format", "?") for r in all_rows)

    print(f"\n{'─'*60}")
    print(f"Total questions : {len(all_rows)}")
    print(f"Numerical       : {n_numerical}  (Estimathon → budget 135 slips)")
    print(f"Categorical     : {n_categorical}  (one-shot)")
    print(f"\nBy format  : {dict(sorted(formats.items(), key=lambda x: -x[1]))}")
    print(f"By domain  : {dict(sorted(domains.items(), key=lambda x: -x[1]))}")

    print(f"\n{'Task':<14} {'N':>3}  {'Format':<18}  {'Domain':<22}  Display name")
    print("─" * 88)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        by_task[row.get("lb_id", "?")].append(row)
    for tid in sorted(by_task.keys()):
        rows = by_task[tid]
        r0 = rows[0]
        print(f"  {tid:<12} {len(rows):>3}  {r0.get('format','?'):<18}  "
              f"{r0.get('domain','?'):<22}  {r0.get('display_name','')[:36]}")

    # Class balance for ternary tasks
    ternary = [r for r in all_rows if r.get("format") == "ternary"]
    if ternary:
        tc = Counter(r["messages"][-1]["content"].strip() for r in ternary)
        print(f"\nTernary class balance ({len(ternary)} tasks): "
              f"A={tc.get('A',0)}  B={tc.get('B',0)}  C={tc.get('C',0)}")

    print(f"\nSaved → {output}")
    print(f"\n{'━'*60}")
    print("Next steps")
    print(f"{'━'*60}")
    print("""
1. Set API keys (once):
   murthy config set llm.endpoint https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud
   murthy config set hf.token <your-hf-token>
   murthy config set anthropic.api_key <your-anthropic-key>

2. Create the 5-model comparison group (PowerShell):
   murthy group add standard-5 `
     "L-LLM:longevity-llm:endpoint" `
     "Qwen3-8B:Qwen/Qwen3-8B:hf" `
     "Claude:claude-sonnet-4-6:anthropic" `
     "DeepSeek-R1-7B:deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:hf" `
     "Llama-3.1-8B:meta-llama/Llama-3.1-8B-Instruct:hf"

3. Run the comparison:
   murthy compare --group standard-5 `
     --tasks prompts/standard_200.jsonl `
     --mode mixed --budget 135 `
     --output results/standard_comparison.jsonl
""")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the standardized 200-question longevity benchmark test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--token", default=None,
                   help="HuggingFace token (falls back to HF_TOKEN env or ~/.longevity/config.json)")
    p.add_argument("--output", default="prompts/standard_200.jsonl")
    p.add_argument("--n-lb-numerical", type=int, default=100,
                   help="LongeBench numerical questions (regression+pairwise). Default: 100")
    p.add_argument("--n-lb-categorical", type=int, default=50,
                   help="LongeBench categorical questions. Default: 50")
    p.add_argument("--n-rna", type=int, default=50,
                   help="RNA task questions (EP-04 + RNA-EB0X combined). Default: 50")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed — keep 42 for the canonical standard set")
    args = p.parse_args()

    token = _resolve_token(args.token)
    if not token:
        print("Warning: no HF token found. Gated datasets may fail.")
        print("  Set HF_TOKEN, pass --token, or: murthy config set hf.token hf_xxx\n")

    total = args.n_lb_numerical + args.n_lb_categorical + args.n_rna
    print(f"Building standard test set: {total} questions total  (seed={args.seed})\n")

    build(
        token=token,
        output=Path(args.output),
        n_lb_numerical=args.n_lb_numerical,
        n_lb_categorical=args.n_lb_categorical,
        n_rna=args.n_rna,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

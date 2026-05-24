#!/usr/bin/env python3
"""
Export benchmark results to the Insilico Medicine Track 01 hackathon submission format.

Produces two files per results file:
  results/submission_<name>.jsonl   — per-row ChatML + evaluation (the main submission)
  results/summary_<name>.json       — per-model performance table

Required fields per row (from Track01 spec):
  lb_id, input_messages, raw_response, tokens_used, pred, gold, metric, correct/f1

Usage (run from inside longivity_hack/):
    python ../scripts/export_hackathon.py results/rank_rna100.jsonl prompts/rna_100.jsonl
    python ../scripts/export_hackathon.py results/rank_rna100_v2.jsonl prompts/rna_100.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _balanced_accuracy(records: list[dict]) -> float | None:
    """Macro-averaged recall across gold classes (balanced accuracy)."""
    by_class: dict[str, dict] = defaultdict(lambda: {"tp": 0, "total": 0})
    for r in records:
        gold = str(r.get("gold", "")).strip()
        pred = str(r.get("pred", "")).strip()
        if not gold:
            continue
        by_class[gold]["total"] += 1
        if pred.lower() == gold.lower():
            by_class[gold]["tp"] += 1
    if not by_class:
        return None
    recalls = [v["tp"] / v["total"] for v in by_class.values() if v["total"] > 0]
    return sum(recalls) / len(recalls) if recalls else None


def _class_distribution(records: list[dict]) -> dict:
    dist: dict[str, int] = defaultdict(int)
    for r in records:
        gold = str(r.get("gold", "")).strip()
        if gold:
            dist[gold] += 1
    return dict(sorted(dist.items()))


# ── main ──────────────────────────────────────────────────────────────────────

def export(results_path: Path, tasks_path: Path | None) -> None:
    results = _load_jsonl(results_path)

    # Build lb_id → task metadata map (for reconstructing input_messages in old results)
    task_map: dict[str, list[dict]] = defaultdict(list)
    if tasks_path and tasks_path.exists():
        for t in _load_jsonl(tasks_path):
            task_map[t.get("lb_id", "")].append(t)
    task_cursors: dict[str, int] = defaultdict(int)

    stem = results_path.stem
    out_dir = results_path.parent
    submission_path = out_dir / f"submission_{stem}.jsonl"
    summary_path    = out_dir / f"summary_{stem}.json"

    # ── per-model collections ────────────────────────────────────────────────
    by_model: dict[str, list[dict]] = defaultdict(list)

    with submission_path.open("w", encoding="utf-8") as fout:
        for rec in results:
            # Skip aggregate/session records (no gold field at row level)
            if "gold" not in rec or rec.get("gold") is None:
                continue

            label = rec.get("model_label", "unknown")
            lb_id = rec.get("lb_id", "")

            # Reconstruct input_messages from task file if not stored in result
            input_messages = rec.get("input_messages")
            if not input_messages and lb_id in task_map:
                idx = task_cursors[f"{label}:{lb_id}"]
                bucket = task_map[lb_id]
                if idx < len(bucket):
                    t = bucket[idx]
                    input_messages = t["messages"][:-1]  # drop gold assistant turn
                    task_cursors[f"{label}:{lb_id}"] += 1

            # Build the full ChatML interaction (input + model response)
            pred = rec.get("pred") or ""
            think = rec.get("think")
            assistant_content = f"<think>{think}</think>\n{pred}" if think else pred

            submission_row = {
                "lb_id":           lb_id,
                "domain":          rec.get("domain", ""),
                "format":          rec.get("format", ""),
                "metric":          rec.get("metric", ""),
                "task":            rec.get("task", ""),
                "model_label":     label,
                # ChatML: input messages + model's assistant turn
                "messages":        (input_messages or []) + [
                    {"role": "assistant", "content": assistant_content}
                ],
                "raw_response":    rec.get("raw_response", assistant_content),
                "pred":            pred,
                "gold":            rec.get("gold", ""),
                "correct":        rec.get("correct"),
                "f1":             rec.get("f1"),
                "tokens_used":     rec.get("tokens_used", 0),
                "think":           think,
                "error":           rec.get("error"),
            }
            # Strip None values for cleaner output
            submission_row = {k: v for k, v in submission_row.items() if v is not None}
            fout.write(json.dumps(submission_row, ensure_ascii=False) + "\n")
            by_model[label].append(submission_row)

    print(f"Submission JSONL → {submission_path}  ({sum(len(v) for v in by_model.values())} rows)")

    # ── per-model summary ────────────────────────────────────────────────────
    summary: dict = {
        "source_results": str(results_path),
        "tasks_file":     str(tasks_path) if tasks_path else None,
        "models": {},
    }

    for label, rows in sorted(by_model.items()):
        n_total   = len(rows)
        n_correct = sum(1 for r in rows if r.get("correct") is True)
        n_error   = sum(1 for r in rows if r.get("error"))
        accuracy  = n_correct / (n_total - n_error) if (n_total - n_error) > 0 else 0.0
        bal_acc   = _balanced_accuracy([r for r in rows if not r.get("error")])
        avg_tok   = sum(r.get("tokens_used", 0) for r in rows) // n_total if n_total else 0

        by_format: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in rows:
            fmt = r.get("format", "unknown")
            by_format[fmt]["n"] += 1
            if r.get("correct") is True:
                by_format[fmt]["correct"] += 1

        summary["models"][label] = {
            "n_tasks":           n_total,
            "n_correct":         n_correct,
            "n_errors":          n_error,
            "accuracy":          round(accuracy, 4),
            "balanced_accuracy": round(bal_acc, 4) if bal_acc is not None else None,
            "avg_tokens":        avg_tok,
            "class_distribution": _class_distribution(rows),
            "by_format": {
                fmt: {
                    "n": s["n"],
                    "correct": s["correct"],
                    "accuracy": round(s["correct"] / s["n"], 4) if s["n"] else 0,
                }
                for fmt, s in sorted(by_format.items())
            },
        }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary JSON      → {summary_path}")

    # ── print leaderboard ────────────────────────────────────────────────────
    print(f"\n{'Model':<16}  {'N':>5}  {'Correct':>7}  {'Bal Acc':>8}  {'Errors':>6}")
    print("─" * 52)
    ranked = sorted(
        summary["models"].items(),
        key=lambda kv: kv[1]["balanced_accuracy"] or 0,
        reverse=True,
    )
    for label, s in ranked:
        ba = s["balanced_accuracy"]
        print(
            f"{label:<16}  {s['n_tasks']:>5}  {s['n_correct']:>7}  "
            f"{f'{ba:.1%}' if ba is not None else 'N/A':>8}  {s['n_errors']:>6}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Export results to hackathon submission format.")
    p.add_argument("results",        help="Path to results JSONL (e.g. results/rank_rna100.jsonl)")
    p.add_argument("tasks", nargs="?", default=None,
                   help="Path to tasks JSONL for reconstructing input_messages (optional)")
    args = p.parse_args()

    results_path = Path(args.results)
    tasks_path   = Path(args.tasks) if args.tasks else None

    if not results_path.exists():
        sys.exit(f"Results file not found: {results_path}")

    export(results_path, tasks_path)


if __name__ == "__main__":
    main()

"""
STEP 4: Score results and generate analysis figures.

Usage:
    python step4_score_and_analyze.py

Outputs:
    results/scores_summary.json
    results/figures/confusion_EP-01.png
    results/figures/confusion_EP-02.png
    results/figures/performance_comparison.png
"""

import json
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_BASELINES = {
    "binary":     0.500,
    "ternary":    0.333,
    "multiclass": 0.333,   # 3-way
}

TASK_LABELS = {
    "EP-01": {"A": "Aging-related", "B": "Not aging-related"},
    "EP-02": {"A": "Aging disease", "B": "Non-aging disease", "C": "No disease"},
    "EP-03": {"A": "Neurodegeneration", "B": "Cardio/metabolic", "C": "Cellular aging"},
}


def load_results(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def score_task(rows, task_id):
    valid = [r for r in rows if r["pred"] is not None]
    golds = [r["gold"] for r in valid]
    preds = [r["pred"] for r in valid]

    bal_acc = balanced_accuracy_score(golds, preds)
    raw_acc = sum(g == p for g, p in zip(golds, preds)) / len(valid)
    parse_rate = len(valid) / len(rows)

    task_format = rows[0]["format"] if rows else "binary"
    baseline = RANDOM_BASELINES.get(task_format, 0.5)

    report = classification_report(golds, preds, output_dict=True)

    return {
        "task_id": task_id,
        "n_total": len(rows),
        "n_valid": len(valid),
        "parse_rate": round(parse_rate, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "raw_accuracy": round(raw_acc, 4),
        "random_baseline": baseline,
        "delta_over_baseline": round(bal_acc - baseline, 4),
        "class_distribution_gold": dict(Counter(golds)),
        "class_distribution_pred": dict(Counter(preds)),
        "per_class_f1": {k: round(v["f1-score"], 4)
                         for k, v in report.items()
                         if k not in ("accuracy", "macro avg", "weighted avg")},
    }


def plot_confusion(rows, task_id, model_name):
    valid = [r for r in rows if r["pred"] is not None]
    golds = [r["gold"] for r in valid]
    preds = [r["pred"] for r in valid]

    labels = sorted(set(golds + preds))
    label_names = [TASK_LABELS.get(task_id, {}).get(l, l) for l in labels]

    cm = confusion_matrix(golds, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"{task_id} — {model_name}\nBalanced Accuracy: "
                 f"{balanced_accuracy_score(golds, preds):.3f}", fontsize=11)
    plt.tight_layout()
    out = f"{FIGURES_DIR}/confusion_{task_id}_{model_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_comparison(all_scores):
    """Bar chart: balanced accuracy across tasks and models vs random baseline."""
    tasks = sorted(set(s["task_id"] for s in all_scores))
    models = sorted(set(s["model"] for s in all_scores))

    x = range(len(tasks))
    width = 0.3
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        scores = []
        for task in tasks:
            match = [s for s in all_scores if s["task_id"] == task and s["model"] == model]
            scores.append(match[0]["balanced_accuracy"] if match else 0)
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], scores, width,
                      label=model, color=colors[i % len(colors)], alpha=0.85)
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{score:.3f}", ha="center", va="bottom", fontsize=8)

    # Random baselines
    baselines = {"EP-01": 0.500, "EP-02": 0.333, "EP-03": 0.333}
    for xi, task in zip(x, tasks):
        ax.hlines(baselines[task], xi - 0.5, xi + 0.5,
                  colors="red", linestyles="dashed", linewidth=1.2, label="Random" if xi == 0 else "")

    ax.set_xticks(list(x))
    ax.set_xticklabels([
        "EP-01\nncRNA–Aging Disease\n(Binary)",
        "EP-02\nRNA Mod Variant\n(Ternary)",
        "EP-03\nAging Category\n(Multiclass)",
    ], fontsize=9)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Epitranscriptome Benchmark — Model Performance vs Random Baseline",
                 fontsize=11)
    ax.legend(loc="upper right")
    ax.axhline(0.5, color="gray", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/performance_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def analyze_failure_modes(rows, task_id):
    """What types of RNA/modifications does the model get wrong?"""
    wrong = [r for r in rows if r["pred"] is not None and not r["correct"]]
    if not wrong:
        return {}

    # Break down failures by RNA type or mod type
    failures = defaultdict(int)
    for r in wrong:
        meta = json.loads(r.get("metadata", "{}"))
        key = meta.get("rna_type") or meta.get("mod_type") or "unknown"
        failures[key] += 1

    total_by_type = defaultdict(int)
    for r in rows:
        if r["pred"] is not None:
            meta = json.loads(r.get("metadata", "{}"))
            key = meta.get("rna_type") or meta.get("mod_type") or "unknown"
            total_by_type[key] += 1

    error_rates = {k: round(failures[k] / total_by_type[k], 3)
                   for k in failures if total_by_type[k] > 0}
    return dict(sorted(error_rates.items(), key=lambda x: -x[1]))


def main():
    result_files = glob.glob(f"{RESULTS_DIR}/*_raw.jsonl")
    if not result_files:
        print("No result files found. Run step3 first.")
        return

    all_scores = []
    summary = {}

    for path in sorted(result_files):
        fname = os.path.basename(path)
        parts = fname.replace("_raw.jsonl", "").split("_")
        task_id = parts[0]
        model_name = "_".join(parts[1:])

        print(f"\nScoring: {task_id} | {model_name}")
        rows = load_results(path)
        scores = score_task(rows, task_id)
        scores["model"] = model_name

        print(f"  Balanced accuracy: {scores['balanced_accuracy']} "
              f"(baseline: {scores['random_baseline']}, "
              f"delta: +{scores['delta_over_baseline']})")
        print(f"  Per-class F1: {scores['per_class_f1']}")

        failure_modes = analyze_failure_modes(rows, task_id)
        scores["failure_modes_error_rate"] = failure_modes
        print(f"  Failure modes: {failure_modes}")

        all_scores.append(scores)
        summary[f"{task_id}_{model_name}"] = scores

        plot_confusion(rows, task_id, model_name)

    plot_comparison(all_scores)

    out_path = f"{RESULTS_DIR}/scores_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out_path}")

    # Print final table
    print("\n" + "="*70)
    print(f"{'Task':<8} {'Model':<12} {'Bal.Acc':>8} {'Baseline':>9} {'Delta':>7} {'Parse%':>7}")
    print("-"*70)
    for s in sorted(all_scores, key=lambda x: (x["task_id"], x["model"])):
        print(f"{s['task_id']:<8} {s['model']:<12} "
              f"{s['balanced_accuracy']:>8.4f} {s['random_baseline']:>9.3f} "
              f"{s['delta_over_baseline']:>+7.4f} {s['parse_rate']:>6.1%}")


if __name__ == "__main__":
    main()

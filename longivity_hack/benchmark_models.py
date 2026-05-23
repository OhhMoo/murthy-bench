#!/usr/bin/env python3
"""
Benchmark multiple models from hf_llm_models.csv on the estimathon benchmark.
Runs models sequentially and collects results.
"""
import csv
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime


def load_models(csv_file: str = "hf_llm_models.csv") -> list[dict]:
    """Load models from CSV."""
    models = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        models = list(reader)
    return models


def run_benchmark(
    model_id: str,
    model_name: str,
    provider: str = "hf",
    tasks: str = "sample",
    mode: str = "estimathon",
    limit: int = 7,
) -> dict:
    """Run benchmark for a single model."""
    # Extract short name from model_id
    short_name = model_id.split("/")[-1]

    print(f"\n{'='*80}")
    print(f"Running: {model_id}")
    print(f"{'='*80}")

    cmd = [
        "python",
        "cli.py",
        "run",
        "--model",
        short_name,
        "--provider",
        provider,
        "--tasks",
        tasks,
        "--mode",
        mode,
        "--limit",
        str(limit),
        "--output",
        f"results_{short_name}.jsonl",
    ]

    try:
        result = subprocess.run(cmd, timeout=600, capture_output=False)
        if result.returncode == 0:
            # Try to read the results file
            try:
                results_file = f"results_{short_name}.jsonl"
                with open(results_file) as f:
                    lines = f.readlines()
                    if lines:
                        last_result = json.loads(lines[-1])
                        return {
                            "model_id": model_id,
                            "model_name": model_name,
                            "status": "✅ success",
                            "final_score": last_result.get("final_score"),
                            "n_good_final": last_result.get("n_good_final"),
                            "n_problems": last_result.get("n_problems"),
                            "slips_used": last_result.get("slips_used"),
                            "refinement_accuracy": last_result.get("refinement_accuracy"),
                        }
            except Exception as e:
                print(f"Warning: Could not parse results: {e}")
                return {
                    "model_id": model_id,
                    "model_name": model_name,
                    "status": "⚠️  completed but could not parse results",
                }
        else:
            return {
                "model_id": model_id,
                "model_name": model_name,
                "status": "❌ failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "model_id": model_id,
            "model_name": model_name,
            "status": "⏱️  timeout (>10min)",
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "model_name": model_name,
            "status": f"❌ error: {e}",
        }


def main():
    """Benchmark multiple models."""
    # Parse args
    if len(sys.argv) < 2:
        print("Usage: python benchmark_models.py <num_models> [provider] [tasks] [limit]")
        print("\nExample:")
        print("  python benchmark_models.py 5           # Top 5 by downloads")
        print("  python benchmark_models.py 10 hf sample 7")
        sys.exit(1)

    num_models = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    provider = sys.argv[2] if len(sys.argv) > 2 else "hf"
    tasks = sys.argv[3] if len(sys.argv) > 3 else "sample"
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    # Load models
    models = load_models()
    sorted_models = sorted(models, key=lambda x: int(x["downloads"]), reverse=True)

    print(f"\n🚀 Benchmarking top {num_models} models")
    print(f"   Provider: {provider}")
    print(f"   Tasks: {tasks}")
    print(f"   Limit: {limit}")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for i, model in enumerate(sorted_models[:num_models], 1):
        result = run_benchmark(
            model["model_id"],
            model["model_name"],
            provider=provider,
            tasks=tasks,
            limit=limit,
        )
        results.append(result)

        # Print progress
        print(
            f"\n[{i}/{num_models}] {result['model_id']}: {result['status']}"
        )
        if result.get("final_score") is not None:
            print(
                f"     Score: {result['final_score']} | "
                f"Solved: {result['n_good_final']}/{result['n_problems']} | "
                f"Refinement: {result.get('refinement_accuracy', 'n/a')}"
            )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for result in results:
        status = result["status"]
        if result.get("final_score") is not None:
            print(
                f"{result['model_id']:<50} score={result['final_score']:>6} "
                f"solved={result['n_good_final']}/{result['n_problems']}"
            )
        else:
            print(f"{result['model_id']:<50} {status}")

    # Save summary to CSV
    summary_file = f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(summary_file, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

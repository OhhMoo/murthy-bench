#!/usr/bin/env python3
"""
Interactive model selector for CLI.
Load models from hf_llm_models.csv and generate CLI commands.
"""
import csv
import sys
from pathlib import Path
from typing import Optional


def load_models(csv_file: str = "hf_llm_models.csv") -> list[dict]:
    """Load models from CSV."""
    models = []
    try:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            models = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found")
        print("Run: python fetch_models.py to generate it first")
        sys.exit(1)
    return models


def search_models(models: list[dict], query: str) -> list[dict]:
    """Search models by name, author, or model_id."""
    query = query.lower()
    return [
        m for m in models
        if query in m["model_id"].lower()
        or query in m["model_name"].lower()
        or query in m["author"].lower()
    ]


def display_model(model: dict, index: int = 1):
    """Display a single model."""
    print(f"\n{index}. {model['model_id']}")
    print(f"   Author: {model['author']}")
    print(f"   Downloads: {int(model['downloads']):,}")
    print(f"   Likes: {model['likes']}")
    print(f"   Gated: {model['gated']}")
    print(f"   URL: {model['hf_url']}")
    print(f"   Inference API: {model['inference_api']}")


def generate_cli_command(model: dict, tasks: str = "sample", mode: str = "estimathon", limit: int = 7) -> str:
    """Generate a CLI command for the model."""
    return f"""python cli.py run \\
  --model {model['model_id'].split('/')[-1]} \\
  --provider hf \\
  --tasks {tasks} \\
  --mode {mode} \\
  --limit {limit}"""


def main():
    """Interactive model selector."""
    models = load_models()
    print(f"✓ Loaded {len(models)} models from HuggingFace")
    print("\nUsage:")
    print("  python model_selector.py list          - Show top 20 models")
    print("  python model_selector.py search llama  - Search for 'llama' models")
    print("  python model_selector.py info 5        - Show details of model #5")
    print("  python model_selector.py cmd 1         - Generate CLI command for model #1")
    print()

    if len(sys.argv) < 2:
        # Show top 20 by downloads
        print("Top 20 models by downloads:")
        print("-" * 80)
        sorted_models = sorted(models, key=lambda x: int(x["downloads"]), reverse=True)
        for i, model in enumerate(sorted_models[:20], 1):
            print(f"{i:2d}. {model['model_id']:<45} ({int(model['downloads']):>12,} downloads)")
        return

    command = sys.argv[1].lower()

    if command == "list":
        sorted_models = sorted(models, key=lambda x: int(x["downloads"]), reverse=True)
        print("Top 20 models by downloads:")
        print("-" * 80)
        for i, model in enumerate(sorted_models[:20], 1):
            print(f"{i:2d}. {model['model_id']:<45} ({int(model['downloads']):>12,} downloads)")

    elif command == "search" and len(sys.argv) > 2:
        query = sys.argv[2]
        results = search_models(models, query)
        print(f"Found {len(results)} models matching '{query}':")
        print("-" * 80)
        for i, model in enumerate(results[:20], 1):
            print(f"{i:2d}. {model['model_id']:<45} ({int(model['downloads']):>12,} downloads)")

    elif command == "info" and len(sys.argv) > 2:
        try:
            idx = int(sys.argv[2]) - 1
            sorted_models = sorted(models, key=lambda x: int(x["downloads"]), reverse=True)
            if 0 <= idx < len(sorted_models):
                display_model(sorted_models[idx], idx + 1)
            else:
                print(f"Model #{sys.argv[2]} not found")
        except ValueError:
            print(f"Invalid index: {sys.argv[2]}")

    elif command == "cmd" and len(sys.argv) > 2:
        try:
            idx = int(sys.argv[2]) - 1
            sorted_models = sorted(models, key=lambda x: int(x["downloads"]), reverse=True)
            if 0 <= idx < len(sorted_models):
                model = sorted_models[idx]
                tasks = sys.argv[3] if len(sys.argv) > 3 else "sample"
                mode = sys.argv[4] if len(sys.argv) > 4 else "estimathon"
                limit = int(sys.argv[5]) if len(sys.argv) > 5 else 7

                print(f"\nGenerated command for {model['model_id']}:")
                print("-" * 80)
                cmd = generate_cli_command(model, tasks, mode, limit)
                print(cmd)
                print("\nCopy this command and run it in the terminal.")
            else:
                print(f"Model #{sys.argv[2]} not found")
        except ValueError:
            print(f"Invalid arguments")

    else:
        print(f"Unknown command: {command}")
        print("Use: python model_selector.py list|search|info|cmd")


if __name__ == "__main__":
    main()

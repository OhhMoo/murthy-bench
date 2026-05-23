#!/usr/bin/env python3
"""
Fetch open-source LLM models from HuggingFace and save to CSV.
Requires HF_TOKEN environment variable to be set.
"""
import os
import csv
import httpx
from pathlib import Path

def fetch_llm_models(max_results: int = 500) -> list[dict]:
    """
    Fetch open-source LLM models from HuggingFace using REST API.

    Returns list of dicts with model info.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    # Search for text-generation models via HF REST API
    print(f"Searching HuggingFace for LLM models (max {max_results})...")

    models_data = []

    # Use HuggingFace's web search API (no auth required for public models)
    url = "https://huggingface.co/api/models"
    params = {
        "pipeline_tag": "text-generation",
        "library": "transformers",
        "sort": "downloads",
        "direction": -1,
        "limit": max_results,
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, params=params, timeout=30.0)
            response.raise_for_status()

            raw_models = response.json()
            if not isinstance(raw_models, list):
                raw_models = [raw_models]

            for i, model_info in enumerate(raw_models, 1):
                if i % 50 == 0:
                    print(f"  Fetched {i} models...")

                model_id = model_info.get("id", "")
                if not model_id:
                    continue

                # Extract key info
                entry = {
                    "model_id": model_id,
                    "model_name": model_id.split("/")[-1],
                    "author": model_id.split("/")[0] if "/" in model_id else "unknown",
                    "downloads": model_info.get("downloads", 0),
                    "likes": model_info.get("likes", 0),
                    "tags": ",".join(model_info.get("tags", [])),
                    "gated": model_info.get("gated", False),
                    "hf_url": f"https://huggingface.co/{model_id}",
                    "inference_api": f"https://api-inference.huggingface.co/models/{model_id}/v1",
                }

                models_data.append(entry)

        print(f"✓ Found {len(models_data)} models")
        return models_data

    except Exception as e:
        print(f"Error fetching models: {e}")
        raise


def save_to_csv(models: list[dict], output_file: str = "hf_llm_models.csv"):
    """Save model list to CSV file."""
    if not models:
        print("No models to save")
        return

    # Define columns
    fieldnames = [
        "model_id",
        "model_name",
        "author",
        "downloads",
        "likes",
        "tags",
        "pipeline_tag",
        "gated",
        "hf_url",
        "inference_api",
    ]

    # Sort by downloads descending
    models = sorted(models, key=lambda x: x.get("downloads", 0), reverse=True)

    # Write CSV
    output_path = Path(output_file)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(models)

    print(f"✓ Saved {len(models)} models to {output_file}")
    print(f"\nTop 10 models by downloads:")
    print("-" * 80)
    for i, model in enumerate(models[:10], 1):
        print(f"{i}. {model['model_id']:<40} ({model['downloads']:>10,} downloads)")


if __name__ == "__main__":
    import sys

    # Check for HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN not set. Set it with:")
        print("  export HF_TOKEN=hf_xxxxxxxxxxxxx")
        sys.exit(1)

    # Parse args
    max_results = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    output_file = sys.argv[2] if len(sys.argv) > 2 else "hf_llm_models.csv"

    # Fetch and save
    models = fetch_llm_models(max_results=max_results)
    save_to_csv(models, output_file)

#!/usr/bin/env python3
"""
Model manager: load/save/edit model list from CSV, interact with HuggingFace.
Integrates with chat commands: /model, /batch, /add
"""
import csv
import json
import subprocess
from pathlib import Path
from typing import Optional
from rich.table import Table
from rich.console import Console

console = Console()

# Default HF models CSV
DEFAULT_CSV = Path(__file__).parent.parent / "hf_llm_models.csv"


class ModelManager:
    """Manage model list from CSV and HuggingFace."""

    def __init__(self, csv_file: str = str(DEFAULT_CSV)):
        self.csv_file = Path(csv_file)
        self.models: list[dict] = []
        self.load()

    def load(self) -> None:
        """Load models from CSV."""
        if not self.csv_file.exists():
            console.print(f"[yellow]Models CSV not found: {self.csv_file}[/yellow]")
            console.print("Run: python fetch_models.py")
            return

        with open(self.csv_file) as f:
            reader = csv.DictReader(f)
            self.models = list(reader)
        console.print(f"[dim]✓ Loaded {len(self.models)} models[/dim]")

    def save(self) -> None:
        """Save models back to CSV."""
        if not self.models:
            return
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.models[0].keys())
            writer.writeheader()
            writer.writerows(self.models)
        console.print(f"[green]✓ Saved {len(self.models)} models[/green]")

    def search(self, query: str, top_n: int = 20) -> list[dict]:
        """Search models by name/author/id."""
        query = query.lower()
        results = [
            m for m in self.models
            if query in m.get("model_id", "").lower()
            or query in m.get("model_name", "").lower()
            or query in m.get("author", "").lower()
        ]
        return sorted(results, key=lambda x: int(x.get("downloads", 0)), reverse=True)[:top_n]

    def get_by_index(self, idx: int) -> Optional[dict]:
        """Get model by 1-indexed position."""
        sorted_models = sorted(self.models, key=lambda x: int(x.get("downloads", 0)), reverse=True)
        if 0 < idx <= len(sorted_models):
            return sorted_models[idx - 1]
        return None

    def refresh_from_hf(self, num_models: int = 300, hf_token: Optional[str] = None) -> None:
        """Update models list from HuggingFace."""
        import os
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            console.print("[red]Error: HF_TOKEN not set[/red]")
            return

        console.print(f"[blue]Fetching {num_models} models from HuggingFace...[/blue]")
        try:
            # Run fetch_models.py
            cmd = [
                "python",
                str(Path(__file__).parent.parent / "fetch_models.py"),
                str(num_models),
                str(self.csv_file),
            ]
            env = os.environ.copy()
            env["HF_TOKEN"] = token
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.load()
                console.print("[green]✓ Models updated from HuggingFace[/green]")
            else:
                console.print(f"[red]Error: {result.stderr}[/red]")
        except Exception as e:
            console.print(f"[red]Error updating models: {e}[/red]")

    def add_model(self, model_id: str, author: str = "custom", **kwargs) -> None:
        """Add a custom model to the list."""
        entry = {
            "model_id": model_id,
            "model_name": model_id.split("/")[-1],
            "author": author,
            "downloads": "0",
            "likes": "0",
            "tags": "",
            "gated": "False",
            "hf_url": f"https://huggingface.co/{model_id}",
            "inference_api": f"https://api-inference.huggingface.co/models/{model_id}/v1",
        }
        entry.update(kwargs)
        self.models.append(entry)
        self.save()
        console.print(f"[green]✓ Added {model_id}[/green]")

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the list."""
        self.models = [m for m in self.models if m.get("model_id") != model_id]
        self.save()
        console.print(f"[green]✓ Removed {model_id}[/green]")

    def list_models(self, limit: int = 20, search: Optional[str] = None) -> None:
        """Display models in a table."""
        if search:
            models = self.search(search, top_n=limit)
            title = f"Models matching '{search}' ({len(models)})"
        else:
            models = sorted(self.models, key=lambda x: int(x.get("downloads", 0)), reverse=True)[:limit]
            title = f"Top {min(limit, len(self.models))} models by downloads"

        table = Table(title=title)
        table.add_column("#", style="cyan", justify="right", width=3)
        table.add_column("Model", style="white")
        table.add_column("Author", style="magenta")
        table.add_column("Downloads", justify="right", style="green")

        for i, model in enumerate(models, 1):
            table.add_row(
                str(i),
                model.get("model_name", "")[:30],
                model.get("author", "")[:20],
                f"{int(model.get('downloads', 0)):,}",
            )

        console.print(table)

    def get_cli_command(self, model_id: str, tasks: str = "sample", mode: str = "estimathon", limit: int = 7) -> str:
        """Generate CLI command for a model."""
        short_name = model_id.split("/")[-1]
        return f"""python cli.py run \\
  --model {short_name} \\
  --provider hf \\
  --tasks {tasks} \\
  --mode {mode} \\
  --limit {limit}"""


if __name__ == "__main__":
    import sys

    mgr = ModelManager()

    if len(sys.argv) < 2:
        mgr.list_models(limit=20)
    elif sys.argv[1] == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        mgr.list_models(search=query)
    elif sys.argv[1] == "refresh":
        mgr.refresh_from_hf()
    elif sys.argv[1] == "add":
        model_id = sys.argv[2] if len(sys.argv) > 2 else ""
        if model_id:
            mgr.add_model(model_id)
        else:
            console.print("Usage: model_manager.py add <model_id>")
    elif sys.argv[1] == "remove":
        model_id = sys.argv[2] if len(sys.argv) > 2 else ""
        if model_id:
            mgr.remove_model(model_id)
        else:
            console.print("Usage: model_manager.py remove <model_id>")

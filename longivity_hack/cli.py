"""
longevity — CLI benchmark runner for Longevity-LLM evaluation.

Usage:
  longevity run   --model <id> --provider <hf|openai|anthropic|endpoint> [options]
  longevity config set <key> <value>
  longevity config get <key>
  longevity config list
  longevity status --model <id> --provider <type> [--api-key <key>] [--endpoint <url>]
  longevity tasks  [--tasks <source>] [--limit <n>]
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from benchmark import config as cfg
from benchmark.client import ModelClient
from benchmark.loader import load_tasks
from benchmark.results import ResultWriter
from benchmark.runner import run_eval

app = typer.Typer(
    name="longevity",
    help="Benchmark runner for Longevity-LLM and any compatible model.",
    add_completion=False,
)
config_app = typer.Typer(help="Manage stored configuration values.")
app.add_typer(config_app, name="config")

console = Console()


class Provider(str, Enum):
    hf = "hf"
    openai = "openai"
    anthropic = "anthropic"
    endpoint = "endpoint"


class EvalMode(str, Enum):
    one_shot = "one-shot"
    iterative = "iterative"


# ---------------------------------------------------------------------------
# longevity run
# ---------------------------------------------------------------------------

@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model ID or name (e.g. meta-llama/Meta-Llama-3-8B-Instruct)"),
    provider: Provider = typer.Option(Provider.hf, "--provider", "-p", help="Model provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (overrides config/env)"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="Custom base URL (for provider=endpoint)"),
    tasks: str = typer.Option("longebench", "--tasks", "-t", help='"longebench", "longebench:extra", or path to .jsonl'),
    mode: EvalMode = typer.Option(EvalMode.one_shot, "--mode", help="Evaluation mode"),
    output: Path = typer.Option(Path("results.jsonl"), "--output", "-o", help="Output file path"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", min=1, max=8, help="Parallel requests (max 8)"),
    budget: int = typer.Option(5, "--budget", "-b", help="Feedback rounds per task (iterative mode)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Cap number of tasks (for dry-runs)"),
    think: bool = typer.Option(False, "--think/--no-think", help="Enable chain-of-thought traces"),
):
    """Run the benchmark against a model and collect results."""
    resolved_key = api_key or cfg.provider_api_key(provider.value)

    # Credentials check
    if provider != Provider.endpoint:
        err = cfg.provider_preflight(provider.value, api_key)
        if err:
            console.print(f"[red]Error:[/red] {err}")
            raise typer.Exit(1)
    elif not resolved_key and provider == Provider.endpoint:
        # endpoint provider can work without a key (some local servers don't require one)
        resolved_key = resolved_key or "none"

    client = ModelClient(
        provider=provider.value,
        model_id=model,
        api_key=resolved_key or "none",
        endpoint_url=endpoint,
    )

    console.print(
        Panel(
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Provider:[/bold] {provider.value}\n"
            f"[bold]Tasks:[/bold] {tasks}  [bold]Mode:[/bold] {mode.value}\n"
            f"[bold]Concurrency:[/bold] {concurrency}  [bold]Budget:[/bold] {budget}  [bold]Think:[/bold] {think}",
            title="[bold green]Longevity Benchmark Run[/bold green]",
            expand=False,
        )
    )

    try:
        task_iter = load_tasks(tasks, limit=limit)
    except (FileNotFoundError, ImportError) as exc:
        console.print(f"[red]Error loading tasks:[/red] {exc}")
        raise typer.Exit(1)

    completed = 0
    errors = 0
    correct_count = 0

    summary_table = Table(title="Results", show_lines=False)
    summary_table.add_column("lb_id", style="cyan", no_wrap=True)
    summary_table.add_column("domain")
    summary_table.add_column("gold", justify="right")
    summary_table.add_column("pred / score", justify="right")
    summary_table.add_column("status", justify="center")

    with ResultWriter(str(output)) as writer:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task_bar = progress.add_task("Running...", total=None)

            def on_result(record: dict):
                nonlocal completed, errors, correct_count
                writer.write(record)
                completed += 1

                if "error" in record:
                    errors += 1
                    status = "[red]ERROR[/red]"
                    pred_str = record["error"][:30]
                elif mode == EvalMode.iterative:
                    score = record.get("task_score", "?")
                    rtc = record.get("rounds_to_correct")
                    status = "[green]OK[/green]" if rtc is not None else "[yellow]MISS[/yellow]"
                    pred_str = f"score={score}"
                else:
                    correct = record.get("correct", False)
                    if correct:
                        correct_count += 1
                    status = "[green]✓[/green]" if correct else "[red]✗[/red]"
                    pred_str = (record.get("pred") or "")[:20]

                summary_table.add_row(
                    record.get("lb_id", ""),
                    record.get("domain", ""),
                    str(record.get("gold", "")),
                    pred_str,
                    status,
                )
                progress.update(task_bar, advance=1, description=f"Done {completed}")

            run_eval(
                tasks=load_tasks(tasks, limit=limit),
                client=client,
                mode=mode.value,
                budget=budget,
                concurrency=concurrency,
                enable_thinking=think,
                on_result=on_result,
            )

    console.print(summary_table)

    if mode == EvalMode.one_shot:
        acc = correct_count / completed if completed else 0
        console.print(
            Panel(
                f"Tasks: {completed}  Correct: {correct_count}  Errors: {errors}\n"
                f"Accuracy: [bold]{acc:.1%}[/bold]",
                title="[bold]Summary[/bold]",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                f"Tasks: {completed}  Errors: {errors}\n"
                f"Results written to [bold]{output}[/bold]",
                title="[bold]Summary[/bold]",
                expand=False,
            )
        )

    console.print(f"\n[dim]Results saved → {output}[/dim]")


# ---------------------------------------------------------------------------
# longevity status
# ---------------------------------------------------------------------------

@app.command()
def status(
    model: str = typer.Option(..., "--model", "-m", help="Model ID"),
    provider: Provider = typer.Option(Provider.hf, "--provider", "-p"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e"),
):
    """Check connectivity and latency to a model endpoint."""
    resolved_key = api_key or cfg.provider_api_key(provider.value) or "none"
    client = ModelClient(
        provider=provider.value,
        model_id=model,
        api_key=resolved_key,
        endpoint_url=endpoint,
    )

    console.print(f"Checking [bold]{model}[/bold] via [cyan]{provider.value}[/cyan]...")
    ok, latency, detail = client.health_check()

    if ok:
        console.print(f"[green]OK[/green]  latency={latency:.2f}s  response={detail!r}")
    else:
        console.print(f"[red]FAIL[/red]  {detail}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# longevity tasks
# ---------------------------------------------------------------------------

@app.command()
def tasks(
    source: str = typer.Option("longebench", "--tasks", "-t"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of tasks to preview"),
):
    """Preview benchmark tasks from a source."""
    console.print(f"Loading tasks from [cyan]{source}[/cyan] (first {limit})...")

    try:
        task_list = list(load_tasks(source, limit=limit))
    except (FileNotFoundError, ImportError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    table = Table(title=f"Tasks — {source}")
    table.add_column("lb_id", style="cyan")
    table.add_column("domain")
    table.add_column("format")
    table.add_column("metric")
    table.add_column("msg_len", justify="right")

    for t in task_list:
        msgs = t.get("messages", [])
        table.add_row(
            t.get("lb_id", ""),
            t.get("domain", ""),
            t.get("format", ""),
            t.get("metric", ""),
            str(len(msgs)),
        )

    console.print(table)
    console.print(f"[dim]Total loaded: {len(task_list)}[/dim]")


# ---------------------------------------------------------------------------
# longevity config
# ---------------------------------------------------------------------------

@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g. hf.token)"),
    value: str = typer.Argument(..., help="Value to store"),
):
    """Store a config value persistently."""
    cfg.set_value(key, value)
    console.print(f"[green]Set[/green] {key} = {value!r}")


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key to retrieve"),
):
    """Read a single config value."""
    val = cfg.get(key)
    if val is None:
        console.print(f"[yellow]{key}[/yellow] is not set")
    else:
        console.print(f"{key} = {val!r}")


@config_app.command("list")
def config_list():
    """Show all current config values."""
    table = Table(title="Config (~/.longevity/config.json + env overrides)")
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    for k, v in sorted(cfg.all_values().items()):
        display = str(v) if v is not None else "[dim]not set[/dim]"
        # Mask tokens/keys
        if v and isinstance(v, str) and len(v) > 8 and any(x in k for x in ("token", "key")):
            display = v[:4] + "..." + v[-4:]
        table.add_row(k, display)

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def entry():
    app()


if __name__ == "__main__":
    entry()

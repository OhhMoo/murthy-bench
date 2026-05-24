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

try:
    # installed as a package: longivity_hack.cli
    from .benchmark import config as cfg
    from .benchmark.chat import run_chat
    from .benchmark.client import ModelClient
    from .benchmark.loader import load_tasks
    from .benchmark.results import ResultWriter
    from .benchmark.runner import run_eval, run_estimathon, run_mixed, _fmt_score, _ESTIMATHON_FORMATS
except ImportError:
    # local dev: python cli.py from inside longivity_hack/
    from benchmark import config as cfg
    from benchmark.chat import run_chat
    from benchmark.client import ModelClient
    from benchmark.loader import load_tasks
    from benchmark.results import ResultWriter
    from benchmark.runner import run_eval, run_estimathon, run_mixed, _fmt_score, _ESTIMATHON_FORMATS

app = typer.Typer(
    name="longevity",
    help="Benchmark runner for Longevity-LLM and any compatible model.",
    add_completion=False,
)
config_app = typer.Typer(help="Manage stored configuration values.")
group_app = typer.Typer(help="Manage named model comparison groups.")
app.add_typer(config_app, name="config")
app.add_typer(group_app, name="group")

console = Console()


class Provider(str, Enum):
    hf = "hf"
    openai = "openai"
    anthropic = "anthropic"
    endpoint = "endpoint"


class EvalMode(str, Enum):
    one_shot   = "one-shot"
    estimathon = "estimathon"
    mixed      = "mixed"


# ---------------------------------------------------------------------------
# longevity run
# ---------------------------------------------------------------------------

@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m", help="Model ID or name"),
    provider: Provider = typer.Option(Provider.hf, "--provider", "-p", help="Model provider"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key (overrides config/env)"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", "-e", help="Custom base URL (for provider=endpoint)"),
    tasks: str = typer.Option("longebench", "--tasks", "-t", help='"longebench", "longebench:extra", or path to .jsonl'),
    mode: EvalMode = typer.Option(EvalMode.one_shot, "--mode", help="one-shot | estimathon | mixed"),
    output: Path = typer.Option(Path("results.jsonl"), "--output", "-o", help="Output file path"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", min=1, max=8, help="Parallel requests for one-shot (max 8)"),
    budget: Optional[int] = typer.Option(None, "--budget", "-b", help="Total slips for estimathon (default: auto floor(18/13 × N))"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Cap number of tasks"),
    think: bool = typer.Option(False, "--think/--no-think", help="Enable chain-of-thought traces"),
    extract: bool = typer.Option(False, "--extract/--no-extract", help="Use Claude to reformat verbose model outputs into required answer format"),
):
    """Run the benchmark against a model and collect results."""
    # Auto-resolve L-LLM endpoint and HF token from config if not supplied
    if model == "longevity-llm":
        if not endpoint:
            endpoint = cfg.get("llm.endpoint")
        if not api_key:
            api_key = cfg.get("hf.token")

    resolved_key = api_key or cfg.provider_api_key(provider.value)
    if provider != Provider.endpoint:
        err = cfg.provider_preflight(provider.value, api_key)
        if err:
            console.print(f"[red]Error:[/red] {err}")
            raise typer.Exit(1)
    resolved_key = resolved_key or "none"

    client = ModelClient(
        provider=provider.value,
        model_id=model,
        api_key=resolved_key,
        endpoint_url=endpoint,
    )

    extractor_client = None
    if extract:
        ant_key = cfg.get("anthropic.api_key")
        if not ant_key:
            console.print("[yellow]Warning:[/yellow] --extract requires anthropic.api_key in config. Extraction disabled.")
        else:
            extractor_client = ModelClient(
                provider="anthropic",
                model_id="claude-haiku-4-5-20251001",
                api_key=ant_key,
            )
            console.print("  [dim]Extraction layer: Claude Haiku will reformat verbose outputs[/dim]")

    console.print(
        Panel(
            f"[bold]Model:[/bold] {model}\n"
            f"[bold]Provider:[/bold] {provider.value}  "
            f"[bold]Mode:[/bold] {mode.value}  "
            f"[bold]Think:[/bold] {think}\n"
            f"[bold]Tasks:[/bold] {tasks}" + (f"  [bold]Limit:[/bold] {limit}" if limit else ""),
            title="[bold green]Longevity Benchmark Run[/bold green]",
            expand=False,
        )
    )

    try:
        task_list = load_tasks(
            tasks,
            limit=limit,
            estimathon=(mode == EvalMode.estimathon),
            mixed=(mode == EvalMode.mixed),
        )
    except (FileNotFoundError, ImportError) as exc:
        console.print(f"[red]Error loading tasks:[/red] {exc}")
        raise typer.Exit(1)

    console.print(f"Loaded [bold]{len(task_list)}[/bold] tasks.")
    if mode == EvalMode.estimathon and tasks.startswith("longebench"):
        console.print("  [dim]Filtered to regression-compatible tasks (interval format)[/dim]")
    if mode == EvalMode.mixed:
        n_num = sum(1 for t in task_list if t.get("format") in _ESTIMATHON_FORMATS)
        n_cat = len(task_list) - n_num
        console.print(f"  [dim]Track 1 (Estimathon): {n_num} numerical  ·  Track 2 (one-shot): {n_cat} categorical[/dim]")

    with ResultWriter(str(output)) as writer:

        # ------------------------------------------------------------------
        # Estimathon mode — single multi-turn session, shared budget
        # ------------------------------------------------------------------
        if mode == EvalMode.estimathon:
            slip_count = 0

            def on_slip(record: dict):
                nonlocal slip_count
                slip_count += 1
                good = record.get("good", False)
                wf = record.get("width_factor")
                score_before = record["score_before"]
                score_after = record["score_after"]
                direction = "↓" if score_after < score_before else ("↑" if score_after > score_before else "=")
                console.print(
                    f"  Slip {record['slip']:2d}  "
                    f"[cyan]{record['pid']}[/cyan]  "
                    f"[{'green' if good else 'red'}]{'GOOD' if good else 'BAD '}[/]"
                    + (f"  width={wf}" if wf is not None else "")
                    + f"  score {_fmt_score(score_before)} {direction} {_fmt_score(score_after)}"
                    + (f"  [yellow]⚠ lost good interval[/yellow]" if record.get("prev_was_good") and not good else "")
                )
                question = record.get("task_content", "")
                if question:
                    console.print(f"     [dim]Q:[/dim] [dim]{question[:300].replace(chr(10), ' ')}[/dim]")
                raw = record.get("raw_response", "")
                if raw:
                    console.print(f"     [dim]→[/dim] [dim]{raw[:200]}[/dim]")

            session_result = run_estimathon(
                tasks=task_list,
                client=client,
                total_budget=budget,
                enable_thinking=think,
                extractor=extractor_client,
                on_slip=on_slip,
            )
            writer.write(session_result)

            ref_acc = session_result.get("refinement_accuracy")
            ref_str = f"{ref_acc:.0%}" if ref_acc is not None else "n/a"
            console.print(
                Panel(
                    f"Final score:  [bold]{_fmt_score(session_result['final_score'])}[/bold]  (lower is better)\n"
                    f"Problems solved:  {session_result['n_good_final']} / {session_result['n_problems']}\n"
                    f"Slips used:  {session_result['slips_used']} / {session_result['total_budget']}\n"
                    f"Refinement accuracy:  {ref_str}  "
                    f"({session_result['refinement_successes']}/{session_result['refinement_attempts']} bets won)",
                    title="[bold]Session Summary[/bold]",
                    expand=False,
                )
            )

        # ------------------------------------------------------------------
        # Mixed mode — Estimathon for numerical, one-shot for categorical
        # ------------------------------------------------------------------
        elif mode == EvalMode.mixed:
            def on_slip_mixed(record: dict):
                good = record.get("good", False)
                wf   = record.get("width_factor")
                sb, sa = record["score_before"], record["score_after"]
                direction = "↓" if sa < sb else ("↑" if sa > sb else "=")
                console.print(
                    f"  [dim]Slip {record['slip']:2d}[/dim]  "
                    f"[cyan]{record['pid']}[/cyan]  "
                    f"[{'green' if good else 'red'}]{'GOOD' if good else 'BAD '}[/]"
                    + (f"  width={wf}" if wf is not None else "")
                    + f"  score {_fmt_score(sb)} {direction} {_fmt_score(sa)}"
                    + ("  [yellow]⚠ lost good[/yellow]" if record.get("prev_was_good") and not good else "")
                )
                question = record.get("task_content", "")
                if question:
                    console.print(f"     [dim]Q:[/dim] [dim]{question[:300].replace(chr(10), ' ')}[/dim]")
                raw = record.get("raw_response", "")
                if raw:
                    console.print(f"     [dim]→[/dim] [dim]{raw[:200]}[/dim]")

            cat_table = Table(show_lines=False)
            cat_table.add_column("lb_id",  style="cyan", no_wrap=True)
            cat_table.add_column("format", style="dim")
            cat_table.add_column("gold",   justify="right")
            cat_table.add_column("pred",   justify="right")
            cat_table.add_column("",       justify="center")

            def on_result_mixed(record: dict):
                writer.write(record)
                correct = record.get("correct", False)
                cat_table.add_row(
                    record.get("lb_id", ""),
                    record.get("format", ""),
                    str(record.get("gold", ""))[:12],
                    (record.get("pred") or "")[:20],
                    "[green]✓[/green]" if correct else "[red]✗[/red]",
                )

            mixed_result = run_mixed(
                tasks=task_list,
                client=client,
                total_budget=budget,
                concurrency=concurrency,
                enable_thinking=think,
                extractor=extractor_client,
                on_slip=on_slip_mixed,
                on_result=on_result_mixed,
            )
            writer.write(mixed_result)

            # --- Estimathon summary ---
            if mixed_result.get("estimathon"):
                er = mixed_result["estimathon"]
                ref_acc = er.get("refinement_accuracy")
                ref_str = f"{ref_acc:.0%}" if ref_acc is not None else "n/a"
                console.print(Panel(
                    f"[bold]Track 1 — Estimathon[/bold]  ({mixed_result['n_numerical']} numerical tasks)\n"
                    f"Final score:  [bold]{_fmt_score(er['final_score'])}[/bold]  (lower is better)\n"
                    f"Problems solved:  {er['n_good_final']} / {er['n_problems']}\n"
                    f"Slips used:  {er['slips_used']} / {er['total_budget']}\n"
                    f"Refinement accuracy:  {ref_str}",
                    title="[bold]Estimathon Summary[/bold]", expand=False,
                ))

            # --- One-shot summary ---
            if mixed_result.get("one_shot"):
                os_r = mixed_result["one_shot"]
                console.print(cat_table)
                fmt_lines = ""
                for fmt, stats in sorted(os_r["by_format"].items()):
                    fmt_lines += f"  {fmt:<14}  {stats['correct']:>3}/{stats['n']:<3}  ({stats['accuracy']:.0%})\n"
                console.print(Panel(
                    f"[bold]Track 2 — One-shot[/bold]  ({mixed_result['n_categorical']} categorical tasks)\n"
                    + fmt_lines.rstrip()
                    + f"\n  {'Overall':<14}  {os_r['n_correct']:>3}/{os_r['n_tasks']:<3}  ({os_r['accuracy']:.0%})",
                    title="[bold]One-shot Summary[/bold]", expand=False,
                ))

        # ------------------------------------------------------------------
        # One-shot mode — parallel, one request per task
        # ------------------------------------------------------------------
        else:
            completed = 0
            correct_count = 0

            summary_table = Table(show_lines=False)
            summary_table.add_column("lb_id", style="cyan", no_wrap=True)
            summary_table.add_column("domain")
            summary_table.add_column("gold", justify="right")
            summary_table.add_column("pred", justify="right")
            summary_table.add_column("", justify="center")

            def on_result(record: dict):
                nonlocal completed, correct_count
                writer.write(record)
                completed += 1
                correct = record.get("correct", False)
                if correct:
                    correct_count += 1
                summary_table.add_row(
                    record.get("lb_id", ""),
                    record.get("domain", ""),
                    str(record.get("gold", ""))[:12],
                    (record.get("pred") or "")[:20],
                    "[green]✓[/green]" if correct else "[red]✗[/red]",
                )

            run_eval(
                tasks=iter(task_list),
                client=client,
                concurrency=concurrency,
                enable_thinking=think,
                extractor=extractor_client,
                on_result=on_result,
            )

            console.print(summary_table)
            acc = correct_count / completed if completed else 0
            console.print(
                Panel(
                    f"Tasks: {completed}  Correct: {correct_count}\n"
                    f"Accuracy: [bold]{acc:.1%}[/bold]",
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
    if model == "longevity-llm":
        if not endpoint:
            endpoint = cfg.get("llm.endpoint")
        if not api_key:
            api_key = cfg.get("hf.token")
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
# longevity group
# ---------------------------------------------------------------------------

@group_app.command("add")
def group_add(
    name: str = typer.Argument(..., help="Group name"),
    entries: list[str] = typer.Argument(
        ..., help='One or more entries as "label:model:provider"'
    ),
):
    """Create or replace a named model group.

    Example:
      longevity group add longevity \\
        "L-LLM:longevity-llm:endpoint" \\
        "Claude:claude-sonnet-4-6:anthropic" \\
        "Qwen3-8B:Qwen/Qwen3-8B-Instruct:hf"
    """
    parsed: list[dict] = []
    for raw in entries:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            console.print(f"[red]Bad entry[/red] {raw!r} — expected label:model:provider")
            raise typer.Exit(1)
        label, model_id, prov = parts
        if prov not in ("hf", "openai", "anthropic", "endpoint"):
            console.print(f"[red]Unknown provider[/red] {prov!r}. Use: hf | openai | anthropic | endpoint")
            raise typer.Exit(1)
        parsed.append({"label": label, "model": model_id, "provider": prov})
    cfg.set_group(name, parsed)
    console.print(f"[green]Saved group[/green] [bold]{name}[/bold] ({len(parsed)} models)")


@group_app.command("list")
def group_list():
    """List all saved group names."""
    groups = cfg.get_groups()
    if not groups:
        console.print("[dim]No groups saved. Use: longevity group add <name> ...[/dim]")
        return
    table = Table(title="Saved Groups")
    table.add_column("Name", style="cyan")
    table.add_column("Models", justify="right")
    for gname, entries in sorted(groups.items()):
        table.add_row(gname, str(len(entries)))
    console.print(table)


@group_app.command("show")
def group_show(
    name: str = typer.Argument(..., help="Group name to inspect"),
):
    """Show all models in a group."""
    entries = cfg.get_group(name)
    if entries is None:
        console.print(f"[red]Group {name!r} not found.[/red]")
        raise typer.Exit(1)
    table = Table(title=f"Group: {name}")
    table.add_column("Label", style="cyan")
    table.add_column("Model")
    table.add_column("Provider")
    for e in entries:
        table.add_row(e["label"], e["model"], e["provider"])
    console.print(table)


@group_app.command("remove")
def group_remove(
    name: str = typer.Argument(..., help="Group name to delete"),
):
    """Delete a saved group."""
    if cfg.remove_group(name):
        console.print(f"[green]Removed group[/green] [bold]{name}[/bold]")
    else:
        console.print(f"[yellow]Group {name!r} not found.[/yellow]")


# ---------------------------------------------------------------------------
# longevity compare
# ---------------------------------------------------------------------------

@app.command()
def compare(
    group: str = typer.Option(..., "--group", "-g", help="Named model group to compare"),
    tasks: str = typer.Option("longebench", "--tasks", "-t", help='"longebench", "longebench:extra", or path to .jsonl'),
    mode: EvalMode = typer.Option(EvalMode.one_shot, "--mode", help="one-shot | mixed"),
    output: Path = typer.Option(Path("compare_results.jsonl"), "--output", "-o"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", min=1, max=8),
    budget: Optional[int] = typer.Option(None, "--budget", "-b", help="Estimathon slip budget (default: auto floor(18/13 × N))"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l"),
    think: bool = typer.Option(False, "--think/--no-think"),
):
    """Run benchmark tasks against every model in a group and print a comparison table."""
    entries = cfg.get_group(group)
    if not entries:
        console.print(f"[red]Group {group!r} not found.[/red] Use: longevity group add {group} ...")
        raise typer.Exit(1)

    try:
        task_list = load_tasks(tasks, limit=limit)
    except (FileNotFoundError, ImportError) as exc:
        console.print(f"[red]Error loading tasks:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]Group:[/bold] {group}  ({len(entries)} models)\n"
            f"[bold]Tasks:[/bold] {tasks}  ({len(task_list)} loaded)\n"
            f"[bold]Mode:[/bold] {mode.value}  [bold]Think:[/bold] {think}",
            title="[bold green]Longevity Compare[/bold green]",
            expand=False,
        )
    )

    summary_rows: list[tuple] = []

    with ResultWriter(str(output)) as writer:
        for entry in entries:
            label = entry["label"]
            model_id = entry["model"]
            prov = entry["provider"]

            # Resolve endpoint for L-LLM shorthand
            ep_url: str | None = entry.get("endpoint")
            if prov == "endpoint" and not ep_url:
                ep_url = cfg.get("llm.endpoint")

            api_key = cfg.provider_api_key(prov) or "none"
            client = ModelClient(
                provider=prov,
                model_id=model_id,
                api_key=api_key,
                endpoint_url=ep_url,
            )

            console.print(f"\n[bold cyan]{label}[/bold cyan] — {model_id} via {prov}")

            n_correct = 0
            n_tasks = 0
            total_tokens = 0

            def _on_result(record: dict):
                nonlocal n_correct, n_tasks, total_tokens
                record["model_label"] = label
                writer.write(record)
                n_tasks += 1
                if record.get("correct"):
                    n_correct += 1
                total_tokens += record.get("tokens_used", 0)

            with Progress(
                SpinnerColumn(), TextColumn("{task.description}"),
                BarColumn(), TaskProgressColumn(),
                console=console, transient=True,
            ) as prog:
                ptask = prog.add_task(f"Evaluating {label}...", total=len(task_list))

                def _on_result_tracked(record: dict, _ptask=ptask):
                    _on_result(record)
                    prog.advance(_ptask)

                if mode == EvalMode.mixed:
                    mixed_result = run_mixed(
                        tasks=task_list,
                        client=client,
                        total_budget=budget,
                        concurrency=concurrency,
                        enable_thinking=think,
                        on_result=_on_result_tracked,
                    )
                    mixed_result["model_label"] = label
                    writer.write(mixed_result)
                else:
                    run_eval(
                        tasks=iter(task_list),
                        client=client,
                        concurrency=concurrency,
                        enable_thinking=think,
                        on_result=_on_result_tracked,
                    )

            avg_tok = total_tokens // n_tasks if n_tasks else 0
            acc = n_correct / n_tasks if n_tasks else 0.0
            summary_rows.append((label, n_tasks, n_correct, acc, avg_tok))
            console.print(
                f"  {label}: {n_correct}/{n_tasks} correct  ({acc:.1%})  avg {avg_tok} tok"
            )

    # Final comparison table
    table = Table(title="[bold]Comparison Summary[/bold]", show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right", style="bold")
    table.add_column("Avg Tokens", justify="right")
    for label, n_tasks, n_correct, acc, avg_tok in summary_rows:
        table.add_row(label, str(n_tasks), str(n_correct), f"{acc:.1%}", str(avg_tok))
    console.print("\n")
    console.print(table)
    console.print(f"\n[dim]Full results → {output}[/dim]")


# ---------------------------------------------------------------------------
# longevity chat
# ---------------------------------------------------------------------------

@app.command()
def chat(
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="Claude model to power the chat"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Anthropic API key (overrides config)"),
):
    """Interactive chat assistant — load datasets, run benchmarks, check models."""
    run_chat(chat_model=model, api_key=api_key)


# Default: open chat when no subcommand given
@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        run_chat()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def entry():
    app()


if __name__ == "__main__":
    entry()

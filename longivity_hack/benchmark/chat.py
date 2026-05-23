"""
Interactive chat session powered by Claude.
Green-themed UI with ASCII banner, slash commands, and live thinking indicators.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field

import anthropic
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style as PTStyle
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from . import config as cfg
from .client import ModelClient
from .loader import load_tasks
from .results import ResultWriter
from .runner import run_estimathon, run_eval

console = Console()

# ---------------------------------------------------------------------------
# ASCII art banner — "MURPHY" with dark→bright green gradient
# ---------------------------------------------------------------------------

_BANNER_ROWS = [
    "  ███╗   ███╗██╗   ██╗██████╗ ██████╗ ██╗  ██╗██╗   ██╗",
    "  ████╗ ████║██║   ██║██╔══██╗██╔══██╗██║  ██║╚██╗ ██╔╝",
    "  ██╔████╔██║██║   ██║██████╔╝██████╔╝███████║ ╚████╔╝ ",
    "  ██║╚██╔╝██║██║   ██║██╔══██╗██╔═══╝ ██╔══██║  ╚██╔╝  ",
    "  ██║ ╚═╝ ██║╚██████╔╝██║  ██║██║     ██║  ██║   ██║   ",
    "  ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ",
]

# Dark bamboo forest → bright bamboo shoot tip: one colour per row
_GRADIENT = [
    "rgb(45,80,25)",
    "rgb(70,110,40)",
    "rgb(100,145,55)",
    "rgb(135,175,70)",
    "rgb(165,200,85)",
    "rgb(195,225,100)",
]

_VERSION = "v0.2.0"
_TAGLINE = "Longevity LLM Benchmark  ·  Estimathon-style evaluation"

# ---------------------------------------------------------------------------
# Slash-command autocomplete (prompt_toolkit)
# ---------------------------------------------------------------------------

_SLASH_META = [
    ("/help",         "Show all commands"),
    ("/exit",         "Exit the chat"),
    ("/clear",        "Clear conversation history"),
    ("/model",        "Show or set benchmark model"),
    ("/provider",     "Show or set provider  (anthropic|openai|hf|endpoint)"),
    ("/tasks",        "Show or set default task source"),
    ("/think",        "Toggle chain-of-thought traces"),
    ("/question_set", "Preview tasks from a source"),
    ("/benchmark",    "Quick-run estimathon with current defaults"),
    ("/status",       "Check model connectivity"),
    ("/config",       "View or set a config value"),
]

_PT_STYLE = PTStyle.from_dict({
    "prompt":                                  "bold #a0c850",
    "completion-menu.completion":              "bg:#152108 #c3df6e",
    "completion-menu.completion.current":      "bg:#3d6018 bold #eaf5a0",
    "completion-menu.meta.completion":         "bg:#152108 #7a9e3e",
    "completion-menu.meta.completion.current": "bg:#3d6018 #c8e064",
    "scrollbar.background":                    "bg:#152108",
    "scrollbar.button":                        "bg:#3d6018",
})


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        word = text.split()[0] if text.split() else text
        for cmd, meta in _SLASH_META:
            if cmd.startswith(word.lower()):
                yield Completion(cmd, start_position=-len(word), display_meta=meta)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class ChatState:
    chat_model:       str  = "claude-sonnet-4-6"
    bench_model:      str  = "claude-haiku-4-5-20251001"
    bench_provider:   str  = "anthropic"
    think_mode:       bool = False
    default_tasks:    str  = "sample"
    conversation:     list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Claude system prompt + tool definitions
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a helpful assistant for the Longevity Benchmark CLI.
You help researchers evaluate LLMs on aging-biology estimation tasks using an Estimathon-style benchmark.

Benchmark mechanics:
- Models submit intervals [min, max] for numerical questions (lifespan, drug extension %, biological age)
- Binary feedback only: GOOD (contains answer) or BAD — no directional hints
- Only the last submission per problem counts toward the final score
- Shared submission budget across all problems
- Score = (10 + Σ floor(max/min) for good final answers) × 2^(N − # good) — lower is better

You have three tools:
- preview_tasks  — load and display tasks from a source
- run_benchmark  — run a full benchmark session against any model
- check_model    — verify connectivity and latency to a model endpoint

Available task sources: "sample" (built-in), "longebench" (HuggingFace, gated), or a local .jsonl path.
Providers: anthropic, openai, hf, endpoint (any OpenAI-compatible URL).

Be concise. Call tools immediately when asked — do not ask for confirmation.
"""

_TOOLS = [
    {
        "name": "preview_tasks",
        "description": "Load and display benchmark tasks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "'sample', 'longebench', 'longebench:extra', or file path"},
                "limit":  {"type": "integer", "description": "Max tasks to show. Default 7."},
            },
            "required": ["source"],
        },
    },
    {
        "name": "run_benchmark",
        "description": "Run a benchmark session (estimathon or one-shot) against a model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model":        {"type": "string"},
                "provider":     {"type": "string", "enum": ["anthropic", "openai", "hf", "endpoint"]},
                "api_key":      {"type": "string"},
                "endpoint_url": {"type": "string"},
                "tasks_source": {"type": "string"},
                "mode":         {"type": "string", "enum": ["estimathon", "one-shot"]},
                "budget":       {"type": "integer"},
                "limit":        {"type": "integer"},
                "think":        {"type": "boolean"},
                "output":       {"type": "string"},
            },
            "required": ["model", "provider", "tasks_source"],
        },
    },
    {
        "name": "check_model",
        "description": "Check connectivity and latency to a model endpoint.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model":        {"type": "string"},
                "provider":     {"type": "string", "enum": ["anthropic", "openai", "hf", "endpoint"]},
                "api_key":      {"type": "string"},
                "endpoint_url": {"type": "string"},
            },
            "required": ["model", "provider"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _resolve_client(model: str, provider: str, api_key: str | None, endpoint_url: str | None) -> ModelClient | str:
    key = api_key or cfg.provider_api_key(provider) or "none"
    if provider != "endpoint" and not api_key:
        err = cfg.provider_preflight(provider, api_key)
        if err:
            return f"Credential error: {err}"
    return ModelClient(provider=provider, model_id=model, api_key=key, endpoint_url=endpoint_url)


def _tool_preview_tasks(source: str, limit: int = 7) -> str:
    try:
        task_list = list(load_tasks(source, limit=limit))
    except Exception as exc:
        return f"Error: {exc}"

    table = Table(title=f"[green]{source}[/green]  ({len(task_list)} tasks)", border_style="green")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Domain")
    table.add_column("Format")
    table.add_column("Gold", justify="right")
    for t in task_list:
        gold = (t.get("messages") or [{}])[-1].get("content", "")[:10]
        table.add_row(t.get("lb_id", ""), t.get("domain", ""), t.get("format", ""), gold)
    console.print(table)
    lines = [f"{t.get('lb_id','')} | {t.get('domain','')} | gold≈{(t.get('messages') or [{}])[-1].get('content','')[:8]}" for t in task_list]
    return f"Loaded {len(task_list)} tasks from '{source}':\n" + "\n".join(lines)


def _tool_check_model(model: str, provider: str, api_key: str | None = None, endpoint_url: str | None = None) -> str:
    client = _resolve_client(model, provider, api_key, endpoint_url)
    if isinstance(client, str):
        return client
    console.print(f"  [dim green]Pinging[/dim green] [cyan]{model}[/cyan] via [green]{provider}[/green]…")
    ok, latency, detail = client.health_check()
    if ok:
        console.print(f"  [green]●[/green] [bold green]Online[/bold green]  {latency:.2f}s  [dim]{detail[:60]}[/dim]")
        return f"OK — {model} via {provider} in {latency:.2f}s"
    console.print(f"  [red]●[/red] [bold red]Offline[/bold red]  {detail[:80]}")
    return f"FAIL — {detail}"


def _tool_run_benchmark(
    model: str, provider: str, tasks_source: str,
    api_key: str | None = None, endpoint_url: str | None = None,
    mode: str = "estimathon", budget: int | None = None,
    limit: int | None = None, think: bool = False, output: str = "results.jsonl",
) -> str:
    client = _resolve_client(model, provider, api_key, endpoint_url)
    if isinstance(client, str):
        return client
    try:
        task_list = list(load_tasks(tasks_source, limit=limit))
    except Exception as exc:
        return f"Error loading tasks: {exc}"
    if not task_list:
        return "No tasks loaded."

    console.print(Panel(
        f"[green]Model:[/green] [cyan]{model}[/cyan]   "
        f"[green]Provider:[/green] {provider}   "
        f"[green]Mode:[/green] {mode}   "
        f"[green]Tasks:[/green] {len(task_list)}   "
        f"[green]Think:[/green] {think}",
        border_style="green", expand=False,
    ))

    with ResultWriter(output) as writer:
        if mode == "estimathon":
            def on_slip(r: dict):
                good = r.get("good", False)
                wf   = r.get("width_factor")
                bar  = "█" * min(wf or 0, 20) if good else "░" * 10
                console.print(
                    f"  [dim]#{r['slip']:02d}[/dim]  "
                    f"[cyan]{r['pid']:<18}[/cyan]  "
                    f"[{'green' if good else 'red'}]{'GOOD' if good else 'BAD '}[/]"
                    + (f"  [green]{bar}[/green] w={wf}" if good else f"  [red]{bar}[/red]")
                    + f"  [dim]{r['score_before']} → {r['score_after']}[/dim]"
                    + (" [yellow]⚠ lost good[/yellow]" if r.get("prev_was_good") and not good else "")
                )
            result = run_estimathon(tasks=task_list, client=client,
                                    total_budget=budget, enable_thinking=think, on_slip=on_slip)
            writer.write(result)
            ref_acc = result.get("refinement_accuracy")
            ref_str = f"{ref_acc:.0%}" if ref_acc is not None else "n/a"
            summary = (
                f"Score: {result['final_score']}  ·  "
                f"Solved: {result['n_good_final']}/{result['n_problems']}  ·  "
                f"Slips: {result['slips_used']}/{result['total_budget']}  ·  "
                f"Refinement accuracy: {ref_str}"
            )
        else:
            records: list[dict] = []
            def on_result(r: dict):
                records.append(r)
                writer.write(r)
                ok = r.get("correct", False)
                console.print(
                    f"  [{'green' if ok else 'red'}]{'✓' if ok else '✗'}[/]  "
                    f"[cyan]{r.get('lb_id',''):<18}[/cyan]  "
                    f"gold=[dim]{str(r.get('gold',''))[:10]}[/dim]  "
                    f"pred=[dim]{str(r.get('pred',''))[:20]}[/dim]"
                )
            run_eval(tasks=iter(task_list), client=client, enable_thinking=think, on_result=on_result)
            n = len(records)
            correct = sum(1 for r in records if r.get("correct"))
            summary = f"{correct}/{n} correct ({correct/n:.0%})" if n else "0 results"

    console.print(Rule(style="green"))
    console.print(f"  [bold green]Done.[/bold green]  {summary}  [dim]→ {output}[/dim]")
    return summary


def _execute_tool(name: str, inputs: dict) -> str:
    if name == "preview_tasks":
        return _tool_preview_tasks(inputs["source"], inputs.get("limit", 7))
    if name == "check_model":
        return _tool_check_model(inputs["model"], inputs["provider"],
                                  inputs.get("api_key"), inputs.get("endpoint_url"))
    if name == "run_benchmark":
        return _tool_run_benchmark(**{k: inputs[k] for k in inputs})
    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def _help_panel() -> None:
    table = Table(border_style="green", show_header=False, box=None, padding=(0, 2))
    table.add_column("cmd",   style="bold green", no_wrap=True)
    table.add_column("args",  style="dim cyan",   no_wrap=True)
    table.add_column("desc",  style="white")
    rows = [
        ("/help",         "",                        "Show this help"),
        ("/exit",         "",                        "Exit the chat"),
        ("/clear",        "",                        "Clear conversation history"),
        ("/model",        "[bench-model]",           "Show or set default benchmark model"),
        ("/provider",     "[provider]",              "Show or set default provider  (anthropic|openai|hf|endpoint)"),
        ("/tasks",        "[source]",                "Show or set default task source  (sample|longebench|<path>)"),
        ("/think",        "",                        "Toggle chain-of-thought traces for benchmark runs"),
        ("/question_set", "[source] [limit]",        "Preview tasks from a source"),
        ("/benchmark",    "[model] [provider] [tasks]","Quick-run estimathon with current defaults"),
        ("/status",       "[model] [provider]",      "Check model connectivity"),
        ("/config",       "[key] [value]",           "View or set a config value"),
    ]
    for r in rows:
        table.add_row(*r)
    console.print(Panel(table, title="[rgb(195,225,100)]murphy  /  commands[/rgb(195,225,100)]", border_style="rgb(100,145,55)", expand=False))


def _handle_slash(cmd: str, args: list[str], state: ChatState) -> bool:
    """Handle a slash command. Returns True if handled."""
    cmd = cmd.lower()

    if cmd == "/help":
        _help_panel()
        return True

    if cmd == "/exit":
        console.print("[dim green]Goodbye.[/dim green]")
        raise SystemExit(0)

    if cmd == "/clear":
        state.conversation.clear()
        console.print("[green]●[/green] Conversation cleared.")
        return True

    if cmd == "/think":
        state.think_mode = not state.think_mode
        console.print(f"[green]●[/green] Think mode [bold]{'ON' if state.think_mode else 'OFF'}[/bold]")
        return True

    if cmd == "/model":
        if args:
            state.bench_model = args[0]
            console.print(f"[green]●[/green] Benchmark model → [cyan]{state.bench_model}[/cyan]")
        else:
            console.print(
                f"  chat model:      [cyan]{state.chat_model}[/cyan]\n"
                f"  bench model:     [cyan]{state.bench_model}[/cyan]\n"
                f"  bench provider:  [cyan]{state.bench_provider}[/cyan]"
            )
        return True

    if cmd == "/provider":
        if args:
            state.bench_provider = args[0]
            console.print(f"[green]●[/green] Provider → [cyan]{state.bench_provider}[/cyan]")
        else:
            console.print(f"  provider: [cyan]{state.bench_provider}[/cyan]")
        return True

    if cmd == "/tasks":
        if args:
            state.default_tasks = args[0]
            console.print(f"[green]●[/green] Default tasks → [cyan]{state.default_tasks}[/cyan]")
        else:
            console.print(f"  tasks: [cyan]{state.default_tasks}[/cyan]")
        return True

    if cmd == "/question_set":
        source = args[0] if args else state.default_tasks
        limit  = int(args[1]) if len(args) > 1 else 7
        _tool_preview_tasks(source, limit)
        return True

    if cmd == "/benchmark":
        model    = args[0] if len(args) > 0 else state.bench_model
        provider = args[1] if len(args) > 1 else state.bench_provider
        tasks    = args[2] if len(args) > 2 else state.default_tasks
        _tool_run_benchmark(model=model, provider=provider, tasks_source=tasks, think=state.think_mode)
        return True

    if cmd == "/status":
        model    = args[0] if len(args) > 0 else state.bench_model
        provider = args[1] if len(args) > 1 else state.bench_provider
        _tool_check_model(model=model, provider=provider)
        return True

    if cmd == "/config":
        if len(args) >= 2:
            cfg.set_value(args[0], args[1])
            console.print(f"[green]●[/green] {args[0]} = [cyan]{args[1]!r}[/cyan]")
        elif len(args) == 1:
            val = cfg.get(args[0])
            console.print(f"  {args[0]} = [cyan]{val!r}[/cyan]")
        else:
            for k, v in sorted(cfg.all_values().items()):
                display = str(v) if v is not None else "[dim]not set[/dim]"
                if v and isinstance(v, str) and len(v) > 8 and any(x in k for x in ("token", "key")):
                    display = v[:4] + "…" + v[-4:]
                console.print(f"  [green]{k}[/green] = [cyan]{display}[/cyan]")
        return True

    console.print(f"[red]Unknown command:[/red] {cmd}  — type [green]/help[/green] for commands")
    return True


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _print_welcome(state: ChatState) -> None:
    # Build gradient banner: each row gets its own green shade
    banner = Text()
    for i, (row, color) in enumerate(zip(_BANNER_ROWS, _GRADIENT)):
        banner.append(row, style=color)
        if i < len(_BANNER_ROWS) - 1:
            banner.append("\n")

    info = Text.from_markup(
        f"\n\n[dim]{_TAGLINE}[/dim]\n"
        f"[dim]{_VERSION}"
        f"  ·  model=[cyan]{state.bench_model}[/cyan]"
        f"  ·  provider=[cyan]{state.bench_provider}[/cyan]"
        f"  ·  tasks=[cyan]{state.default_tasks}[/cyan][/dim]\n\n"
        "[white]Type a message, or [green]/help[/green] for slash commands.[/white]"
    )
    console.print(Panel(
        Text.assemble(banner, info),
        title="[rgb(195,225,100)]murphy[/rgb(195,225,100)]",
        border_style="rgb(100,145,55)",
        padding=(0, 2),
    ))


@contextmanager
def _thinking(label: str = "Thinking"):
    spinner = Spinner("dots2", text=f" [rgb(160,200,80)]{label}…[/rgb(160,200,80)]")
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        yield


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def run_chat(chat_model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
    key = api_key or cfg.get("anthropic.api_key")
    if not key:
        console.print(
            "[red]No Anthropic API key found.[/red]\n"
            "Run: [green]python cli.py config set anthropic.api_key <key>[/green]"
        )
        return

    anthropic_client = anthropic.Anthropic(api_key=key)
    state = ChatState(chat_model=chat_model)
    pt_session: PromptSession = PromptSession(
        completer=SlashCompleter(),
        complete_while_typing=True,
        style=_PT_STYLE,
    )
    _print_welcome(state)

    while True:
        # Prompt
        try:
            console.print()
            user_input = pt_session.prompt([("class:prompt", "> ")])
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim green]Goodbye.[/dim green]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash command handling
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()
            args = parts[1:]
            try:
                _handle_slash(cmd, args, state)
            except SystemExit:
                raise
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
            continue

        # Regular message → send to Claude
        state.conversation.append({"role": "user", "content": user_input})

        # Agentic tool-use loop
        while True:
            with _thinking("Thinking"):
                response = anthropic_client.messages.create(
                    model=state.chat_model,
                    max_tokens=4096,
                    system=_SYSTEM,
                    tools=_TOOLS,
                    messages=state.conversation,
                )

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if text:
                    console.print(Rule(characters="─", style="dim green"))
                    console.print(Markdown(text))
                state.conversation.append({"role": "assistant", "content": response.content})
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    # Show tool call
                    args_preview = json.dumps(block.input, ensure_ascii=False)
                    if len(args_preview) > 100:
                        args_preview = args_preview[:97] + "…"
                    console.print(
                        f"  [dim green]⟳[/dim green]  "
                        f"[green]{block.name}[/green][dim]({args_preview})[/dim]"
                    )
                    with _thinking(f"Running {block.name}"):
                        result = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                state.conversation.append({"role": "assistant", "content": response.content})
                state.conversation.append({"role": "user", "content": tool_results})
                continue

            break  # unexpected stop_reason

# Murphy-Health — Longevity Benchmark CLI Dev Log

Iterative refinement benchmark for Longevity-LLM (Track 01 · Insilico Medicine Hackathon).
A CLI that evaluates any model on aging-biology tasks from LongeBench or custom interval tasks.

---

## 2026-05-23 — Initial CLI Design

### Context
The project idea (see `idea.md`) proposes an iterative refinement benchmark on top of
LongeBench: instead of one-shot Q&A, the model submits intervals [min, max] for numerical
tasks and receives yes/no feedback each round. The benchmark tests calibration and convergence,
not just final-answer accuracy.

Before writing any code, we evaluated the idea against Track 01's four scoring criteria:
- **Utility**: strong — tests a genuine failure mode (right answer, wrong reasoning) that
  one-shot metrics miss entirely.
- **Diversity**: strong for numerical tasks (AnAge lifespan, DrugAge extension %); needs care
  to avoid mixing binary/MCQ formats under the same estimathon scoring formula.
- **Retrieval resistance**: strong — interval format makes memorisation hard to exploit even
  if the ground truth value was in pretraining data.
- **Statistical rigor**: clean — estimathon score is mathematically well-defined; needs a
  baseline (e.g. random interval) reported for interpretability.

**Risk flagged**: NHANES and GEO are already in LongeBench — using them as sources would
hurt the Retrieval Resistance score. Decided to target AnAge and DrugAge instead.

### CLI design decisions

**Framework**: Typer + Rich, mirroring SPEQTRO CLI architecture.
- Config persisted at `~/.longevity/config.json`, env vars override.
- JSONL result files (one record per task), same pattern as SPEQTRO trajectories.
- `ThreadPoolExecutor(max_workers=8)` for concurrency — hard cap matches shared endpoint policy.

**Model abstraction**: all providers go through OpenAI SDK by setting `base_url`.
- Open-source models (HuggingFace): `https://api-inference.huggingface.co/models/{id}/v1`
- Custom endpoints (L-LLM, vLLM): user-provided URL
- OpenAI: standard SDK defaults
- Anthropic: separate `anthropic` SDK, mapped to same return shape

**Two eval modes**:
- `one-shot`: standard LongeBench evaluation — send prompt, score final answer
- `iterative`: full feedback loop — interval submission, yes/no signal, convergence tracking

### Files created this session
- `devlog.md` (this file)
- `requirements.txt`
- `benchmark/config.py`
- `benchmark/client.py`
- `benchmark/loader.py`
- `benchmark/runner.py`
- `benchmark/results.py`
- `cli.py`

### Next steps
- Build dataset (Area 1): pull AnAge + DrugAge, design interval task prompts, write tasks.jsonl
- Test CLI against L-LLM endpoint with `--limit 5` dry-run
- Implement trace scorer (Area 3) once first results.jsonl is collected

---

## 2026-05-23 — Estimathon redesign + standalone mode

### Corrections from real Estimathon rules

Read the official Estimathon rules PDF. Our initial implementation had three bugs:

**Scoring formula was wrong.** We had `2^(# misses)` as the exponent. Correct formula:
```
(10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good final answers)
```
The exponent is `N − # good`, not `# misses`. These only match if every problem has exactly
one submission — they diverge with re-submissions.

**Only last submission counts — we missed this entirely.** The real game: if you have a good
interval and your refinement misses, you lose that problem. Every re-submission is a bet.
Our previous implementation tracked any correct round; the correct logic tracks only the
final state of each problem.

**Budget is shared across all problems.** 18 slips for 13 problems in the original; we now
default to `floor(1.38 × N)` to match this ratio. Previously we used per-problem budgets.

### Feedback redesign

Changed from directional ("too high / too low") to binary-only ("GOOD / BAD").
Directional hints let a model converge mechanically without biological reasoning — just shift
bounds in the indicated direction. Binary-only feedback forces the model to use its domain
knowledge to infer direction after a miss. This is a purer test of biological understanding.

Model now sees after each slip:
- Good/bad result
- Score before → score after (with ↓/↑ indicator)
- Live standings table for all problems
- Remaining slips in total budget
- Warning if a refinement replaced a previously good interval

### Key inference signal: refinement accuracy

After a confirmed "good" interval, any re-submission is a bet. We now track:
- `refinement_attempts`: how many times the model bet on improving a good interval
- `refinement_successes`: how many bets paid off
- `refinement_accuracy`: success rate

A biologically-informed model only bets when its internal reasoning is confident. Random
guessing after binary-only feedback succeeds ~50% of the time. Significantly above 50%
indicates the model is genuinely reasoning about the biology to infer direction.

### Standalone mode (no HuggingFace required)

Added `--tasks sample` with 7 built-in tasks sourced from real published data:
- 5 × multispecies lifespan (AnAge): bowhead whale (211y), naked mole rat (32y),
  little brown bat (34y), European hedgehog (16y), common lab mouse
- 2 × drug lifespan extension (DrugAge): rapamycin in mice (+14%), caloric restriction (+40%)
- 1 × clinical biological age (NHANES-style): biomarker panel → 57y

Gold values are all verifiable from primary literature. Tasks cover three distinct domains
to test breadth of biological knowledge.

The CLI now runs entirely through `--provider anthropic` with `--tasks sample`:
no HuggingFace token, no dataset download, no endpoint dependency.

### Files modified
- `benchmark/runner.py`: full rewrite — `EstimathonSession`, `run_estimathon()`, binary feedback
- `benchmark/loader.py`: added `_SAMPLE_TASKS` + `sample` keyword
- `cli.py`: wired estimathon mode, live slip-by-slip console output, refinement summary panel
- `README.md`: rewritten around Claude/standalone workflow

---

## 2026-05-23 — MURPHY banner, bamboo green palette, slash command fix

### MURPHY banner

Replaced the "LONGEVITY" ASCII art banner with "MURPHY" using the same block-character style.
Six rows, one per gradient color, built with Rich `Text.append(row, style=color)`.

### Bamboo green color scheme

Changed the gradient from pure forest green (`rgb(0,70,0)` → `rgb(90,255,135)`) to a
bamboo yellow-green spectrum:

```
rgb(45,80,25)    — deep bamboo forest (row 1)
rgb(70,110,40)   — mature stalk (row 2)
rgb(100,145,55)  — mid bamboo (row 3)
rgb(135,175,70)  — light stalk (row 4)
rgb(165,200,85)  — young bamboo (row 5)
rgb(195,225,100) — shoot tip (row 6)
```

The panel borders and titles use mid-bamboo `rgb(100,145,55)` for borders and shoot-tip
`rgb(195,225,100)` for titles. The thinking spinner and prompt `>` use `rgb(160,200,80)`.

### Slash command fix

**Root cause:** `Prompt.ask` from Rich does not always flush `sys.stdout` before blocking on
`input()`, causing the prompt to be invisible on some Windows terminals. Also, `shlex.split`
uses POSIX quoting rules, which can mangle paths containing Windows backslashes.

**Fix:** Replaced `Prompt.ask` with `console.print(prompt, end="")` + `input()`. This uses
the native Python `input()` with Rich handling the colored prompt line separately.
Replaced `shlex.split` with a plain `user_input.split()` — sufficient for all slash commands
since none of their arguments contain spaces. Wrapped the dispatch in `try/except` so a bad
slash command doesn't crash the loop; `SystemExit` (from `/exit`) is re-raised.

### Files modified
- `benchmark/chat.py`: `_GRADIENT`, `_print_welcome`, `_help_panel`, `_thinking`, `run_chat`
- `devlog.md` (this file)

---

## 2026-05-23 — Slash command autocomplete (prompt_toolkit)

### Motivation

Users had no discoverability for slash commands — you had to already know `/help` existed to
find the command list. Wanted: type `/` and see all available commands immediately, Tab to
cycle and select.

### Implementation

Replaced `input()` with `prompt_toolkit.PromptSession`. Added a `SlashCompleter(Completer)`
subclass: `get_completions` only fires when the input starts with `/`, matches against an
`_SLASH_META` list of `(command, description)` pairs, and yields `Completion` objects with
`display_meta` set to the description string. The completion dropdown appears the moment `/`
is typed; Tab cycles forward, Shift+Tab cycles backward.

Styled the completion menu to match the bamboo palette using `PTStyle.from_dict`:
- Menu background: `#152108` (dark bamboo)
- Menu text: `#c3df6e` (bamboo yellow-green)
- Selected row: `#3d6018` bg / `#eaf5a0` text

The `PromptSession` is created once before the loop and reused across turns (preserves
session-level history for free via `InMemoryHistory` default).

The `input()` → `PromptSession` change also fixes the Windows prompt-not-visible regression
from the previous entry: `prompt_toolkit` handles its own ANSI rendering independently of
Rich's console state.

### Dependency added
- `prompt_toolkit>=3.0.0` → `requirements.txt`

### Files modified
- `benchmark/chat.py`: added `_SLASH_META`, `_PT_STYLE`, `SlashCompleter`; updated `run_chat`
- `requirements.txt`: added `prompt_toolkit>=3.0.0`
- `devlog.md` (this file)

---

## 2026-05-23 — LongeBench → Estimathon adapter (chat integration)

### Context

After pulling main, the loader was reset to the original iterator version (no `estimathon` flag).
The new `chat.py` module also calls `load_tasks()` without it. This caused a silent bug when
running LongeBench in estimathon mode via chat:

1. `_tool_run_benchmark(tasks_source="longebench", mode="estimathon")`
2. `load_tasks("longebench")` yields raw rows (gold = multi-line reasoning)
3. `_extract_gold` calls `float(messages[-1]["content"])` → `ValueError` → returns `None`
4. All golds become `None` → every submission marked `BAD` → session degrades immediately

No error was raised — the session just silently ran with broken gold values.

### Solution: Re-apply transformation layer + wire through chat pipeline

**loader.py:** Re-added `_extract_lb_gold()`, `_transform_lb_to_estimathon()`,
`ESTIMATHON_COMPATIBLE_FORMATS`, and updated `load_tasks()` signature to accept `estimathon` flag.
Return type changed from `Iterator[dict]` to `list[dict]` (already used as list in chat.py).

- `_extract_lb_gold()` — robust numeric extractor for LongeBench assistant content
- `_transform_lb_to_estimathon()` — filters to regression/pairwise, rewrites prompts, converts gold to float
- `load_tasks(source, limit, estimathon=False)` — applies transform when `estimathon=True` and source is longebench*

**Filtering strategy:**
- `"sample"` → no transform (already estimathon-ready)
- `"longebench*"` + `estimathon=True` → filter to regression/pairwise, rewrite prompts
- `"<local.jsonl>"` → no transform (user confirmed: assume already correct format)

**chat.py:** Updated tool implementations:
- `_tool_run_benchmark()` — now passes `estimathon=(mode == "estimathon")` to `load_tasks()`
  Adds filter note to console when longebench + estimathon
- `_tool_preview_tasks()` — keeps `estimathon=False` (raw view) but adds tip about filtering

**cli.py:** Same `estimathon=` flag pass-through in `run` command, plus filter note.

### Design decisions

**Preview shows raw tasks (all formats):** Intuitive for dataset exploration. Estimathon mode
filters on the backend when actually running; users see the truth on `/question_set`.

**Local JSONL skips transform:** Custom task files are assumed already estimathon-ready if the
user is running them in estimathon mode. Safer than auto-transforming and potentially breaking them.

### Files modified
- `benchmark/loader.py`: added transformation functions; updated `load_tasks()` signature/return type
- `benchmark/chat.py`: pass `estimathon=` flag; add filter notes and hints
- `cli.py`: pass `estimathon=` flag; add filter note
- `idea.md`: documented LongeBench integration strategy
- `devlog.md` (this file)

---

## 2026-05-23 — CLI Pipeline Logic & Documentation

### Summary

Completed a comprehensive review and documentation of how the Murphy CLI works end-to-end.
The system supports two evaluation modes (one-shot and estimathon) with different answer-grabbing
and feedback strategies.

### Key Architecture

**One-shot mode** (parallel evaluation):
- Spawns up to 8 concurrent threads, each evaluating one task independently
- Sends `messages[:-1]` (stripping gold) to the model
- Uses naive string comparison: `pred.lower() == gold.lower()`
- Records: {correct: bool, gold, pred, think}
- Fragile: "211 years" ≠ "211" → marked incorrect despite containing the right number

**Estimathon mode** (iterative session):
- Single sequential conversation with shared budget (default `floor(1.38 × N)` slips)
- Model submits structured: `PROBLEM <N>\nINTERVAL [min, max]`
- Strict regex parsing: extracts problem number and interval bounds as floats
- Robust gold comparison: `pmin <= gold <= pmax`
- Live feedback loop: GOOD/BAD signal + score delta + standings table
- Refinement accuracy: tracks bets (re-submissions on GOOD intervals) and success rate

**Answer-grabbing flow:**
1. Load tasks → extract golds from `messages[-1]["content"]`
2. Create client (routes to Anthropic/OpenAI/HuggingFace/custom endpoint)
3. Build conversation (system prompt + all problems)
4. Loop: send → parse response → check gold → give feedback → repeat
5. Exit when budget exhausted or 3 parse failures
6. Compute final score and refinement accuracy

**Model client abstraction** (`benchmark/client.py`):
- Provider-agnostic interface: routes to appropriate SDK
- Anthropic SDK for `provider="anthropic"`
- OpenAI SDK for `provider="openai"` (default), `"hf"` (HuggingFace endpoint), `"endpoint"` (custom)
- Thinking trace extraction: separates `<think>...</think>` blocks from answer
- Temperature: always 0.0 (deterministic)
- Max tokens: 500 (one-shot), 600–3000 (estimathon, depends on thinking)

### Integration Points

- **Loader** → supplies tasks with golds and message format
- **Runner** → orchestrates session logic and feedback loop
- **Client** → sends/receives from models
- **Chat UI** → provides interactive tools and slash commands
- **CLI** → entry point for command-line runs

### Testing Done

✅ Loader transformation (LongeBench → estimathon format)
✅ Gold extraction (robust numeric parsing)
✅ One-shot parallel evaluation
✅ Estimathon multi-turn session with parsing
✅ Chat tool integration (`_tool_run_benchmark`, `_tool_preview_tasks`)
✅ Model client routing (Anthropic, OpenAI APIs)
✅ Thinking trace splitting

### Documentation Added

- `idea.md` — LongeBench integration strategy (filtering, gold extraction, prompt transformation)
- `devlog.md` — this entry (CLI pipeline logic and architecture)
- Detailed comments in `loader.py`, `runner.py`, `client.py`

---

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

## 2026-05-23 — InSilico LongeBench → Estimathon Adapter

### Motivation

LongeBench (`insilicomedicine/longebench`) contains 32K+ tasks across 6 formats: binary, multiclass,
ternary, pairwise, regression, and generation. Our estimathon runner is designed for numeric
interval-based tasks — the gold value must be extractable as a float, and user prompts must ask
for interval submissions.

The raw LongeBench rows couldn't work in estimathon mode because:
1. Assistant messages contain reasoning traces (sometimes multi-paragraph), not bare numbers
2. Prompts ask for single answers, not intervals
3. Non-regression formats (binary, MCQ, etc.) have categorical gold values — incompatible with
   interval scoring formula

### Solution: Transformation layer in `loader.py`

**Constants added:**
- `ESTIMATHON_COMPATIBLE_FORMATS = {"regression", "pairwise"}` — only these formats have
  numeric gold values
- `_INTERVAL_INSTRUCTION` — standard suffix appended to user prompts

**New functions:**
- `_extract_lb_gold(content)` — robust numeric extractor. Tries direct `float()` first, then
  regex search for the last number in the text. Handles "67 years", multi-line reasoning traces,
  and scientific notation.
- `_transform_lb_to_estimathon(task)` — filters by format, extracts numeric gold, rewrites
  user prompt to request intervals, converts gold to bare float. Returns None if incompatible.

**API change:**
- `load_tasks(source, limit, estimathon=False)` — new parameter. When `estimathon=True` and
  source is `longebench*`, applies transformation and filters incompatible tasks.
- Return type changed from `Iterator[dict]` to `list[dict]` (already used as list in cli.py).

**Filtering strategy:**
From the 32K total LongeBench tasks, only regression/pairwise tasks survive. These are the
biological age prediction, biomarker prediction, and similar numeric estimation tasks — a
natural subset of the benchmark. Non-numeric tasks (binary classification, multiclass, etc.)
are silently dropped.

### CLI integration

In `cli.py`:
- `run` command passes `estimathon=(mode == EvalMode.estimathon)` to `load_tasks()`
- Added a status line when loading from longebench in estimathon mode:
  `"Filtered to regression-compatible tasks (interval format)"`

### Backward compatibility

- One-shot mode still loads raw LongeBench rows (`estimathon=False`)
- Sample tasks completely unaffected
- Existing local JSONL files can opt-in to transformation by passing `estimathon=True`

### Testing needed

1. `python cli.py tasks --tasks longebench --limit 10` — verify interval prompts in output
2. `python cli.py run --model claude-haiku-... --provider anthropic --tasks longebench --mode estimathon --limit 5` — full session test
3. Check `results.jsonl` to confirm `format: "interval"` and clean numeric golds in slip_log

---

## 2026-05-23 — Mixed eval mode + /setup wizard + LongeBench token fix

### Problem: non-numerical LongeBench tasks had nowhere to go

LongeBench contains 6 task formats: regression, pairwise, binary, multiclass, ternary,
generation. The Estimathon mechanic (interval [min, max], shared budget, binary feedback)
only makes sense for numerical tasks — you can't submit an interval for "yes or no".
Previously `--mode estimathon` silently dropped all non-numerical tasks.

### Solution: two-track hybrid (`--mode mixed`)

Split by `format` field, run each pile through the right pipeline:

| Track | Formats | Eval function | Scoring |
|---|---|---|---|
| Estimathon | regression, pairwise, interval | `run_estimathon()` | Estimathon score |
| One-shot | binary, multiclass, ternary, generation | `run_eval()` | Accuracy / F1 |

`_ESTIMATHON_FORMATS = {"regression", "pairwise", "interval"}` is the canonical routing set.

For generation tasks (gene lists), added token F1 scoring in `_score_task()`: splits pred
and gold on whitespace/commas, computes precision/recall/F1, marks `correct=True` if F1≥0.5.
Binary/multiclass/ternary use exact-match (case-insensitive).

### How the evaluated model knows which mode it's in

Three layers signal the answer type to the model being tested:

1. **System prompt** — Estimathon tasks are presented inside a session with a full game-rules
   system prompt ("You have N slips, submit PROBLEM X / INTERVAL [min, max], binary feedback").
   One-shot tasks have no such context.

2. **Task prompt rewrite** — `_transform_lb_to_estimathon()` appends
   "Submit an interval [min, max] for your answer. Reply with only: [min, max]"
   to numerical tasks. Categorical tasks keep their original prompt unchanged.

3. **Separate conversations** — The Estimathon session is one long multi-turn conversation
   (all numerical problems at once). Each categorical task is an independent single-turn call.
   The model never needs to decide which mode it's in.

### Loader changes

Added `mixed=False` parameter to `load_tasks()`. When `mixed=True`:
- Numerical tasks → `_transform_lb_to_estimathon()` → interval format + float gold
- Categorical tasks → pass through unchanged
- Both returned in the same list; `run_mixed()` splits them by `format`

Also fixed a bug: `_load_longebench()` was not passing the HuggingFace token to
`load_dataset()`. Gated datasets require `token=` to be passed explicitly. Fixed by reading
`cfg.get("hf.token")` or `os.environ.get("HF_TOKEN")`.

### /setup wizard

Added `_setup_wizard()` triggered by `/setup` slash command. Three-step interactive wizard:
1. Anthropic API key (password-masked input via `PromptSession(is_password=True)`)
2. HuggingFace token — saved to config, then immediately verified by connecting to
   `insilicomedicine/longebench` with `streaming=True` and fetching one row
3. OpenAI API key (optional, skippable)

If LongeBench access fails, the wizard prints the dataset URL and tells the user to request
access before re-running `/setup`.

### Files modified
- `benchmark/runner.py`: `_ESTIMATHON_FORMATS`, `_score_task()`, updated `_run_one_shot()`, `run_mixed()`
- `benchmark/loader.py`: `mixed=False` param, HF token fix in `_load_longebench()`
- `benchmark/chat.py`: `_setup_wizard()`, `/setup` in `_SLASH_META` + `_handle_slash` + help panel, `run_mixed` import
- `cli.py`: `EvalMode.mixed`, mixed dispatch block with two summary panels
- `devlog.md` (this file), `README.md`, `CLAUDE.md`

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

## 2026-05-23 — Benchmark bugs fixed + [1,100] trivial interval observation

### Bugs fixed this session

**1. Python 3.11 integer-to-string limit crash**
`10 × 2^N` for N=14,000+ tasks exceeds Python 3.11's 4300-digit int→str conversion guard.
Fix: `sys.set_int_max_str_digits(0)` at runner import time + `_fmt_score()` helper that
renders large scores as `1.23e45` using pure integer arithmetic (bit_length → exponent)
to avoid float overflow on astronomically large values.

**2. `pairwise` tasks routed to Estimathon**
LongeBench `format="pairwise"` tasks ask "which individual is older? A or B" — gold is a
letter, not a number. Our `_ESTIMATHON_FORMATS` included `"pairwise"`, so `_extract_gold`
tried `float("A")` → `None`, and every submission was permanently BAD.
Fix: removed `"pairwise"` from `_ESTIMATHON_FORMATS`. Pairwise tasks now go to the one-shot
track and are scored by exact-match.

**3. Duplicate lb_id collapses all problems into one**
All 100 numerical tasks sampled from LongeBench shared `lb_id = "LB-0010"`. This meant
`session.last_submissions["LB-0010"]` was overwritten on every slip — the model was
effectively submitting to the same single problem regardless of which PROBLEM number it chose.
Fix: pids are now position-based (`P1`, `P2`, …, `P100`). The lb_id is preserved as a
display label but is not used as a dict key.

**4. No per-problem attempt cap**
The model could waste the entire budget on one problem. Added `_MAX_SLIPS_PER_PROBLEM = 3`.
After 3 submissions on a problem it is locked — the next attempt to submit to it is rejected
without consuming a slip and the model is told to choose a different problem.
Each feedback message now includes "Attempts left on Problem Px: Y/3" so the model knows
when to move on.

**5. Model outputting reasoning prose instead of structured format**
The model was responding with multi-paragraph reasoning instead of the mandatory two-line
`PROBLEM N / INTERVAL [min, max]` block, causing parse failures.
Fix: added explicit constraint to system prompt: *"Your entire response must be ONLY these
two lines. No explanation. No reasoning. No other text. Any other format will be ignored."*
Nudge message on parse failure now says "FORMAT ERROR" (not "could not parse").

### Observation: [1, 100] trivial interval exploit

**Discovered during live run with sonnet-4-5 on LongeBench age estimation tasks.**

All 100 numerical tasks sampled were from `LB-0010` (age estimation from DNA methylation,
regression format). The model immediately found a dominant strategy:

```
INTERVAL [1, 100]   →   w = floor(100/1) = 100, GOOD for every task
```

Since human ages fall in [1, 100], this single interval is correct for all age estimation
problems. With 100 problems covered at w=100:

```
score = (10 + 100 × 100) × 2^0 = 10,010
```

vs. any unsolved problems:

```
1 unsolved:   (10 + 99 × 100) × 2^1 = 19,820   →  worse than 10,010
```

The model rationally chose to cover all problems first even at w=100 rather than try to
narrow any single problem and risk leaving others unsolved.

**Implication for benchmark design:**
This is a valid finding. A trivial baseline of `[1, 100]` scores 10,010 on any all-age
dataset. A biologically-informed model should score significantly better by narrowing
intervals based on demographic and methylation clues. This makes `[1, 100]` a natural
baseline: if the L-LLM cannot beat 10,010, it has no advantage over random wide guessing.

**To reproduce:**
```bash
python cli.py run --model claude-sonnet-4-5 --provider anthropic \
  --tasks longebench --mode estimathon --limit 100
```

### /test command updated

Changed `/test` from 7 built-in sample tasks to a proper trial run:
- **20 regression tasks** from LongeBench (requires HF token)
- **40-slip budget** (roughly 2× the default `floor(1.38 × 20) = 27`)
- Mode: `estimathon` (pure interval track, no categorical mixing)

This gives the model room to both cover all problems AND attempt refinements, revealing
whether it can beat the [1, 100] baseline through domain reasoning.

### Files modified
- `benchmark/runner.py`: `_fmt_score`, `sys.set_int_max_str_digits`, `_ESTIMATHON_FORMATS`,
  position-based pids, `_MAX_SLIPS_PER_PROBLEM`, attempt tracking, format-constraint system prompt
- `benchmark/chat.py`: `/test` updated to 20 tasks / 40-slip budget; debug Q+response lines
  in `_slip_line`; `lb_id` + `attempts_left` display
- `cli.py`: `_fmt_score` in all score display sites
- `devlog.md` (this file)

---

## 2026-05-23 — Model Discovery & Batch Benchmarking

### Context

To improve the L-LLM model iteratively, we need to:
1. Discover which open-source LLMs are available and comparable
2. Run estimathon benchmarks across multiple models sequentially
3. Compare performance to understand where L-LLM excels/struggles

Previous work had added `fetch_models.py` to fetch 300+ models from HuggingFace; now we
integrate that data into the chat and CLI for easy selection and batch testing.

### New Module: ModelManager (`benchmark/model_manager.py`)

Self-contained class for managing a CSV of HuggingFace models:
- **`load()`** — reads from `hf_llm_models.csv` (300+ rows, sorted by downloads)
- **`save()`** — persists model list back to CSV after edits
- **`search(query, top_n=20)`** — filters by model_id, model_name, or author
- **`get_by_index(idx)`** — retrieves by 1-indexed position (for `/model 5` selection)
- **`refresh_from_hf(num_models=300)`** — updates the CSV from HuggingFace API (calls `fetch_models.py`)
- **`add_model(model_id, author)`** — adds custom model to CSV
- **`remove_model(model_id)`** — removes model from CSV
- **`list_models(limit, search)`** — displays formatted Rich table of models
- **`get_cli_command(model_id)`** — generates a CLI command for testing

### Chat Commands: Three New Slash Commands

**`/model`** — model browser and selector
- `/model` — show current chat/bench models and settings
- `/model list [limit]` — display top N models by downloads (default 20)
- `/model search <query>` — search for models (e.g., "llama", "qwen")
- `/model <index>` — select model by position (e.g., `/model 5` selects the 5th-ranked model)
- `/model <id>` — select model by full ID or shorthand (e.g., `/model sonnet`)

**`/batch`** — sequential multi-model benchmarking
- `/batch 1 2 3` — run estimathon on models at positions 1, 2, 3
- `/batch llama qwen` — search for "llama" and "qwen", then run both
- `/batch 1 2 hf sample 7` — specify provider (hf), task source (sample), and limit (7)
- Runs each model sequentially, appending results to separate output files

**`/add`** — CSV management and refresh
- `/add meta-llama/Llama-3.2-8B` — add a new model to the CSV
- `/add my-model myorg` — add with custom author
- `/add refresh 500` — fetch latest 500 models from HuggingFace, replace CSV
- Integrates with the model manager's `refresh_from_hf()` method

### Design Decisions

**CSV vs. in-memory database:** Chose CSV for simplicity, portability, and git-friendliness.
No database setup required. Easy to inspect and edit manually if needed.

**HuggingFace API integration:** Models are fetched once (expensive) and cached in CSV.
`/add refresh` can be run manually when you want fresh data. Keeps the chat responsive.

**Index-based selection:** `/model 5` is faster than typing the full model_id. Sorts by
downloads (popularity) so common models are in the top 10.

**Batch runs sequentially, not parallel:** Simpler error handling, no concurrency issues.
Each run writes to a separate results file (`results_<model_id>.jsonl`) for easy comparison.

### Integration with Loader Pipeline

The model manager works seamlessly with the existing transformation layer:
- Estimathon mode automatically applies `_transform_lb_to_estimathon()` to LongeBench
- Any model can be tested via the same unified pipeline
- `/batch` defaults to `tasks=sample` (no HF token needed) but can run any source

### Files created/modified
- **New:** `benchmark/model_manager.py` — 120 lines, ModelManager class
- **New:** `hf_llm_models.csv` — 300 models from HuggingFace (generated by `fetch_models.py`)
- **Modified:** `benchmark/chat.py` — added three new slash commands, ModelManager import
- **Modified:** `_SLASH_META` list with help text for /model, /batch, /add

### Testing
✅ ModelManager loads 300 models from CSV
✅ `search()` filters by keyword
✅ `get_by_index()` retrieves by position
✅ `/model list` displays formatted table
✅ `/model search` works with multi-word queries
✅ `/batch` resolves indices to model IDs and runs sequentially

---

## 2026-05-23 — Diverse sampling, model shorthands, progress bar, /explore

### Problem: all /test tasks were the same question type

LongeBench is ordered by lb_id. The first 6,095 regression rows are all `LB-0010`
(DNA methylation age estimation). Running `/test` with limit=20 produced 20 identical
question types — not a useful cross-domain evaluation.

### Fix: diverse sampling in loader

Added `_load_longebench_diverse()` to `loader.py`. Strategy:

1. Scan the first `_DIVERSE_SCAN_CAP = 3000` rows of LongeBench
2. Group eligible (estimathon-compatible) tasks by `lb_id` into a pool
3. Compute `per_type = max(1, limit // n_types)` slots per type
4. Distribute remainder slots to the first N types
5. Fill shortfalls from leftovers, shuffle the final list

`load_tasks()` now accepts `diverse=True`. `/test` passes `diverse=True` so its
20-task trial samples evenly across all regression task types found in the first 3000 rows.

**Added /explore command** to show what's in that pool: scans 3000 rows, groups by lb_id,
prints a Rich table of all unique task types with their format, domain, example gold value,
Estimathon-compatibility flag (✓/✗), and row count. Useful for understanding what the
benchmark is actually testing before running it.

### Model shorthands

`_resolve_model()` maps short names to full Claude model IDs:
- `sonnet` → `claude-sonnet-4-6`
- `haiku` → `claude-haiku-4-5-20251001`
- `opus` → `claude-opus-4-7`

`/test` now accepts an optional model argument: `/test sonnet`, `/test haiku`, `/test opus`,
or any full model ID. `/model <shorthand>` also resolves through the same map.

### Progress bar

Replaced ad-hoc slip printing with a Rich `Progress` bar showing two tracks:
- **slips**: slip number out of total budget
- **solved**: number of currently-GOOD problems out of total

Uses bamboo green palette (`rgb(45,80,25)` → `rgb(195,225,100)`). Slip details (Q preview,
model response, GOOD/BAD, width, score delta) are printed above the bar using
`_progress.console.print()` so the bar stays pinned at the bottom.

### Budget ratio corrected

Changed from the approximate `floor(1.38 × N)` to the exact `floor(18/13 × N)`.
At N=13 the old formula gave 17 slips instead of the correct 18.
Updated everywhere: `runner.py`, `cli.py`, `CLAUDE.md`, `idea.md`.

### Files modified
- `benchmark/loader.py`: `_DIVERSE_SCAN_CAP`, `diverse` param, `_load_longebench_diverse()`
- `benchmark/chat.py`: `_resolve_model()`, `_MODEL_SHORTHANDS`, `/test` diverse+model arg,
  `/explore` command, Rich Progress bar with two tracks

---

## 2026-05-23 — pip packaging, first-run setup wizard, GitHub Actions CI

### pip package: murthy-bench

Added `pyproject.toml` at the repo root. Package name: `murthy-bench`.
Entry points: `murthy` and `murthy-bench` both launch `longivity_hack.cli:entry`.

Dependencies moved from `requirements.txt` to `pyproject.toml` as required deps —
including `datasets>=2.0.0` (previously optional) since LongeBench is the main task source.
Requires Python ≥ 3.11 (needed for `sys.set_int_max_str_digits`).

### First-run setup wizard

`run_chat()` previously exited with an error if no Anthropic key was set.
Changed: if no key is found on startup, show a "First Run Setup" welcome panel and call
`_setup_wizard()` automatically before opening the chat loop.

After the wizard, the key is re-read from config. If still not set (user skipped all steps),
show the manual config command and exit cleanly.

**User flow on a fresh machine:**
```
pip install murthy-bench
murthy
→  First Run Setup wizard
→  [1/3] Anthropic key
→  [2/3] HuggingFace token + live LongeBench access check
→  [3/3] OpenAI key (optional)
→  Chat opens
```

### GitHub Actions: auto-publish on version tag

Added `.github/workflows/publish.yml`. Triggers on `v*` tags pushed to main.
Uses `pypa/gh-action-pypi-publish` with OIDC trusted publishing — no API tokens stored
in GitHub Secrets. One-time setup: add a trusted publisher on pypi.org pointing to this
repo and workflow file.

**Release flow:**
```
git add . && git commit -m "release v0.x.x"
git tag v0.x.x
git push origin main --tags
```

GitHub Actions builds the wheel and sdist, runs `twine check`, and uploads to PyPI.

### Files created/modified
- **New:** `pyproject.toml` — package metadata, deps, entry points
- **New:** `.github/workflows/publish.yml` — CI publish workflow
- **Modified:** `benchmark/chat.py` — first-run setup wizard in `run_chat()`; version bump to v0.2.1
- **Modified:** `.gitignore` — added `dist/`, `build/`, `*.egg-info/`, `*.jsonl`
- **Modified:** `README.md` — rewritten for `pip install murthy-bench` workflow

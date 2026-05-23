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

# Murphy — Developer Guide

Longevity LLM benchmark CLI (Track 01 · Insilico Medicine Hackathon).
Read `idea.md` for the benchmark design rationale. This file is for contributors extending the code.

**L-LLM endpoint:** `https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud`
**Dataset:** `insilicomedicine/longebench` on HuggingFace (gated — request access first)

---

## Current state

The CLI is fully built. All three original hackathon areas map as follows:

| Area | Status | Notes |
|---|---|---|
| Area 1 — Custom dataset | **TODO** | See below — need AnAge/DrugAge tasks.jsonl |
| Area 2 — Eval loop | **Done** | `benchmark/runner.py` + `cli.py` |
| Area 3 — Trace scorer | **TODO** | See below — scorer.py not yet written |

---

## Quick start

1. **Setup environment**:
```bash
cd longivity_hack
cp .env.example .env
# Edit .env and add your HF_TOKEN (from ask.py or generate at https://huggingface.co/settings/tokens)
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Load environment and run**:
```bash
source .env
python cli.py          # opens interactive chat
```

In chat, run `/setup` to configure API keys and verify LongeBench access.

**Test L-LLM directly** (no other API keys needed):
```bash
source .env
python cli.py run --model longevity-llm --provider endpoint \
  --endpoint $L_LLM_ENDPOINT --api-key $HF_TOKEN \
  --tasks sample --mode estimathon --limit 7
```

Or run a full LongeBench benchmark:
```bash
python cli.py run --model longevity-llm --provider endpoint \
  --endpoint $L_LLM_ENDPOINT --api-key $HF_TOKEN \
  --tasks longebench --mode mixed --limit 100
```

---

## Codebase map

```
benchmark/
├── config.py     Config at ~/.longevity/config.json; env vars override
├── client.py     ModelClient — wraps all providers behind one .chat() interface
├── loader.py     load_tasks() — sample / LongeBench (with token) / local JSONL
├── runner.py     run_estimathon(), run_eval(), run_mixed(), scoring helpers
├── results.py    ResultWriter / read_results — JSONL append
└── chat.py       Interactive chat UI — Claude tool-use, /setup, slash autocomplete
cli.py            Typer app — run / chat / status / tasks / config commands
```

---

## Key design decisions

### Scoring formula
```
score = (10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good final answers)
```
Lower is better. Only the **last** submission per problem counts.
Default budget: `floor(18/13 × N)` slips, matching the real Estimathon's 18-slip / 13-problem ratio.

### Binary-only feedback
The model receives GOOD or BAD — **no directional hints** ("too high / too low").
Directional hints let a model converge by bracket-searching without any biological understanding.
Binary-only forces genuine domain reasoning. The original CLAUDE.md had directional hints — that
version was wrong. Do not reintroduce them.

### Refinement accuracy
After a GOOD interval is confirmed, any re-submission is a voluntary bet.
`refinement_accuracy = successes / attempts`. Random guessing wins ~50%.
A model above 50% is reasoning about biology to infer direction; at 50% it is guessing.
This is the primary signal for comparing L-LLM vs baselines.

### Two-track evaluation (mixed mode)
LongeBench has 6 task formats. Only regression and pairwise have numeric gold values suitable
for interval scoring. Mixed mode routes by the `format` field:

```python
_ESTIMATHON_FORMATS = {"regression", "pairwise", "interval"}
# tasks with these formats → run_estimathon()
# all other formats → run_eval() with format-aware scoring
```

Generation tasks (gene lists) use token F1 ≥ 0.5 as the correctness threshold.
Binary / multiclass / ternary use exact-match (case-insensitive).

### How the model knows which mode it's in
Three layers — the model never has to guess:
1. **Estimathon system prompt** explicitly states game rules (budget, PROBLEM N / INTERVAL format)
2. **Task prompt** for numerical tasks has the interval instruction appended by `_transform_lb_to_estimathon()`
3. **Separate conversations** — Estimathon is one long multi-turn session; one-shot tasks are independent calls

---

## Provider wiring

All providers go through `benchmark/client.py`. HuggingFace and custom endpoints use the
OpenAI SDK with `base_url` overridden. Anthropic uses the `anthropic` SDK directly.

| Provider | SDK | base_url |
|---|---|---|
| `anthropic` | `anthropic.Anthropic` | n/a |
| `openai` | `openai.OpenAI` | default |
| `hf` | `openai.OpenAI` | `https://api-inference.huggingface.co/models/{id}/v1` |
| `endpoint` | `openai.OpenAI` | user-supplied URL |

To connect to L-LLM:
```bash
python cli.py status \
  --model longevity-llm \
  --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud \
  --api-key <hf-token>
```

---

## Area 1 — Custom dataset (TODO)

Build interval-based tasks from aging biology sources not already in LongeBench.
NHANES, GTEx, GEO are already in LongeBench — avoid them (hurts retrieval resistance score).

**Recommended sources:**

| Dataset | URL | Task type |
|---|---|---|
| AnAge | genomics.senescence.info/download | Multispecies lifespan regression |
| DrugAge | genomics.senescence.info/download | Drug lifespan extension % regression |
| MGI mouse phenotypes | informatics.jax.org/downloads | Mutation → murine lifespan |

**Task format** (JSONL, one row per instance):
```json
{
  "lb_id": "LB-CUSTOM-001",
  "task": "multispecies_lifespan_regression",
  "domain": "comparative_biology",
  "format": "regression",
  "metric": "estimathon_score",
  "messages": [
    {"role": "system", "content": "You are an expert in comparative biology and aging science."},
    {"role": "user",   "content": "Given the following biological profile...\n\nSubmit an interval [min, max] for the maximum lifespan of this species in years.\nReply with only: [min, max]"},
    {"role": "assistant", "content": "45.2"}
  ]
}
```
The last assistant message is the **bare numeric gold value** (no units, no brackets).
`runner.py` extracts it with `float(messages[-1]["content"].strip())`.

**Split strategy** — prevent data leakage:
- AnAge → split by taxonomic Order
- DrugAge → split by drug mechanism class
- MGI → split by genetic background strain

**Deliverable:** `tasks.jsonl` (≥50 test instances) + `data_notes.md`.
Load with: `python cli.py tasks --tasks path/to/tasks.jsonl`

---

## Area 3 — Reasoning trace scorer (TODO)

A programmatic scorer that evaluates the *quality of reasoning* in `<think>` traces collected
during `--think` runs. Scorer must be automatable and correlated with biological correctness.

**Input:** `results.jsonl` from `--mode estimathon --think`
**Output:** `scored_traces.jsonl` — one record per slip with added quality scores

### Three signals to implement

**Signal 1 — Entity verification**
Extract biological entities from the think trace and check them against authoritative lists:
```python
# Gene symbols → HGNC list
# CpG site IDs → Illumina 450K/EPIC manifest
# Drug names → DrugAge / DrugBank
def entity_hallucination_rate(think_trace: str) -> float:
    # returns fraction of mentioned entities that don't exist
    pass
```

**Signal 2 — Monotonic convergence**
For each problem, does the width factor decrease monotonically across GOOD submissions?
```python
def convergence_score(slip_log: list[dict]) -> float:
    good_slips = [s for s in slip_log if s["good"]]
    widths = [s["width_factor"] for s in good_slips if s["width_factor"] is not None]
    if len(widths) < 2:
        return 1.0
    violations = sum(widths[i] > widths[i-1] for i in range(1, len(widths)))
    return 1.0 - violations / (len(widths) - 1)
```

**Signal 3 — Trace-answer consistency**
Does the reasoning in the think block predict the interval actually submitted?
Use Claude Haiku as a judge (cheap, one call per slip):
```python
# "The think trace says X. The submitted interval is Y. Are they consistent? yes/no"
def trace_answer_consistency(think: str, interval: str) -> bool:
    pass
```

**Composite score:**
```python
def trace_quality(think, slip_log):
    return (1 - entity_hallucination_rate(think)) \
           * convergence_score(slip_log) \
           * trace_answer_consistency(think, slip_log[-1])
```

**Validation:** check that composite score correlates with `refinement_accuracy` across problems.
A model with high trace quality but low refinement accuracy is a red flag worth reporting.

**Deliverable:** `scorer.py` (reads results.jsonl, writes scored_traces.jsonl) + `scorer_notes.md`.

---

## Model Discovery & Selection

Use `model_selector.py` to browse 300+ open-source LLMs from HuggingFace:

```bash
# List top 20 models by downloads
python model_selector.py list

# Search for specific models (e.g., Llama)
python model_selector.py search llama

# Show details of a model
python model_selector.py info 6

# Generate CLI command for a model
python model_selector.py cmd 6
```

**Update model database** (requires HF_TOKEN):
```bash
HF_TOKEN=hf_xxxxx python fetch_models.py 500 hf_llm_models.csv
```

This creates `hf_llm_models.csv` with:
- model_id, author, downloads, likes
- HuggingFace URL and inference API endpoint
- Tags and gating status

You can then use the generated commands to benchmark any model:
```bash
python cli.py run --model Llama-3.1-8B-Instruct --provider hf \
  --tasks sample --mode estimathon --limit 7
```

---

## Adding a new eval mode

1. Add `my_mode = "my-mode"` to `EvalMode` in `cli.py`
2. Implement `run_my_mode(tasks, client, ...) -> dict` in `benchmark/runner.py`
3. Add the dispatch block inside `with ResultWriter(...) as writer:` in `cli.py`
4. Update `_tool_run_benchmark` in `benchmark/chat.py` and the tool schema's `mode` enum

## Adding a new task source

1. Add a branch in `load_tasks()` in `benchmark/loader.py`
2. Return a list of task dicts with at minimum: `lb_id`, `format`, `messages`, and a numeric gold
   value as the last assistant message content

## Adding a new provider

1. Add to the `Provider` enum in `cli.py`
2. Handle it in `ModelClient.__init__` and `ModelClient.chat()` in `benchmark/client.py`
3. Add to `provider_api_key()` and `provider_preflight()` in `benchmark/config.py`

---

## Config reference

Stored at `~/.longevity/config.json`. Env vars override file values.

| Key | Env var | Used for |
|---|---|---|
| `anthropic.api_key` | `ANTHROPIC_API_KEY` | Chat UI + anthropic provider |
| `hf.token` | `HF_TOKEN` | LongeBench dataset + hf provider inference |
| `openai.api_key` | `OPENAI_API_KEY` | openai provider |

```bash
python cli.py config set anthropic.api_key sk-ant-...
python cli.py config list
```

---

## Common issues

**LongeBench access denied**
Request access at `huggingface.co/datasets/insilicomedicine/longebench`, then run `/setup`
to re-verify. The token must be passed explicitly — setting `HF_TOKEN` env var also works.

**Model returns no valid interval**
The parser in `runner.py parse_interval()` accepts bare numbers, `[min, max]`, and loose
whitespace. If the model writes text around the interval (e.g. "I think [20, 40] years"),
the regex still finds it. After 3 consecutive parse failures the session terminates early.

**prompt_toolkit not found**
```bash
pip install -r requirements.txt
```
`prompt_toolkit>=3.0.0` is required for the `/` autocomplete menu in the chat UI.

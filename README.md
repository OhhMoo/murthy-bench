# Murphy — Longevity Benchmark CLI

Evaluate any LLM on aging-biology estimation tasks using an **Estimathon-style** benchmark.
Models submit intervals `[min, max]` for numerical questions (lifespan, drug extension %, biological age),
receive only **binary feedback** (GOOD / BAD), and must manage a shared submission budget across all problems.
Lower score is better.

---

## Install

```bash
cd longivity_hack
pip install -r requirements.txt
```

---

## Interactive chat (recommended entry point)

The easiest way to use Murphy is the interactive chat powered by Claude.
Run it with no arguments:

```bash
python cli.py
```

Or explicitly:

```bash
python cli.py chat --model claude-sonnet-4-6
```

You'll see the **MURPHY** ASCII banner in bamboo green, then a live prompt.
Type a message in plain English — Claude handles the rest, calling tools to load datasets,
run benchmarks, and check models on your behalf.

### Slash commands

Type `/` at the prompt and a **completion menu** appears showing all available commands.
Press **Tab** to cycle through them, **Enter** to select.

| Command | Args | Description |
|---|---|---|
| `/help` | | Show all commands |
| `/exit` | | Exit the chat |
| `/clear` | | Clear conversation history |
| `/model` | `[model-id]` | Show or set benchmark model |
| `/provider` | `[provider]` | Show or set provider (`anthropic\|openai\|hf\|endpoint`) |
| `/tasks` | `[source]` | Show or set default task source |
| `/think` | | Toggle chain-of-thought traces for benchmark runs |
| `/question_set` | `[source] [limit]` | Preview tasks from a source |
| `/benchmark` | `[model] [provider] [tasks]` | Quick-run estimathon with current defaults |
| `/status` | `[model] [provider]` | Check model connectivity |
| `/config` | `[key] [value]` | View or set a config value |

---

## Quick start — no HuggingFace required

The CLI ships with 7 built-in sample tasks from real AnAge, DrugAge, and NHANES data.
Use `--tasks sample` and `--provider anthropic` to run entirely through the Anthropic API.

```bash
# Store your Anthropic key once
python cli.py config set anthropic.api_key sk-ant-...

# Run Estimathon session against Claude (7 problems, auto budget ~10 slips)
python cli.py run \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --tasks sample \
  --mode estimathon \
  --think

# One-shot baseline (same tasks, no feedback loop)
python cli.py run \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --tasks sample \
  --mode one-shot
```

---

## Estimathon rules

Scoring formula (lower is better):
```
(10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good final answers)
```

- Only your **last** submission per problem counts
- Refining a good interval is a bet — if the new interval misses, you lose that problem
- You receive **only binary feedback**: GOOD or BAD — no directional hints ("too high / too low")
- No directional hints forces the model to reason from biology, not just bracket-search
- Default budget: `floor(1.38 × N)` slips (matches the real Estimathon's 18/13 ratio)

### Key output metric: refinement accuracy

`refinement_accuracy` = fraction of voluntary bets on GOOD intervals that paid off.
Random guessing on binary feedback succeeds ~50%. A model significantly above 50% is genuinely
reasoning about biology to infer direction, not just guessing.

---

## Connecting to the L-LLM endpoint

```bash
python cli.py status \
  --model longevity-llm \
  --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud \
  --api-key <hf-token>

python cli.py run \
  --model longevity-llm \
  --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud \
  --api-key <hf-token> \
  --tasks sample \
  --mode estimathon \
  --think
```

---

## Using HuggingFace models

```bash
python cli.py config set hf.token hf_...
python cli.py run \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --provider hf \
  --tasks sample \
  --mode estimathon
```

---

## Providers

| `--provider` | What it connects to | Credential |
|---|---|---|
| `anthropic` | Anthropic API (Claude) | `anthropic.api_key` / `ANTHROPIC_API_KEY` |
| `endpoint` | Any OpenAI-compatible URL | `--api-key` flag + `--endpoint` |
| `hf` | HuggingFace Inference API | `hf.token` / `HF_TOKEN` |
| `openai` | OpenAI API | `openai.api_key` / `OPENAI_API_KEY` |

---

## Task sources

| `--tasks` | What it loads |
|---|---|
| `sample` | 7 built-in tasks — no network required |
| `longebench` | Full LongeBench benchmark (HuggingFace, gated) |
| `longebench:extra` | LongeBench extra split |
| `path/to/tasks.jsonl` | Local JSONL file |

---

## CLI commands

```
python cli.py              Open interactive chat (default)
python cli.py chat         Open interactive chat
python cli.py run          Run benchmark (one-shot or estimathon)
python cli.py status       Check model endpoint connectivity
python cli.py tasks        Preview tasks from a source
python cli.py config       Get/set/list stored config values
```

---

## Output

Results written to `results.jsonl`. Each session record includes:
- `final_score` — Estimathon score (lower is better)
- `n_good_final` — number of problems solved at session end
- `slips_used` / `total_budget` — budget utilisation
- `slip_log` — every submission with GOOD/BAD, width factor, score before/after
- `refinement_accuracy` — fraction of refinement bets that improved the score
- `think` — per-slip chain-of-thought trace (when `--think` is enabled)

---

## Project structure

```
longivity_hack/
├── cli.py                  entry point (Typer app)
├── requirements.txt
├── benchmark/
│   ├── chat.py             interactive chat UI (Claude tool-use, slash autocomplete)
│   ├── client.py           unified model client (all providers)
│   ├── runner.py           Estimathon session + one-shot eval loop
│   ├── loader.py           task loading (sample / LongeBench / local JSONL)
│   ├── config.py           ~/.longevity/config.json
│   └── results.py          JSONL writer/reader
├── devlog.md               development log
└── idea.md                 benchmark design document
```

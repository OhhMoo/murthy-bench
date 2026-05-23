# Murphy-Health — Longevity Benchmark CLI

Evaluate any LLM on aging-biology estimation tasks using an **Estimathon-style** benchmark.
Models submit intervals `[min, max]` for numerical questions (lifespan, drug extension %, biological age),
receive only binary feedback (good/bad), and must manage a shared submission budget across all problems.
Lower score is better.

## Install

```bash
cd longivity_hack
pip install -r requirements.txt
```

## Quick start — no HuggingFace required

The CLI ships with 7 built-in sample tasks (real AnAge, DrugAge, and NHANES-derived data).
Use `--tasks sample` and `--provider anthropic` to run entirely through the Anthropic API.

```bash
# Store your Anthropic key once
python cli.py config set anthropic.api_key sk-ant-...

# Run Estimathon session against Claude (7 problems, auto budget)
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

## Estimathon rules

Scoring formula (lower is better):
```
(10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good final answers)
```

- Only your **last** submission per problem counts
- Refining a good interval is a bet — if the new interval misses, you lose that problem
- You receive **only binary feedback**: good or bad, plus your live running score
- No directional hints ("too high / too low") — the model must reason from biology

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

## Using HuggingFace models

```bash
python cli.py config set hf.token hf_...
python cli.py run \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --provider hf \
  --tasks sample \
  --mode estimathon
```

## Providers

| `--provider` | What it connects to | Credential |
|---|---|---|
| `anthropic` | Anthropic API (Claude) | `anthropic.api_key` / `ANTHROPIC_API_KEY` |
| `endpoint` | Any OpenAI-compatible URL | `--api-key` flag or `--endpoint` |
| `hf` | HuggingFace Inference API | `hf.token` / `HF_TOKEN` |
| `openai` | OpenAI API | `openai.api_key` / `OPENAI_API_KEY` |

## Task sources

| `--tasks` | What it loads |
|---|---|
| `sample` | 7 built-in tasks — no network required |
| `longebench` | Full LongeBench benchmark (HuggingFace, gated) |
| `longebench:extra` | LongeBench extra split |
| `path/to/tasks.jsonl` | Local JSONL file |

## Commands

```
python cli.py run       Run benchmark (one-shot or estimathon)
python cli.py status    Check model endpoint connectivity
python cli.py tasks     Preview tasks from a source
python cli.py config    Get/set/list stored config values
```

## Output

Results written to `results.jsonl`. Each session record includes:
- `final_score` — Estimathon score (lower is better)
- `slip_log` — every submission with good/bad, width factor, score before/after
- `refinement_accuracy` — fraction of refinement bets that improved the score
- `think` — per-slip chain-of-thought trace (when `--think` is enabled)

## Project structure

```
longivity_hack/
├── cli.py              entry point
├── benchmark/
│   ├── client.py       unified model client (all providers)
│   ├── runner.py       Estimathon session + one-shot eval loop
│   ├── loader.py       task loading (sample / LongeBench / local JSONL)
│   ├── config.py       ~/.longevity/config.json
│   └── results.py      JSONL writer/reader
└── devlog.md           development log
```

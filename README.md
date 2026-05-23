# Murphy-Health — Longevity Benchmark CLI

Evaluate any LLM on aging-biology tasks from [LongeBench](https://huggingface.co/datasets/insilicomedicine/longebench).
Supports one-shot scoring and iterative interval-refinement (Estimathon-style) evaluation.

## Install

```bash
cd longivity_hack
pip install -r requirements.txt
```

## Quick start

```bash
# Store credentials once
python cli.py config set hf.token <your-hf-token>
python cli.py config set openai.api_key <your-openai-key>

# Check connectivity
python cli.py status --model longevity-llm --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud

# Run one-shot benchmark (5-task dry-run)
python cli.py run --model longevity-llm --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud \
  --api-key <token> --limit 5

# Run iterative refinement benchmark with thinking traces
python cli.py run --model longevity-llm --provider endpoint \
  --endpoint https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud \
  --api-key <token> --mode iterative --think --output results.jsonl

# Run any HuggingFace model via Inference API
python cli.py run --model meta-llama/Meta-Llama-3-8B-Instruct --provider hf --limit 20

# Preview benchmark tasks
python cli.py tasks --limit 10
```

## Providers

| Flag | What it connects to |
|---|---|
| `--provider hf` | HuggingFace Inference API (needs `hf.token`) |
| `--provider openai` | OpenAI API (needs `openai.api_key`) |
| `--provider anthropic` | Anthropic API (needs `anthropic.api_key`) |
| `--provider endpoint` | Any OpenAI-compatible URL (`--endpoint <url>`) |

## Output

Results are written as JSONL to `results.jsonl` (configurable via `--output`).
Each line is one task record with gold answer, model prediction, and full thinking traces.

## Project structure

```
longivity_hack/
├── cli.py            entry point
├── benchmark/
│   ├── client.py     model client (all providers)
│   ├── runner.py     eval loop (one-shot + iterative)
│   ├── loader.py     task loading (LongeBench or local JSONL)
│   ├── config.py     ~/.longevity/config.json
│   └── results.py    JSONL writer/reader
└── devlog.md         development log
```

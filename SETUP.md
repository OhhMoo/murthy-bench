# Local Setup Guide

End-to-end guide for cloning **Murphy-Health** and running everything from
source: the `murthy` benchmark CLI, the standardized 200-task builder, and
the custom RNA Q-bank pipeline (EP-01/02/03).

> For the *user-facing* `pip install murthy-bench` workflow, see [README.md](README.md).
> This document is for **developers and hackathon teammates** working out of the repo.

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | **≥ 3.11** | `pyproject.toml` allows 3.10, but `runner.py` calls `sys.set_int_max_str_digits(0)` which only exists in 3.11+. |
| `git` | any | for cloning |
| `pip` | recent | bundled with Python |
| HuggingFace account | — | required for the gated `insilicomedicine/longebench` dataset and the L-LLM endpoint |
| Anthropic / OpenAI account | optional | only needed if you target those providers |

---

## 2. Clone & virtual environment

```bash
git clone https://github.com/OhhMoo/Murphy-Health.git
cd Murphy-Health

python3 -m venv .venv
source .venv/bin/activate          # zsh / bash
# .venv\Scripts\activate           # PowerShell on Windows
```

---

## 3. Install

Pick **one** of the two install modes:

### Mode A — editable package install (recommended)

Installs as `murthy-bench`, gives you the `murthy` and `murthy-bench` entry
points anywhere in your shell, and picks up your edits live.

```bash
pip install -e .
```

### Mode B — `requirements.txt`, run from source

Use this if you'd rather not install the package at all.

```bash
pip install -r longivity_hack/requirements.txt
```

Then run via the module path:

```bash
python -m longivity_hack.cli ...
# or, from inside longivity_hack/:
python cli.py ...
```

Both modes pull the same dependency set: `openai`, `anthropic`, `typer`,
`rich`, `prompt_toolkit`, `requests`, `datasets`.

---

## 4. Configure credentials

Credentials live in two places that the CLI checks in this priority order:

1. **Environment variables** (highest) — see table below
2. **`~/.longevity/config.json`** — managed by `murthy config set …`

> **Precedence note.** Env vars shadow file values, so if you already have
> `HF_TOKEN` exported in your shell, the wizard's "Current:" line and
> `/config` listing will both show the env value rather than whatever is in
> `config.json`. When you save a new value via `/setup` or `murthy config
> set`, the running process also exports the matching env var so the rest of
> *this session* sees the update — but new shells will still get whatever
> your `.env` or `~/.zshrc` exports. If you want the new value to be
> permanent, update your `.env` (or just `unset HF_TOKEN` before relaunching
> so the file value takes over).

### 4a. Quickest: copy `.env` and source it

```bash
cp longivity_hack/.env.example .env
```

Edit `.env` in your editor and fill in real values, then export everything
for the current shell:

```bash
set -a
source .env
set +a
```

> zsh tip: if you paste a multi-line block that includes `#` comments, prefix
> with `setopt interactive_comments` once, or just strip the comments —
> interactive zsh treats `#` as a command name by default.

`.env.example` shows every var the project understands:

| Env var | Maps to | Used for |
|---|---|---|
| `HF_TOKEN` | `hf.token` | LongeBench dataset access + HuggingFace inference + L-LLM endpoint |
| `L_LLM_ENDPOINT` | `llm.endpoint` | Insilico L-LLM target URL |
| `ANTHROPIC_API_KEY` | `anthropic.api_key` | Chat UI + `--provider anthropic` runs |
| `OPENAI_API_KEY` | `openai.api_key` | `--provider openai` runs |

### 4b. Or: interactive wizard (first launch)

```bash
murthy            # first run with no Anthropic key triggers a 3-step wizard
```

The wizard prompts for the Anthropic key, then the HF token (with a *live*
LongeBench access check), then the optional OpenAI key. Re-runnable any time
inside chat with `/setup`.

### 4c. Or: per-key CLI

```bash
murthy config set hf.token            hf_xxx
murthy config set anthropic.api_key   sk-ant-xxx
murthy config set llm.endpoint        https://<your-endpoint>.huggingface.cloud
murthy config list                    # masks tokens
```

### 4d. LongeBench gate

LongeBench is a **gated** dataset. Even with a valid `HF_TOKEN`, you must
first click *Request access* at
<https://huggingface.co/datasets/insilicomedicine/longebench>.
Approval is usually instant. Re-run `/setup` to re-verify.

---

## 5. Connect to the Longevity-LLM endpoint

The L-LLM is a fine-tuned aging-biology model hosted as a HuggingFace
inference endpoint. It is the primary benchmark target for this project.

> Endpoint URLs in this section are illustrative — the active one rotates;
> use whichever URL the hackathon organisers gave you (or the value in
> `longivity_hack/.env.example`).

### 5a. What needs to be configured

Three values, all of which can live in `~/.longevity/config.json` or
in your shell environment:

| Config key | Env var | Required? | What it's for |
|---|---|---|---|
| `llm.endpoint` | `L_LLM_ENDPOINT` | yes | HuggingFace inference URL for the L-LLM |
| `hf.token` | `HF_TOKEN` | yes | Sent as the bearer token to the endpoint AND used to access the gated LongeBench dataset |
| `llm.model` | — | optional | Display alias the CLI matches against. Defaults to `longevity-llm`; only change it if you renamed the model. |

### 5b. Set the values — pick any one

**Option A — `.env` (recommended for repeat use):**

```bash
cp longivity_hack/.env.example .env
# edit .env, fill in L_LLM_ENDPOINT and HF_TOKEN
set -a; source .env; set +a
```

**Option B — CLI:**

```bash
murthy config set llm.endpoint https://<your-endpoint>.huggingface.cloud
murthy config set hf.token     hf_xxxxxxxxxxxxx
```

**Option C — interactive wizard:**

```bash
murthy            # then type /setup inside the chat
```

The wizard also live-checks the HF token against the LongeBench gate.

### 5c. Verify the connection

```bash
murthy status \
  --model    longevity-llm \
  --provider endpoint \
  --endpoint $L_LLM_ENDPOINT \
  --api-key  $HF_TOKEN
```

`OK  latency=…s` confirms the endpoint accepts your token and is reachable.
If you get `401` the token is wrong/expired; if you get a connection error
the endpoint URL is wrong or the endpoint is paused.

### 5d. How the CLI handles L-LLM automatically

Once `llm.endpoint` and `hf.token` are in config, the model name
`longevity-llm` is special-cased throughout the CLI:

- `/test longevity-llm`, `/benchmark longevity-llm`, `/status longevity-llm`
  and any tool-use call that names this model automatically swap
  `provider=endpoint` and inject `llm.endpoint` + `hf.token`. You do
  not need `/provider endpoint` first.
- Whenever the resolved provider is `endpoint`, Estimathon runs in
  **isolated per-problem context mode** — each slip is a fresh single-turn
  API call containing only the target problem and its prior wrong
  intervals (~210 tokens per call, vs. ~24k for the legacy shared-budget
  flow). Many HF-hosted models have no continuous conversation memory,
  so this gives them the best chance to actually use the feedback.
- The run panel shows `Mode: estimathon (isolated context)` in yellow
  when this is active.

### 5e. Run a quick smoke test

```bash
murthy run \
  --model    longevity-llm \
  --provider endpoint \
  --endpoint $L_LLM_ENDPOINT \
  --api-key  $HF_TOKEN \
  --tasks    sample \
  --mode     estimathon \
  --limit    7 \
  --budget   14
```

Or from inside the chat:

```bash
murthy
> /test longevity-llm
```

Both write to `results.jsonl` and print a multi-attempt summary table
at the end showing how the model evolved (or didn't) across attempts.
Turn on `/cheat` first to dump every raw request and response with the
per-slip scheduling order.

### 5f. L-LLM-specific gotchas

| Symptom | Cause |
|---|---|
| `404 not_found_error: model: longevity-llm` | The provider wasn't auto-routed. Make sure you're on the current build (`git pull && pip install -e .`) — the auto-route fix landed in commit `1eb9ae0`. |
| `401 invalid x-api-key` | Stale `HF_TOKEN` env var shadowing the config value. `unset HF_TOKEN` and rerun, or update your `.env`. |
| Model keeps submitting the same interval (e.g. `[50, 55]`) repeatedly | The L-LLM is essentially doing one-shot inference and ignoring the FORBIDDEN history — see the multi-attempt table at the end of a run to confirm. The runner will bail after 3 consecutive no-progress turns rather than loop forever. |
| Run hangs at constant slip count | Older builds had a degenerate-interval infinite loop; pull past `1eb9ae0` for the bail-out fix. |

---

## 6. Verify the install

```bash
murthy --help                                    # lists subcommands
murthy config list                               # shows current config (tokens masked)
murthy tasks --tasks sample --limit 3            # 3 built-in tasks, no network
murthy status --model claude-sonnet-4-6 --provider anthropic
```

If `status` reports `OK  latency=…s`, the SDK + key combo works.

---

## 7. Two ways to run a benchmark

### 7a. Interactive chat (default)

```bash
murthy
```

Type naturally — Claude routes to the right tools. Type `/` to see slash
commands (Tab-cycle through them):

| Command | What it does |
|---|---|
| `/setup` | Re-run the API-key + HF-token wizard |
| `/test [model]` | 20-task LongeBench Estimathon trial, 40-slip budget |
| `/benchmark [model] [provider] [tasks]` | One-shot run with current defaults |
| `/explore` | Show all LongeBench task types + Estimathon eligibility |
| `/model list \| search <q> \| <id>` | Browse / select benchmark model |
| `/batch <models> [provider]` | Sequentially benchmark multiple models |
| `/add <model_id> \| refresh` | Add a model to the local CSV or refresh from HF |
| `/think` | Toggle `<think>` chain-of-thought capture |
| `/status [model] [provider]` | Connectivity check |
| `/config [key] [value]` | View or set a config value |

### 7b. Non-interactive CLI

Full LongeBench, mixed mode (regression → Estimathon, categorical → one-shot):

```bash
murthy run \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --tasks longebench \
  --mode mixed \
  --limit 50
```

Estimathon-only on the 7 built-in sample tasks:

```bash
murthy run \
  --model claude-sonnet-4-6 \
  --provider anthropic \
  --tasks sample \
  --mode estimathon \
  --think
```

Against the Insilico L-LLM endpoint:

```bash
murthy run \
  --model longevity-llm \
  --provider endpoint \
  --endpoint $L_LLM_ENDPOINT \
  --api-key  $HF_TOKEN \
  --tasks    longebench \
  --mode     mixed \
  --limit    50
```

Compare a saved group of models in one shot:

```bash
murthy group add  baseline   claude-sonnet-4-6 anthropic
murthy group add  baseline   gpt-4o-mini       openai
murthy compare    --group baseline --tasks longebench --mode mixed --limit 30
```

Results stream to `results.jsonl` (or `--output <path>`). See `idea.md` for
the score formula and the *refinement accuracy* metric.

---

## 8. Build the standardized 200-question set

The Track 01 submission uses a single fixed 200-question evaluation set.
Build it from upstream sources with:

```bash
python scripts/build_standard200.py
# optional overrides
python scripts/build_standard200.py --token hf_xxx \
       --n-lb-numerical 100 --n-lb-categorical 50 --n-rna 50
```

Composition: 100 LongeBench numerical (regression + pairwise) + 50 LongeBench
categorical + 50 RNA (40 from `carolw/EP04`, 10 fill from `sarahliu/rna-eb0x`).
Estimathon budget defaults to 135 slips for the numerical track.

---

## 9. Custom RNA Q-bank pipeline (EP-01 / EP-02 / EP-03)

Four-step pipeline that builds and scores our adversarial RNA banks. Run
sequentially from the repo root:

```bash
python scripts/step1_build_rnadisease_tasks.py   # EP-01 binary + EP-03 multiclass (RNADisease v4)
python scripts/step2_build_rmdisease_task.py     # EP-02 ternary (RMDisease v2.0)
python scripts/step3_run_llm_eval.py --task all --model longevity
python scripts/step4_score_and_analyze.py        # writes results/scores_summary.json + confusion PNGs
```

Step 3 also accepts `--task EP-01 | EP-02 | EP-03` and `--model gpt-4o` if
you have `OPENAI_API_KEY` set.

---

## 10. Repo map

```
Murphy-Health/
├── pyproject.toml                package metadata, entry points
├── README.md                     pip-install user guide
├── SETUP.md                      this file — local dev guide
├── longivity_hack/               main package (note: misspelled, kept for
│   │                             continuity with PyPI distribution)
│   ├── cli.py                    Typer app: run / chat / status / tasks /
│   │                             config / group / compare
│   ├── .env.example              copy → .env
│   ├── requirements.txt          pinned deps for mode B install
│   ├── idea.md                   benchmark design rationale
│   ├── devlog.md                 chronological dev log
│   ├── CLAUDE.md                 dev guide for contributors
│   ├── hf_llm_models.csv         300+ open-source models for /model search
│   ├── benchmark/
│   │   ├── client.py             unified ModelClient (anthropic / openai /
│   │   │                         hf / endpoint)
│   │   ├── config.py             ~/.longevity/config.json + env overrides
│   │   ├── loader.py             sample / LongeBench / local JSONL
│   │   ├── runner.py             EstimathonSession, run_eval, run_mixed
│   │   ├── results.py            JSONL writer / reader
│   │   ├── chat.py               interactive UI (slash commands, wizard)
│   │   └── model_manager.py      CSV-backed model browser
│   └── results_*.jsonl           per-model outputs
└── scripts/
    ├── build_standard200.py      builds the 200-Q hackathon test set
    ├── step1_build_rnadisease_tasks.py
    ├── step2_build_rmdisease_task.py
    ├── step3_run_llm_eval.py
    └── step4_score_and_analyze.py
```

---

## 11. Common issues

**`ImportError: cannot import name 'set_int_max_str_digits'`**
You're on Python ≤ 3.10. Upgrade to 3.11+ (or stop running with `--limit
> ~14000`, which is what triggers the guard).

**`DatasetNotFoundError: insilicomedicine/longebench`**
Either your `HF_TOKEN` is missing or you haven't requested access yet. Visit
the dataset URL above and click *Request access*.

**`prompt_toolkit not found`**
Mode B install missed a dep. Run `pip install -r longivity_hack/requirements.txt`
again, or switch to `pip install -e .`.

**LongeBench load returns *0 tasks***
You forgot `--estimathon` filtering, or the source kwarg is wrong. Use
`longebench` (not `LongeBench`) and pair `--mode estimathon` with the right
loader flag — the chat wrapper handles this automatically.

**Model returns no valid interval**
Parser in `runner.py:parse_interval()` accepts `[min, max]` and bare numbers;
3 consecutive parse failures terminates the session early. Check the model's
last response in `results.jsonl` to see what it actually wrote.

---

## 12. House-keeping notes

- `longivity_hack/=3.0.0` is a stray file created by a Windows `pip install
  prompt_toolkit >=3.0.0` invocation that the shell parsed as a redirect.
  Safe to delete locally; leaving alone here so the repo state matches CI.
- `longivity_hack/.gitignore` partially duplicates the root `.gitignore`.
  Both are harmless; the root one takes priority for top-level files.
- Empty `results_*.jsonl` files in `longivity_hack/` are placeholders from
  the `/batch` command — they get overwritten on the next run for that model.

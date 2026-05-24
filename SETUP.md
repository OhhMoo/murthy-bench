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

Reading from `longivity_hack/benchmark/config.py`:

```python
_DEFAULTS = {
    "llm.endpoint":  None,             # ← you MUST set this
    "llm.model":     "longevity-llm",  # ← already correct; only change if renamed
    "hf.token":      None,             # ← you MUST set this
    ...
}
_ENV_OVERRIDES = {
    "llm.endpoint":  "L_LLM_ENDPOINT",
    "hf.token":      "HF_TOKEN",
    ...
}
```

So for L-LLM you need exactly two values:

| Config key | Env var | What it's for |
|---|---|---|
| `llm.endpoint` | `L_LLM_ENDPOINT` | The HuggingFace inference URL for the L-LLM (provided by the hackathon organisers; the working tree's `longivity_hack/.env.example` has the current default) |
| `hf.token` | `HF_TOKEN` | Sent as the bearer token to the endpoint AND used to access the gated `insilicomedicine/longebench` dataset |

Env vars shadow file values per `config.get()` precedence: env first, then
`~/.longevity/config.json`, then the defaults above.

### 5b. Set the values

**Option A — `.env` (recommended; sets both at once):**

```bash
cp longivity_hack/.env.example .env
# edit .env, fill in real values for HF_TOKEN and L_LLM_ENDPOINT
set -a; source .env; set +a
```

`set -a` makes every variable assignment in `.env` an export, so
`HF_TOKEN` and `L_LLM_ENDPOINT` land in the environment of the shell
you then run `murthy` from.

**Option B — `murthy config set` (writes to `~/.longevity/config.json`):**

```bash
murthy config set llm.endpoint https://<your-endpoint>.huggingface.cloud
murthy config set hf.token     hf_xxxxxxxxxxxxx
```

Note: `set_value()` also exports the matching env var for the current
process, so a subsequent `murthy` invocation in the same shell sees the
new value immediately. New shells pick it up from `config.json`.

**Option C — the `/setup` wizard inside chat (HF token only):**

```bash
murthy
> /setup
```

The wizard (`_setup_wizard()` in `longivity_hack/benchmark/chat.py`)
walks three steps: Anthropic key, HF token (with a live LongeBench access
check via `datasets.load_dataset(..., streaming=True)`), and OpenAI key.

> **The wizard does NOT prompt for `llm.endpoint`.** You still have to
> set the endpoint via Option A or B. Use the wizard for the HF token
> half, then add the endpoint separately.

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
`longevity-llm` (or whatever `llm.model` is set to) is special-cased
throughout the CLI by two helpers in `longivity_hack/benchmark/chat.py`:

- `_is_longevity_llm(model)` matches both the literal alias and the
  configured `llm.model` value.
- `_route_longevity_llm(model, provider, api_key, endpoint_url)`
  rewrites the call to `provider="endpoint"` and fills `endpoint_url`
  + `api_key` from config — unless the caller already passed them.

This routing fires at the top of `_tool_run_benchmark` and inside
`_resolve_client`, so every entry point benefits:

- `/test longevity-llm` (chat.py:780)
- `/benchmark longevity-llm` (chat.py:892)
- `/status longevity-llm` (chat.py:898)
- `/model longevity-llm` — pins it AND flips `state.bench_provider`
  to `endpoint` (chat.py:866) so the panel and subsequent commands
  stay coherent.
- Claude tool-use calls naming this model.

Once routing has produced `provider == "endpoint"` and the mode is
`estimathon` or `mixed`, `_tool_run_benchmark` sets
`use_isolated = True` (chat.py:472) and dispatches to
`run_estimathon_isolated()` (`runner.py`). That runner:

- Sends one fresh single-turn API call per slip (no growing conversation)
- Includes only the target problem + the prior wrong intervals for THAT
  problem + remaining-slip count
- Uses a round-robin scheduler so the model never has to choose which
  problem to attempt — it just answers the one in front of it
- Per-call prompt drops from ~24k tokens (legacy shared-budget flow)
  to ~210 tokens, which matters because many HF-hosted endpoint models
  have no continuous conversation memory

The run panel shows `Mode: estimathon (isolated context)` in yellow
when this path is active. Turn on `/cheat` to see a green-background
`Slip N/B → Pn (attempt k/3)` header before each request dump.

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

---
title: "Track01-LongevityLLM Benchmarking"
source: "https://www.notion.so/Track-01-LongevityLLM-Benchmarking-36733e38a36f80dc9440fb8a06ed5d43"
author:
published:
created: 2026-05-23
description: "A collaborative AI workspace, built on your company context. Build and orchestrate agents right alongside your team's projects, meetings, and connected apps."
tags:
  - "clippings"
---
![Page icon](https://www.notion.so/icons/golf_orange.svg?mode=dark)

## Track 01 · LongevityLLM Benchmarking

This track sits at the intersection of AI and aging biology. You will be working with a fine-tuned large language model built specifically for longevity science, and your job is to push its limits, find its gaps, and build something useful with it. If you care about how AI performs in biomedical research and want your weekend's work to end up in a journal, this is your track.

| Criterion | Weight |
| --- | --- |
| Utility: Does the benchmark assess an LLM capability that genuinely matters for aging biology? Does the gap analysis focus on failure modes that would affect whether a real user could trust the model? | 5 pts |
| Diversity: Are samples, phrasings, and answer formats varied enough to prevent shortcut learning? A “compound → senescence effect” benchmark dominated by rapamycin experiments fails on the data axis. An MCQ where the correct option is always “A” fails on the semantic axis. A single rigid format that cannot be turned into binary, ternary, regression, or set-generation variants fails on the format axis — a model that has never seen negative DEG results will ace “Is this gene overexpressed in condition X?” but will break down once “Not changed” becomes a possible answer. | 5 pts |
| Retrieval resistance: How likely is a SotA model to have already seen the answer in its training corpus? Tasks based on PubMed abstracts score poorly. Tasks based on processed raw datasets or obscure repositories score well. | 5 pts |
| Statistical rigor: Are class imbalances handled, is the chosen metric appropriate (F1, balanced accuracy, MAE, Jaccard…), and is a baseline reported so that the headline number is interpretable? | 5 pts |

### Prize: $1000 per team and a mention in a peer-reviewed publication with the Insilico Medicine research team.

## The Task

Longevity Bench (LB) is a collection of datasets intended for measuring LLMs’ ability to derive high-level phenotypes from low-level data. Most tasks in it focus on age and mortality prediction, life extension experiments, and associations between genes and species lifespan.

You may access some tasks used to evaluate LLMs [here](https://dx.doi.org/10.57967/hf/8851).

### Currently explored domains:

Epigenetics: GEO Datasets on DNAm arrays;

Clinical biodata: NHANES;

Transcriptomics: GTEx;

Genetics: OpenGenes, SynergyAge, CellAge;

Proteomics: public Olink datasets.

### Task ideas:

Multispecies aging clocks: LB is mostly human-centric. A task asking an LLM to predict the age of a cat, dog, or other animal could be used to check if a model has generalized the concept of aging across different species;

Anti-aging target potential: The repositories documenting genes involved in longevity or small-molecule targets that can prolong human lifespan are strongly biased toward well-studied pathways and genes. Is there a way to quantify the potential of a gene to slow down aging based only on omics data?

Mouse strain longevity: MGI provides detailed annotations of murine strains, including detailed phenotype and allele profiles. Can a model identify the effect of a mutation on murine lifespan?

Factual accuracy: L-LLM can explain its decisions in terms of aging biology and by appealing to aging clocks. But are these arguments true or just seem plausible? Are the CpG sites and genes it mentions really used by the clocks? Are they truly involved in the pathways L-LLM says they are?

### Task specifications:

All tasks should be split into train-test cohorts, based on covariates that prevent data leakage. E.g. for GTEx data — by patient ID, for GEO data — by GSE accessions, for proteins — for protein families, for in vivo experiments — by species, for in vitro — by cell line. Even if the task is small, there still must be a way to split it multiple ways.

All tasks need to have verifiable ground truth.

Preferred task formats: binary classification, multiple choice question (MCQ), ternary classification (Yes, No, None/Either), pairwise comparison (choose sample A or B), regression, set generation;

Freetext output format that requires expert/agent review is discouraged;

Each task needs to have a formally defined statistic for measuring performance. F1, balanced accuracy, MAE, Jaccard similarity etc;

No prompt shall exceed 30K tokens after tokenizing with cl100k\_base;

File format for final submission: JSONL files with ChatML system-user-assistant interaction;

Tasks involving sequence modalities should be avoided. E.g. predicting the effect of an SNP on human lifespan requires an LLM to operate with a DNA sequence.

Minimal N prompts in a task is 50.

### Extra credit: a training signal for reasoning

Once your benchmark produces a concrete performance metric, you have a substrate for asking a harder question: does the model arrive at its answer through trustworthy reasoning, or does it land on the right answer for the wrong reasons?

Citing a gene that does not exist, attributing it to the wrong chromosome or pathway, or contradicting itself between its thinking trace and its final token are all failure modes that survive any final-answer-only metric. We want a fast, programmatic signal that scores the trace, not just the final answer — cheap enough to call millions of times when used as a training signal for reasoning, and resistant to surface-level hacking. A regex over capitalized tokens to detect genes, for instance, fails the moment a model stops capitalizing, or the moment we evaluate on murine and C. elegans genes whose symbols are not capitalized.

We will share access to a HuggingFace endpoint of Longevity-LLM (L-LLM), a fine-tuned Qwen3.5-9B checkpoint that outperforms much larger models on the LB tasks. Run L-LLM on the benchmark you built, collect its reasoning traces, and prototype a scoring function over them. Promising directions include verifying that mentioned genes, CpG sites, or strains exist and have the properties claimed; checking mutual consistency between the thinking trace and the final answer; grounding claimed pathway memberships in known biology. The strongest submissions will describe a scoring function that is automatable, hard to hack, and demonstrably correlated with biological correctness on a held-out set of traces from your own benchmark.

## Guide: How to connect to the hosted Longevity-LLM endpoint and run the LongeBench evaluation tasks against it.

### Longevity-LLM Hackathon — Participant Guide

#### What you're talking to

Model: Longevity-LLM, a fine-tuned Qwen3.5-9B trained on a curated aging-biology corpus (DNA methylation, transcriptomics, plasma proteomics, clinical measurements, gene knowledge). API: OpenAI-compatible (/v1/chat/completions, /v1/completions, /v1/models) served by vLLM. Context window: up to 32,000 tokens. Long inputs (~25–28k) are supported but cost noticeably more latency. Output format: the model is post-trained to answer LongeBench prompts directly — usually a single letter, a number, or a short list, depending on the task format.

#### Endpoint URL

The organizers will share the live URL on the day. Throughout this guide it's referenced as:

ENDPOINT\_URL = https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud

Use it in the

Authorization: Bearer <token>

header (or as

api\_key

in the OpenAI SDK).

#### Quick sanity check

bash

export EP=https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud # Health curl -s $EP/health # Model name curl -s $EP/v1/models | python3 -m json.tool # First request curl -s -H "Content-Type: application/json" \\ -X POST $EP/v1/chat/completions \\ -d '{ "model": "longevity-llm", "messages": \[{"role":"user","content":"Name one well-validated epigenetic aging clock."}\], "max\_tokens": 100, "temperature": 0.0 }' | python3 -m json.tool

If the model id from /v1/models is different from

longevity-llm

, use whatever id is shown.

bash

pip install openai datasets

python

from openai import OpenAI client = OpenAI( base\_url="https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud/v1", api\_key="<token-or-any-string>", ) r = client.chat.completions.create( model="longevity-llm", messages=\[ {"role": "user", "content": "Name one well-validated epigenetic aging clock."} \], max\_tokens=200, temperature=0.0, ) print(r.choices\[0\].message.content) print("tokens:", r.usage)

#### CLI — ask.py (stdlib, no dependencies)

A small stdlib-only CLI sits next to this guide for quick interactive queries. Paste your HF token into

HF\_TOKEN

and (if needed) the endpoint URL into

ENDPOINT

at the top of the file, then:

bash

\# Basic chat call (default mode) python3 ask.py -q "Name one well-validated epigenetic aging clock." # Tune generation python3 ask.py -q "..." -n 1000 -t 0.0 # Thinking mode (Qwen3.5 chain-of-thought) python3 ask.py -q "..." --think -n 3000 # Save a structured record python3 ask.py -q "..." --think -o answer.json python3 ask.py -q "..." -o log.jsonl -a # Hit /v1/completions or /generate instead of /v1/chat/completions python3 ask.py -q "..." -m completions python3 ask.py -q "..." -m generate # Raw JSON response python3 ask.py -q "..." --raw # Long generations: bump the HTTP read timeout (default 900s) python3 ask.py -q "..." -n 8000 --think --timeout 1800

Flags at a glance:

| Flag | Default | Notes |
| --- | --- | --- |
| \-q / --query | (required) | Prompt to send. |
| \-n / --max-new-tokens | 256 | Bump to 2000–4000 when --think is on. |
| \-t / --temperature | 0.7 | Set 0.0 for deterministic eval. |
| \-m / --mode | chat | chat, completions, or generate. |
| \--think / --no-think | \--no-think | Toggles chat\_template\_kwargs.enable\_thinking. |
| \-o / --output | stdout | Writes a JSON record (or --raw body) to file. |
| \-a / --append | overwrite | With -o, appends instead of overwriting. |
| \--timeout | 900 (s) | HTTP read timeout — raise for very long generations. |
| \--raw | off | Dump the raw JSON response unchanged. |

When stdout (no -o), thinking traces are printed as a human-readable === THINKING === / === ANSWER === split, followed by a \[finish\_reason=... usage=...\] footer.

#### Loading the LongeBench dataset

The eval data lives at [https://huggingface.co/datasets/insilicomedicine/longebench](https://huggingface.co/datasets/insilicomedicine/longebench).

python

from datasets import load\_dataset # Main set — 22 tasks, 32k rows bench = load\_dataset("insilicomedicine/longebench", "benchmark", split="eval") # Held-out set — 5 tasks, 2k rows extra = load\_dataset("insilicomedicine/longebench", "extra", split="eval") print(bench\[0\]\["lb\_id"\], bench\[0\]\["display\_name"\], bench\[0\]\["metric"\])

Each row has:

messages

— chat messages to send to the model (system + user). The last entry is the gold answer — do NOT send it to the model.

lb\_id

,

pool

,

display\_name

,

domain

,

format

,

metric

,

units

— task metadata.

task

— free-text task description.

metadata

— JSON-encoded provenance (sample IDs etc). The dataset's

messages

field includes both the input and the gold answer. When calling the model, send

messages\[:-1\]

(everything except the assistant's gold response) and compare the model's prediction against

messages\[-1\]\['content'\]

.

#### Running one task end-to-end

python

import json, re from openai import OpenAI from datasets import load\_dataset client = OpenAI( base\_url="https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud/v1", api\_key="<token>", ) ds = load\_dataset("insilicomedicine/longebench", "benchmark", split="eval") ds\_38 = ds.filter(lambda r: r\["lb\_id"\] == "LB-0038") # NHANES Age / Regression preds, golds = \[\], \[\] for row in ds\_38.select(range(20)): msgs = row\["messages"\]\[:-1\] # drop gold gold = row\["messages"\]\[-1\]\["content"\].strip() r = client.chat.completions.create( model="longevity-llm", messages=msgs, max\_tokens=500, temperature=0.0, ) pred = r.choices\[0\].message.content.strip() preds.append(pred) golds.append(gold) # Quick MAE for the regression task def extract\_int(s): m = re.search(r"-?\\d+", s) return int(m.group()) if m else None errors = \[abs(extract\_int(p) - int(g)) for p, g in zip(preds, golds) if extract\_int(p) is not None\] print(f"n={len(errors)} MAE={sum(errors)/len(errors):.1f}")

#### Concurrency

The endpoint handles multiple in-flight requests via vLLM's continuous batcher. Recommended client-side concurrency is 4–8.

python

from concurrent.futures import ThreadPoolExecutor, as\_completed def call(row): r = client.chat.completions.create( model="longevity-llm", messages=row\["messages"\]\[:-1\], max\_tokens=500, temperature=0.0, ) return row\["lb\_id"\], r.choices\[0\].message.content.strip() rows = ds\_38.select(range(100)) with ThreadPoolExecutor(max\_workers=8) as ex: futs = \[ex.submit(call, r) for r in rows\] for f in as\_completed(futs): lb\_id, pred = f.result() # store, score, etc.

Please don't exceed concurrency 8 per team — the endpoint is shared across all teams and runaway scripts will starve everyone.

#### Thinking mode (Qwen3.5 chain-of-thought)

The model has a built-in thinking mode. When enabled it emits a

<think>...</think>

block of reasoning before its final answer. Off by default for speed.

python

\# Disable thinking (fast, default) client.chat.completions.create( model="longevity-llm", messages=msgs, max\_tokens=500, temperature=0.0, extra\_body={"chat\_template\_kwargs": {"enable\_thinking": False}}, ) # Enable thinking (slower, often more accurate on multi-hop tasks) client.chat.completions.create( model="longevity-llm", messages=msgs, max\_tokens=3000, # bump this! temperature=0.0, extra\_body={"chat\_template\_kwargs": {"enable\_thinking": True}}, )

Notes:

Don't use a

/nothink

system prompt. Only the

enable\_thinking

kwarg works.Bump

max\_tokens

to 2000–4000 when thinking is on. The think trace alone often eats 500–1500 tokens before the answer appears.Strip

<think>

blocks before scoring. Split on the closing tag:

python

import re raw = r.choices\[0\].message.content m = re.search(r"(?:<think>)?(.\*?)</think>\\s\*", raw, flags=re.DOTALL) if m and "</think>" in raw: think = m.group(1).strip() answer = raw\[m.end():\].strip() else: think, answer = None, raw.strip()

If your endpoint was launched with a

\--reasoning-parser

, the reasoning is in

r.choices\[0\].message.reasoning\_content

and

content

is already clean — check that field first.

Thinking-on runs are 3–5× slower and consume 3–5× more endpoint capacity. Use sparingly on full sweeps.

#### Inference parameters

| Param | Recommended | Notes |
| --- | --- | --- |
| temperature | 0.0 | Deterministic — important for reproducible eval |
| top\_p | omit (or 1.0) | irrelevant at temp 0 |
| max\_tokens | 200–1000 depending on task | Multiclass/binary: 50 is plenty. Generation/freetext: 500+ |
| seed | any fixed int | Helps reproducibility if you ever use temp > 0 |
| stop | \["</s>"\] (optional) | Some clients add it automatically |

#### Common pitfalls

Sending the gold answer to the model. Always slice

messages\[:-1\]

.Tiny max\_tokens. If you set

max\_tokens=10

for a task where the model produces a brief justification before its answer, the response gets truncated and accuracy looks like 0%. Give it 500 when in doubt.Parsing prose answers. For multiple-choice tasks, use

re.findall(r"(?<!\[A-Za-z\])(\[A-F\])(?!\[A-Za-z\])", text)\[-1\]

to extract the final letter rather than checking the whole string.

Long-context tasks at full sweep. LB-0002, LB-0010, LB-0014 have ~22–27k token prompts. A single request can take 30s+. Test on a small sample first.

Client-side read timeouts. Set the client timeout to several minutes (15+ min for --think with max\_tokens >= 4000).

Conversation history. The model is stateless — every request must include the full message history.

#### Reproducibility

For your final submission, log per-row:

lb\_id

input messages (or a hash)

raw response.choices\[0\].message.content

usage (token counts)

the extracted answer + the gold

the task metric (accuracy / MAE / jaccard)
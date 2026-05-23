# Longevity Hackathon — Team Research Guide

## What We're Building

An iterative refinement benchmark for Longevity-LLM (L-LLM), a fine-tuned Qwen3.5-9B model for aging biology. Instead of one-shot Q&A, we give the model interval-based numerical tasks, feed it yes/no correctness signals after each attempt, and measure how fast and efficiently it converges to the right answer. Read `idea.md` for the full concept before diving into your area.

**L-LLM endpoint:** `https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud`
**Dataset:** `insilicomedicine/longebench` on HuggingFace (gated — request access if you haven't)
**Submission format:** JSONL files in ChatML system-user-assistant format

---

## How the Three Areas Connect

```
Area 1: Dataset         Area 2: Eval Loop        Area 3: Trace Scorer
─────────────────       ─────────────────────    ────────────────────
Build interval tasks  → Run L-LLM iteratively  → Score thinking traces
Output: tasks.jsonl     Output: results.jsonl    Output: scorer + report
```

Area 2 depends on Area 1's output. Area 3 depends on Area 2's output. Start Area 1 first, then Area 2 can begin as soon as the first 50 tasks are ready. Area 3 can prototype independently using existing LongeBench thinking traces.

---

## Area 1 — Dataset Construction

**Goal:** Build a set of interval-based benchmark tasks grounded in real aging biology data. Minimum 50 task instances, leakage-free train/test split, verifiable ground truth.

### Your job

1. **Pick two source datasets** from the list below and download them. Aim for ones not already in LongeBench (NHANES, GTEx, GEO are already used — pick something fresher for retrieval resistance).

   | Dataset | URL | Best for |
   |---|---|---|
   | AnAge | genomics.senescence.info/download | Multispecies lifespan regression |
   | DrugAge | genomics.senescence.info/download | Drug lifespan extension % regression |
   | MGI mouse phenotypes | informatics.jax.org/downloads | Mutation effect on murine lifespan |
   | GEO (pick obscure GSE) | ncbi.nlm.nih.gov/geo | DNAm-based age regression |

2. **Design the task prompt.** The user message should contain a real biological profile (a row of measurements), and the model should respond with an interval [min, max]. Example for AnAge:

   ```
   System: You are an expert in comparative biology and aging.
   User: Given the following biological traits of an unknown mammal species:
     - Body mass: 22.4 kg
     - Metabolic rate: 0.38 W/kg
     - Gestation time: 290 days
     - Litter size: 1.2
   Submit an interval [min, max] for the maximum lifespan of this species in years.
   Respond with only: [min, max]
   ```

3. **Build the leakage-free split.** Split by a covariate that prevents data leakage:
   - AnAge → split by taxonomic Order
   - DrugAge → split by drug mechanism class
   - GEO → split by GSE accession
   - MGI → split by genetic background strain

4. **Output format.** Each task instance as a JSONL row:
   ```json
   {
     "lb_id": "LB-ITER-001",
     "task": "multispecies_lifespan_regression",
     "domain": "comparative_biology",
     "format": "interval",
     "metric": "estimathon_score",
     "budget": 5,
     "messages": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "...biological profile..."},
       {"role": "assistant", "content": "[20, 60]"}
     ],
     "gold": 45.2,
     "split": "test"
   }
   ```
   The final assistant message is the gold answer range hint — Area 2 will strip this and run the actual iterative loop.

### Deliverable
`tasks.jsonl` — at least 50 test instances, plus train/val splits. Include a short `data_notes.md` describing the source, split rationale, and any preprocessing decisions.

---

## Area 2 — Evaluation Loop

**Goal:** Implement the iterative feedback protocol, run L-LLM against Area 1's tasks, and collect full multi-round records including thinking traces.

### Your job

1. **Set up the client.** Install dependencies and verify endpoint access:
   ```bash
   pip install openai datasets
   ```
   ```python
   from openai import OpenAI
   client = OpenAI(
       base_url="https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud/v1",
       api_key="<token>",
   )
   ```

2. **Implement the feedback loop.** For each task instance:
   - Send the prompt to L-LLM (with thinking enabled)
   - Parse the model's interval response `[min, max]`
   - Check if the gold answer falls within the interval
   - Generate the feedback message (see feedback rules below)
   - Append feedback to conversation history and repeat
   - Stop when: budget exhausted, or model submits a correct interval with width factor ≤ 2

   **Feedback rules:**
   ```python
   def generate_feedback(pred_min, pred_max, gold, attempt, budget):
       contains = pred_min <= gold <= pred_max
       width_factor = int(pred_max / pred_min)
       if contains:
           return f"Your interval contains the answer. Width factor: {width_factor}. {budget - attempt} submissions remaining."
       else:
           direction = "too high" if pred_min > gold else "too low"
           return f"Your interval does not contain the answer — it is {direction}. {budget - attempt} submissions remaining."
   ```

3. **Parse interval responses.** Models don't always format cleanly:
   ```python
   import re
   def parse_interval(text):
       m = re.search(r'\[?\s*([0-9.e+\-]+)\s*,\s*([0-9.e+\-]+)\s*\]?', text)
       if m:
           return float(m.group(1)), float(m.group(2))
       return None, None
   ```

4. **Extract thinking traces.** Strip `<think>` blocks before scoring but save them:
   ```python
   def split_think(raw):
       m = re.search(r'<think>(.*?)</think>\s*', raw, flags=re.DOTALL)
       if m:
           return m.group(1).strip(), raw[m.end():].strip()
       return None, raw.strip()
   ```

5. **Concurrency.** Max 8 parallel requests to the shared endpoint. Use `ThreadPoolExecutor(max_workers=8)`.

6. **Output format.** One JSONL record per task instance with full conversation history:
   ```json
   {
     "lb_id": "LB-ITER-001",
     "gold": 45.2,
     "budget": 5,
     "rounds": [
       {
         "round": 1,
         "think": "The metabolic rate suggests...",
         "answer": "[20, 60]",
         "pred_min": 20, "pred_max": 60,
         "contains_gold": true,
         "width_factor": 3,
         "feedback": "Correct. Width factor: 3. 4 submissions remaining."
       },
       {
         "round": 2,
         "think": "I can narrow this...",
         "answer": "[35, 55]",
         "pred_min": 35, "pred_max": 55,
         "contains_gold": true,
         "width_factor": 1,
         "feedback": "Correct. Width factor: 1. 3 submissions remaining."
       }
     ],
     "rounds_to_correct": 1,
     "final_width_factor": 1,
     "task_score": 1,
     "budget_used": 2
   }
   ```

### Deliverable
`results.jsonl` — full multi-round records for all test instances. Include a `eval_notes.md` with any parsing edge cases, timeout issues, or anomalies observed.

---

## Area 3 — Reasoning Trace Scorer

**Goal:** Build a programmatic scoring function that evaluates the *quality of reasoning* in the thinking traces collected by Area 2. The scorer must be automatable, hard to hack, and correlated with biological correctness.

### Your job

You can prototype independently using existing LongeBench traces while waiting for Area 2 results. To get traces quickly:
```python
from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    base_url="https://saujlffcxf20v74m.us-east-2.aws.endpoints.huggingface.cloud/v1",
    api_key="<token>",
)
ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval")
# pick a small subset, run with enable_thinking=True, collect raw traces
```

### Three scoring signals to implement

**Signal 1 — Entity verification**

Extract biological entities from the thinking trace and verify they exist:
```python
# Gene symbols: check against HGNC or a local gene list
# CpG site IDs: check against Illumina 450K/EPIC manifest
# Drug names: check against DrugAge/DrugBank

def verify_entities(think_trace, entity_type="gene"):
    # extract candidates
    # lookup against authoritative list
    # return: (n_mentioned, n_valid, n_hallucinated)
    pass
```

**Signal 2 — Monotonic convergence check**

Across rounds, does the model's interval narrow consistently?
```python
def convergence_score(rounds):
    widths = [r["width_factor"] for r in rounds if r["contains_gold"]]
    if len(widths) < 2:
        return 1.0
    # penalize if width ever increases after being correct
    violations = sum(widths[i] > widths[i-1] for i in range(1, len(widths)))
    return 1.0 - violations / (len(widths) - 1)
```

**Signal 3 — Trace-answer consistency**

Does the model's stated reasoning in the think block predict the interval it actually submits?
```python
# If the think trace says "this species likely lives 40-50 years"
# but the submitted interval is [10, 200], that's inconsistent
# Use an LLM judge (cheapest Claude model) to score consistency 0/1
# This is the one signal that requires a model call — keep it cheap
```

### Composite scorer

Combine signals into a single trace quality score:
```python
def trace_quality(think_trace, rounds, gold):
    entity_score = entity_hallucination_rate(think_trace)   # lower = better
    convergence  = convergence_score(rounds)                 # higher = better
    consistency  = trace_answer_consistency(think_trace, rounds[-1]["answer"])
    return {
        "entity_hallucination_rate": entity_score,
        "convergence_score": convergence,
        "trace_answer_consistency": consistency,
        "composite": (1 - entity_score) * convergence * consistency
    }
```

### Validate the scorer

On your held-out set, check: does composite score correlate with whether the model actually converged to the correct answer? If a model with a bad trace still gets the right answer, that's a red flag — worth flagging in your write-up.

### Deliverable
`scorer.py` — runnable script that takes `results.jsonl` and outputs `scored_traces.jsonl`. Include a `scorer_notes.md` describing: what each signal measures, failure modes, and correlation with final answer accuracy on your validation set.

---

## Final Integration

When all three areas are done, run:
```
tasks.jsonl → eval_loop.py → results.jsonl → scorer.py → scored_traces.jsonl
```

Final submission to the hackathon: all JSONL files + a short writeup covering benchmark design, scoring formula, and trace scorer validation.

**Aggregate score formula (lower is better):**
```
total_score = (10 + Σ width_factors for all correct final intervals) × 2^(# tasks never solved)
```

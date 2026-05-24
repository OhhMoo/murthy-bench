# Iterative Refinement Benchmark for Longevity-LLM

## Core Idea

Standard LongeBench evaluation is one-shot: the model answers, you score it, done. This benchmark tests something different — **how well L-LLM updates its beliefs when given partial feedback**. Instead of asking for a final answer upfront, we give the model iterative yes/no signals about which parts of its answer are correct, then measure how quickly and efficiently it converges to the right answer.

This captures a failure mode that final-answer-only metrics miss entirely: a model that lands on the right answer for the wrong reasons will fall apart the moment it's told "part of that is wrong."

---

## Two Mechanisms, Combined

### 1. Interval-Based Answers (Estimathon mechanic)

For numerical tasks (age prediction, lifespan, fold change, drug extension %), the model does not submit a single number. It submits an **interval [min, max]**.

An interval is **good** if it contains the correct answer. But a wide interval is penalized. The **session-level** score across N problems is:

```
score = (10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good final answers)
```

- Lower score is better
- Only the **last** submission per problem counts — refining a good interval is a bet
- A correct but wide interval [1, 200] scores worse than a tight correct interval [58, 72]
- An unsolved problem doubles the score — covering all N problems beats nailing one perfectly
- This forces the model to be both **calibrated** (interval contains the truth) and **confident** (narrow interval)

The model can resubmit a refined interval after each slip. This directly tests whether the model can reason about its own uncertainty and update it using only biological domain knowledge.

### 2. Partial Feedback (Binary, Shared Budget)

After each submission the model receives **only binary feedback** — no directional hints:

| Signal | Meaning |
|---|---|
| `GOOD` | Interval contains the correct answer |
| `BAD`  | Interval does not contain the correct answer |

No "too high / too low." The model must use its biological domain knowledge to infer *why* it missed and in which direction to move. Directional hints would let the model converge mechanically without any reasoning — binary-only feedback forces genuine understanding.

After each slip the model also sees:
- Its current score (before → after, with ↓/↑)
- A live standings table for every problem
- Remaining slips in the shared budget
- A warning if it replaced a previously GOOD interval with a BAD one

**Shared budget**: One pool of slips across all N problems (default `floor(18/13 × N)`, matching the real Estimathon's exact 18-slip / 13-problem ratio). The model must decide how to allocate: spend more slips refining a hard problem, or lock in an answer early and move on.

**Per-problem cap**: Each problem can be attempted at most 3 times (`_MAX_SLIPS_PER_PROBLEM = 3`). Attempts beyond the cap are rejected without consuming a slip — the model is told to move on. This prevents the degenerate strategy of spending the entire budget on one problem.

---

## What We Measure

Per session:

| Metric | Description |
|---|---|
| `final_score` | `(10 + Σ floor(max/min) for GOOD final answers) × 2^(N − # good)` — lower is better |
| `n_good_final` | Problems solved at session end |
| `slips_used` | Total submissions made |
| `refinement_accuracy` | Of bets placed on improving a GOOD interval, fraction that succeeded |
| `thinking_trace` | Full `<think>` block per slip — recorded for reasoning analysis |

**Refinement accuracy** is the primary inference quality signal. After a confirmed GOOD interval, any re-submission is a voluntary bet. Random guessing on binary-only feedback succeeds ~50% of the time. A model significantly above 50% is genuinely reasoning about biology to infer the direction of refinement; a model at 50% is guessing.

```
total_score = (10 + Σ floor(max/min) for all GOOD final intervals) × 2^(N − # good final answers)
```

Lower total score = better. A model that hedges with wide intervals will lose to a model that is genuinely calibrated, even if both solve the same number of problems.

---

## Why This Is Novel vs. Standard LongeBench

| Property | Standard LongeBench | This benchmark |
|---|---|---|
| Answer format | Single point prediction | Interval with iterative refinement |
| What is scored | Final answer correctness | Calibration + convergence speed + budget use |
| Reasoning trace role | Extra credit | Core signal — trace at each round recorded |
| Failure mode tested | Wrong answer | Right answer wrong reasoning, inability to update |
| Retrieval resistance | Depends on task | Interval format makes memorization harder to exploit |

---

## Example Walkthrough

**Task**: Predict the chronological age of a subject from 353 CpG methylation beta values (ground truth = 67 years). 8 problems total, shared budget of 11 slips.

**Slip 1** — Problem: CpG age
- Model submits: `[40, 90]` — width factor = floor(90/40) = 2
- Feedback: `GOOD`  width=2.  Score 1024 → 522 ↓.  Slips remaining: 10.

**Slip 2** — Model bets on tightening the interval
- Thinking trace: *"GOOD means 67 is inside [40, 90]. The methylation pattern at cg07364285 skews older — most subjects with this profile are 60–75. I'll bet on [60, 80]."*
- Model submits: `[60, 80]` — width factor = floor(80/60) = 1
- Feedback: `GOOD`  width=1.  Score 522 → 270 ↓.  Slips remaining: 9.

**Slip 3** — Model moves on to another problem
- Thinking trace: *"Width factor 1 is as tight as I can go without risking a miss. Expected value of refining further is negative. Using remaining slips on unsolved problems."*

**Refinement accuracy signal**: the model placed 1 bet (slip 2) and succeeded. A model that bets randomly after binary-only feedback wins ~50%; this model used its biological reasoning to infer direction correctly.

**Contrast with a poorly calibrated model**: submits `[1, 120]` every round — always GOOD, but width factor = 120 per problem. Same "correct" count, 60× worse score.

---

## Thinking Trace Analysis (Extra Credit)

Because we collect a thinking trace at every round, we have a multi-turn record of how the model's internal reasoning evolves. This is the substrate for the reasoning scorer described in the extra credit section of Track 1.

Specific signals to extract:
- **Gene/CpG mentions**: Do the biological entities the model cites actually exist and have the properties claimed?
- **Monotonic narrowing**: Does the model's interval consistently shrink, or does it widen (sign of confused reasoning)?
- **Self-consistency**: Does the model's stated reasoning in round N+1 follow logically from the feedback received in round N?
- **Hallucination under pressure**: Does the model start fabricating biological justifications when its first guess is wrong?

---

## LongeBench Integration & Estimathon Adapter

The benchmark can run against the full InSilico LongeBench dataset (32K+ tasks) via an automated transformation layer:

### Task Filtering
LongeBench contains 6 task formats: regression, pairwise, binary, multiclass, ternary, generation.
Only **regression** tasks are compatible with interval-based estimathon scoring (numeric gold values).
`pairwise` tasks ask "which individual is older? A or B" — gold is a letter, not a number, so they
are routed to the one-shot track instead. Non-numeric tasks are filtered out of Estimathon automatically.

### Prompt Transformation
Raw LongeBench tasks ask for a single point answer. The adapter rewrites them to ask for intervals:

**Before (raw LongeBench):**
```
"Given methylation profile M1, M2, M3, predict the age."
Gold: "65.3"
```

**After (estimathon format):**
```
"Given methylation profile M1, M2, M3, predict the age.
Submit an interval [min, max] for your answer. Reply with only: [min, max]"
Gold: 65.3 (as float)
```

### Gold Extraction
LongeBench assistant messages often contain multi-line reasoning. The adapter extracts numeric gold robustly:
- Tries direct float parse first
- Falls back to regex for the last number in text
- Handles units ("67 years"), scientific notation ("5e2"), reasoning traces

This enables safe merging of 5K+ regression tasks from LongeBench into estimathon sessions without manual curation.

---

## Dataset Construction

Source data: AnAge (multispecies lifespan), GEO DNAm arrays (epigenetic age), NHANES (clinical biomarkers). Split by species class / GSE accession / survey cycle to prevent leakage.

Each task instance becomes a multi-turn JSONL record:

```jsonl
{
  "lb_id": "LB-ITER-001",
  "task": "age_regression",
  "domain": "epigenomics",
  "format": "interval",
  "metric": "estimathon_score",
  "budget": 5,
  "rounds": [
    {"role": "user", "content": "...CpG profile..."},
    {"role": "assistant", "content": "[40, 90]", "think": "..."},
    {"role": "user", "content": "Feedback: interval is good. Width factor: 2."},
    {"role": "assistant", "content": "[60, 80]", "think": "..."},
    ...
  ],
  "gold": 67,
  "final_score": 1
}
```

Minimum 50 instances per task. Budget fixed at 5 rounds per task (or shared pool across tasks).

---

## Scoring Summary

| Component | Formula | Direction |
|---|---|---|
| Width factor per problem | `floor(max/min)` of last GOOD interval | Lower better |
| Unsolved penalty | `2^(N − # good final answers)` multiplier | Lower better |
| Session score | `(10 + Σ width_factors) × unsolved_penalty` | Lower better |
| Refinement accuracy | Fraction of voluntary bets on GOOD intervals that succeeded | Higher better |
| Budget used | `slips_used / total_budget` | Lower better |

**Default budget**: `floor(18/13 × N)` slips for N problems (e.g. 18 slips for 13 problems — exact Estimathon ratio).  
**Trivial baseline**: submitting `[1, 100]` for all age-estimation tasks scores ~10,010 (w=100, all GOOD). A biologically-informed model must beat this by narrowing intervals using domain knowledge.  
**Random baseline**: a random-interval agent achieves ~50% refinement accuracy — any well-calibrated model should beat both.

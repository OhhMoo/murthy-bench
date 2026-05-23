# Iterative Refinement Benchmark for Longevity-LLM

## Core Idea

Standard LongeBench evaluation is one-shot: the model answers, you score it, done. This benchmark tests something different — **how well L-LLM updates its beliefs when given partial feedback**. Instead of asking for a final answer upfront, we give the model iterative yes/no signals about which parts of its answer are correct, then measure how quickly and efficiently it converges to the right answer.

This captures a failure mode that final-answer-only metrics miss entirely: a model that lands on the right answer for the wrong reasons will fall apart the moment it's told "part of that is wrong."

---

## Two Mechanisms, Combined

### 1. Interval-Based Answers (Estimathon mechanic)

For numerical tasks (age prediction, lifespan, fold change, drug extension %), the model does not submit a single number. It submits an **interval [min, max]**.

An interval is **good** if it contains the correct answer. But a wide interval is penalized. The score for a single task across K rounds is:

```
score = (10 + Σ floor(max/min) for good intervals) × 2^(rounds_missed)
```

- Lower score is better
- A correct but wide interval [1, 200] scores worse than a tight correct interval [58, 72]
- A wrong interval doubles the penalty
- This forces the model to be both **calibrated** (contains the truth) and **confident** (narrow interval)

The model can resubmit a refined interval each round after receiving feedback. This directly tests whether the model can reason about its own uncertainty and update it.

### 2. Partial Feedback (Yes/No Budget)

After each submission, the model receives structured feedback:

| Task type | Feedback signal |
|---|---|
| Numerical | "Your interval does not contain the answer" / "Correct, but your interval spans factor X" |
| Binary / MCQ | "Correct" / "Wrong" |
| Set generation | "N of your K genes are correct" (no hint which ones) |
| Ternary | "Correct" / "Wrong direction" / "No effect is not the answer" |

**Budget constraint**: Each model run gets a fixed total number of feedback tokens (e.g., 18 signals across 13 tasks — directly borrowed from Estimathon's 18 slips / 13 problems ratio). The model must decide how to allocate: spend more rounds refining a hard task, or lock in an answer early and move on.

---

## What We Measure

For each task instance:

| Metric | Description |
|---|---|
| `rounds_to_correct` | How many feedback rounds before the model's submission is correct |
| `final_interval_score` | `floor(max/min)` of the last submitted interval (lower = more confident) |
| `convergence_trajectory` | Sequence of interval widths across rounds — does it narrow monotonically? |
| `budget_efficiency` | Did the model waste submissions on a task it had already solved? |
| `thinking_trace` | Full `<think>` block at each round — recorded for reasoning analysis |

Aggregate across all tasks:

```
total_score = (10 + Σ floor(max/min) for all good final intervals) × 2^(# tasks with no correct submission)
```

Lower total score = better. A model that hedges with wide intervals will lose to a model that is genuinely calibrated, even if both get the same number of tasks "correct."

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

**Task**: Predict the chronological age of a subject from 353 CpG methylation beta values (regression task, ground truth = 67 years).

**Round 1**
- Model submits: `[40, 90]` — interval width = floor(90/40) = 2
- Feedback: "Your interval contains the answer. Width factor: 2."

**Round 2**
- Model's thinking trace: *"The interval is valid but wide. The methylation pattern at cg07364285 suggests an older subject, probably above 60. I'll narrow to [60, 80]."*
- Model submits: `[60, 80]` — width = floor(80/60) = 1
- Feedback: "Your interval contains the answer. Width factor: 1."

**Round 3**
- Model's thinking trace: *"Width factor 1 means floor(max/min) = 1, so max < 2×min. I'm already tight. I'll lock this in and use remaining budget elsewhere."*
- Model does not resubmit. Final score contribution: 1 (good interval, width factor 1).

**Contrast with a poorly calibrated model**: submits `[1, 120]` every round, always "correct" but width = floor(120/1) = 120. Scores 120 per task vs. 1.

---

## Thinking Trace Analysis (Extra Credit)

Because we collect a thinking trace at every round, we have a multi-turn record of how the model's internal reasoning evolves. This is the substrate for the reasoning scorer described in the extra credit section of Track 1.

Specific signals to extract:
- **Gene/CpG mentions**: Do the biological entities the model cites actually exist and have the properties claimed?
- **Monotonic narrowing**: Does the model's interval consistently shrink, or does it widen (sign of confused reasoning)?
- **Self-consistency**: Does the model's stated reasoning in round N+1 follow logically from the feedback received in round N?
- **Hallucination under pressure**: Does the model start fabricating biological justifications when its first guess is wrong?

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
| Per-task interval score | `floor(max/min)` of final good interval | Lower better |
| Miss penalty | `× 2` per task with no correct submission | Lower better |
| Total | `(10 + Σ width_factors) × 2^misses` | Lower better |
| Budget efficiency | Rounds used / rounds available | Lower better |
| Convergence rate | Rounds to first correct interval | Lower better |

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .client import ModelClient

_MAX_CONCURRENCY = 8
_PROBLEM_CONTENT_LIMIT = 1500  # chars per problem shown in session prompt


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_interval(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"\[?\s*([0-9.e+\-]+)\s*,\s*([0-9.e+\-]+)\s*\]?", text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None, None


# ---------------------------------------------------------------------------
# Estimathon session (shared budget, last-submission-counts, binary feedback)
# ---------------------------------------------------------------------------

@dataclass
class EstimathonSession:
    tasks: list
    total_budget: int
    slips_used: int = 0
    last_submissions: dict = field(default_factory=dict)   # pid → (min, max)
    golds: dict = field(default_factory=dict)              # pid → float
    slip_log: list = field(default_factory=list)

    @property
    def n_problems(self) -> int:
        return len(self.tasks)

    @property
    def slips_remaining(self) -> int:
        return self.total_budget - self.slips_used

    def current_score(self) -> int:
        n_good = 0
        width_sum = 0
        for pid, (pmin, pmax) in self.last_submissions.items():
            gold = self.golds.get(pid)
            if gold is not None and pmin <= gold <= pmax and pmin > 0:
                n_good += 1
                width_sum += math.floor(pmax / pmin)
        return (10 + width_sum) * (2 ** (self.n_problems - n_good))

    def submit(self, pid: str, pmin: float, pmax: float) -> dict:
        gold = self.golds.get(pid)
        good = gold is not None and pmin <= gold <= pmax
        width_factor = math.floor(pmax / pmin) if good and pmin > 0 else None

        prev = self.last_submissions.get(pid)
        prev_good = False
        if prev is not None:
            pg, ph = prev
            prev_good = gold is not None and pg <= gold <= ph

        score_before = self.current_score()
        self.last_submissions[pid] = (pmin, pmax)
        score_after = self.current_score()
        self.slips_used += 1

        record = {
            "slip": self.slips_used,
            "pid": pid,
            "pred_min": pmin,
            "pred_max": pmax,
            "gold": gold,
            "good": good,
            "width_factor": width_factor,
            "was_refinement": prev is not None,
            "prev_was_good": prev_good,
            "score_before": score_before,
            "score_after": score_after,
            "slips_remaining": self.slips_remaining,
        }
        self.slip_log.append(record)
        return record


def _extract_gold(task: dict) -> float | None:
    try:
        return float(task["messages"][-1]["content"].strip())
    except (ValueError, KeyError, IndexError):
        return None


def _problem_content(task: dict) -> str:
    """Return the user-facing content of a task, truncated for session prompt."""
    msgs = task.get("messages", [])
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    content = user_msgs[0]["content"] if user_msgs else ""
    if len(content) > _PROBLEM_CONTENT_LIMIT:
        content = content[:_PROBLEM_CONTENT_LIMIT] + "\n[...truncated — full data available in context above]"
    return content


def _build_system_prompt(n_problems: int, total_budget: int) -> str:
    return f"""You are playing Estimathon — a scientific estimation game.

Rules:
- You have {total_budget} slips total across {n_problems} problems.
- Each slip: choose a problem number and submit an interval [min, max].
- Intervals must be positive (no zero, no negatives).
- An interval is GOOD if it contains the correct answer.
- ONLY your LAST submission for each problem counts toward your final score.
- If you refine a GOOD interval and your new interval is BAD, you lose that problem.

Scoring formula (lower is better):
  (10 + sum of floor(max/min) for all GOOD final intervals) × 2^({n_problems} − number of GOOD final answers)

Strategy:
- Wide intervals are safe but expensive (high floor(max/min)).
- Narrow intervals score better but risk being wrong.
- Unsolved problems double your score — cover all {n_problems} problems first.
- You receive ONLY binary feedback: GOOD or BAD. No direction hints.

After each submission you will see your live score and standings.

Format your submissions exactly as:
PROBLEM <number>
INTERVAL [min, max]"""


def _build_standings(session: EstimathonSession) -> str:
    lines = []
    for i, task in enumerate(session.tasks):
        pid = task.get("lb_id", f"P{i+1}")
        gold = session.golds.get(pid)
        if pid in session.last_submissions:
            pmin, pmax = session.last_submissions[pid]
            is_good = gold is not None and pmin <= gold <= pmax
            wf = math.floor(pmax / pmin) if is_good and pmin > 0 else None
            status = f"GOOD  width={wf}" if is_good else "BAD"
            lines.append(f"  [{i+1:2d}] [{pmin}, {pmax}]  →  {status}")
        else:
            lines.append(f"  [{i+1:2d}] —  no submission yet")
    return "\n".join(lines)


def _build_initial_user_message(session: EstimathonSession) -> str:
    parts = ["Here are your problems:\n"]
    for i, task in enumerate(session.tasks):
        domain = task.get("domain", "")
        metric = task.get("metric", "")
        content = _problem_content(task)
        parts.append(f"--- Problem {i+1}  [{domain}  |  {metric}] ---\n{content}\n")

    parts.append(
        f"\nStarting score: {session.current_score()}  "
        f"(all {session.n_problems} problems unsolved)\n"
        f"Slips remaining: {session.total_budget}\n\n"
        "Submit your first interval:\n"
        "PROBLEM <number>\n"
        "INTERVAL [min, max]"
    )
    return "\n".join(parts)


def _build_feedback(result: dict, session: EstimathonSession) -> str:
    good_str = "GOOD" if result["good"] else "BAD"
    width_note = f"  Width factor: {result['width_factor']}." if result["good"] else ""

    warning = ""
    if result["was_refinement"] and result["prev_was_good"] and not result["good"]:
        warning = (
            "\n⚠  You had a GOOD interval for this problem. "
            "Your new BAD submission replaced it — this problem now counts as wrong."
        )

    score_line = f"Score: {result['score_before']} → {result['score_after']}"
    if result["score_after"] < result["score_before"]:
        score_line += "  ↓ improved"
    elif result["score_after"] > result["score_before"]:
        score_line += "  ↑ worsened"

    standings = _build_standings(session)

    if session.slips_remaining == 0:
        next_prompt = "No slips remaining. Session complete."
    else:
        next_prompt = (
            f"Slips remaining: {session.slips_remaining}\n\n"
            "Next submission:\n"
            "PROBLEM <number>\n"
            "INTERVAL [min, max]"
        )

    return (
        f"Problem [{result['pid']}]: {good_str}.{width_note}{warning}\n"
        f"{score_line}\n\n"
        f"Standings:\n{standings}\n\n"
        f"{next_prompt}"
    )


def _parse_estimathon_response(text: str) -> tuple[int | None, float | None, float | None]:
    """Parse 'PROBLEM N\\nINTERVAL [min, max]' from model response."""
    prob_m = re.search(r"PROBLEM\s+(\d+)", text, re.IGNORECASE)
    interval_m = re.search(r"INTERVAL\s*\[?\s*([0-9.e+\-]+)\s*,\s*([0-9.e+\-]+)\s*\]?", text, re.IGNORECASE)
    if prob_m and interval_m:
        try:
            return int(prob_m.group(1)), float(interval_m.group(1)), float(interval_m.group(2))
        except ValueError:
            pass
    return None, None, None


def run_estimathon(
    tasks: list[dict],
    client: ModelClient,
    total_budget: int | None = None,
    enable_thinking: bool = False,
    on_slip: Callable[[dict], None] | None = None,
) -> dict:
    """
    Run a full Estimathon session: shared budget, binary feedback, last-submission-counts.

    total_budget defaults to floor(1.38 × n_problems) matching the real 18-slip / 13-problem ratio.
    """
    n = len(tasks)
    if total_budget is None:
        total_budget = max(n + 1, math.floor(1.38 * n))

    session = EstimathonSession(tasks=tasks, total_budget=total_budget)

    for task in tasks:
        pid = task.get("lb_id", "")
        gold = _extract_gold(task)
        if gold is not None:
            session.golds[pid] = gold

    conversation = [
        {"role": "system", "content": _build_system_prompt(n, total_budget)},
        {"role": "user", "content": _build_initial_user_message(session)},
    ]

    parse_failures = 0

    while session.slips_remaining > 0:
        resp = client.chat(
            conversation,
            max_tokens=3000 if enable_thinking else 600,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )

        problem_num, pmin, pmax = _parse_estimathon_response(resp.answer)

        if problem_num is None or pmin is None:
            parse_failures += 1
            if parse_failures >= 3:
                break  # give up if model can't format
            nudge = (
                "Could not parse your submission. Reply with exactly:\n"
                "PROBLEM <number>\n"
                "INTERVAL [min, max]\n\n"
                f"Slips remaining: {session.slips_remaining}"
            )
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({"role": "user", "content": nudge})
            continue

        parse_failures = 0

        task_idx = problem_num - 1
        if task_idx < 0 or task_idx >= n:
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({"role": "user", "content": f"Problem {problem_num} does not exist. Choose 1–{n}."})
            continue

        pid = tasks[task_idx].get("lb_id", f"P{problem_num}")
        result = session.submit(pid, pmin, pmax)
        result["think"] = resp.think
        result["raw_response"] = resp.answer

        if on_slip:
            on_slip(result)

        feedback = _build_feedback(result, session)
        conversation.append({"role": "assistant", "content": resp.answer})
        conversation.append({"role": "user", "content": feedback})

    # Compute refinement accuracy (key inference signal)
    refinements = [r for r in session.slip_log if r["was_refinement"] and r["prev_was_good"]]
    ref_success = sum(1 for r in refinements if r["good"])
    ref_total = len(refinements)

    return {
        "mode": "estimathon",
        "n_problems": n,
        "total_budget": total_budget,
        "slips_used": session.slips_used,
        "final_score": session.current_score(),
        "n_good_final": sum(
            1 for pid, (pmin, pmax) in session.last_submissions.items()
            if session.golds.get(pid) is not None
            and pmin <= session.golds[pid] <= pmax
        ),
        "refinement_attempts": ref_total,
        "refinement_successes": ref_success,
        "refinement_accuracy": ref_success / ref_total if ref_total else None,
        "slip_log": session.slip_log,
    }


# ---------------------------------------------------------------------------
# One-shot eval
# ---------------------------------------------------------------------------

_ESTIMATHON_FORMATS = {"regression", "pairwise", "interval"}


def _score_task(pred: str, gold: str, fmt: str) -> tuple[bool, float | None]:
    """Format-aware scoring. Returns (correct, f1_or_None)."""
    if fmt == "generation":
        pred_tokens = {t.strip().lower() for t in re.split(r"[,;\s]+", pred) if t.strip()}
        gold_tokens = {t.strip().lower() for t in re.split(r"[,;\s]+", gold) if t.strip()}
        if not gold_tokens:
            return False, 0.0
        tp = len(pred_tokens & gold_tokens)
        p = tp / len(pred_tokens) if pred_tokens else 0.0
        r = tp / len(gold_tokens)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1 >= 0.5, f1
    else:
        exact = pred.strip().lower() == gold.strip().lower()
        return exact, None


def _run_one_shot(task: dict, client: ModelClient, enable_thinking: bool) -> dict:
    messages = task["messages"]
    gold = messages[-1]["content"].strip()
    fmt = task.get("format", "")
    resp = client.chat(
        messages[:-1],
        max_tokens=500,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    pred = resp.answer.strip()
    correct, f1 = _score_task(pred, gold, fmt)
    result = {
        "lb_id": task.get("lb_id", ""),
        "domain": task.get("domain", ""),
        "format": fmt,
        "metric": task.get("metric", ""),
        "mode": "one-shot",
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "think": resp.think,
        "tokens_used": resp.tokens_used,
    }
    if f1 is not None:
        result["f1"] = f1
    return result


def run_eval(
    tasks: Iterator[dict],
    client: ModelClient,
    concurrency: int = 4,
    enable_thinking: bool = False,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    concurrency = min(concurrency, _MAX_CONCURRENCY)
    task_list = list(tasks)

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_one_shot, t, client, enable_thinking): t for t in task_list}
        for fut in as_completed(futures):
            try:
                record = fut.result()
            except Exception as exc:
                task = futures[fut]
                record = {"lb_id": task.get("lb_id", ""), "mode": "one-shot", "error": str(exc)}
            results.append(record)
            if on_result:
                on_result(record)

    return results


# ---------------------------------------------------------------------------
# Mixed eval — two-track: Estimathon for numerical, one-shot for categorical
# ---------------------------------------------------------------------------

def run_mixed(
    tasks: list[dict],
    client: ModelClient,
    total_budget: int | None = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    on_slip: Callable[[dict], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> dict:
    """
    Split tasks by format:
      numerical (regression / pairwise / interval) → Estimathon with shared budget
      categorical (binary / multiclass / ternary / generation) → one-shot accuracy

    Returns a combined result dict with 'estimathon' and 'one_shot' sub-dicts.
    """
    numerical   = [t for t in tasks if t.get("format") in _ESTIMATHON_FORMATS]
    categorical = [t for t in tasks if t.get("format") not in _ESTIMATHON_FORMATS]

    estimathon_result: dict | None = None
    one_shot_result:   dict | None = None

    # --- Track 1: Estimathon ---
    if numerical:
        estimathon_result = run_estimathon(
            tasks=numerical,
            client=client,
            total_budget=total_budget,
            enable_thinking=enable_thinking,
            on_slip=on_slip,
        )

    # --- Track 2: one-shot categorical ---
    if categorical:
        records: list[dict] = []

        def _collect(r: dict) -> None:
            records.append(r)
            if on_result:
                on_result(r)

        run_eval(
            tasks=iter(categorical),
            client=client,
            concurrency=concurrency,
            enable_thinking=enable_thinking,
            on_result=_collect,
        )

        by_format: dict[str, dict] = {}
        for r in records:
            fmt = r.get("format", "unknown")
            bucket = by_format.setdefault(fmt, {"n": 0, "correct": 0})
            bucket["n"] += 1
            if r.get("correct"):
                bucket["correct"] += 1
        for bucket in by_format.values():
            bucket["accuracy"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0

        n_total   = len(records)
        n_correct = sum(1 for r in records if r.get("correct"))
        one_shot_result = {
            "mode":      "one-shot",
            "n_tasks":   n_total,
            "n_correct": n_correct,
            "accuracy":  n_correct / n_total if n_total else 0.0,
            "by_format": by_format,
            "records":   records,
        }

    return {
        "mode":          "mixed",
        "n_numerical":   len(numerical),
        "n_categorical": len(categorical),
        "estimathon":    estimathon_result,
        "one_shot":      one_shot_result,
    }

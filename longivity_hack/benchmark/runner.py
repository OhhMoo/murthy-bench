import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .client import ModelClient

# Python 3.11+ added a 4300-digit limit on int→str conversion (CVE guard).
# Estimathon scores are legitimately huge (10 × 2^N) — remove the cap.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def _fmt_score(n: int) -> str:
    """Compact display for Estimathon scores. Uses integer arithmetic to avoid
    float overflow on astronomically large values (e.g. 10 × 2^14000)."""
    if n < 1_000_000:
        return str(n)
    bits = n.bit_length() - 1            # floor(log2(n))
    exp = int(bits * 0.30103)            # floor(log10(n)) estimate
    p10 = 10 ** exp
    if p10 * 10 <= n:                    # adjust if off by one
        exp += 1
        p10 *= 10
    q = n // p10
    frac = (n % p10) * 100 // p10       # two decimal places
    return f"{q}.{frac:02d}e{exp}"

_MAX_CONCURRENCY = 8
_PROBLEM_CONTENT_LIMIT = 1500  # chars per problem shown in session prompt
# Estimathon sends ALL problems in one context window. Cap to avoid blowing
# token limits when running mixed mode against the full LongeBench dataset.
_MAX_ESTIMATHON_PROBLEMS = 100
# Maximum number of slips a model may spend on a single problem.
_MAX_SLIPS_PER_PROBLEM = 3


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
    per_problem_slips: dict = field(default_factory=dict)  # pid → attempt count
    tried_intervals: dict = field(default_factory=dict)    # pid → list of (min, max) that scored BAD

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
        if not good:
            self.tried_intervals.setdefault(pid, []).append((pmin, pmax))

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
- Each slip: submit ONE interval for ONE problem only. Do not submit multiple problems in a single message.
- Intervals must have positive values (min > 0, max > min).
- An interval is GOOD if it contains the correct numeric answer.
- ONLY your LAST submission for each problem counts toward your final score.
- If you refine a GOOD interval and your new interval is BAD, you lose that problem.
- You may attempt each problem AT MOST {_MAX_SLIPS_PER_PROBLEM} times. After that it is locked — choose a different problem.

Scoring formula (lower is better):
  (10 + sum of floor(max/min) for all GOOD final intervals) × 2^({n_problems} − number of GOOD final answers)

CRITICAL STRATEGY — read this carefully:

Outcome priority (best → worst):
  1. GOOD & narrow  → ideal: correct answer, small max/min ratio
  2. GOOD & wide    → safe:  correct answer, large ratio — still FAR better than any BAD
  3. BAD  & wide    → bad:   wrong — your ENTIRE score doubles
  4. BAD  & narrow  → worst: wrong AND you wasted precision — score still doubles

Every unsolved problem doubles your entire score.
A correct wide interval is ALWAYS better than a wrong narrow one.

Optimal play:
  Step 1 — Secure GOOD first: submit a WIDE interval to guarantee correctness.
           When uncertain, start very wide: e.g. [1, 1000] or [10, 100000].
  Step 2 — Narrow only when confident: use remaining slips to tighten the interval.
           Only submit a narrow interval if you are sure it contains the answer.
           IMPORTANT: a refinement must have a SMALLER max/min ratio than your current GOOD interval.
           Submitting a WIDER interval for a problem you already have GOOD always worsens your score.
  Step 3 — Cover all problems: spread slips across problems before refining any one.

You receive ONLY binary feedback: GOOD or BAD. No directional hints.
Never start narrow — a missed narrow guess costs a doubled score AND burns a slip.

OUTPUT FORMAT — THIS IS MANDATORY:
Your entire response must be ONLY these two lines. No explanation. No reasoning. No other text.

PROBLEM <number>
INTERVAL [min, max]

Example:
PROBLEM 3
INTERVAL [45, 78]"""


def _build_standings(session: EstimathonSession) -> str:
    lines = []
    for i, task in enumerate(session.tasks):
        pid = task.get("lb_id", f"P{i+1}")
        gold = session.golds.get(pid)
        if pid in session.last_submissions:
            pmin, pmax = session.last_submissions[pid]
            is_good = gold is not None and pmin <= gold <= pmax
            wf = math.floor(pmax / pmin) if is_good and pmin > 0 else None
            if is_good:
                status = f"GOOD  width={wf}  (score contribution: {wf})"
            else:
                status = "BAD"
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
        f"\nStarting score: {_fmt_score(session.current_score())}  "
        f"(all {session.n_problems} problems unsolved)\n"
        f"Slips remaining: {session.total_budget}\n\n"
        "REMINDER: Start each problem with a WIDE interval to secure GOOD first.\n"
        "A correct wide answer is always better than a wrong narrow one.\n\n"
        "Submit your first interval:\n"
        "PROBLEM <number>\n"
        "INTERVAL [min, max]"
    )
    return "\n".join(parts)


def _build_feedback(result: dict, session: EstimathonSession) -> str:
    good_str = "GOOD" if result["good"] else "BAD"
    if result["good"]:
        wf = result["width_factor"]
        width_note = f"  Width factor: {wf}."
        if wf and wf > 2:
            width_note += " You may narrow this interval to improve your score — only if confident."
    else:
        width_note = ""

    warning = ""
    if result["was_refinement"] and result["prev_was_good"] and not result["good"]:
        warning = (
            "\n⚠  You had a GOOD interval for this problem. "
            "Your new BAD submission replaced it — this problem now counts as wrong."
        )
    elif result["was_refinement"] and result["prev_was_good"] and result["good"] and result["score_after"] > result["score_before"]:
        warning = (
            "\n⚠  Your new interval is WIDER than your previous GOOD interval — this WORSENED your score. "
            "To improve, submit a NARROWER interval (smaller max/min ratio) than what you had before."
        )

    score_line = f"Score: {_fmt_score(result['score_before'])} → {_fmt_score(result['score_after'])}"
    if result["score_after"] < result["score_before"]:
        score_line += "  ↓ improved"
    elif result["score_after"] > result["score_before"]:
        score_line += "  ↑ worsened"

    standings = _build_standings(session)

    # Per-problem attempt status
    attempts_used = result.get("attempts_used", 1)
    attempts_left = result.get("attempts_left", 0)
    if attempts_left == 0:
        attempt_note = f"  Problem {result['pid']} is now LOCKED (all {_MAX_SLIPS_PER_PROBLEM} attempts used)."
    else:
        attempt_note = f"  Attempts left on Problem {result['pid']}: {attempts_left}/{_MAX_SLIPS_PER_PROBLEM}."

    if session.slips_remaining == 0:
        next_prompt = "No slips remaining. Session complete."
    else:
        unsolved = []
        can_narrow = []
        for i, t in enumerate(session.tasks):
            _pid = t.get("lb_id", f"P{i+1}")
            _gold = session.golds.get(_pid)
            if _pid not in session.last_submissions:
                unsolved.append(i + 1)
            else:
                _mn, _mx = session.last_submissions[_pid]
                _is_good = _gold is not None and _mn <= _gold <= _mx
                if not _is_good:
                    unsolved.append(i + 1)
                else:
                    _wf = math.floor(_mx / _mn) if _mn > 0 else None
                    if _wf and _wf > 1:
                        can_narrow.append((i + 1, _mn, _mx, _wf))

        guidance_lines = []
        if unsolved:
            guidance_lines.append(
                f"PRIORITY — unsolved problems (submit wide to secure GOOD): {unsolved}"
            )
        if can_narrow:
            guidance_lines.append(
                "REFINEMENT opportunities (submit a NARROWER interval to lower your score):"
            )
            for pnum, mn, mx, wf in can_narrow:
                guidance_lines.append(
                    f"  Problem {pnum}: current [{mn}, {mx}] width={wf} — "
                    f"submit smaller max/min ratio to reduce score contribution"
                )
        if not unsolved and not can_narrow:
            guidance_lines.append("All problems solved and fully refined.")

        guidance = "\n".join(guidance_lines)
        next_prompt = (
            f"Slips remaining: {session.slips_remaining}\n\n"
            f"{guidance}\n\n"
            "Next submission:\n"
            "PROBLEM <number>\n"
            "INTERVAL [min, max]"
        )

    pid = result["pid"]
    wrong_history = session.tried_intervals.get(pid, [])
    wrong_note = ""
    if wrong_history and not result["good"]:
        wrong_note = (
            f"Wrong intervals tried so far for Problem {pid}: "
            + ", ".join(f"[{a}, {b}]" for a, b in wrong_history)
            + " — none of these contained the answer.\n"
        )

    return (
        f"Problem [{pid}]: {good_str}.{width_note}{warning}\n"
        f"{wrong_note}"
        f"{attempt_note}\n"
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
    extractor: "ModelClient | None" = None,
    on_slip: Callable[[dict], None] | None = None,
) -> dict:
    """
    Run a full Estimathon session: shared budget, binary feedback, last-submission-counts.

    total_budget defaults to floor(18/13 × n_problems) matching the real Estimathon ratio.
    extractor: optional secondary ModelClient (e.g. Claude) used to parse verbose responses.
    """
    n = len(tasks)
    if total_budget is None:
        total_budget = max(n + 1, math.floor(18 / 13 * n))

    session = EstimathonSession(tasks=tasks, total_budget=total_budget)

    # Use position-based pids so duplicate lb_ids don't collapse into one entry.
    for i, task in enumerate(tasks):
        pid = f"P{i + 1}"
        gold = _extract_gold(task)
        if gold is not None:
            session.golds[pid] = gold

    conversation = [
        {"role": "system", "content": _build_system_prompt(n, total_budget)},
        {"role": "user", "content": _build_initial_user_message(session)},
    ]

    parse_failures = 0

    while session.slips_remaining > 0:
        try:
            resp = client.chat(
                conversation,
                max_tokens=3000 if enable_thinking else 120,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = "rate_limit" in msg or "429" in msg
            is_too_large = "too large" in msg or "request too large" in msg
            if is_rate_limit and is_too_large:
                # Request itself exceeds TPM limit — retrying won't help, need fewer tasks
                est_tokens = sum(len(m.get("content", "")) for m in conversation) // 4
                print(
                    f"\n[Estimathon] Request too large (~{est_tokens} tokens). "
                    f"Current session has {n} problems. "
                    f"Use --limit to reduce (e.g. --limit {max(1, n // 2)}) and retry."
                )
                break
            elif is_rate_limit:
                gave_up = False
                for attempt in range(1, 5):
                    wait = 15 * attempt
                    print(f"\n[Estimathon] Rate limit hit — waiting {wait}s (attempt {attempt}/4)...")
                    time.sleep(wait)
                    try:
                        resp = client.chat(
                            conversation,
                            max_tokens=3000 if enable_thinking else 120,
                            temperature=0.0,
                            enable_thinking=enable_thinking,
                        )
                        break
                    except Exception as retry_exc:
                        if attempt == 4:
                            print(f"\n[Estimathon] Rate limit persists after 4 retries. Returning partial results.")
                            gave_up = True
                        exc = retry_exc
                if gave_up:
                    break
            elif "context" in msg or "too long" in msg or "token" in msg:
                print(
                    f"\n[Estimathon] Context too long ({n} problems). "
                    f"Use --limit to cap tasks. Returning partial results."
                )
                break
            else:
                print(f"\n[Estimathon] API error: {exc}. Returning partial results.")
                break

        problem_num, pmin, pmax = _parse_estimathon_response(resp.answer)

        if problem_num is None or pmin is None:
            # Try Claude extractor before giving up / nudging
            if extractor is not None:
                problem_summaries = "\n".join(
                    f"Problem {i+1}: {t.get('lb_id','?')} ({t.get('domain','?')})"
                    for i, t in enumerate(tasks)
                )
                problem_num, pmin, pmax = _extract_interval_from_verbose(
                    resp.answer, problem_summaries, extractor
                )

        if problem_num is None or pmin is None:
            parse_failures += 1
            if parse_failures >= 3:
                break  # give up if model can't format
            nudge = (
                "FORMAT ERROR. Output ONLY these two lines, nothing else:\n"
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

        pid = f"P{problem_num}"

        # Reject degenerate intervals (min == max can never contain a value).
        if pmin == pmax:
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({
                "role": "user",
                "content": (
                    f"INVALID: [{pmin}, {pmax}] — min and max cannot be equal.\n\n"
                    f"Slips remaining: {session.slips_remaining}"
                ),
            })
            continue

        # Warn on identical repeat submissions but still process (burn the slip).
        repeat_note = ""
        if pid in session.last_submissions and session.last_submissions[pid] == (pmin, pmax):
            repeat_note = f"NOTE: [{pmin}, {pmax}] for Problem {problem_num} was already submitted and scored BAD.\n"

        # Enforce per-problem attempt cap — reject without burning a slip.
        attempts_so_far = session.per_problem_slips.get(pid, 0)
        if attempts_so_far >= _MAX_SLIPS_PER_PROBLEM:
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({
                "role": "user",
                "content": (
                    f"Problem {problem_num} is LOCKED — you have already used all "
                    f"{_MAX_SLIPS_PER_PROBLEM} attempts on it. "
                    f"Choose a different problem.\n\n"
                    f"Slips remaining: {session.slips_remaining}"
                ),
            })
            continue

        session.per_problem_slips[pid] = attempts_so_far + 1
        attempts_used = attempts_so_far + 1
        attempts_left = _MAX_SLIPS_PER_PROBLEM - attempts_used

        result = session.submit(pid, pmin, pmax)
        result["think"] = resp.think
        result["raw_response"] = resp.answer
        result["task_content"] = _problem_content(tasks[task_idx])
        result["lb_id"] = tasks[task_idx].get("lb_id", pid)
        result["attempts_used"] = attempts_used
        result["attempts_left"] = attempts_left

        if on_slip:
            on_slip(result)

        feedback = repeat_note + _build_feedback(result, session)
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

# "pairwise" tasks in LongeBench are categorical comparisons (answer = "A" or "B"),
# not numerical predictions. Only regression/interval have numeric gold values.
_ESTIMATHON_FORMATS = {"regression", "interval"}

_FORMAT_REMINDERS: dict[str, str] = {
    "binary":     "\n\nReply with ONLY a single letter: A or B. No explanation. No other text.",
    "pairwise":   "\n\nReply with ONLY a single letter: A or B. No explanation. No other text.",
    "multiclass": "\n\nReply with ONLY a single letter (A, B, C, D, or E). No explanation. No other text.",
    "generation": "\n\nReply with ONLY a semicolon-separated list of gene symbols. No explanation. No other text.",
    # ternary omitted — LongeBench ternary uses task-specific text labels (e.g. Same/Different/Unknown)
}

_FORMAT_VALID: dict[str, str] = {
    "binary":     "A or B",
    "ternary":    "A, B, or C",
    "pairwise":   "A or B",
    "multiclass": "A, B, C, D, or E",
    "generation": "semicolon-separated gene symbols",
}


def _inject_format_reminder(messages: list[dict], fmt: str) -> list[dict]:
    """Append a format reminder to the last user message if one exists for this format."""
    reminder = _FORMAT_REMINDERS.get(fmt)
    if not reminder:
        return messages
    out = [dict(m) for m in messages]
    for i in range(len(out) - 1, -1, -1):
        if out[i]["role"] == "user":
            out[i] = {**out[i], "content": out[i]["content"] + reminder}
            break
    return out


def _looks_like_answer(pred: str, fmt: str) -> bool:
    """Return True if pred already matches the expected short-form answer."""
    p = pred.strip().upper()
    if fmt in ("binary", "pairwise"):
        return p in ("A", "B")
    if fmt == "ternary":
        return p in ("A", "B", "C")
    if fmt == "multiclass":
        return p in ("A", "B", "C", "D", "E")
    return True  # generation / unknown: skip extraction


def _extract_answer(raw: str, fmt: str, extractor: "ModelClient") -> str:
    """Ask the extractor model to pull the answer letter from a verbose response."""
    valid = _FORMAT_VALID.get(fmt)
    if not valid:
        return raw
    prompt = (
        f"A language model gave this verbose response to a {fmt} classification question "
        f"(valid answers: {valid}):\n\n"
        f"<response>\n{raw[:3000]}\n</response>\n\n"
        f"Extract the answer. Reply with ONLY the answer ({valid}). Nothing else."
    )
    try:
        resp = extractor.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.0,
        )
        return resp.answer.strip()
    except Exception:
        return raw


def _extract_interval_from_verbose(
    raw: str, problem_summaries: str, extractor: "ModelClient"
) -> tuple[int | None, float | None, float | None]:
    """Ask the extractor to identify which problem and interval a verbose response addresses."""
    prompt = (
        "A model responded to an Estimathon problem with verbose text instead of the required format.\n\n"
        f"Available problems:\n{problem_summaries}\n\n"
        f"Model response:\n<response>\n{raw[:2000]}\n</response>\n\n"
        "Which problem number is it addressing and what numeric interval does the text imply?\n"
        "Reply with ONLY:\nPROBLEM <number>\nINTERVAL [min, max]"
    )
    try:
        resp = extractor.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.0,
        )
        return _parse_estimathon_response(resp.answer)
    except Exception:
        return None, None, None


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


def _run_one_shot(
    task: dict,
    client: ModelClient,
    enable_thinking: bool,
    extractor: "ModelClient | None" = None,
) -> dict:
    messages = task["messages"]
    gold = messages[-1]["content"].strip()
    fmt = task.get("format", "")
    call_messages = _inject_format_reminder(messages[:-1], fmt)
    try:
        resp = client.chat(
            call_messages,
            max_tokens=3000 if enable_thinking else 200,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )
    except Exception as exc:
        return {
            "lb_id": task.get("lb_id", ""),
            "domain": task.get("domain", ""),
            "format": fmt,
            "metric": task.get("metric", ""),
            "mode": "one-shot",
            "error": str(exc),
            "input_messages": call_messages,
            "raw_response": None,
            "pred": None,
            "gold": gold,
            "correct": None,
            "think": None,
            "tokens_used": 0,
        }
    raw_response = (resp.think or "") + resp.answer if resp.think else resp.answer
    pred = resp.answer.strip()
    if extractor is not None and not _looks_like_answer(pred, fmt):
        pred = _extract_answer(pred, fmt, extractor)
    correct, f1 = _score_task(pred, gold, fmt)
    result = {
        "lb_id": task.get("lb_id", ""),
        "domain": task.get("domain", ""),
        "format": fmt,
        "metric": task.get("metric", ""),
        "task": task.get("task", ""),
        "mode": "one-shot",
        "input_messages": call_messages,
        "raw_response": raw_response,
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
    extractor: "ModelClient | None" = None,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    concurrency = min(concurrency, _MAX_CONCURRENCY)
    task_list = list(tasks)

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_one_shot, t, client, enable_thinking, extractor): t
            for t in task_list
        }
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
    extractor: "ModelClient | None" = None,
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

    if len(numerical) > _MAX_ESTIMATHON_PROBLEMS:
        print(
            f"[run_mixed] {len(numerical)} numerical tasks found — "
            f"capping Estimathon at {_MAX_ESTIMATHON_PROBLEMS} "
            f"(all problems must fit in one context window). "
            f"Use --limit to control this."
        )
        numerical = numerical[:_MAX_ESTIMATHON_PROBLEMS]

    estimathon_result: dict | None = None
    one_shot_result:   dict | None = None

    # --- Track 1: Estimathon ---
    if numerical:
        estimathon_result = run_estimathon(
            tasks=numerical,
            client=client,
            total_budget=total_budget,
            enable_thinking=enable_thinking,
            extractor=extractor,
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
            extractor=extractor,
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

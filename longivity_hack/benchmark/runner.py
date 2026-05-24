import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import client as _client_mod
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


# Point-estimate fallback for the isolated runner only. Smaller endpoint
# models (e.g. L-LLM) sometimes return a bare number like "42.0" when they
# have high confidence — meaningful behaviour we shouldn't throw away.
# We treat it as the centre of an interval with ±30% margin: max/min stays
# below 2 so the width-factor in the scoring formula remains 1 (tight GOOD
# is the best possible outcome), but the band is wide enough to catch
# estimates that are off by up to ~30%.
_BARE_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_interval_with_point_fallback(
    text: str,
) -> tuple[float | None, float | None, str]:
    """Return (pmin, pmax, source).

    source ∈ {'interval', 'point', 'none'}:
      interval — parsed a real [min, max] pair
      point    — parsed a single number, expanded to ±30% margin
      none     — couldn't find any usable numbers
    """
    pmin, pmax = parse_interval(text)
    if pmin is not None and pmax is not None and pmin > 0 and pmax > pmin:
        return pmin, pmax, "interval"

    m = _BARE_NUMBER_RE.search(text)
    if m:
        try:
            n = float(m.group())
        except ValueError:
            return None, None, "none"
        if n > 0:
            margin = max(n * 0.30, 1.0)
            return n - margin, n + margin, "point"
    return None, None, "none"


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


def _format_rules_core(format_shape: str, format_lines_count: str, ok_example: str, wrong_extra: str = "") -> str:
    """Shared format-strict + anti-repetition system-prompt core used by both
    runners. Keeps the two prompts the same shape and length so models see
    the same rules whether running shared-budget multi-turn or isolated."""
    return (
        f"OUTPUT FORMAT (mandatory) — reply with EXACTLY {format_lines_count}, nothing else:\n"
        f"\n"
        f"{format_shape}\n"
        f"\n"
        f"min and max are positive numbers with min < max.\n"
        f"\n"
        f"Examples that are CORRECT:\n"
        f"{ok_example}\n"
        f"\n"
        f"Examples that are WRONG (never output these shapes):\n"
        f"  42.0                  ← single number, missing the range\n"
        f"  The answer is 42.     ← prose / explanation\n"
        f"  INTERVAL [50, 50]     ← min must be strictly less than max\n"
        f"  INTERVAL [-3, 10]     ← min must be positive\n"
        f"{wrong_extra}"
        f"\n"
        f"Hard rules:\n"
        f"- A wide CORRECT interval beats any wrong narrow one.\n"
        f"- NEVER repeat an interval marked FORBIDDEN. Repetition is the WORST failure mode — the runner will lock the problem after 2 repeats.\n"
        f"- If a previous attempt was wrong, the answer is OUTSIDE that range. Shift the center OR widen — do NOT keep the same numbers.\n"
        f"- Output nothing other than the required line(s). No reasoning, no markdown."
    )


def _build_system_prompt(n_problems: int, total_budget: int) -> str:
    # Multi-turn shared-budget shape. Same core rules as isolated; only
    # the format block and a one-line operational note differ.
    core = _format_rules_core(
        format_shape="PROBLEM <n>\nINTERVAL [min, max]",
        format_lines_count="two lines",
        ok_example=(
            "  PROBLEM 3\n"
            "  INTERVAL [30, 90]"
        ),
        wrong_extra="  PROBLEM 3 INTERVAL [1,2]   ← must be two separate lines\n",
    )
    return (
        f"{core}\n"
        f"\n"
        f"Game: {n_problems} problems, {total_budget} slips, max {_MAX_SLIPS_PER_PROBLEM}/problem. "
        f"ONLY your LAST submission per problem counts. Binary feedback (GOOD/BAD), no hints."
    )


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
        next_prompt = (
            f"Slips remaining: {session.slips_remaining}\n\n"
            "Next submission:\n"
            "PROBLEM <number>\n"
            "INTERVAL [min, max]"
        )

    pid = result["pid"]
    wrong_history = session.tried_intervals.get(pid, [])

    # Same FORBIDDEN block format as the isolated runner — REPEAT markers,
    # concrete numeric alternatives, and an extra ⚠ when the model has
    # already re-submitted the same interval for this problem.
    forbidden_note = ""
    if wrong_history and not result["good"]:
        seen: set = set()
        lines = []
        has_repeat = False
        for a, b in wrong_history:
            if (a, b) in seen:
                lines.append(f"  ✗ [{a}, {b}]   ← REPEAT — never submit this again")
                has_repeat = True
            else:
                lines.append(f"  ✗ [{a}, {b}]")
                seen.add((a, b))

        lo_all = min(a for a, _ in wrong_history)
        hi_all = max(b for _, b in wrong_history)
        span = max((hi_all - lo_all) * 5, 30.0)
        suggest_wider = f"INTERVAL [{max(1, lo_all - span):g}, {hi_all + span:g}]"
        suggest_below = f"INTERVAL [{max(1, lo_all * 0.2):g}, {max(1.5, lo_all * 0.7):g}]"
        suggest_above = f"INTERVAL [{hi_all * 1.3:g}, {hi_all * 3:g}]"

        repeat_warning = ""
        if has_repeat:
            repeat_warning = (
                "\n⚠  You have ALREADY submitted the same interval more than once "
                "for this problem. Repeating it AGAIN is forbidden — change the numbers."
            )

        forbidden_note = (
            f"\n━━━ FORBIDDEN INTERVALS for Problem {pid} ━━━\n"
            + "\n".join(lines)
            + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            + "Correct answer is OUTSIDE every range above. Your new INTERVAL for "
            + f"Problem {pid} MUST differ from every line above."
            + repeat_warning
            + "\nConcrete options for this problem (any DIFFERENT range works):\n"
            + f"  • {suggest_wider}   ← much wider\n"
            + f"  • {suggest_below}   ← shifted lower\n"
            + f"  • {suggest_above}   ← shifted higher\n"
        )

    return (
        f"Problem [{pid}]: {good_str}.{width_note}{warning}\n"
        f"{attempt_note}\n"
        f"{score_line}"
        f"{forbidden_note}\n\n"
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


# ---------------------------------------------------------------------------
# Isolated-context Estimathon — one problem per API call
# ---------------------------------------------------------------------------
#
# Designed for smaller models (e.g. L-LLM on a HF endpoint) that drown in the
# accumulating shared-budget conversation used by run_estimathon. Each slip is
# a FRESH single-turn API call containing only:
#   - compact rules
#   - the ONE problem being attempted (full text)
#   - prior wrong intervals on THIS problem (if any)
#   - slips remaining
# Round-robin scheduler picks the next unlocked problem.

# Tighter content cap for isolated mode — the model isn't using the bulk of
# the CpG / gene list anyway (its outputs are clearly a learned prior), so we
# keep just the demographic preamble + a handful of features.
_ISOLATED_PROBLEM_CONTENT_LIMIT = 600


def _isolated_problem_content(task: dict) -> str:
    msgs = task.get("messages", [])
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    content = user_msgs[0]["content"] if user_msgs else ""
    if len(content) > _ISOLATED_PROBLEM_CONTENT_LIMIT:
        content = content[:_ISOLATED_PROBLEM_CONTENT_LIMIT] + " […]"
    return content


def _build_isolated_system_prompt() -> str:
    # Single-turn shape. Same shared core as the legacy multi-turn prompt —
    # only the format block differs (one line instead of two).
    return _format_rules_core(
        format_shape="INTERVAL [min, max]",
        format_lines_count="ONE line",
        ok_example=(
            "  INTERVAL [30, 90]\n"
            "  INTERVAL [12.5, 18.0]"
        ),
    )


def _build_isolated_user_message(
    task: dict, pid: str, problem_num: int,
    wrong_intervals: list[tuple[float, float]], slips_remaining: int,
) -> str:
    content = _isolated_problem_content(task)

    if not wrong_intervals:
        return (
            f"{content}\n\n"
            "Reply with ONE line in this exact format:\n"
            "INTERVAL [min, max]"
        )

    # Build the FORBIDDEN list with REPEAT markers for any interval that
    # appears more than once — makes it visually obvious which exact
    # numbers the model has been re-emitting.
    seen: set = set()
    wrong_lines = []
    has_repeat = False
    for a, b in wrong_intervals:
        if (a, b) in seen:
            wrong_lines.append(f"  ✗ [{a}, {b}]   ← REPEAT — never submit this again")
            has_repeat = True
        else:
            wrong_lines.append(f"  ✗ [{a}, {b}]")
            seen.add((a, b))

    # Concrete alternative ranges, derived from the bounding box of all
    # forbidden intervals. Models that ignore prose still often copy
    # explicit example numbers, so each suggestion must be visibly
    # different from anything in the forbidden list.
    lo_all = min(a for a, _ in wrong_intervals)
    hi_all = max(b for _, b in wrong_intervals)
    # Aggressive widening: 5× the current span (or 30, whichever is bigger)
    # so the "wider" suggestion is unmistakably broader than the forbidden
    # band, not just nudged a few units outward.
    span = max((hi_all - lo_all) * 5, 30.0)
    suggest_wider = f"INTERVAL [{max(1, lo_all - span):g}, {hi_all + span:g}]"
    suggest_below = f"INTERVAL [{max(1, lo_all * 0.2):g}, {max(1.5, lo_all * 0.7):g}]"
    suggest_above = f"INTERVAL [{hi_all * 1.3:g}, {hi_all * 3:g}]"

    repeat_warning = ""
    if has_repeat:
        repeat_warning = (
            "\n⚠  You have ALREADY submitted the same interval more than once "
            "for this problem. Repeating it AGAIN is forbidden — change the numbers."
        )

    return (
        f"{content}\n\n"
        "━━━ FORBIDDEN INTERVALS for this problem ━━━\n"
        f"{chr(10).join(wrong_lines)}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "The correct answer is OUTSIDE every range above.\n"
        "Your new INTERVAL MUST differ from every line above — "
        "shift the center OR widen the range."
        f"{repeat_warning}\n\n"
        "Concrete options you may pick (any DIFFERENT range works):\n"
        f"  • {suggest_wider}   ← much wider\n"
        f"  • {suggest_below}   ← shifted lower\n"
        f"  • {suggest_above}   ← shifted higher\n\n"
        "Reply with ONE line in this exact format:\n"
        "INTERVAL [min, max]"
    )


def _build_estimathon_result(session: EstimathonSession, n: int, total_budget: int, mode_label: str) -> dict:
    refinements = [r for r in session.slip_log if r["was_refinement"] and r["prev_was_good"]]
    ref_success = sum(1 for r in refinements if r["good"])
    ref_total = len(refinements)
    return {
        "mode": mode_label,
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


def run_estimathon_isolated(
    tasks: list[dict],
    client: ModelClient,
    total_budget: int | None = None,
    enable_thinking: bool = False,
    on_slip: Callable[[dict], None] | None = None,
) -> dict:
    """Per-problem single-turn variant. Each slip = one fresh API call with
    only the target problem + its prior wrong intervals. Round-robin picks
    the next unlocked problem so the model never has to choose."""
    n = len(tasks)
    if total_budget is None:
        total_budget = max(n + 1, math.floor(18 / 13 * n))

    session = EstimathonSession(tasks=tasks, total_budget=total_budget)
    for i, task in enumerate(tasks):
        pid = f"P{i + 1}"
        gold = _extract_gold(task)
        if gold is not None:
            session.golds[pid] = gold

    cursor = 0
    parse_failures = 0

    while session.slips_remaining > 0:
        # Round-robin to the next problem that still has attempts left.
        start = cursor
        while True:
            pid = f"P{cursor + 1}"
            if session.per_problem_slips.get(pid, 0) < _MAX_SLIPS_PER_PROBLEM:
                break
            cursor = (cursor + 1) % n
            if cursor == start:
                # All problems locked — nothing more to do.
                return _build_estimathon_result(session, n, total_budget, "estimathon-iso")

        task_idx = cursor
        task = tasks[task_idx]

        messages = [
            {"role": "system", "content": _build_isolated_system_prompt()},
            {"role": "user", "content": _build_isolated_user_message(
                task, pid, cursor + 1,
                session.tried_intervals.get(pid, []),
                session.slips_remaining,
            )},
        ]

        # In cheat mode, surface the scheduling order before the request dump.
        _client_mod.cheat_header(
            f"Slip {session.slips_used + 1}/{total_budget}  →  {pid} "
            f"(attempt {session.per_problem_slips.get(pid, 0) + 1}/{_MAX_SLIPS_PER_PROBLEM})"
        )

        try:
            resp = client.chat(
                messages,
                max_tokens=3000 if enable_thinking else 120,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            print(f"\n[Estimathon-iso] API error: {exc}. Returning partial results.")
            break

        pmin, pmax, parse_src = parse_interval_with_point_fallback(resp.answer)
        if parse_src == "none":
            parse_failures += 1
            if parse_failures >= 3:
                print(f"\n[Estimathon-iso] 3 consecutive parse failures. Returning partial results.")
                break
            cursor = (cursor + 1) % n  # skip this problem this round
            continue

        # Hard server-side exact-repeat rejection. Burns a per-problem
        # attempt slot so the problem locks faster, but does NOT burn
        # a slip — the model gets a chance to actually change its answer.
        if session.last_submissions.get(pid) == (pmin, pmax):
            session.per_problem_slips[pid] = session.per_problem_slips.get(pid, 0) + 1
            parse_failures += 1
            if parse_failures >= 3:
                print(f"\n[Estimathon-iso] 3 consecutive exact-repeat submissions. Returning partial results.")
                break
            cursor = (cursor + 1) % n  # advance to a different problem
            continue
        parse_failures = 0

        session.per_problem_slips[pid] = session.per_problem_slips.get(pid, 0) + 1
        attempts_used = session.per_problem_slips[pid]
        attempts_left = _MAX_SLIPS_PER_PROBLEM - attempts_used

        result = session.submit(pid, pmin, pmax)
        result["think"] = resp.think
        result["raw_response"] = resp.answer
        result["task_content"] = _problem_content(task)
        result["lb_id"] = task.get("lb_id", pid)
        result["attempts_used"] = attempts_used
        result["attempts_left"] = attempts_left

        if on_slip:
            on_slip(result)

        cursor = (cursor + 1) % n

    return _build_estimathon_result(session, n, total_budget, "estimathon-iso")


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
            if "rate_limit" in msg or "429" in msg:
                print(
                    f"\n[Estimathon] Rate limit hit after {session.slips_used} slips. "
                    f"Use --limit to run fewer tasks (current: {n} problems). "
                    f"Returning partial results."
                )
            elif "context" in msg or "too long" in msg or "token" in msg:
                print(
                    f"\n[Estimathon] Context too long ({n} problems). "
                    f"Use --limit to cap tasks. Returning partial results."
                )
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

        # Reject degenerate intervals (min == max can never contain a value).
        # Count as a parse failure BEFORE the reset below — otherwise a
        # stateless model that keeps returning the same degenerate response
        # would have its counter zeroed every turn and loop forever.
        if pmin == pmax or pmin <= 0 or pmax < pmin:
            parse_failures += 1
            if parse_failures >= 3:
                print(
                    f"\n[Estimathon] 3 consecutive invalid intervals "
                    f"(last: [{pmin}, {pmax}]). Returning partial results."
                )
                break
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({
                "role": "user",
                "content": (
                    f"INVALID: [{pmin}, {pmax}] — need positive min < max.\n\n"
                    f"Slips remaining: {session.slips_remaining}"
                ),
            })
            continue

        task_idx = problem_num - 1
        if task_idx < 0 or task_idx >= n:
            parse_failures += 1
            if parse_failures >= 3:
                print(
                    f"\n[Estimathon] 3 consecutive references to nonexistent problem "
                    f"(last: {problem_num}, valid: 1–{n}). Returning partial results."
                )
                break
            conversation.append({"role": "assistant", "content": resp.answer})
            conversation.append({"role": "user", "content": f"Problem {problem_num} does not exist. Choose 1–{n}."})
            continue

        pid = f"P{problem_num}"

        # Hard server-side rejection of exact-repeat submissions: a stateless
        # model that doesn't read the FORBIDDEN block can no longer waste
        # slips by re-emitting the same numbers. First repeat = free retry
        # with a stronger nudge. Second repeat = lock the problem so the
        # model is forced to move on.
        already_tried = pid in session.last_submissions and session.last_submissions[pid] == (pmin, pmax)
        if already_tried:
            session.per_problem_slips[pid] = session.per_problem_slips.get(pid, 0) + 1
            attempts_after = session.per_problem_slips[pid]
            parse_failures += 1
            if parse_failures >= 3:
                print(
                    f"\n[Estimathon] 3 consecutive exact-repeat submissions. "
                    f"Returning partial results."
                )
                break
            if attempts_after >= _MAX_SLIPS_PER_PROBLEM:
                conversation.append({"role": "assistant", "content": resp.answer})
                conversation.append({
                    "role": "user",
                    "content": (
                        f"REJECTED: [{pmin}, {pmax}] for Problem {problem_num} is an EXACT REPEAT — "
                        f"Problem {problem_num} is now LOCKED. Choose a different problem.\n\n"
                        f"Slips remaining: {session.slips_remaining}"
                    ),
                })
            else:
                conversation.append({"role": "assistant", "content": resp.answer})
                conversation.append({
                    "role": "user",
                    "content": (
                        f"REJECTED: [{pmin}, {pmax}] for Problem {problem_num} is an EXACT REPEAT of a prior wrong submission. "
                        f"No slip burned. You MUST submit a DIFFERENT interval — shift the center or widen the range.\n\n"
                        f"Slips remaining: {session.slips_remaining}"
                    ),
                })
            continue

        # Enforce per-problem attempt cap — reject without burning a slip.
        # Bail if the model keeps picking the same locked problem (stateless
        # models that don't read the rejection message would loop forever).
        attempts_so_far = session.per_problem_slips.get(pid, 0)
        if attempts_so_far >= _MAX_SLIPS_PER_PROBLEM:
            parse_failures += 1
            if parse_failures >= 3:
                print(
                    f"\n[Estimathon] 3 consecutive picks of LOCKED problem {problem_num}. "
                    f"Returning partial results."
                )
                break
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

        # Reset the no-progress counter — we're about to make a real submission.
        parse_failures = 0
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
        # Legacy runner uses the strict parser only — no point-estimate
        # fallback. Tag every slip as "interval" for downstream consumers
        # that key off parse_source.
        result["parse_source"] = "interval"

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
    resp = client.chat(
        call_messages,
        max_tokens=3000 if enable_thinking else 200,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    pred = resp.answer.strip()
    if extractor is not None and not _looks_like_answer(pred, fmt):
        pred = _extract_answer(pred, fmt, extractor)
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
    isolated: bool = False,
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
        if isolated:
            estimathon_result = run_estimathon_isolated(
                tasks=numerical,
                client=client,
                total_budget=total_budget,
                enable_thinking=enable_thinking,
                on_slip=on_slip,
            )
        else:
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

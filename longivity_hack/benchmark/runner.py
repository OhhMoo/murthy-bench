import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterator

from .client import ModelClient

_MAX_CONCURRENCY = 8


def parse_interval(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"\[?\s*([0-9.e+\-]+)\s*,\s*([0-9.e+\-]+)\s*\]?", text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None, None


def generate_feedback(
    pred_min: float,
    pred_max: float,
    gold: float,
    attempt: int,
    budget: int,
) -> str:
    contains = pred_min <= gold <= pred_max
    remaining = budget - attempt - 1
    if contains:
        width = int(pred_max / pred_min) if pred_min > 0 else pred_max - pred_min
        return (
            f"Your interval contains the answer. Width factor: {width}. "
            f"{remaining} submission(s) remaining."
        )
    direction = "too high" if pred_min > gold else "too low"
    return (
        f"Your interval does not contain the answer — it is {direction}. "
        f"{remaining} submission(s) remaining."
    )


def _run_one_shot(task: dict, client: ModelClient, enable_thinking: bool) -> dict:
    messages = task["messages"]
    gold = messages[-1]["content"].strip()
    resp = client.chat(
        messages[:-1],
        max_tokens=500,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    pred = resp.answer.strip()
    correct = pred.lower() == gold.lower()
    return {
        "lb_id": task.get("lb_id", ""),
        "domain": task.get("domain", ""),
        "format": task.get("format", ""),
        "metric": task.get("metric", ""),
        "mode": "one-shot",
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "think": resp.think,
        "tokens_used": resp.tokens_used,
    }


def _run_iterative(task: dict, client: ModelClient, budget: int, enable_thinking: bool) -> dict:
    messages = task["messages"]
    gold_raw = messages[-1]["content"].strip()
    try:
        gold = float(gold_raw)
    except ValueError:
        # Fall back to one-shot if gold isn't numeric
        return _run_one_shot(task, client, enable_thinking)

    history = list(messages[:-1])
    rounds = []
    rounds_to_correct = None
    final_width_factor = None

    for attempt in range(budget):
        resp = client.chat(
            history,
            max_tokens=3000 if enable_thinking else 500,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )
        pred_min, pred_max = parse_interval(resp.answer)

        if pred_min is None or pred_max is None:
            feedback = f"Could not parse an interval from your response. Please reply with [min, max]. {budget - attempt - 1} submission(s) remaining."
            round_record = {
                "round": attempt + 1,
                "think": resp.think,
                "answer": resp.answer,
                "pred_min": None,
                "pred_max": None,
                "contains_gold": False,
                "width_factor": None,
                "feedback": feedback,
                "parse_error": True,
            }
        else:
            contains = pred_min <= gold <= pred_max
            width = int(pred_max / pred_min) if pred_min > 0 else int(pred_max - pred_min)
            feedback = generate_feedback(pred_min, pred_max, gold, attempt, budget)
            round_record = {
                "round": attempt + 1,
                "think": resp.think,
                "answer": resp.answer,
                "pred_min": pred_min,
                "pred_max": pred_max,
                "contains_gold": contains,
                "width_factor": width,
                "feedback": feedback,
                "parse_error": False,
            }
            if contains:
                if rounds_to_correct is None:
                    rounds_to_correct = attempt + 1
                final_width_factor = width
                if width <= 1:
                    rounds.append(round_record)
                    break

        rounds.append(round_record)
        history.append({"role": "assistant", "content": resp.answer})
        history.append({"role": "user", "content": feedback})

    # Estimathon score: (10 + sum of good width factors) * 2^misses
    good_widths = [r["width_factor"] for r in rounds if r.get("contains_gold") and r["width_factor"] is not None]
    misses = 0 if rounds_to_correct is not None else 1
    task_score = (10 + sum(good_widths)) * (2 ** misses) if good_widths or misses else 10 * (2 ** misses)

    return {
        "lb_id": task.get("lb_id", ""),
        "domain": task.get("domain", ""),
        "format": task.get("format", ""),
        "metric": "estimathon_score",
        "mode": "iterative",
        "gold": gold,
        "budget": budget,
        "rounds": rounds,
        "rounds_to_correct": rounds_to_correct,
        "final_width_factor": final_width_factor,
        "task_score": task_score,
        "budget_used": len(rounds),
    }


def run_eval(
    tasks: Iterator[dict],
    client: ModelClient,
    mode: str = "one-shot",
    budget: int = 5,
    concurrency: int = 4,
    enable_thinking: bool = False,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    concurrency = min(concurrency, _MAX_CONCURRENCY)
    task_list = list(tasks)

    def process(task):
        if mode == "iterative":
            return _run_iterative(task, client, budget, enable_thinking)
        return _run_one_shot(task, client, enable_thinking)

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process, t): t for t in task_list}
        for fut in as_completed(futures):
            try:
                record = fut.result()
            except Exception as exc:
                task = futures[fut]
                record = {
                    "lb_id": task.get("lb_id", ""),
                    "mode": mode,
                    "error": str(exc),
                }
            results.append(record)
            if on_result:
                on_result(record)

    return results

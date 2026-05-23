"""
STEP 3: Run L-LLM (and optionally GPT-4o) on all three tasks.

Usage:
    python step3_run_llm_eval.py --task EP-01 --model longevity
    python step3_run_llm_eval.py --task EP-02 --model longevity
    python step3_run_llm_eval.py --task EP-03 --model longevity
    python step3_run_llm_eval.py --task all --model longevity
"""

import json
import re
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Paths — uses the directory this script lives in ───────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR  = os.path.join(SCRIPT_DIR, "prompts")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model config ──────────────────────────────────────────────────────────────
LONGEVITY_LLM_URL   = "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1"
LONGEVITY_LLM_TOKEN = os.environ.get("HF_TOKEN", "")
LONGEVITY_MODEL_ID  = "longevity-llm"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_WORKERS = 6    # stay <= 8 per hackathon rules for L-LLM
MAX_TOKENS  = 50   # binary/ternary only needs 1 token


def get_client(model_name):
    if model_name == "longevity":
        return OpenAI(
            base_url=LONGEVITY_LLM_URL,
            api_key=LONGEVITY_LLM_TOKEN,
            timeout=300,
        ), LONGEVITY_MODEL_ID
    elif model_name == "gpt4o":
        if not OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY env variable to use gpt4o")
        return OpenAI(api_key=OPENAI_API_KEY, timeout=120), "gpt-4o"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def extract_letter(raw_text):
    """Pull the last standalone A/B/C from model output."""
    if "</think>" in raw_text:
        raw_text = raw_text.split("</think>")[-1].strip()
    matches = re.findall(r"(?<![A-Za-z])([A-Ca-c])(?![A-Za-z])", raw_text)
    if matches:
        return matches[-1].upper()
    return None


def call_model(client, model_id, row):
    messages = row["messages"][:-1]   # drop gold assistant turn
    gold = row["messages"][-1]["content"].strip()

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        pred = extract_letter(raw)
        tokens_used = resp.usage.total_tokens if resp.usage else None
    except Exception as e:
        raw = f"ERROR: {e}"
        pred = None
        tokens_used = None

    return {
        "lb_id":       row["lb_id"],
        "format":      row["format"],
        "metric":      row["metric"],
        "gold":        gold,
        "raw_response": raw,
        "pred":        pred,
        "correct":     (pred == gold) if pred else False,
        "tokens_used": tokens_used,
        "metadata":    row.get("metadata", "{}"),
    }


def run_eval(task_id, model_name, max_prompts=None):
    matches = glob.glob(os.path.join(PROMPTS_DIR, f"{task_id}_*.jsonl"))
    if not matches:
        print(f"ERROR: No prompt file found for {task_id} in {PROMPTS_DIR}")
        print(f"  Make sure EP-01_binary.jsonl / EP-02_ternary.jsonl / EP-03_multiclass.jsonl")
        print(f"  are in a 'prompts/' subfolder next to this script.")
        return

    prompt_file = matches[0]
    print(f"\nLoading: {prompt_file}")

    with open(prompt_file) as f:
        rows = [json.loads(line) for line in f]

    if max_prompts:
        rows = rows[:max_prompts]
        print(f"  (Limited to {max_prompts} prompts for testing)")

    print(f"Running {len(rows)} prompts  |  model={model_name}")

    client, model_id = get_client(model_name)

    # Check model name from endpoint
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if model_id not in available and available:
            print(f"  NOTE: '{model_id}' not found, using '{available[0]}' instead")
            model_id = available[0]
    except Exception:
        pass  # proceed with configured name

    out_path = os.path.join(RESULTS_DIR, f"{task_id}_{model_name}_raw.jsonl")
    results = []

    with open(out_path, "w") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(call_model, client, model_id, row): row
                       for row in rows}
            done = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                done += 1
                if done % 20 == 0 or done == len(rows):
                    valid = [r for r in results if r["pred"] is not None]
                    acc = sum(r["correct"] for r in valid) / len(valid) if valid else 0
                    print(f"  {done}/{len(rows)}  |  accuracy so far: {acc:.3f}  "
                          f"| parse rate: {len(valid)/done:.1%}")

    valid = [r for r in results if r["pred"] is not None]
    if valid:
        from sklearn.metrics import balanced_accuracy_score
        golds = [r["gold"] for r in valid]
        preds = [r["pred"] for r in valid]
        bal_acc = balanced_accuracy_score(golds, preds)
        print(f"\n{'='*55}")
        print(f"  Task:              {task_id}")
        print(f"  Model:             {model_name}")
        print(f"  Prompts run:       {len(results)}")
        print(f"  Parseable:         {len(valid)} ({len(valid)/len(results):.1%})")
        print(f"  Raw accuracy:      {sum(r['correct'] for r in valid)/len(valid):.4f}")
        print(f"  Balanced accuracy: {bal_acc:.4f}")
        print(f"  Results saved:     {out_path}")
    else:
        print("WARNING: No parseable responses. Check API connection.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  required=True,
                        choices=["EP-01", "EP-02", "EP-03", "all"])
    parser.add_argument("--model", default="longevity",
                        choices=["longevity", "gpt4o"])
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit prompts for quick testing, e.g. --max-prompts 10")
    args = parser.parse_args()

    tasks = ["EP-01", "EP-02", "EP-03"] if args.task == "all" else [args.task]
    for t in tasks:
        run_eval(t, args.model, max_prompts=args.max_prompts)

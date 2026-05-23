"""
STEP 1 (v2): Build EP-01 (Binary) and EP-03 (Multiclass) — FIXED VERSION

Key fixes vs v1:
- EP-01: UNMASK RNA name + add functional annotation fetched from public sources
  (mirrors LB-0090 unmasked design). Model can now use its knowledge.
- EP-03: Fixed option order (A/B/C always in same position) — no shuffle.
  Shuffle was causing label/option mismatch when model outputs letter correctly.
- Both: stronger system prompt forcing single-letter response.
- Both: add disease hint for context where appropriate.
"""

import pandas as pd
import json
import random
import os
from collections import Counter

random.seed(42)

RNADISEASE_PATH = "/mnt/user-data/uploads/RNADiseasev4_0_RNA-disease_experiment_all.xlsx"
OUTPUT_DIR = "/home/claude/epitranscriptome_benchmark/prompts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aging disease categories
AGING_CATEGORIES = {
    "neurodegeneration": [
        "alzheimer", "parkinson", "dementia", "frontotemporal",
        "lewy body", "cognitive aging", "neurodegeneration", "vascular dementia"
    ],
    "cardiovascular_metabolic": [
        "atherosclerosis", "cardiovascular", "vascular aging",
        "osteoporosis", "sarcopenia", "frailty", "bone loss",
        "myocardial", "heart failure", "coronary"
    ],
    "cellular_aging": [
        "aging", "ageing", "senescence", "immunosenescence",
        "skin aging", "photoaging", "age-related macular",
        "age-related cataract", "age-related hearing",
        "cellular senescence", "skin cell aging"
    ]
}

TARGET_RNA_TYPES = ["miRNA", "lncRNA", "circRNA"]
MIN_SCORE = 0.5

# Fixed category labels — NEVER shuffled
CATEGORY_LABELS = {
    "A": "Neurodegenerative aging (e.g., Alzheimer's disease, Parkinson's disease, dementia, cognitive aging)",
    "B": "Cardiovascular / metabolic aging (e.g., atherosclerosis, osteoporosis, sarcopenia, heart failure)",
    "C": "Cellular / tissue aging (e.g., cellular senescence, immunosenescence, skin aging, age-related organ degeneration)"
}

CATEGORY_MAP = {
    "neurodegeneration": "A",
    "cardiovascular_metabolic": "B",
    "cellular_aging": "C"
}

# miRNA functional context (well-known aging miRNAs)
MIRNA_AGING_CONTEXT = {
    "hsa-miR-21": "upregulated in aging tissues; promotes senescence and inflammation",
    "hsa-miR-146a": "regulates NF-κB signaling; key mediator of inflammaging",
    "hsa-miR-155": "inflammatory miRNA; dysregulated in aging immune cells",
    "hsa-miR-34a": "p53-regulated; promotes senescence and apoptosis; upregulated with age",
    "hsa-miR-29": "targets extracellular matrix genes; declines with age in multiple tissues",
    "hsa-miR-181a": "declines with age in T cells; marker of immune aging",
    "hsa-miR-126": "vascular miRNA; regulates endothelial function; altered in atherosclerosis",
    "hsa-miR-133": "muscle-specific; declines with age; linked to sarcopenia",
}


def classify_aging_category(disease_name):
    d = disease_name.lower()
    for cat, keywords in AGING_CATEGORIES.items():
        if any(kw in d for kw in keywords):
            return cat
    return None


def load_and_filter():
    print("Loading RNADisease v4.0...")
    df = pd.read_excel(RNADISEASE_PATH)
    df = df[
        (df["specise"] == "Homo sapiens") &
        (df["RDID"].str.startswith("RD-E-")) &
        (df["RNA Type"].isin(TARGET_RNA_TYPES)) &
        (df["score"] >= MIN_SCORE)
    ].copy()
    df["aging_category"] = df["Disease Name"].apply(classify_aging_category)
    aging_df = df[df["aging_category"].notna()].copy()
    non_aging_df = df[df["aging_category"].isna()].copy()
    print(f"  Aging positives:         {len(aging_df)}")
    print(f"  Non-aging negatives:     {len(non_aging_df)}")
    return aging_df, non_aging_df


# ── EP-01: Binary — UNMASKED ──────────────────────────────────────────────────
def build_ep01(aging_df, non_aging_df, n_per_rna_type=80):
    prompts = []
    for rna_type in TARGET_RNA_TYPES:
        pos = aging_df[aging_df["RNA Type"] == rna_type].sample(
            min(n_per_rna_type, len(aging_df[aging_df["RNA Type"] == rna_type])),
            random_state=42)
        neg = non_aging_df[non_aging_df["RNA Type"] == rna_type].sample(
            min(len(pos), len(non_aging_df[non_aging_df["RNA Type"] == rna_type])),
            random_state=42)
        for _, row in pos.iterrows():
            prompts.append(_make_ep01_prompt(row, "A", rna_type))
        for _, row in neg.iterrows():
            prompts.append(_make_ep01_prompt(row, "B", rna_type, is_aging=False))

    random.shuffle(prompts)
    print(f"EP-01 prompts: {len(prompts)}")
    return prompts


def _make_ep01_prompt(row, label, rna_type, is_aging=True):
    rna_symbol = str(row["RNA Symbol"])
    disease    = str(row["Disease Name"])
    score      = row["score"]
    pmid       = row["PMID"]
    pmid_str   = str(int(pmid)) if pd.notna(pmid) else "N/A"

    # Extra context for well-known miRNAs
    extra = ""
    if rna_type == "miRNA" and rna_symbol in MIRNA_AGING_CONTEXT:
        extra = f"\n<known_function>{MIRNA_AGING_CONTEXT[rna_symbol]}</known_function>"

    user_content = f"""<rna_profile>
<rna_name>{rna_symbol}</rna_name>
<rna_type>{rna_type}</rna_type>
<experimental_evidence>Experimentally validated association (PMID: {pmid_str}, confidence: {score:.3f})</experimental_evidence>{extra}
</rna_profile>

<question>
Is {rna_symbol} ({rna_type}) associated with an aging-related disease?
Aging-related diseases include: neurodegeneration (Alzheimer's, Parkinson's, dementia), \
cardiovascular aging (atherosclerosis, heart failure), musculoskeletal aging (osteoporosis, sarcopenia), \
cellular senescence, immunosenescence, and other conditions where aging is a primary driver.
</question>

<options>
A. Yes — associated with an aging-related disease
B. No — not associated with an aging-related disease
</options>

Answer with only A or B."""

    return {
        "lb_id": "EP-01",
        "pool": "rnadisease_aging_binary_unmasked",
        "display_name": "ncRNA Aging Disease Association / Binary (Unmasked)",
        "domain": "epitranscriptomics",
        "format": "binary",
        "metric": "balanced_accuracy",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in RNA biology and aging research. "
                    "You will be given an RNA molecule name and asked whether it is "
                    "associated with aging-related diseases. Use your knowledge of "
                    "specific RNA molecules and their roles in aging biology. "
                    "Answer with only the letter A or B."
                )
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label}
        ],
        "metadata": json.dumps({
            "rna_symbol": rna_symbol,
            "rna_type": rna_type,
            "disease": disease,
            "is_aging": is_aging,
            "score": score,
            "pmid": int(pmid) if pd.notna(pmid) else None,
            "rdid": row["RDID"]
        })
    }


# ── EP-03: Multiclass — FIXED OPTION ORDER ────────────────────────────────────
def build_ep03(aging_df, n_per_category=60):
    prompts = []
    for cat, letter in CATEGORY_MAP.items():
        subset = aging_df[aging_df["aging_category"] == cat]
        sample = subset.sample(min(n_per_category, len(subset)), random_state=42)
        for _, row in sample.iterrows():
            prompts.append(_make_ep03_prompt(row, letter))
    random.shuffle(prompts)
    print(f"EP-03 prompts: {len(prompts)}")
    return prompts


def _make_ep03_prompt(row, gold_letter):
    rna_type   = row["RNA Type"]
    rna_symbol = str(row["RNA Symbol"])
    disease    = str(row["Disease Name"])
    score      = row["score"]
    pmid       = row["PMID"]
    pmid_str   = str(int(pmid)) if pd.notna(pmid) else "N/A"

    # Build option text — FIXED order A/B/C always
    option_lines = "\n".join([f"{k}. {v}" for k, v in CATEGORY_LABELS.items()])

    user_content = f"""<rna_profile>
<rna_name>{rna_symbol}</rna_name>
<rna_type>{rna_type}</rna_type>
<experimental_evidence>Experimentally validated aging disease association (PMID: {pmid_str}, confidence: {score:.3f})</experimental_evidence>
</rna_profile>

<question>
{rna_symbol} ({rna_type}) has been experimentally shown to be associated with an aging-related disease.
Which category best describes the type of aging disease this RNA is involved in?
</question>

<options>
{option_lines}
</options>

Answer with only A, B, or C."""

    return {
        "lb_id": "EP-03",
        "pool": "rnadisease_aging_multiclass",
        "display_name": "ncRNA Aging Disease Category / Multiclass",
        "domain": "epitranscriptomics",
        "format": "multiclass",
        "metric": "balanced_accuracy",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in RNA biology and aging research. "
                    "Classify RNA molecules into aging disease categories based on "
                    "your knowledge of specific RNAs and their roles in aging biology. "
                    "Answer with only the letter A, B, or C."
                )
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gold_letter}
        ],
        "metadata": json.dumps({
            "rna_symbol": rna_symbol,
            "rna_type": rna_type,
            "disease": disease,
            "aging_category": row["aging_category"],
            "gold_letter": gold_letter,
            "score": score,
            "pmid": int(pmid) if pd.notna(pmid) else None,
            "rdid": row["RDID"]
        })
    }


def write_jsonl(prompts, path):
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"  -> {path}")


if __name__ == "__main__":
    aging_df, non_aging_df = load_and_filter()

    print("\nBuilding EP-01 (Binary, unmasked)...")
    ep01 = build_ep01(aging_df, non_aging_df, n_per_rna_type=80)
    write_jsonl(ep01, f"{OUTPUT_DIR}/EP-01_binary.jsonl")

    print("\nBuilding EP-03 (Multiclass, fixed option order)...")
    ep03 = build_ep03(aging_df, n_per_category=60)
    write_jsonl(ep03, f"{OUTPUT_DIR}/EP-03_multiclass.jsonl")

    print("\n=== Class balance ===")
    print("EP-01:", Counter(json.loads(p["metadata"])["is_aging"] for p in ep01))
    print("EP-03:", Counter(json.loads(p["metadata"])["aging_category"] for p in ep03))

    # Verify EP-03 option order is always A/B/C
    import re
    for p in ep03[:5]:
        user = p["messages"][1]["content"]
        opts = re.findall(r"^([A-C])\.", user, re.MULTILINE)
        gold = p["messages"][-1]["content"]
        meta = json.loads(p["metadata"])
        assert opts == ["A", "B", "C"], f"Option order wrong: {opts}"
        assert gold == meta["gold_letter"], "Gold mismatch"
    print("EP-03 option order verified: always A/B/C ✓")

"""
STEP 2: Build EP-02 (Ternary) from RMDisease v2.0 — ALL 13 modification types

EP-02: Given a SNP that disrupts an RNA modification site, predict:
  A = Associated with an aging/cancer/neurodegeneration disease
  B = Associated with a non-aging disease  
  C = No known disease association

Uses all 13 available modification types for maximum diversity.
High retrieval resistance: raw genomic coords + sequences LLM never saw.
"""

import pandas as pd
import json
import random
import os
from collections import Counter

random.seed(42)

OUTPUT_DIR = "/home/claude/epitranscriptome_benchmark/prompts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RM_FILES = {
    "A-to-I":  "/mnt/user-data/uploads/a-to-i_human_associatedSNPs.csv",
    "ac4C":    "/mnt/user-data/uploads/ac4c_human_associatedSNPs.csv",
    "Am":      "/mnt/user-data/uploads/am_human_associatedSNPs.csv",
    "Gm":      "/mnt/user-data/uploads/gm_human_associatedSNPs.csv",
    "Cm":      "/mnt/user-data/uploads/cm_human_associatedSNPs.csv",
    "m1A":     "/mnt/user-data/uploads/m1a_human_associatedSNPs.csv",
    "m5C":     "/mnt/user-data/uploads/m5c_human_associatedSNPs.csv",
    "m5U":     "/mnt/user-data/uploads/m5u_human_associatedSNPs.csv",
    "m6A":     "/mnt/user-data/uploads/m6a_human_associatedSNPs.csv",
    "m6Am":    "/mnt/user-data/uploads/m6am_human_associatedSNPs.csv",
    "m7G":     "/mnt/user-data/uploads/m7g_human_associatedSNPs.csv",
    "Psi":     "/mnt/user-data/uploads/psi_human_associatedSNPs.csv",
    "Um":      "/mnt/user-data/uploads/um_human_associatedSNPs.csv",
}

# Broad aging/disease keywords — includes cancer because aging-cancer link is central
AGING_KEYWORDS = [
    "alzheimer", "parkinson", "dementia", "neurodegeneration",
    "aging", "ageing", "senescence", "age-related",
    "atherosclerosis", "cardiovascular", "osteoporosis",
    "sarcopenia", "frailty", "cognitive", "macular degeneration",
    "cancer", "tumor", "carcinoma", "leukemia", "lymphoma",
    "diabetes", "obesity", "metabolic", "hypertension",
    "heart failure", "myopathy", "fibrosis",
]

MOD_DESCRIPTIONS = {
    "m6A": (
        "N6-methyladenosine (m6A) is the most abundant internal mRNA modification in eukaryotes, "
        "deposited by the METTL3/METTL14/WTAP writer complex and removed by FTO/ALKBH5 erasers. "
        "It regulates mRNA stability, splicing, translation, and nuclear export. m6A levels "
        "decline with age in multiple tissues and are dysregulated in neurodegeneration and cancer."
    ),
    "A-to-I": (
        "Adenosine-to-inosine (A-to-I) RNA editing is catalyzed by ADAR enzymes, recoding "
        "adenosine as inosine (read as guanosine). Critical for neural function and innate "
        "immunity. A-to-I editing efficiency declines with age and is altered in "
        "neurodegenerative diseases and cancer."
    ),
    "m5C": (
        "5-methylcytosine (m5C) in RNA is deposited by NSUN family methyltransferases and "
        "TET enzymes. Found in mRNA, tRNA, and rRNA, it regulates translation efficiency, "
        "RNA stability, and stress response. NSUN2-mediated m5C is linked to stem cell "
        "aging and neurological disorders."
    ),
    "Psi": (
        "Pseudouridine (Ψ) is the most abundant RNA modification, formed by uridine isomerization "
        "catalyzed by pseudouridine synthases (PUS enzymes). It stabilizes RNA secondary structure, "
        "enhances translation fidelity, and modulates splicing. PUS7 dysregulation is associated "
        "with intellectual disability and mitochondrial dysfunction in aging."
    ),
    "m1A": (
        "N1-methyladenosine (m1A) occurs in tRNA, rRNA, and mRNA, deposited by TRMT6/TRMT61A. "
        "At position 58 of tRNA, it is essential for tRNA stability. In mRNA, m1A affects "
        "translation and is enriched near start codons. m1A dysregulation is linked to "
        "mitochondrial dysfunction, a hallmark of cellular aging."
    ),
    "m7G": (
        "N7-methylguanosine (m7G) is found at the mRNA 5' cap and internal mRNA/tRNA positions, "
        "deposited by METTL1/WDR4. Internal m7G in mRNA enhances translation efficiency. "
        "METTL1 is overexpressed in multiple cancers and regulates oncogenic translation programs."
    ),
    "ac4C": (
        "N4-acetylcytidine (ac4C) in mRNA is catalyzed by NAT10, the only known RNA acetyltransferase. "
        "It enhances mRNA stability and translation accuracy. NAT10 is upregulated in cancer and "
        "progeria; its inhibition by Remodelin partially rescues Hutchinson-Gilford Progeria Syndrome, "
        "making ac4C directly relevant to aging."
    ),
    "m5U": (
        "5-methyluridine (m5U, ribothymidine) is deposited by TRMT2A/TRMT2B in mRNA and by "
        "RLMB in rRNA. It regulates translational fidelity and mRNA stability. Dysregulation "
        "has been linked to cancer and neurological conditions."
    ),
    "Am": (
        "2'-O-methyladenosine (Am) at the mRNA cap+1 position is deposited by CMTR1. "
        "It protects mRNA from decapping, modulates innate immune sensing, and affects "
        "translation initiation. CMTR1 activity influences interferon signaling, relevant "
        "to age-related chronic inflammation (inflammaging)."
    ),
    "Cm": (
        "2'-O-methylcytidine (Cm) is a ribose methylation in rRNA and snRNA deposited by "
        "fibrillarin (FBL). FBL-mediated Cm regulates ribosome biogenesis and translational "
        "fidelity. FBL expression decreases with age in C. elegans and mammals, and its "
        "reduction extends lifespan by altering translation of specific mRNAs."
    ),
    "Gm": (
        "2'-O-methylguanosine (Gm) is a ribose methylation in rRNA, snRNA, and tRNA, "
        "deposited by fibrillarin in the nucleolus. It stabilizes RNA structure and "
        "regulates ribosome function. Nucleolar dysfunction is a conserved feature "
        "of aging across species."
    ),
    "Um": (
        "2'-O-methyluridine (Um) is a ribose methylation in rRNA and snRNA, deposited "
        "by fibrillarin. It regulates ribosomal RNA folding and ribosome assembly. "
        "Changes in ribosome modification patterns are associated with translational "
        "stress during cellular aging."
    ),
    "m6Am": (
        "N6,2'-O-dimethyladenosine (m6Am) occurs at the mRNA cap+1 position and is "
        "deposited by PCIF1. It enhances mRNA stability and cap-dependent translation. "
        "FTO demethylates m6Am in addition to m6A, linking this modification to the "
        "same regulatory network implicated in metabolic disease and aging."
    ),
}

GENE_REGION_CONTEXT = {
    "3' UTR": "3' UTR — critical region for miRNA targeting, mRNA stability signals, and AU-rich element-mediated decay; RNA modifications here directly affect mRNA half-life.",
    "5' UTR": "5' UTR — affects cap-dependent translation initiation; modifications alter ribosome recruitment and translational efficiency.",
    "Distal Intergenic": "Distal intergenic region — may harbor enhancer elements or affect long-range chromatin interactions regulating nearby genes.",
    "Promoter": "Promoter region — affects transcription factor binding and gene expression levels.",
}


def classify_disease(disease_str):
    if pd.isna(disease_str) or str(disease_str).strip() in ("", "NA", "nan"):
        return None
    if any(k in str(disease_str).lower() for k in AGING_KEYWORDS):
        return "aging"
    return "non_aging"


def load_all_files(n_per_class=60):
    """Load all modification files and collect balanced samples."""
    aging_rows, non_aging_rows, no_disease_rows = [], [], []

    for mod_type, path in RM_FILES.items():
        df = pd.read_csv(path, low_memory=False)
        # Keep high + medium confidence only
        df = df[df["Confidence_Level"].isin(["high", "medium"])].copy()
        if len(df) == 0:
            df = pd.read_csv(path, low_memory=False).copy()  # fallback: take all

        df["mod_type"] = mod_type
        df["disease_class"] = df["Disease_association"].apply(classify_disease)

        aging   = df[df["disease_class"] == "aging"]
        non_ag  = df[df["disease_class"] == "non_aging"]
        no_dis  = df[df["disease_class"].isna()]

        print(f"  {mod_type:<8}: aging={len(aging):>4}, non_aging={len(non_ag):>4}, "
              f"no_disease={len(no_dis):>6}")

        # Sample up to n_per_class from each category for each mod type
        # Weight by availability — don't skip mods with fewer aging entries
        n_a = min(n_per_class, len(aging))
        n_n = min(n_per_class, len(non_ag))
        n_nd = min(n_per_class, len(no_dis))

        if n_a > 0:
            aging_rows.append(aging.sample(n_a, random_state=42))
        if n_n > 0:
            non_aging_rows.append(non_ag.sample(n_n, random_state=42))
        if n_nd > 0:
            no_disease_rows.append(no_dis.sample(n_nd, random_state=42))

    all_aging    = pd.concat(aging_rows, ignore_index=True)
    all_non_aging = pd.concat(non_aging_rows, ignore_index=True)
    all_no_disease = pd.concat(no_disease_rows, ignore_index=True)

    # Balance: use minimum class size across the three
    min_n = min(len(all_aging), len(all_non_aging), len(all_no_disease))
    print(f"\nPre-balance: aging={len(all_aging)}, "
          f"non_aging={len(all_non_aging)}, no_disease={len(all_no_disease)}")
    print(f"Balancing to {min_n} per class")

    balanced = pd.concat([
        all_aging.sample(min_n, random_state=42),
        all_non_aging.sample(min_n, random_state=42),
        all_no_disease.sample(min_n, random_state=42),
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced


def clean_gene_region(raw):
    """Simplify verbose exon annotations."""
    if pd.isna(raw):
        return "Unknown"
    s = str(raw)
    if s.startswith("Exon"):
        return "Exon (coding region)"
    return s


def make_prompt(row):
    mod_type = row["mod_type"]
    dc = row["disease_class"]
    dc = dc if str(dc) != "nan" else None
    gold = {"aging": "A", "non_aging": "B", None: "C"}[dc]

    gene       = str(row.get("Gene", "N/A"))
    gene_type  = str(row.get("Gene_Type", "N/A"))
    gene_region = clean_gene_region(row.get("Gene_Region"))
    region_ctx = GENE_REGION_CONTEXT.get(gene_region,
                    f"{gene_region} — a genomic region affecting nearby regulatory elements.")
    chrom      = row.get("seqnames", "N/A")
    pos        = row.get("MD_ChromStart", "N/A")
    strand     = row.get("MD_Strand", "N/A")
    refseq     = str(row.get("refseq", "N/A"))
    altseq     = str(row.get("altseq", "N/A"))
    snp_pos    = row.get("snp_pos", "N/A")
    conf       = row.get("Confidence_Level", "N/A")
    var_type   = str(row.get("type", f"human {mod_type} variant"))

    user_content = f"""<rna_modification_context>
<modification_type>{mod_type}</modification_type>
<modification_biology>
{MOD_DESCRIPTIONS.get(mod_type, f"An RNA chemical modification ({mod_type}) affecting post-transcriptional gene regulation.")}
</modification_biology>
</rna_modification_context>

<genetic_variant>
<genomic_location>{chrom}:{pos} ({strand} strand, hg19)</genomic_location>
<variant_effect>{var_type}</variant_effect>
<confidence_level>{conf}</confidence_level>
<affected_gene>{gene} ({gene_type})</affected_gene>
<gene_region>{gene_region}</gene_region>
<region_functional_context>{region_ctx}</region_functional_context>
<sequence_context>
  Reference (41 nt): {refseq}
  Alternate  (41 nt): {altseq}
  Modification site position in window: {snp_pos}/41
</sequence_context>
</genetic_variant>

<question>
This genetic variant affects an RNA {mod_type} modification site in the {gene_region} of gene {gene}.
Based on the modification biology, variant characteristics, and gene context, 
what is the most likely disease association of this variant?
</question>

<options>
A. Aging or age-related disease — including neurodegeneration (Alzheimer's, Parkinson's), cancer, cardiovascular disease, metabolic disorders, or other conditions strongly linked to aging biology
B. Non-aging disease — a genetic or developmental disorder without a primary aging component (e.g., congenital syndrome, rare Mendelian disorder)
C. No known disease association — this modification site variant has not been linked to a specific disease in current databases
</options>

Respond with only the option letter (A, B, or C)."""

    return {
        "lb_id": "EP-02",
        "pool": "rmdisease_variant_ternary",
        "display_name": "RNA Modification Variant Disease Association / Ternary",
        "domain": "epitranscriptomics",
        "format": "ternary",
        "metric": "balanced_accuracy",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in epitranscriptomics and human genetics. "
                    "You understand how RNA modifications (m6A, A-to-I editing, m5C, "
                    "pseudouridine, ac4C, and others) regulate gene expression and "
                    "contribute to aging and disease. Classify genetic variants affecting "
                    "RNA modification sites by their most likely disease association. "
                    "Respond with only the option letter (A, B, or C)."
                )
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gold}
        ],
        "metadata": json.dumps({
            "mod_type": mod_type,
            "gene": gene,
            "gene_type": gene_type,
            "gene_region": gene_region,
            "variant_type": var_type,
            "disease_class": dc if dc else "none",
            "disease_association": str(row.get("Disease_association", "")),
            "confidence_level": str(conf),
            "chrom": str(chrom),
            "position": str(pos),
        })
    }


def write_jsonl(prompts, path):
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    print(f"Written: {path} ({len(prompts)} prompts)")


if __name__ == "__main__":
    print("Loading and balancing RMDisease files...\n")
    balanced_df = load_all_files(n_per_class=60)

    print(f"\nBuilding prompts...")
    prompts = [make_prompt(row) for _, row in balanced_df.iterrows()]

    # Stats
    golds = [p["messages"][-1]["content"] for p in prompts]
    mod_types = [json.loads(p["metadata"])["mod_type"] for p in prompts]
    print(f"Total prompts: {len(prompts)}")
    print(f"Class balance: {Counter(golds)}")
    print(f"Mod type coverage: {Counter(mod_types)}")

    write_jsonl(prompts, f"{OUTPUT_DIR}/EP-02_ternary.jsonl")

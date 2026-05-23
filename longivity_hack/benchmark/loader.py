import json
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Built-in sample tasks (no HuggingFace required)
# Real data from AnAge, DrugAge, NHANES. Gold values are verifiable.
# ---------------------------------------------------------------------------

_SYSTEM_BIO = "You are an expert in comparative biology and aging science."
_SYSTEM_DRUG = "You are an expert in geroscience and longevity pharmacology."
_SYSTEM_CLINICAL = "You are an expert clinician specialising in biological aging and longevity medicine."

_SAMPLE_TASKS = [
    {
        "lb_id": "LB-SAMPLE-001",
        "task": "multispecies_lifespan_regression",
        "domain": "comparative_biology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_BIO},
            {"role": "user", "content": (
                "Given the following biological profile of an unknown mammal species:\n"
                "  Body mass: 54,431 kg\n"
                "  Heart rate at rest: ~8 bpm\n"
                "  Core body temperature: 33–37 °C (variable, regulates poorly)\n"
                "  Gestation period: 365 days\n"
                "  Litter size: 1\n"
                "  Habitat: sub-zero Arctic/subarctic waters year-round\n"
                "  Order: Artiodactyla (infraorder Cetacea)\n"
                "  Distinguishing trait: extremely slow growth rate; "
                "Bowhead-specific ERCC1 and PCNA gene variants associated with enhanced DNA repair\n\n"
                "Submit an interval [min, max] for the maximum recorded lifespan of this species in years.\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "211"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-002",
        "task": "multispecies_lifespan_regression",
        "domain": "comparative_biology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_BIO},
            {"role": "user", "content": (
                "Given the following biological profile of an unknown mammal species:\n"
                "  Body mass: 35 g\n"
                "  Heart rate at rest: ~210 bpm\n"
                "  Reproductive strategy: eusocial (single reproductive queen, ~300 non-breeding workers)\n"
                "  Habitat: sealed underground burrow systems in sub-Saharan Africa\n"
                "  Metabolic rate: suppressed; tolerates sustained hypoxia and near-anoxia\n"
                "  Cancer incidence: near-zero (high-molecular-weight hyaluronan implicated)\n"
                "  Order: Rodentia\n"
                "  Distinguishing trait: negligible senescence; "
                "mortality rate does not increase with age after maturity\n\n"
                "Submit an interval [min, max] for the maximum recorded lifespan of this species in years.\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "32"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-003",
        "task": "multispecies_lifespan_regression",
        "domain": "comparative_biology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_BIO},
            {"role": "user", "content": (
                "Given the following biological profile of an unknown mammal species:\n"
                "  Body mass: 8 g\n"
                "  Wing morphology: adapted for sustained aerial insectivory\n"
                "  Hibernation: yes, 5–7 months annually (torpor, body temp drops to ~2 °C)\n"
                "  Heart rate during torpor: ~8 bpm\n"
                "  Habitat: temperate North America; roosts in caves and mines\n"
                "  Order: Chiroptera\n"
                "  Distinguishing trait: longest-lived small mammal relative to body size; "
                "telomere length does not shorten with age in wild populations\n\n"
                "Submit an interval [min, max] for the maximum recorded lifespan of this species in years.\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "34"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-004",
        "task": "drug_lifespan_extension_regression",
        "domain": "pharmacology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_DRUG},
            {"role": "user", "content": (
                "A longevity intervention study was conducted in male C57BL/6J mice.\n"
                "Treatment: oral rapamycin (encapsulated, 14 ppm in chow)\n"
                "Treatment start age: 9 months (roughly equivalent to ~35 human years)\n"
                "Control median lifespan: 24.0 months\n"
                "Study design: NIA Interventions Testing Program (ITP), multi-site\n"
                "Mechanism: mTORC1 inhibition; reduces translational activity and "
                "autophagy upregulation\n\n"
                "Submit an interval [min, max] for the percent increase in median lifespan "
                "relative to control (e.g. 10 means +10%).\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "14"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-005",
        "task": "clinical_biological_age_regression",
        "domain": "clinical_biomarkers",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_CLINICAL},
            {"role": "user", "content": (
                "Given the following clinical biomarker panel from an adult participant "
                "(sex: male, race/ethnicity: non-Hispanic white):\n"
                "  Serum albumin: 4.1 g/dL\n"
                "  Serum creatinine: 1.0 mg/dL\n"
                "  Serum glucose (fasting): 102 mg/dL\n"
                "  C-reactive protein (log): 0.4 mg/L\n"
                "  Systolic blood pressure: 138 mmHg\n"
                "  BMI: 27.4 kg/m²\n"
                "  Forced expiratory volume (FEV1): 83% predicted\n"
                "  White blood cell count: 6.8 × 10³/μL\n"
                "  Total cholesterol: 201 mg/dL\n"
                "  HDL cholesterol: 44 mg/dL\n\n"
                "Submit an interval [min, max] for the chronological age of this participant in years.\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "57"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-006",
        "task": "drug_lifespan_extension_regression",
        "domain": "pharmacology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_DRUG},
            {"role": "user", "content": (
                "A longevity intervention study was conducted in both male and female C57BL/6J mice.\n"
                "Treatment: caloric restriction (CR), 40% reduction from ad libitum intake\n"
                "Treatment start age: 2 months\n"
                "Control median lifespan: 24.0 months\n"
                "CR is the most reproducible and well-characterised longevity intervention "
                "in mammals; reduces IGF-1 signalling, lowers core body temperature, "
                "and activates AMPK/SIRT1 pathways.\n\n"
                "Submit an interval [min, max] for the percent increase in median lifespan "
                "relative to control (e.g. 20 means +20%).\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "40"},
        ],
    },
    {
        "lb_id": "LB-SAMPLE-007",
        "task": "multispecies_lifespan_regression",
        "domain": "comparative_biology",
        "format": "regression",
        "metric": "estimathon_score",
        "messages": [
            {"role": "system", "content": _SYSTEM_BIO},
            {"role": "user", "content": (
                "Given the following biological profile of an unknown mammal species:\n"
                "  Body mass: 800 g\n"
                "  Diet: insectivorous (earthworms, beetles, slugs)\n"
                "  Hibernation: yes, October–April\n"
                "  Litter size: 4–5 per year\n"
                "  Habitat: European gardens, hedgerows, woodland edges\n"
                "  Order: Erinaceomorpha\n"
                "  Distinguishing trait: covered in ~5,000 modified hair spines; "
                "rolls into a ball as primary defence; well-studied in European wildlife ecology\n\n"
                "Submit an interval [min, max] for the maximum recorded lifespan of this species in years.\n"
                "Reply with only: [min, max]"
            )},
            {"role": "assistant", "content": "16"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_tasks(source: str, limit: int | None = None) -> Iterator[dict]:
    """
    Yield task dicts from:
      "sample"           — built-in tasks, no network required
      "longebench"       — HuggingFace insilicomedicine/longebench benchmark split
      "longebench:extra" — HuggingFace longebench extra split
      <path>             — local .jsonl file
    """
    if source == "sample":
        yield from _load_sample(limit)
    elif source.startswith("longebench"):
        yield from _load_longebench(source, limit)
    else:
        yield from _load_jsonl(source, limit)


def _load_sample(limit: int | None) -> Iterator[dict]:
    tasks = _SAMPLE_TASKS
    if limit is not None:
        tasks = tasks[:limit]
    yield from tasks


def _load_longebench(source: str, limit: int | None) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    config_name = "extra" if source == "longebench:extra" else "benchmark"
    ds = load_dataset("insilicomedicine/longebench", config_name, split="eval")

    count = 0
    for row in ds:
        if limit is not None and count >= limit:
            break
        yield dict(row)
        count += 1


def _load_jsonl(path: str, limit: int | None) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    count = 0
    with p.open() as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1

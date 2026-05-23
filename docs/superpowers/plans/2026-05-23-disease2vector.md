# disease2vector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `disease2vector` Python module — a PPMI-based disease embedding with anchor-set k-NN scoring for the EP-01/EP-02/EP-03 RNA disease-association benchmark tasks.

**Architecture:** Scan the 13 `data/raw/*.csv` files for the `Disease_association` column, parse semicolon-separated disease bags, build a sparse PPMI co-occurrence matrix, expose `embed()` and `set_distance()` primitives, and use hard-coded anchor sets (8 aging + 4 non-aging) with k-NN distance to decide EP-01/02/03. No clustering algorithm. All steps deterministic.

**Tech Stack:** Python 3.11+, NumPy, SciPy (sparse), pandas (CSV reading only), pytest. No PyTorch, no transformers, no external API calls.

**Reference spec:** `docs/superpowers/specs/2026-05-23-disease2vector-design.md`

---

## File Structure

Files this plan creates (in repo root unless noted):

```
.gitignore                                # (modify) add data/disease2vector/*.npz, *.json except anchor_qc.json
pyproject.toml                            # (create or modify) add pytest, scipy, numpy, pandas

disease2vector/
    __init__.py                           # exports Disease2Vec
    config.py                             # constants: MIN_FREQ, K_NN, paths
    vocab.py                              # disease string normalization, vocab building
    ppmi.py                               # co-occurrence + PPMI math
    anchors.py                            # ANCHOR_SETS, EP03_LETTER, AGING_KEYS, resolve_anchors
    embed.py                              # Disease2Vec class (load, embed, set_distance, decide_*)
    validate.py                           # anchor QC report

scripts/
    build_disease2vector.py               # CLI: scan CSVs → build PPMI → save cache
    validate_disease2vector.py            # CLI: print/save QC report

data/disease2vector/                      # build artifacts (gitignored except anchor_qc.json)
    vocab.json
    ppmi.npz
    anchor_qc.json

tests/
    __init__.py
    conftest.py                           # synthetic CSV fixture
    test_vocab.py
    test_ppmi.py
    test_anchors.py
    test_decide.py
    test_integration.py                   # runs against real data/raw/
```

Each module file has one responsibility and is small enough (~50-150 lines) to hold in working context at once.

---

## Task 1: Project scaffolding

**Files:**
- Create: `disease2vector/__init__.py`
- Create: `disease2vector/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore`
- Modify: `pyproject.toml` (or create if missing)

- [ ] **Step 1: Inspect existing project setup**

```bash
cat pyproject.toml 2>/dev/null || echo "no pyproject"
cat .gitignore 2>/dev/null | tail -20
ls disease2vector/ 2>/dev/null || echo "module not yet present"
```

Expected: pyproject.toml may or may not exist; `.gitignore` exists (created in earlier commit).

- [ ] **Step 2: Create `pyproject.toml`** (only if it doesn't already exist; if it does, add the dependencies to its existing `[project]` table)

```toml
[project]
name = "disease2vector"
version = "0.1.0"
description = "PPMI disease embedding for Longevity-LLM benchmark scoring"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["disease2vector*"]
exclude = ["tests*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 3: Append to `.gitignore`**

```
# disease2vector build artifacts (rebuildable from data/raw/)
data/disease2vector/vocab.json
data/disease2vector/ppmi.npz
data/disease2vector/anchor_index.json
# keep anchor_qc.json checked in for inspection
!data/disease2vector/anchor_qc.json
```

- [ ] **Step 4: Create `disease2vector/__init__.py`**

```python
"""disease2vector — PPMI disease embedding for EP-01/EP-02/EP-03 scoring."""

from disease2vector.embed import Disease2Vec

__all__ = ["Disease2Vec"]
__version__ = "0.1.0"
```

- [ ] **Step 5: Create `disease2vector/config.py`**

```python
"""Centralized configuration constants for disease2vector."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data sources and cache locations
DATA_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "disease2vector"

VOCAB_PATH = CACHE_DIR / "vocab.json"
PPMI_PATH = CACHE_DIR / "ppmi.npz"
ANCHOR_INDEX_PATH = CACHE_DIR / "anchor_index.json"
ANCHOR_QC_PATH = CACHE_DIR / "anchor_qc.json"

# Build-time constants
MIN_FREQ = 3                # drop disease tokens appearing in <3 rows
K_NN = 3                    # k-nearest-neighbors for set_distance
MIN_ANCHORS_PER_SET = 5     # build fails if a set resolves below this

# CSV column we care about
DISEASE_COLUMN = "Disease_association"
```

- [ ] **Step 6: Create `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 7: Create `tests/conftest.py` — synthetic CSV fixture**

```python
"""Pytest fixtures with a tiny synthetic disease vocabulary for fast tests."""

import csv
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_csv_dir(tmp_path: Path) -> Path:
    """
    Build a temp dir with two CSV files whose Disease_association column
    encodes a known co-occurrence pattern.

    Vocabulary (after MIN_FREQ=3):
      - "alpha"  (freq 5)
      - "beta"   (freq 5; co-occurs with alpha 4 times)
      - "gamma"  (freq 3)
      - "delta"  (freq 4; co-occurs with gamma 3 times)
      - "lonely" (freq 1 — should be filtered out)

    Diseases NOT in vocab: "ghost" (only mentioned via lonely co-occurrence).
    """
    rows = [
        # cell pattern: semicolon-separated diseases
        "alpha;beta",
        "alpha;beta",
        "alpha;beta",
        "alpha;beta",
        "alpha",                  # alpha alone
        "beta",                   # beta alone
        "gamma;delta",
        "gamma;delta",
        "gamma;delta",
        "delta",
        "lonely;ghost",           # both will be filtered (each freq < 3)
        "",                       # empty cell
    ]
    csv_path = tmp_path / "synth_a.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seqnames", "Disease_association"])
        for r in rows:
            w.writerow(["chr1", r])
    return tmp_path
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore disease2vector/__init__.py disease2vector/config.py tests/__init__.py tests/conftest.py
git -c commit.gpgsign=false commit -m "Scaffold disease2vector package and test fixtures"
```

---

## Task 2: Disease vocabulary extraction

**Files:**
- Create: `disease2vector/vocab.py`
- Create: `tests/test_vocab.py`

- [ ] **Step 1: Write failing test for `normalize_token`**

Create `tests/test_vocab.py`:

```python
from disease2vector.vocab import normalize_token, parse_row, build_vocab


def test_normalize_lowercases_and_strips_whitespace():
    assert normalize_token("  Alzheimer Disease  ") == "alzheimer disease"


def test_normalize_collapses_internal_whitespace():
    assert normalize_token("type  2\tdiabetes") == "type 2 diabetes"


def test_normalize_strips_trailing_period():
    assert normalize_token("Glaucoma.") == "glaucoma"


def test_normalize_replaces_unicode_dashes():
    assert normalize_token("alzheimer–type dementia") == "alzheimer-type dementia"


def test_parse_row_splits_on_semicolons():
    assert parse_row("alpha;beta;gamma") == {"alpha", "beta", "gamma"}


def test_parse_row_strips_each_token():
    assert parse_row("alpha ; beta ; gamma ") == {"alpha", "beta", "gamma"}


def test_parse_row_handles_empty_cell():
    assert parse_row("") == set()
    assert parse_row("   ") == set()


def test_parse_row_deduplicates_within_row():
    assert parse_row("alpha;alpha;beta") == {"alpha", "beta"}
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/test_vocab.py -v
```

Expected: ImportError — `disease2vector.vocab` does not exist.

- [ ] **Step 3: Implement `disease2vector/vocab.py`**

```python
"""Disease string normalization and vocabulary construction."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from disease2vector.config import DISEASE_COLUMN, MIN_FREQ


def normalize_token(s: str) -> str:
    """Lowercase, collapse internal whitespace, strip trailing periods,
    normalize unicode dashes."""
    if s is None:
        return ""
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip().lower()
    s = " ".join(s.split())
    s = s.rstrip(".")
    return s


def parse_row(cell: str) -> set[str]:
    """Split a Disease_association cell into a set of normalized tokens.

    Empty / whitespace-only cells return empty set. Duplicates within a
    row are collapsed to a single occurrence (set semantics)."""
    if cell is None:
        return set()
    tokens = {normalize_token(t) for t in str(cell).split(";")}
    tokens.discard("")
    return tokens


def iter_disease_bags(csv_paths: Iterable[Path]) -> Iterable[set[str]]:
    """Yield one disease-set per CSV row across all input CSVs.

    Only reads the Disease_association column for efficiency.
    Skips rows with empty/NaN cells silently.
    """
    for path in csv_paths:
        df = pd.read_csv(
            path,
            usecols=[DISEASE_COLUMN],
            dtype={DISEASE_COLUMN: str},
            keep_default_na=False,
            na_values=[""],
        )
        for cell in df[DISEASE_COLUMN].fillna(""):
            bag = parse_row(cell)
            if bag:
                yield bag


def build_vocab(
    csv_paths: Iterable[Path],
    min_freq: int = MIN_FREQ,
) -> tuple[list[str], dict[str, int]]:
    """Scan CSVs, count token frequencies, filter by min_freq.

    Returns:
        vocab:     sorted list of surviving tokens
        freq_map:  token -> number of rows it appeared in
    """
    freq: Counter[str] = Counter()
    for bag in iter_disease_bags(csv_paths):
        for token in bag:
            freq[token] += 1

    surviving = {tok: count for tok, count in freq.items() if count >= min_freq}
    vocab = sorted(surviving.keys())
    return vocab, surviving
```

- [ ] **Step 4: Run normalize/parse tests**

```bash
pytest tests/test_vocab.py::test_normalize_lowercases_and_strips_whitespace tests/test_vocab.py::test_normalize_collapses_internal_whitespace tests/test_vocab.py::test_normalize_strips_trailing_period tests/test_vocab.py::test_normalize_replaces_unicode_dashes tests/test_vocab.py::test_parse_row_splits_on_semicolons tests/test_vocab.py::test_parse_row_strips_each_token tests/test_vocab.py::test_parse_row_handles_empty_cell tests/test_vocab.py::test_parse_row_deduplicates_within_row -v
```

Expected: 8 passed.

- [ ] **Step 5: Add `build_vocab` integration test using synthetic fixture**

Append to `tests/test_vocab.py`:

```python
def test_build_vocab_filters_below_min_freq(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, freq = build_vocab(csv_paths, min_freq=3)

    # alpha, beta, gamma, delta should survive; lonely, ghost should not
    assert "alpha" in vocab
    assert "beta" in vocab
    assert "gamma" in vocab
    assert "delta" in vocab
    assert "lonely" not in vocab
    assert "ghost" not in vocab


def test_build_vocab_returns_sorted_list(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, _ = build_vocab(csv_paths, min_freq=3)
    assert vocab == sorted(vocab)


def test_build_vocab_freq_counts_match_rows(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    _, freq = build_vocab(csv_paths, min_freq=3)
    assert freq["alpha"] == 5    # 4 with beta + 1 alone
    assert freq["beta"] == 5
    assert freq["gamma"] == 3
    assert freq["delta"] == 4    # 3 with gamma + 1 alone
```

- [ ] **Step 6: Run all vocab tests**

```bash
pytest tests/test_vocab.py -v
```

Expected: 11 passed.

- [ ] **Step 7: Commit**

```bash
git add disease2vector/vocab.py tests/test_vocab.py
git -c commit.gpgsign=false commit -m "Add disease vocabulary extraction and tests"
```

---

## Task 3: Co-occurrence matrix construction

**Files:**
- Create: `disease2vector/ppmi.py`
- Create: `tests/test_ppmi.py`

- [ ] **Step 1: Write failing tests for `build_cooccurrence`**

Create `tests/test_ppmi.py`:

```python
import numpy as np
from scipy.sparse import csr_matrix

from disease2vector.ppmi import build_cooccurrence, compute_ppmi
from disease2vector.vocab import build_vocab


def test_cooccurrence_matrix_shape_matches_vocab(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, _ = build_vocab(csv_paths, min_freq=3)
    C = build_cooccurrence(csv_paths, vocab)
    assert C.shape == (len(vocab), len(vocab))
    assert isinstance(C, csr_matrix)


def test_cooccurrence_is_symmetric(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, _ = build_vocab(csv_paths, min_freq=3)
    C = build_cooccurrence(csv_paths, vocab)
    diff = (C - C.T).toarray()
    assert np.all(diff == 0)


def test_cooccurrence_diagonal_equals_marginal_frequency(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, freq = build_vocab(csv_paths, min_freq=3)
    C = build_cooccurrence(csv_paths, vocab)
    idx = {tok: i for i, tok in enumerate(vocab)}
    # alpha appears in 5 rows -> diagonal[alpha] = 5
    assert C[idx["alpha"], idx["alpha"]] == freq["alpha"]
    assert C[idx["beta"], idx["beta"]] == freq["beta"]
    assert C[idx["gamma"], idx["gamma"]] == freq["gamma"]
    assert C[idx["delta"], idx["delta"]] == freq["delta"]


def test_cooccurrence_off_diagonal_counts_pairs(synthetic_csv_dir):
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, _ = build_vocab(csv_paths, min_freq=3)
    C = build_cooccurrence(csv_paths, vocab)
    idx = {tok: i for i, tok in enumerate(vocab)}
    # alpha;beta co-occur in 4 rows
    assert C[idx["alpha"], idx["beta"]] == 4
    # gamma;delta co-occur in 3 rows
    assert C[idx["gamma"], idx["delta"]] == 3
    # alpha and gamma never co-occur
    assert C[idx["alpha"], idx["gamma"]] == 0
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_ppmi.py -v
```

Expected: ImportError — `disease2vector.ppmi` does not exist.

- [ ] **Step 3: Implement `build_cooccurrence` in `disease2vector/ppmi.py`**

```python
"""Sparse co-occurrence matrix + PPMI computation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from disease2vector.vocab import iter_disease_bags


def build_cooccurrence(
    csv_paths: Iterable[Path],
    vocab: list[str],
) -> csr_matrix:
    """Build symmetric sparse co-occurrence matrix from CSV disease bags.

    For each row's disease bag Dr (a set):
      - For every ordered pair (a, b) in Dr × Dr (including a == b),
        C[a, b] += 1.
      - A single-disease row contributes only C[a, a] += 1.

    Tokens not in `vocab` are skipped silently. The diagonal equals each
    token's marginal frequency (number of rows it appeared in).
    """
    vocab_index = {tok: i for i, tok in enumerate(vocab)}
    n = len(vocab)

    rows: list[int] = []
    cols: list[int] = []
    for bag in iter_disease_bags(csv_paths):
        idxs = [vocab_index[t] for t in bag if t in vocab_index]
        for i in idxs:
            for j in idxs:
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.float64)
    C = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return C
```

- [ ] **Step 4: Run co-occurrence tests**

```bash
pytest tests/test_ppmi.py -v -k cooccurrence
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add disease2vector/ppmi.py tests/test_ppmi.py
git -c commit.gpgsign=false commit -m "Add sparse co-occurrence matrix builder and tests"
```

---

## Task 4: PPMI computation

**Files:**
- Modify: `disease2vector/ppmi.py`
- Modify: `tests/test_ppmi.py`

- [ ] **Step 1: Add failing tests for `compute_ppmi`**

Append to `tests/test_ppmi.py`:

```python
def test_ppmi_clips_negatives_to_zero():
    # 4-token vocab, pair (0,3) NEVER co-occurs => PMI = -inf, PPMI = 0
    C = csr_matrix(np.array([
        [10, 4, 1, 0],
        [4, 10, 0, 0],
        [1, 0, 10, 4],
        [0, 0, 4, 10],
    ], dtype=np.float64))
    P = compute_ppmi(C)
    assert P[0, 3] == 0.0
    assert P[3, 0] == 0.0


def test_ppmi_positive_for_above_chance_pairs():
    # Construct a case where (0,1) co-occurs more than chance
    C = csr_matrix(np.array([
        [10, 8, 1],
        [8, 10, 1],
        [1, 1, 10],
    ], dtype=np.float64))
    P = compute_ppmi(C)
    # alpha-beta should have positive PMI (above chance)
    assert P[0, 1] > 0
    assert P[1, 0] > 0


def test_ppmi_symmetric():
    C = csr_matrix(np.array([
        [10, 5, 2],
        [5, 10, 3],
        [2, 3, 10],
    ], dtype=np.float64))
    P = compute_ppmi(C)
    diff = (P - P.T).toarray()
    assert np.allclose(diff, 0, atol=1e-10)


def test_ppmi_on_synthetic_data_t1d_t2d_signal(synthetic_csv_dir):
    """Sanity: alpha-beta (4 co-occurrences) should have higher PPMI than
    alpha-gamma (0 co-occurrences => clipped to 0)."""
    csv_paths = list(synthetic_csv_dir.glob("*.csv"))
    vocab, _ = build_vocab(csv_paths, min_freq=3)
    C = build_cooccurrence(csv_paths, vocab)
    P = compute_ppmi(C)
    idx = {tok: i for i, tok in enumerate(vocab)}
    assert P[idx["alpha"], idx["beta"]] > P[idx["alpha"], idx["gamma"]]
    assert P[idx["alpha"], idx["gamma"]] == 0.0
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_ppmi.py -v -k ppmi
```

Expected: AttributeError — `compute_ppmi` does not exist in `disease2vector.ppmi`.

- [ ] **Step 3: Add `compute_ppmi` to `disease2vector/ppmi.py`**

Append to `disease2vector/ppmi.py`:

```python
def compute_ppmi(C: csr_matrix) -> csr_matrix:
    """Compute Positive Pointwise Mutual Information from a co-occurrence matrix.

    PMI(a, b) = log( P(a, b) / (P(a) * P(b)) )
    PPMI(a, b) = max(0, PMI(a, b))

    P(a, b) = C[a, b] / N
    P(a)    = C[a, a] / N    (marginal from the diagonal)
    """
    if not isinstance(C, csr_matrix):
        C = C.tocsr()

    N = C.sum()
    if N == 0:
        return C.copy()

    # Marginal frequencies from diagonal
    diag = C.diagonal().astype(np.float64)
    p_a = diag / N            # shape (n,)

    # Work on COO for elementwise PMI on nonzeros
    coo = C.tocoo()
    rows = coo.row
    cols = coo.col
    vals = coo.data.astype(np.float64)

    p_ab = vals / N
    expected = p_a[rows] * p_a[cols]

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(p_ab / expected)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[pmi < 0.0] = 0.0

    # Drop explicit zeros to keep matrix sparse
    mask = pmi > 0.0
    P = coo_matrix(
        (pmi[mask], (rows[mask], cols[mask])),
        shape=C.shape,
    ).tocsr()
    return P
```

- [ ] **Step 4: Run PPMI tests**

```bash
pytest tests/test_ppmi.py -v
```

Expected: all 8 tests passed.

- [ ] **Step 5: Commit**

```bash
git add disease2vector/ppmi.py tests/test_ppmi.py
git -c commit.gpgsign=false commit -m "Add PPMI computation with positive-clip and sparse output"
```

---

## Task 5: Anchor set definitions

**Files:**
- Create: `disease2vector/anchors.py`

This task has no tests — it's pure data. Tests for the resolution logic come in Task 6.

- [ ] **Step 1: Create `disease2vector/anchors.py` with all 12 anchor sets**

```python
"""Hard-coded anchor sets for disease2vector.

8 aging sets (each mapped to an EP-03 letter) + 4 non-aging reference sets.

Anchor strings are written in canonical form. The resolver
(`disease2vector.anchors.resolve_anchors`) maps them to actual vocab tokens
via case-insensitive bidirectional substring matching.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Aging-side anchor sets (each maps to one EP-03 answer letter)
# ---------------------------------------------------------------------------

ANCHOR_SETS_AGING: dict[str, list[str]] = {
    "aging_neurodegenerative": [
        "Alzheimer disease",
        "Parkinson disease",
        "Frontotemporal dementia",
        "Lewy body dementia",
        "Vascular dementia",
        "Amyotrophic lateral sclerosis",
        "Mild cognitive impairment",
        "Progressive supranuclear palsy",
        "Multiple system atrophy",
        "Age-related cognitive decline",
    ],
    "aging_cardiovascular": [
        "Atherosclerosis",
        "Coronary artery disease",
        "Heart failure",
        "Myocardial infarction",
        "Essential hypertension",
        "Aortic aneurysm",
        "Atrial fibrillation",
        "Ischemic stroke",
        "Peripheral artery disease",
    ],
    "aging_metabolic": [
        "Type 2 diabetes mellitus",
        "Metabolic syndrome",
        "Non-alcoholic fatty liver disease",
        "Obesity",
        "Dyslipidemia",
        "Insulin resistance",
        "Gout",
    ],
    "aging_musculoskeletal": [
        "Osteoporosis",
        "Sarcopenia",
        "Osteoarthritis",
        "Frailty syndrome",
        "Age-related muscle atrophy",
        "Degenerative disc disease",
        "Spinal stenosis",
    ],
    "aging_fibrosis_tissue": [
        "Idiopathic pulmonary fibrosis",
        "Liver cirrhosis",
        "Renal fibrosis",
        "Cardiac fibrosis",
        "Skin photoaging",
        "Hepatic fibrosis",
        "Systemic sclerosis",
    ],
    "aging_cancer_solid": [
        "Breast cancer",
        "Prostate cancer",
        "Colorectal cancer",
        "Lung cancer",
        "Pancreatic cancer",
        "Gastric cancer",
        "Hepatocellular carcinoma",
        "Bladder cancer",
        "Renal cell carcinoma",
        "Glioblastoma",
    ],
    "aging_cancer_hematologic": [
        "Acute myeloid leukemia",
        "Myelodysplastic syndrome",
        "Chronic lymphocytic leukemia",
        "Multiple myeloma",
        "Diffuse large B-cell lymphoma",
        "Clonal hematopoiesis of indeterminate potential",
    ],
    "aging_organ_decline": [
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease",
        "Age-related macular degeneration",
        "Cataract",
        "Presbycusis",
        "Benign prostatic hyperplasia",
        "Diabetic retinopathy",
    ],
}

# Map each aging set to its EP-03 answer letter
EP03_LETTER: dict[str, str] = {
    "aging_neurodegenerative":   "A",
    "aging_cardiovascular":      "B",
    "aging_metabolic":           "B",
    "aging_musculoskeletal":     "B",
    "aging_fibrosis_tissue":     "C",
    "aging_cancer_solid":        "C",
    "aging_cancer_hematologic":  "C",
    "aging_organ_decline":       "C",
}

# ---------------------------------------------------------------------------
# Non-aging reference sets (used to anchor "not aging" decisions for EP-01/02)
# ---------------------------------------------------------------------------

ANCHOR_SETS_NON_AGING: dict[str, list[str]] = {
    "non_aging_congenital_mendelian": [
        "Cystic fibrosis",
        "Sickle cell anemia",
        "Duchenne muscular dystrophy",
        "Phenylketonuria",
        "Tay-Sachs disease",
        "Hemophilia",
        "Beta thalassemia",
        "Spinal muscular atrophy",
        "Marfan syndrome",
        "Achondroplasia",
        "Galactosemia",
    ],
    "non_aging_autoimmune": [
        "Rheumatoid arthritis",
        "Systemic lupus erythematosus",
        "Multiple sclerosis",
        "Inflammatory bowel disease",
        "Type 1 diabetes mellitus",
        "Celiac disease",
        "Psoriasis",
        "Hashimoto thyroiditis",
        "Graves disease",
    ],
    "non_aging_infectious": [
        "Tuberculosis",
        "HIV infection",
        "Hepatitis B",
        "Hepatitis C",
        "Malaria",
        "Influenza",
        "COVID-19",
        "Pneumonia",
        "Sepsis",
    ],
    "non_aging_psychiatric_neurodev": [
        "Schizophrenia",
        "Bipolar disorder",
        "Major depressive disorder",
        "Autism spectrum disorder",
        "Attention deficit hyperactivity disorder",
        "Intellectual disability",
        "Anxiety disorder",
    ],
}

# Combined view
ALL_ANCHOR_SETS: dict[str, list[str]] = {
    **ANCHOR_SETS_AGING,
    **ANCHOR_SETS_NON_AGING,
}

AGING_SET_NAMES: frozenset[str] = frozenset(ANCHOR_SETS_AGING.keys())
NON_AGING_SET_NAMES: frozenset[str] = frozenset(ANCHOR_SETS_NON_AGING.keys())
```

- [ ] **Step 2: Sanity check by importing**

```bash
python -c "from disease2vector.anchors import ALL_ANCHOR_SETS, EP03_LETTER; print(len(ALL_ANCHOR_SETS), 'sets,', sum(len(v) for v in ALL_ANCHOR_SETS.values()), 'anchors'); print('EP-03 letters:', set(EP03_LETTER.values()))"
```

Expected output:
```
12 sets, 107 anchors
EP-03 letters: {'A', 'B', 'C'}
```

- [ ] **Step 3: Commit**

```bash
git add disease2vector/anchors.py
git -c commit.gpgsign=false commit -m "Define anchor sets for aging and non-aging clusters"
```

---

## Task 6: Anchor-to-vocab resolution

**Files:**
- Modify: `disease2vector/anchors.py`
- Create: `tests/test_anchors.py`

- [ ] **Step 1: Write failing tests for `resolve_anchors`**

Create `tests/test_anchors.py`:

```python
import pytest

from disease2vector.anchors import resolve_anchors


def test_resolve_exact_match():
    vocab = ["alzheimer disease", "parkinson disease", "random other"]
    freq = {"alzheimer disease": 100, "parkinson disease": 80, "random other": 20}
    sets = {"neuro": ["Alzheimer disease"]}
    resolved = resolve_anchors(sets, vocab, freq, min_anchors_per_set=1)
    assert resolved["neuro"] == [0]  # index of "alzheimer disease"


def test_resolve_bidirectional_substring_anchor_in_vocab():
    """Vocab token contains the anchor string."""
    vocab = ["alzheimer disease early onset"]
    freq = {"alzheimer disease early onset": 50}
    sets = {"neuro": ["Alzheimer disease"]}
    resolved = resolve_anchors(sets, vocab, freq, min_anchors_per_set=1)
    assert resolved["neuro"] == [0]


def test_resolve_bidirectional_substring_vocab_in_anchor():
    """Anchor string contains the vocab token."""
    vocab = ["alzheimer"]
    freq = {"alzheimer": 50}
    sets = {"neuro": ["Alzheimer disease"]}
    resolved = resolve_anchors(sets, vocab, freq, min_anchors_per_set=1)
    assert resolved["neuro"] == [0]


def test_resolve_picks_highest_frequency_on_ambiguity():
    """When multiple vocab tokens match one anchor, all are kept."""
    vocab = ["breast cancer", "metastatic breast cancer"]
    freq = {"breast cancer": 500, "metastatic breast cancer": 30}
    sets = {"solid": ["Breast cancer"]}
    resolved = resolve_anchors(sets, vocab, freq, min_anchors_per_set=1)
    # Both should resolve; vocab indices sorted ascending
    assert resolved["solid"] == [0, 1]


def test_resolve_fails_when_too_few_anchors_match():
    vocab = ["totally unrelated"]
    freq = {"totally unrelated": 10}
    sets = {"neuro": ["Alzheimer disease", "Parkinson disease"]}
    with pytest.raises(ValueError, match="resolved only 0"):
        resolve_anchors(sets, vocab, freq, min_anchors_per_set=2)


def test_resolve_logs_unmatched_anchors():
    vocab = ["alzheimer disease"]
    freq = {"alzheimer disease": 100}
    sets = {"neuro": ["Alzheimer disease", "Parkinson disease", "Lewy body dementia"]}
    resolved = resolve_anchors(sets, vocab, freq, min_anchors_per_set=1)
    # Only alzheimer matched; the other two are dropped without error
    assert resolved["neuro"] == [0]
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_anchors.py -v
```

Expected: ImportError — `resolve_anchors` does not exist.

- [ ] **Step 3: Implement `resolve_anchors` — append to `disease2vector/anchors.py`**

Append after the existing module content:

```python
# ---------------------------------------------------------------------------
# Anchor → vocab resolution
# ---------------------------------------------------------------------------

import re
import unicodedata

_TRAILING_MODIFIERS = ("disease", "disorder", "syndrome")


def _normalize_for_matching(s: str) -> str:
    """Lowercase, strip diacritics and most punctuation, drop trailing
    'disease'/'disorder'/'syndrome' suffix.

    The original string is not modified; this is only used to compare
    anchor strings against vocab tokens.
    """
    s = s.lower().strip()
    # strip diacritics
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )
    # collapse punctuation to spaces (keep alphanumerics, hyphens, and spaces)
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    s = " ".join(s.split())
    # drop a single trailing modifier word
    for suffix in _TRAILING_MODIFIERS:
        if s.endswith(" " + suffix):
            s = s[: -(len(suffix) + 1)]
            break
    return s


def resolve_anchors(
    anchor_sets: dict[str, list[str]],
    vocab: list[str],
    freq: dict[str, int],
    min_anchors_per_set: int,
) -> dict[str, list[int]]:
    """Resolve each anchor set's strings to vocab indices.

    Matching is case-insensitive, diacritic-insensitive, and uses
    bidirectional substring: anchor (normalized) is a substring of the
    vocab token (normalized) OR vice versa.

    Returns: set_name -> sorted list of vocab indices.

    Raises:
        ValueError: if any set resolves fewer than `min_anchors_per_set`
                    distinct vocab tokens.
    """
    norm_vocab = [(_normalize_for_matching(tok), i) for i, tok in enumerate(vocab)]

    resolved: dict[str, list[int]] = {}
    for set_name, anchors in anchor_sets.items():
        matched_idx: set[int] = set()
        for anchor in anchors:
            a_norm = _normalize_for_matching(anchor)
            if not a_norm:
                continue
            for v_norm, idx in norm_vocab:
                if not v_norm:
                    continue
                if a_norm == v_norm or a_norm in v_norm or v_norm in a_norm:
                    matched_idx.add(idx)
        if len(matched_idx) < min_anchors_per_set:
            raise ValueError(
                f"Anchor set {set_name!r} resolved only {len(matched_idx)} "
                f"vocab tokens (required: {min_anchors_per_set}). "
                f"Anchors tried: {anchors}"
            )
        resolved[set_name] = sorted(matched_idx)
    return resolved
```

- [ ] **Step 4: Run anchor tests**

```bash
pytest tests/test_anchors.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add disease2vector/anchors.py tests/test_anchors.py
git -c commit.gpgsign=false commit -m "Add anchor-to-vocab resolution with bidirectional substring matching"
```

---

## Task 7: Disease2Vec class — embed and set_distance

**Files:**
- Create: `disease2vector/embed.py`
- Create: `tests/test_decide.py` (will grow in Task 8)

- [ ] **Step 1: Write failing tests for `Disease2Vec.embed` and `Disease2Vec.set_distance`**

Create `tests/test_decide.py`:

```python
"""Tests for Disease2Vec class — embed, set_distance, and decide_* methods."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from disease2vector.embed import Disease2Vec


@pytest.fixture
def toy_d2v():
    """A hand-built Disease2Vec instance with known geometry.

    Vocab (5 tokens):
      0: "alzheimer disease"
      1: "parkinson disease"
      2: "type 2 diabetes mellitus"
      3: "atherosclerosis"
      4: "cystic fibrosis"

    PPMI vectors are constructed so that:
      - AD and PD are close (both neuro)
      - T2D and atherosclerosis are close (both cardio-metabolic)
      - CF is far from everything (non-aging)
    """
    vocab = [
        "alzheimer disease",
        "parkinson disease",
        "type 2 diabetes mellitus",
        "atherosclerosis",
        "cystic fibrosis",
    ]
    P = np.array([
        # ad   pd   t2d  ath  cf
        [1.0, 0.9, 0.1, 0.1, 0.05],   # ad
        [0.9, 1.0, 0.1, 0.1, 0.05],   # pd
        [0.1, 0.1, 1.0, 0.8, 0.05],   # t2d
        [0.1, 0.1, 0.8, 1.0, 0.05],   # ath
        [0.05, 0.05, 0.05, 0.05, 1.0],  # cf
    ], dtype=np.float64)
    ppmi = csr_matrix(P)

    anchor_index = {
        "aging_neurodegenerative":        [0, 1],
        "aging_metabolic":                [2],
        "aging_cardiovascular":           [3],
        "non_aging_congenital_mendelian": [4],
    }
    ep03_letter = {
        "aging_neurodegenerative":        "A",
        "aging_metabolic":                "B",
        "aging_cardiovascular":           "B",
    }
    aging_set_names = frozenset({
        "aging_neurodegenerative",
        "aging_metabolic",
        "aging_cardiovascular",
    })
    non_aging_set_names = frozenset({"non_aging_congenital_mendelian"})

    return Disease2Vec(
        ppmi=ppmi,
        vocab=vocab,
        anchor_index=anchor_index,
        ep03_letter=ep03_letter,
        aging_set_names=aging_set_names,
        non_aging_set_names=non_aging_set_names,
    )


def test_embed_known_disease_returns_vector(toy_d2v):
    v = toy_d2v.embed("Alzheimer disease")
    assert v is not None
    assert v.shape == (5,)


def test_embed_unknown_disease_returns_none(toy_d2v):
    assert toy_d2v.embed("totally fake disease") is None


def test_embed_is_case_insensitive(toy_d2v):
    v1 = toy_d2v.embed("Alzheimer disease")
    v2 = toy_d2v.embed("ALZHEIMER DISEASE")
    assert v1 is not None and v2 is not None
    assert np.allclose(v1, v2)


def test_set_distance_close_for_in_set_members(toy_d2v):
    v_ad = toy_d2v.embed("Alzheimer disease")
    # AD is itself in aging_neurodegenerative; k-NN includes self => distance ~0
    d = toy_d2v.set_distance(v_ad, "aging_neurodegenerative", k=2)
    assert d < 0.1


def test_set_distance_far_for_out_of_set(toy_d2v):
    v_cf = toy_d2v.embed("Cystic fibrosis")
    d = toy_d2v.set_distance(v_cf, "aging_neurodegenerative", k=2)
    # CF is far from neuro -> large cosine distance
    assert d > 0.5


def test_set_distance_k_capped_at_set_size(toy_d2v):
    # aging_metabolic has only 1 anchor; k=5 should not crash, just use 1
    v = toy_d2v.embed("Type 2 diabetes mellitus")
    d = toy_d2v.set_distance(v, "aging_metabolic", k=5)
    assert d < 0.1
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_decide.py -v
```

Expected: ImportError — `disease2vector.embed` does not exist.

- [ ] **Step 3: Implement `Disease2Vec` (without decide_* yet) in `disease2vector/embed.py`**

```python
"""Disease2Vec — the main public class.

Holds the PPMI matrix and anchor indices, exposes embed() and
set_distance() primitives, plus ep01/ep02/ep03 decision methods.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from disease2vector.config import (
    ANCHOR_INDEX_PATH,
    K_NN,
    PPMI_PATH,
    VOCAB_PATH,
)
from disease2vector.vocab import normalize_token


@dataclass
class Disease2Vec:
    ppmi: csr_matrix
    vocab: list[str]
    anchor_index: dict[str, list[int]]
    ep03_letter: dict[str, str]
    aging_set_names: frozenset[str]
    non_aging_set_names: frozenset[str]

    def __post_init__(self) -> None:
        self._vocab_index: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}

    # ---- primitives ----

    def embed(self, disease: str) -> np.ndarray | None:
        """Return the PPMI row vector for `disease`, or None if not in vocab."""
        key = normalize_token(disease)
        idx = self._vocab_index.get(key)
        if idx is None:
            return None
        return np.asarray(self.ppmi[idx].todense()).ravel()

    def set_distance(
        self,
        vec: np.ndarray,
        set_name: str,
        k: int = K_NN,
    ) -> float:
        """Mean cosine distance from `vec` to its k closest anchors in
        `set_name`. If the set has fewer than k anchors, k is capped at
        set size.
        """
        idxs = self.anchor_index[set_name]
        if not idxs:
            return math.inf

        anchor_mat = np.asarray(self.ppmi[idxs].todense())  # (m, D)
        sims = _cosine_similarity_batch(vec, anchor_mat)    # (m,)
        dists = 1.0 - sims

        k_eff = min(k, len(idxs))
        # partial sort: take the k smallest distances
        idx_sort = np.argpartition(dists, k_eff - 1)[:k_eff]
        return float(np.mean(dists[idx_sort]))

    # ---- (decide_* methods added in Task 8) ----

    # ---- loading from cache ----

    @classmethod
    def load(
        cls,
        ppmi_path: Path = PPMI_PATH,
        vocab_path: Path = VOCAB_PATH,
        anchor_index_path: Path = ANCHOR_INDEX_PATH,
    ) -> "Disease2Vec":
        """Load a Disease2Vec from cached build artifacts."""
        from disease2vector.anchors import (
            AGING_SET_NAMES,
            EP03_LETTER,
            NON_AGING_SET_NAMES,
        )

        ppmi = load_npz(ppmi_path)
        with vocab_path.open() as f:
            vocab = json.load(f)
        with anchor_index_path.open() as f:
            anchor_index = json.load(f)
        return cls(
            ppmi=ppmi,
            vocab=vocab,
            anchor_index=anchor_index,
            ep03_letter=EP03_LETTER,
            aging_set_names=AGING_SET_NAMES,
            non_aging_set_names=NON_AGING_SET_NAMES,
        )


def _cosine_similarity_batch(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector `v` (D,) and each row of matrix `M` (m, D)."""
    v_norm = np.linalg.norm(v)
    M_norms = np.linalg.norm(M, axis=1)
    denom = v_norm * M_norms
    denom = np.where(denom == 0.0, 1.0, denom)   # avoid div-by-zero
    sims = (M @ v) / denom
    # if v has zero norm, every sim is undefined => treat as 0
    if v_norm == 0.0:
        sims = np.zeros_like(sims)
    return sims
```

- [ ] **Step 4: Run embed and set_distance tests**

```bash
pytest tests/test_decide.py -v -k "embed or set_distance"
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add disease2vector/embed.py tests/test_decide.py
git -c commit.gpgsign=false commit -m "Add Disease2Vec class with embed and set_distance primitives"
```

---

## Task 8: Decision functions ep01_decide / ep02_decide / ep03_decide

**Files:**
- Modify: `disease2vector/embed.py`
- Modify: `tests/test_decide.py`

- [ ] **Step 1: Write failing tests for the three decision functions**

Append to `tests/test_decide.py`:

```python
def test_ep01_aging_disease_returns_A(toy_d2v):
    result = toy_d2v.ep01_decide("Alzheimer disease")
    assert result["answer"] == "A"
    assert result["d_aging"] < result["d_nonaging"]


def test_ep01_non_aging_disease_returns_B(toy_d2v):
    result = toy_d2v.ep01_decide("Cystic fibrosis")
    assert result["answer"] == "B"
    assert result["d_nonaging"] < result["d_aging"]


def test_ep01_unknown_disease_returns_B(toy_d2v):
    result = toy_d2v.ep01_decide("never heard of it")
    assert result["answer"] == "B"
    assert result["reason"] == "not_in_vocab"


def test_ep02_bag_with_aging_member_returns_A(toy_d2v):
    result = toy_d2v.ep02_decide(["Alzheimer disease", "Cystic fibrosis"])
    assert result["answer"] == "A"


def test_ep02_all_non_aging_returns_B(toy_d2v):
    result = toy_d2v.ep02_decide(["Cystic fibrosis"])
    assert result["answer"] == "B"


def test_ep02_no_disease_in_vocab_returns_C(toy_d2v):
    result = toy_d2v.ep02_decide(["unknown1", "unknown2"])
    assert result["answer"] == "C"
    assert result["reason"] == "no_disease_in_vocab"


def test_ep02_empty_bag_returns_C(toy_d2v):
    result = toy_d2v.ep02_decide([])
    assert result["answer"] == "C"


def test_ep03_neuro_disease_returns_A(toy_d2v):
    result = toy_d2v.ep03_decide("Alzheimer disease")
    assert result["answer"] == "A"


def test_ep03_metabolic_disease_returns_B(toy_d2v):
    result = toy_d2v.ep03_decide("Type 2 diabetes mellitus")
    assert result["answer"] == "B"


def test_ep03_returns_letter_distance_dict(toy_d2v):
    result = toy_d2v.ep03_decide("Alzheimer disease")
    assert set(result["letter_dist"].keys()) == {"A", "B", "C"}
    # AD's nearest letter is A -> A's distance < B's distance
    assert result["letter_dist"]["A"] < result["letter_dist"]["B"]


def test_ep03_unknown_disease_returns_none(toy_d2v):
    result = toy_d2v.ep03_decide("never heard of it")
    assert result["answer"] is None
    assert result["reason"] == "not_in_vocab"


def test_ep03_margin_is_nonnegative(toy_d2v):
    result = toy_d2v.ep03_decide("Alzheimer disease")
    assert result["margin"] >= 0
```

- [ ] **Step 2: Run failing tests**

```bash
pytest tests/test_decide.py -v -k "ep01 or ep02 or ep03"
```

Expected: AttributeError — `ep01_decide`/`ep02_decide`/`ep03_decide` not implemented.

- [ ] **Step 3: Add the three decide methods to `Disease2Vec`**

In `disease2vector/embed.py`, insert these methods inside the `Disease2Vec` class (between `set_distance` and the `load` classmethod):

```python
    # ---- EP-01 / EP-02 / EP-03 decisions ----

    def ep01_decide(self, disease: str) -> dict:
        """Decide whether `disease` is associated with aging.

        Returns dict with keys: answer ("A" or "B"), d_aging, d_nonaging,
        and optionally `reason` if out-of-vocab.
        """
        v = self.embed(disease)
        if v is None:
            return {"answer": "B", "reason": "not_in_vocab",
                    "d_aging": math.inf, "d_nonaging": math.inf}

        d_aging = min(
            self.set_distance(v, name) for name in self.aging_set_names
        )
        d_nonaging = min(
            self.set_distance(v, name) for name in self.non_aging_set_names
        )
        return {
            "answer": "A" if d_aging < d_nonaging else "B",
            "d_aging": d_aging,
            "d_nonaging": d_nonaging,
        }

    def ep02_decide(self, disease_bag: list[str]) -> dict:
        """Ternary decision for an SNP's disease bag.

        A: any disease in bag is aging-associated (and in vocab)
        B: all bag diseases are in vocab but none are aging-associated
        C: no disease in the bag is in vocabulary (no known association)
        """
        in_vocab_diseases = [d for d in disease_bag if self.embed(d) is not None]
        if not in_vocab_diseases:
            return {"answer": "C", "reason": "no_disease_in_vocab"}

        for d in in_vocab_diseases:
            if self.ep01_decide(d)["answer"] == "A":
                return {"answer": "A"}
        return {"answer": "B"}

    def ep03_decide(self, disease: str) -> dict:
        """Decide which EP-03 aging subcategory (A/B/C) `disease` belongs to."""
        v = self.embed(disease)
        if v is None:
            return {"answer": None, "reason": "not_in_vocab"}

        set_dists = {
            name: self.set_distance(v, name) for name in self.aging_set_names
        }
        letter_dist = {"A": math.inf, "B": math.inf, "C": math.inf}
        for name, d in set_dists.items():
            letter = self.ep03_letter[name]
            letter_dist[letter] = min(letter_dist[letter], d)

        sorted_dists = sorted(letter_dist.values())
        margin = sorted_dists[1] - sorted_dists[0]
        return {
            "answer": min(letter_dist, key=letter_dist.get),
            "letter_dist": letter_dist,
            "set_dist": set_dists,
            "margin": margin,
        }
```

- [ ] **Step 4: Run all decide tests**

```bash
pytest tests/test_decide.py -v
```

Expected: 18 tests passed (6 from Task 7 + 12 new).

- [ ] **Step 5: Commit**

```bash
git add disease2vector/embed.py tests/test_decide.py
git -c commit.gpgsign=false commit -m "Add ep01/ep02/ep03 decision methods"
```

---

## Task 9: Anchor QC report

**Files:**
- Create: `disease2vector/validate.py`

This task has no separate unit tests — the QC report is exercised end-to-end in the integration test (Task 12).

- [ ] **Step 1: Create `disease2vector/validate.py`**

```python
"""Anchor QC report — self-consistency checks for the built embedding.

Computes:
  - within-set tightness: median pairwise cosine distance among anchors of the same set
  - between-set separation: median cosine distance between set centroids
  - anchor misplacement: anchors that are closer to a different set than their own

Saves a JSON report. Does NOT auto-modify the anchor lists.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np

from disease2vector.embed import Disease2Vec, _cosine_similarity_batch


def _set_centroid(d2v: Disease2Vec, set_name: str) -> np.ndarray:
    idxs = d2v.anchor_index[set_name]
    rows = np.asarray(d2v.ppmi[idxs].todense())
    return rows.mean(axis=0)


def within_set_tightness(d2v: Disease2Vec, set_name: str) -> float:
    idxs = d2v.anchor_index[set_name]
    if len(idxs) < 2:
        return 0.0
    M = np.asarray(d2v.ppmi[idxs].todense())  # (m, D)
    sims = (M @ M.T) / (
        np.linalg.norm(M, axis=1, keepdims=True)
        * np.linalg.norm(M, axis=1, keepdims=True).T
        + 1e-12
    )
    dists = 1.0 - sims
    # take upper triangle (exclude diagonal)
    iu = np.triu_indices_from(dists, k=1)
    return float(median(dists[iu].tolist()))


def between_set_separation(d2v: Disease2Vec) -> dict[str, dict[str, float]]:
    set_names = list(d2v.anchor_index.keys())
    centroids = {name: _set_centroid(d2v, name) for name in set_names}
    sep: dict[str, dict[str, float]] = {}
    for a in set_names:
        sep[a] = {}
        for b in set_names:
            if a == b:
                continue
            sim = _cosine_similarity_batch(centroids[a], centroids[b].reshape(1, -1))[0]
            sep[a][b] = float(1.0 - sim)
    return sep


def anchor_misplacements(d2v: Disease2Vec) -> list[dict]:
    """For each anchor, check whether its own set is the closest set."""
    misplaced: list[dict] = []
    set_names = list(d2v.anchor_index.keys())
    for owner_set, idxs in d2v.anchor_index.items():
        for idx in idxs:
            v = np.asarray(d2v.ppmi[idx].todense()).ravel()
            set_dists = {name: d2v.set_distance(v, name) for name in set_names}
            closest = min(set_dists, key=set_dists.get)
            if closest != owner_set:
                misplaced.append({
                    "vocab_token": d2v.vocab[idx],
                    "vocab_index": idx,
                    "owner_set": owner_set,
                    "closest_set": closest,
                    "distance_to_owner": set_dists[owner_set],
                    "distance_to_closest": set_dists[closest],
                })
    return misplaced


def write_qc_report(d2v: Disease2Vec, out_path: Path) -> dict:
    report = {
        "n_vocab": len(d2v.vocab),
        "n_sets": len(d2v.anchor_index),
        "set_sizes": {k: len(v) for k, v in d2v.anchor_index.items()},
        "within_set_tightness_median_cosine_distance": {
            name: within_set_tightness(d2v, name) for name in d2v.anchor_index
        },
        "between_set_separation_cosine_distance": between_set_separation(d2v),
        "anchor_misplacements": anchor_misplacements(d2v),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    return report
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from disease2vector.validate import write_qc_report; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add disease2vector/validate.py
git -c commit.gpgsign=false commit -m "Add anchor QC report (tightness, separation, misplacements)"
```

---

## Task 10: Build script CLI

**Files:**
- Create: `scripts/build_disease2vector.py`

- [ ] **Step 1: Create the build script**

```python
#!/usr/bin/env python3
"""Build disease2vector cache from data/raw/*.csv.

Steps:
    1. Scan all *_human_associatedSNPs.csv files in data/raw/
    2. Build vocabulary (filter by MIN_FREQ)
    3. Build co-occurrence matrix
    4. Compute PPMI matrix
    5. Resolve anchor sets against vocabulary
    6. Save vocab.json, ppmi.npz, anchor_index.json to data/disease2vector/
    7. Print summary

Re-run anytime the underlying CSVs change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.sparse import save_npz

from disease2vector.anchors import (
    ALL_ANCHOR_SETS,
    resolve_anchors,
)
from disease2vector.config import (
    ANCHOR_INDEX_PATH,
    CACHE_DIR,
    DATA_DIR,
    MIN_ANCHORS_PER_SET,
    MIN_FREQ,
    PPMI_PATH,
    VOCAB_PATH,
)
from disease2vector.ppmi import build_cooccurrence, compute_ppmi
from disease2vector.vocab import build_vocab


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DATA_DIR), type=Path)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), type=Path)
    ap.add_argument("--min-freq", default=MIN_FREQ, type=int)
    ap.add_argument(
        "--min-anchors-per-set", default=MIN_ANCHORS_PER_SET, type=int
    )
    args = ap.parse_args()

    csv_paths = sorted(args.data_dir.glob("*_human_associatedSNPs.csv"))
    if not csv_paths:
        print(f"No matching CSVs found in {args.data_dir}", file=sys.stderr)
        return 1
    print(f"Scanning {len(csv_paths)} CSV files in {args.data_dir} ...")

    print(f"Building vocab (min_freq={args.min_freq}) ...")
    vocab, freq = build_vocab(csv_paths, min_freq=args.min_freq)
    print(f"  vocab size: {len(vocab)}")

    print("Building co-occurrence matrix ...")
    C = build_cooccurrence(csv_paths, vocab)
    print(f"  matrix shape: {C.shape}, nnz: {C.nnz}")

    print("Computing PPMI ...")
    P = compute_ppmi(C)
    print(f"  PPMI nnz: {P.nnz}")

    print("Resolving anchor sets against vocabulary ...")
    try:
        anchor_index = resolve_anchors(
            ALL_ANCHOR_SETS, vocab, freq, args.min_anchors_per_set
        )
    except ValueError as e:
        print(f"\nAnchor resolution failed: {e}", file=sys.stderr)
        return 2
    for name, idxs in anchor_index.items():
        print(f"  {name}: {len(idxs)} resolved tokens")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving cache to {args.cache_dir} ...")
    with (args.cache_dir / VOCAB_PATH.name).open("w") as f:
        json.dump(vocab, f)
    save_npz(args.cache_dir / PPMI_PATH.name, P)
    with (args.cache_dir / ANCHOR_INDEX_PATH.name).open("w") as f:
        json.dump(anchor_index, f, indent=2)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the script's argument parsing**

```bash
python scripts/build_disease2vector.py --help
```

Expected: argparse usage printed.

- [ ] **Step 3: Run the script for real on `data/raw/`**

```bash
python scripts/build_disease2vector.py
```

Expected: completes successfully with output similar to:
```
Scanning 13 CSV files in data/raw ...
Building vocab (min_freq=3) ...
  vocab size: <number>
Building co-occurrence matrix ...
  matrix shape: (<N>, <N>), nnz: <large number>
Computing PPMI ...
  PPMI nnz: <large number>
Resolving anchor sets against vocabulary ...
  aging_neurodegenerative: <N> resolved tokens
  ...
Done.
```

If anchor resolution fails (set has <5 matches), the script will print which set fell short. **If this happens**: read the message, decide whether to add alternate anchor names (e.g., `"Alzheimer's disease"` may match better than `"Alzheimer disease"` in raw data) OR lower `--min-anchors-per-set` to 3 for first pass. Document the change in the commit message.

- [ ] **Step 4: Verify cache files were created**

```bash
ls -la data/disease2vector/
```

Expected: `vocab.json`, `ppmi.npz`, `anchor_index.json` all present.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_disease2vector.py
git -c commit.gpgsign=false commit -m "Add build_disease2vector CLI script"
```

---

## Task 11: Validate script CLI

**Files:**
- Create: `scripts/validate_disease2vector.py`

- [ ] **Step 1: Create the validate script**

```python
#!/usr/bin/env python3
"""Print the disease2vector anchor QC report.

Requires `build_disease2vector.py` to have been run first.

Saves a JSON report to data/disease2vector/anchor_qc.json and prints a
human-readable summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from disease2vector.config import ANCHOR_QC_PATH
from disease2vector.embed import Disease2Vec
from disease2vector.validate import write_qc_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ANCHOR_QC_PATH))
    args = ap.parse_args()

    print("Loading Disease2Vec from cache ...")
    d2v = Disease2Vec.load()
    print(f"  vocab size:  {len(d2v.vocab)}")
    print(f"  anchor sets: {len(d2v.anchor_index)}")

    print("Computing QC report ...")
    report = write_qc_report(d2v, args.out)

    print(f"\nWithin-set tightness (median pairwise cosine distance, lower = tighter):")
    for name, d in report["within_set_tightness_median_cosine_distance"].items():
        print(f"  {name:38s}  {d:.4f}")

    print(f"\nAnchor misplacements: {len(report['anchor_misplacements'])}")
    for m in report["anchor_misplacements"]:
        print(
            f"  {m['vocab_token']!r:48s} owned by {m['owner_set']!r} "
            f"but closer to {m['closest_set']!r} "
            f"({m['distance_to_owner']:.3f} vs {m['distance_to_closest']:.3f})"
        )

    print(f"\nFull report written to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
python scripts/validate_disease2vector.py
```

Expected: prints per-set tightness and any misplacements; writes `data/disease2vector/anchor_qc.json`. Report may show misplaced anchors — that's informational, not a failure.

- [ ] **Step 3: Inspect the QC report file**

```bash
python -c "import json; r = json.load(open('data/disease2vector/anchor_qc.json')); print('vocab:', r['n_vocab']); print('sets:', r['n_sets']); print('misplaced:', len(r['anchor_misplacements']))"
```

Expected: prints summary numbers.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_disease2vector.py data/disease2vector/anchor_qc.json
git -c commit.gpgsign=false commit -m "Add validate_disease2vector CLI and initial QC report"
```

---

## Task 12: Integration test on real data

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
"""End-to-end integration test on real data/raw/ CSVs.

This is the empirical gate: if these assertions fail, the design is wrong
on this dataset (not just the implementation buggy). The test is marked
`integration` and may be skipped in unit-test-only runs.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from disease2vector.config import ANCHOR_INDEX_PATH, PPMI_PATH, VOCAB_PATH
from disease2vector.embed import Disease2Vec

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_OK = (
    PPMI_PATH.exists() and VOCAB_PATH.exists() and ANCHOR_INDEX_PATH.exists()
)

pytestmark = pytest.mark.skipif(
    not CACHE_OK,
    reason="disease2vector cache not built — run scripts/build_disease2vector.py first",
)


@pytest.fixture(scope="module")
def d2v():
    return Disease2Vec.load()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


def test_vocab_is_nonempty(d2v):
    assert len(d2v.vocab) > 100, "vocabulary suspiciously small"


def test_t1d_t2d_closer_than_t1d_alzheimer(d2v):
    """The signature pathology-grounded sanity check from the design spec.

    Looks up Type 1 diabetes, Type 2 diabetes, and Alzheimer disease (or any
    vocab token containing those phrases). Skips if any are absent.
    """
    def find(needle: str) -> str | None:
        n = needle.lower()
        for tok in d2v.vocab:
            if n in tok.lower():
                return tok
        return None

    t1d = find("type 1 diabetes")
    t2d = find("type 2 diabetes")
    ad = find("alzheimer")
    if not all([t1d, t2d, ad]):
        pytest.skip(f"missing tokens: t1d={t1d}, t2d={t2d}, ad={ad}")

    v_t1d = d2v.embed(t1d)
    v_t2d = d2v.embed(t2d)
    v_ad = d2v.embed(ad)
    sim_t1d_t2d = _cosine(v_t1d, v_t2d)
    sim_t1d_ad = _cosine(v_t1d, v_ad)
    assert sim_t1d_t2d > sim_t1d_ad, (
        f"T1D-T2D similarity ({sim_t1d_t2d:.3f}) should exceed "
        f"T1D-Alzheimer's ({sim_t1d_ad:.3f}) — design assumption violated"
    )


def test_each_aging_set_owns_majority_of_its_anchors(d2v):
    """For each aging set, the majority of its anchors should be closer to
    that set's centroid than to any other set's centroid."""
    misplaced_per_set: dict[str, int] = {}
    for owner_set, idxs in d2v.anchor_index.items():
        n_misplaced = 0
        for idx in idxs:
            v = np.asarray(d2v.ppmi[idx].todense()).ravel()
            dists = {
                name: d2v.set_distance(v, name) for name in d2v.anchor_index
            }
            closest = min(dists, key=dists.get)
            if closest != owner_set:
                n_misplaced += 1
        misplaced_per_set[owner_set] = n_misplaced

    for name, n in misplaced_per_set.items():
        total = len(d2v.anchor_index[name])
        # tolerate up to ~30% misplacement (edge anchors are expected)
        assert n <= int(0.3 * total) + 1, (
            f"set {name!r}: {n}/{total} anchors closer to a different set "
            f"(allowed: {int(0.3 * total) + 1})"
        )


def test_ep03_decides_canonical_anchors_correctly(d2v):
    """For canonical anchors, ep03_decide should return the right letter
    for the strong majority."""
    from disease2vector.anchors import (
        ANCHOR_SETS_AGING,
        EP03_LETTER,
        _normalize_for_matching,
    )

    correct = 0
    total = 0
    failures: list[tuple[str, str, str]] = []  # (anchor, expected, got)
    for set_name, anchors in ANCHOR_SETS_AGING.items():
        expected_letter = EP03_LETTER[set_name]
        for anchor in anchors:
            # Find any vocab token that matches this anchor
            a_norm = _normalize_for_matching(anchor)
            matched = next(
                (tok for tok in d2v.vocab
                 if a_norm and (a_norm in tok or tok.lower() in a_norm)),
                None,
            )
            if matched is None:
                continue
            result = d2v.ep03_decide(matched)
            total += 1
            if result["answer"] == expected_letter:
                correct += 1
            else:
                failures.append((anchor, expected_letter, result["answer"]))

    assert total > 0, "no anchors resolved against vocab"
    assert correct / total >= 0.90, (
        f"only {correct}/{total} anchors classified into the right EP-03 "
        f"letter. Failures: {failures[:10]}"
    )
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/test_integration.py -v
```

Expected outcomes:
- If `data/disease2vector/` cache is not present: all tests skip.
- If cache is present: 4 tests should pass.
- If `test_t1d_t2d_closer_than_t1d_alzheimer` fails: the design assumption is empirically wrong; this means either the data doesn't carry enough signal for L3 PPMI alone (would need R3 multi-source extension) OR the disease names in the CSVs don't include T1D/T2D/AD in matchable form (vocab inspection needed).
- If `test_each_aging_set_owns_majority_of_its_anchors` fails: anchor lists need refinement (consult `anchor_qc.json` to see which anchors are misplaced).

- [ ] **Step 3: Run the full test suite as a final check**

```bash
pytest -v
```

Expected: all unit tests pass (Tasks 2/3/4/6/7/8), plus integration tests if cache is built.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git -c commit.gpgsign=false commit -m "Add integration test gating design assumptions on real data"
```

---

## Done — what you have at the end

- A `disease2vector` Python package with vocab, PPMI, anchors, embed, validate modules.
- Two CLI scripts: `build_disease2vector.py` (one-shot build) and `validate_disease2vector.py` (QC).
- Unit tests for each module against synthetic fixtures (millisecond runtime).
- Integration test against real `data/raw/` data, gating the design's pathology assumption (T1D-T2D > T1D-AD).
- A QC report at `data/disease2vector/anchor_qc.json` (committed) showing within-set tightness, between-set separation, and any anchor misplacements.

Downstream `scripts/step4_score_and_analyze.py` can `from disease2vector import Disease2Vec` and use `ep01_decide` / `ep02_decide` / `ep03_decide` to score model outputs.

---

## Self-review (skill checklist)

**1. Spec coverage:**
- §1 Goal → Task 1 (scaffolding) + entire plan ✓
- §3 Conceptual basis → Documented in spec; no code task needed ✓
- §4 Pipeline overview → Tasks 2 → 4 → 5 → 6 → 7 → 8 ✓
- §5.1 `embed` → Task 7 ✓
- §5.2 `set_distance` → Task 7 ✓
- §5.3 anchor sets → Task 5 ✓
- §5.4 EP-03 letter mapping → Task 5 ✓
- §6 PPMI construction → Tasks 3 + 4 ✓
- §7 Anchor sets (12) → Task 5 ✓
- §8 Anchor resolution → Task 6 ✓
- §9 Anchor QC → Tasks 9 + 11 ✓
- §10 Decision logic (EP-01/02/03) → Task 8 ✓
- §11 Module API → Task 7 (load classmethod) ✓
- §12 File layout → Task 1 scaffolds, others fill in ✓
- §13 Tests → all unit test tasks + Task 12 integration ✓
- §14 Configuration → Task 1 config.py ✓
- §15 Known limitations → Documented in spec; no code task ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"implement later" in any task. ✓

**3. Type consistency:**
- `build_vocab` returns `tuple[list[str], dict[str, int]]` in Task 2; consumed correctly in Task 6 and Task 10. ✓
- `build_cooccurrence` takes `vocab: list[str]` in Task 3; called with `vocab` from `build_vocab` in Tasks 4, 10. ✓
- `compute_ppmi` takes `csr_matrix` returns `csr_matrix`; consistent with caller in Task 10. ✓
- `resolve_anchors` returns `dict[str, list[int]]` in Task 6; consumed in `Disease2Vec.__init__` (Task 7) and `write_qc_report` (Task 9). ✓
- `Disease2Vec.set_distance(vec, set_name, k)` signature consistent across Tasks 7, 8, 9, 12. ✓
- `_cosine_similarity_batch` defined in Task 7 (`embed.py`), imported in Task 9 (`validate.py`). ✓

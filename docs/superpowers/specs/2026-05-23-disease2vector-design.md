# disease2vector — Design Spec

**Date:** 2026-05-23
**Branch:** `disease2vector`
**Status:** Draft — pending user review

## 1. Goal

A helper module that embeds disease names into a vector space and answers three
benchmark scoring questions for the Longevity-LLM RNA evaluation suite (EP-01,
EP-02, EP-03). Closeness in the space reflects shared molecular etiology (T1D
and T2D close; T1D and Alzheimer's far). No clustering algorithm is used.

## 2. Non-goals

- Not a general-purpose biomedical NER tool.
- Not a replacement for hand-curated disease ontologies (DO/MeSH); we use them
  as inputs but do not reproduce them.
- Not a model-call dependency at scoring time — the module is purely numerical
  once built.
- Not handling rare/novel disease names not present in our data source.

## 3. Conceptual basis

Two papers ground the approach (`1301.3781v3.pdf`, `1607.00653v1.pdf`):

- **Word2Vec (Mikolov 2013)** — distributional hypothesis: items appearing in
  similar contexts end up near each other in vector space.
- **node2vec (Grover & Leskovec 2016)** — extends the same logic to graphs via
  biased random walks → Skip-gram.

We use the underlying *statistical logic* without the neural-net machinery.
Levy & Goldberg (2014) proved Skip-gram implicitly factorizes a shifted PMI
matrix; we compute Positive PMI (PPMI) directly. This gives a deterministic,
debuggable, ~50-line numerical pipeline.

Pathology grounding draws on the disease-network literature:
- Goh et al. (PNAS 2007) — *The human disease network*.
- Hidalgo, Blumm, Barabási, Christakis (PLOS CB 2009).
- Menche et al. (Science 2015) — *Disease modules in the interactome*.
- López-Otín et al. (Cell 2023) — *Hallmarks of aging: An expanding universe*.

The matrix we build is a **molecular comorbidity network**: two diseases share
mass in the matrix if the same RNA-modification-disrupting variants have been
independently associated with both.

## 4. Pipeline overview

```
data/raw/*.csv  ──► vocab + co-occurrence ──► PPMI matrix ──► embed(disease)
                                                                  │
                                                                  ▼
                          anchor sets (12)  ──────────►  set_distance(v, S, k=3)
                                                                  │
                                                                  ▼
                                              ep01_decide / ep02_decide / ep03_decide
```

## 5. The four primitives

### 5.1 `embed(disease) -> np.ndarray | None`

Returns the PPMI row vector for the disease, or `None` if not in vocabulary.
Vocabulary out-of-band signals EP-02 "no association" (class C).

### 5.2 `set_distance(vec, anchor_set, k=3) -> float`

Mean cosine distance from `vec` to its `k` closest anchors in the set. k-NN
rather than centroid because anchor sets are internally heterogeneous (the
solid-cancer set contains both breast and glioblastoma); centroids would
average that signal away.

### 5.3 12 anchor sets (see §7)

Hard-coded dictionary. 8 aging-side + 4 non-aging-side reference sets.

### 5.4 EP-03 letter mapping

Hard-coded dict from aging-set name → EP-03 letter (A/B/C). Multiple sets can
map to the same letter — this is intentional for geometric resolution.

## 6. PPMI construction (v1 vector space)

### 6.1 Vocabulary

Scan all 13 CSVs in `data/raw/`. For each row, parse the `Disease_association`
cell:
- Lowercase, strip leading/trailing whitespace.
- Split on `;` to get the disease bag for that row.
- Light normalization on each token: collapse multiple spaces, strip trailing
  periods, normalize unicode dashes to ASCII hyphens. **No deeper normalization
  in v1.**
- Drop empty strings.
- Tokens appearing in fewer than `MIN_FREQ = 3` rows are discarded (too noisy
  to embed). This threshold is in `config.py`.

Vocabulary `V` = the set of surviving disease tokens.

### 6.2 Co-occurrence matrix

Build symmetric `C ∈ ℝ^{|V|×|V|}` (sparse, scipy CSR):
- Deduplicate each row's disease bag to a set `Dr`.
- For each row, for each ordered pair `(a, b) ∈ Dr × Dr` (including `a == b`),
  `C[a, b] += 1`. The diagonal `C[a, a]` is the marginal frequency of disease
  `a` (number of rows it appears in).
- A row with a single disease contributes `C[a, a] += 1` only — no off-diagonal
  contribution.

### 6.3 PPMI

```
N        = sum of C
P_ab     = C[a, b] / N
P_a      = C[a, a] / N         # marginal from the diagonal
PMI[a,b] = log( P_ab / (P_a · P_b) )
PPMI[a,b]= max(0, PMI[a,b])
```

Empty rows (a disease with no positive PPMI co-occurrences) are flagged in QC
output but kept in the matrix.

### 6.4 Vector form

Each disease's vector is its row of the PPMI matrix. Cosine similarity is used
for distance (not Euclidean) because the vector magnitude reflects how
well-studied a disease is — we want the *profile* of associations, not their
volume.

### 6.5 R3 extension (v2, deferred)

The module's `embed()` interface is the only entry point downstream code uses.
This makes it straightforward to later swap or augment the PPMI vector with
per-disease feature blocks (gene/HPO/pathway/ontology) and a TruncatedSVD
projection, without changing any scoring code. Out of scope for v1.

## 7. Anchor sets

8 aging + 4 non-aging. Each anchor is the canonical name; matching is
case-insensitive and tolerates trivial variants (e.g. `Alzheimer's disease` ↔
`Alzheimer disease`).

### 7.1 Aging side (each maps to an EP-03 letter)

| Set | EP-03 | Anchors |
|---|---|---|
| `aging_neurodegenerative` | **A** | Alzheimer disease, Parkinson disease, Frontotemporal dementia, Lewy body dementia, Vascular dementia, Amyotrophic lateral sclerosis, Mild cognitive impairment, Progressive supranuclear palsy, Multiple system atrophy, Age-related cognitive decline |
| `aging_cardiovascular` | **B** | Atherosclerosis, Coronary artery disease, Heart failure, Myocardial infarction, Essential hypertension, Aortic aneurysm, Atrial fibrillation, Ischemic stroke, Peripheral artery disease |
| `aging_metabolic` | **B** | Type 2 diabetes mellitus, Metabolic syndrome, Non-alcoholic fatty liver disease, Obesity, Dyslipidemia, Insulin resistance, Gout |
| `aging_musculoskeletal` | **B** | Osteoporosis, Sarcopenia, Osteoarthritis, Frailty syndrome, Age-related muscle atrophy, Degenerative disc disease, Spinal stenosis |
| `aging_fibrosis_tissue` | **C** | Idiopathic pulmonary fibrosis, Liver cirrhosis, Renal fibrosis, Cardiac fibrosis, Skin photoaging, Hepatic fibrosis, Systemic sclerosis |
| `aging_cancer_solid` | **C** | Breast cancer, Prostate cancer, Colorectal cancer, Lung cancer, Pancreatic cancer, Gastric cancer, Hepatocellular carcinoma, Bladder cancer, Renal cell carcinoma, Glioblastoma |
| `aging_cancer_hematologic` | **C** | Acute myeloid leukemia, Myelodysplastic syndrome, Chronic lymphocytic leukemia, Multiple myeloma, Diffuse large B-cell lymphoma, Clonal hematopoiesis of indeterminate potential |
| `aging_organ_decline` | **C** | Chronic kidney disease, Chronic obstructive pulmonary disease, Age-related macular degeneration, Cataract, Presbycusis, Benign prostatic hyperplasia, Diabetic retinopathy |

### 7.2 Non-aging side (reference centroids; no EP-03 letter)

| Set | Anchors |
|---|---|
| `non_aging_congenital_mendelian` | Cystic fibrosis, Sickle cell anemia, Duchenne muscular dystrophy, Phenylketonuria, Tay-Sachs disease, Hemophilia, Beta thalassemia, Spinal muscular atrophy, Marfan syndrome, Achondroplasia, Galactosemia |
| `non_aging_autoimmune` | Rheumatoid arthritis, Systemic lupus erythematosus, Multiple sclerosis, Inflammatory bowel disease, Type 1 diabetes mellitus, Celiac disease, Psoriasis, Hashimoto thyroiditis, Graves disease |
| `non_aging_infectious` | Tuberculosis, HIV infection, Hepatitis B, Hepatitis C, Malaria, Influenza, COVID-19, Pneumonia, Sepsis |
| `non_aging_psychiatric_neurodev` | Schizophrenia, Bipolar disorder, Major depressive disorder, Autism spectrum disorder, Attention deficit hyperactivity disorder, Intellectual disability, Anxiety disorder |

Anchors are intentionally exemplars, not exhaustive. Each set has 7–11 members
so that k=3 nearest-neighbor distances are meaningful but no single member
dominates.

## 8. Anchor → vocabulary resolution

Anchors are written in canonical form. At module-build time we run a matcher
to find which raw vocabulary tokens correspond to each anchor:

1. Lowercase both sides.
2. Strip diacritics and punctuation; remove trailing modifiers (`disease`,
   `disorder`, `syndrome`) for matching only — original strings preserved.
3. Match candidates from vocab using either:
   - **exact** equality on the normalized form, or
   - **bidirectional substring**: anchor normalized form is a substring of the
     vocab token, *or* the vocab token is a substring of the anchor. This
     catches both `"type 2 diabetes mellitus"` ↔ `"type 2 diabetes"` and
     `"alzheimer"` ↔ `"alzheimer's disease early onset"`.
4. Per anchor, log:
   - matched vocab tokens (with row counts in original data)
   - unmatched anchors (warning)
   - ambiguous matches (multiple vocab strings — pick the highest-count one,
     log the rejected alternatives)
5. An anchor resolves to the union of all its matched vocab tokens. A single
   anchor may pull in several vocab strings; their PPMI rows are *averaged*
   to form that anchor's effective vector.

If a set ends up with **< 5 resolved anchors**, the build fails with a clear
error — that set's centroid would be unreliable. Build script lists the
shortfall and the candidate vocab terms it considered.

## 9. Anchor validation (sanity check, runs after PPMI build)

Self-consistency report saved to `data/disease2vector/anchor_qc.json`:

- **Within-set tightness**: median pairwise cosine distance among anchors of
  the same set. Smaller is better.
- **Between-set separation**: median cosine distance between centroids of
  different sets. Larger is better.
- **Anchor misplacement**: for each anchor `a` in set `S`, check whether
  `set_distance(v_a, S)` is the minimum across all 12 sets. Flag (do not
  auto-fix) anchors that are closer to a different set than their own.

The build does not auto-adjust; it logs. The user decides whether to refine
the anchor lists in `anchors.py` based on the QC report.

## 10. Decision logic

### 10.1 EP-01 — binary aging?

```python
def ep01_decide(disease: str) -> dict:
    v = embed(disease)
    if v is None:
        return {"answer": "B", "reason": "not_in_vocab"}
    d_aging    = min(set_distance(v, S) for S in AGING_SETS.values())
    d_nonaging = min(set_distance(v, S) for S in NON_AGING_SETS.values())
    answer = "A" if d_aging < d_nonaging else "B"
    return {"answer": answer, "d_aging": d_aging, "d_nonaging": d_nonaging}
```

Rationale: `min over sets` not `mean over sets` — a disease only needs to be
near *some* aging anchor set to be aging, not equidistant to all of them.

### 10.2 EP-02 — ternary (aging+cancer / non-aging / no association)

```python
def ep02_decide(disease_bag: list[str]) -> dict:
    in_vocab = [d for d in disease_bag if embed(d) is not None]
    if not in_vocab:
        return {"answer": "C", "reason": "no_disease_in_vocab"}
    if any(ep01_decide(d)["answer"] == "A" for d in in_vocab):
        return {"answer": "A"}
    return {"answer": "B"}
```

The `C` (no association) class is identified by **vocabulary membership only**,
not by geometric position. PPMI space cannot distinguish "no data" from
"distant" — only vocabulary membership can.

### 10.3 EP-03 — multiclass aging subcategory

```python
def ep03_decide(disease: str) -> dict:
    v = embed(disease)
    if v is None:
        return {"answer": None, "reason": "not_in_vocab"}
    set_dists = {name: set_distance(v, S) for name, S in AGING_SETS.items()}
    letter_dist = {"A": inf, "B": inf, "C": inf}
    for name, d in set_dists.items():
        letter_dist[EP03_LETTER[name]] = min(letter_dist[EP03_LETTER[name]], d)
    answer = min(letter_dist, key=letter_dist.get)
    margin = sorted(letter_dist.values())[1] - sorted(letter_dist.values())[0]
    return {"answer": answer, "letter_dist": letter_dist, "margin": margin}
```

`margin` is exposed so downstream scoring can weight by confidence
(small margin = borderline case, model error more forgivable).

## 11. Module API

```python
from disease2vector import Disease2Vec

d2v = Disease2Vec.load()   # loads cached PPMI + anchors

d2v.embed("Alzheimer disease")          # → np.ndarray or None
d2v.set_distance(vec, "aging_neurodegenerative")  # → float
d2v.ep01_decide("Alzheimer disease")    # → {"answer": "A", ...}
d2v.ep02_decide(["Type 2 diabetes mellitus", "Alzheimer disease"])
d2v.ep03_decide("Type 2 diabetes mellitus")
```

## 12. File layout

```
disease2vector/
    __init__.py
    config.py            # constants: MIN_FREQ, K_NN, paths
    vocab.py             # vocabulary extraction + normalization
    ppmi.py              # PPMI matrix builder
    anchors.py           # ANCHOR_SETS dict, EP03_LETTER mapping
    embed.py             # Disease2Vec class: embed, set_distance, decide_*
    validate.py          # anchor QC report
scripts/
    build_disease2vector.py     # CLI: build PPMI + anchors → cache
    validate_disease2vector.py  # CLI: print QC report
data/disease2vector/
    vocab.json           # cached vocabulary (built artifact, gitignored)
    ppmi.npz             # cached sparse PPMI matrix (gitignored)
    anchor_qc.json       # QC report (committed for inspection)
tests/
    test_vocab.py
    test_ppmi.py
    test_anchors.py
    test_decide.py
```

`data/disease2vector/*.npz` and `*.json` (except `anchor_qc.json`) are added to
`.gitignore` — they are rebuildable from `data/raw/`.

## 13. Tests

Each test targets one primitive. All use a tiny synthetic vocab fixture
(8 fake diseases, 20 fake rows) so they run in milliseconds.

- `test_vocab.py` — vocabulary extraction, semicolon split, normalization,
  `MIN_FREQ` filtering.
- `test_ppmi.py` — co-occurrence counts on synthetic data; PMI math against
  hand-computed values; clipping behavior; sparse output sanity.
- `test_anchors.py` — anchor-to-vocab resolution on synthetic data;
  build failure when set has <5 resolved members.
- `test_decide.py` — synthetic vocab placed in known geometry; all three
  `decide_*` functions return the expected answer.

One **integration test** runs the build on real `data/raw/` and asserts:
- T1D–T2D cosine similarity > T1D–Alzheimer cosine similarity (sanity check
  this approach actually works on real data).
- Every aging anchor set's centroid is closer to its own members (median) than
  to any other set's centroid.
- EP-03 decisions for the canonical anchors return the right letter for ≥90%
  of anchors.

This integration test is the gate — if it fails, the design is empirically
wrong, not just the implementation.

## 14. Configuration knobs (centralized in `config.py`)

| Constant | Default | Meaning |
|---|---|---|
| `MIN_FREQ` | `3` | Drop vocab tokens appearing in fewer rows |
| `K_NN` | `3` | k in set_distance k-NN |
| `MIN_ANCHORS_PER_SET` | `5` | Build fails below this |
| `DATA_DIR` | `data/raw` | Source CSVs |
| `CACHE_DIR` | `data/disease2vector` | Build artifacts |

## 15. Known limitations (to document in module README and writeup)

- Disease vocabulary limited to terms in our 13 CSV files. Out-of-vocab
  diseases return `None` from `embed()`.
- Symmetric similarity — does not capture causal/temporal relations
  (diabetes → nephropathy is not directional in this space).
- Phenotypic similarity without shared genetics is not captured.
- Anchor sets are exemplars; edge-of-category diseases may sit between sets
  with low margins. The exposed `margin` field surfaces this.
- v1 uses only SNP co-occurrence; v2 with multi-source features (HPO,
  DisGeNET, pathway) is a planned extension via the same `embed()` interface.

## 16. What this design does *not* commit to

- Whether scoring uses the decisions as hard predictions or soft signals.
  The module returns rich dicts (`answer`, `d_aging`, `d_nonaging`, `margin`,
  per-set distances) so downstream scoring code in `scripts/step4_*.py` can
  pick its policy.
- Whether anchor sets get refined iteratively after seeing the QC report.
  The QC report is informational; manual tuning of `anchors.py` is allowed
  but not required.

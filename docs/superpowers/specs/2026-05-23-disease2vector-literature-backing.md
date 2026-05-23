# disease2vector — Literature Backing

**Companion to:** `2026-05-23-disease2vector-design.md`
**Purpose:** Map each pathology and methodology claim in the design to peer-reviewed primary literature. Flag where we extend beyond direct precedent.

## Summary of support

| Claim | Status | Primary reference(s) |
|---|---|---|
| Disease-disease networks built from shared molecular features reflect real pathobiology | **STRONG** | Goh 2007 PNAS · Menche 2015 *Science* · Park 2009 *MSB* · Hidalgo 2009 *PLoS CB* |
| Shared genetic variants (SNPs) across diseases carry pathology-relevant signal | **STRONG** | Solovieff 2013 *Nat Rev Genet* · cross-disease GWAS literature |
| PMI / PPMI is a valid similarity measure for biomedical concepts | **STRONG** | Levy & Goldberg 2014 NIPS · Beam 2018 PSB · Choi 2016 KDD |
| Aging diseases cluster by shared mechanisms (hallmarks of aging) | **STRONG** | López-Otín 2023 *Cell* |
| Source data (RMDisease, RNADisease) is peer-reviewed | **STRONG** | RMDisease v2.0 NAR 2023 · RNADisease v4.0 NAR 2023 |
| The specific 12-anchor-set partition + k-NN scoring | **OUR EXTENSION** | Grounded in disease-module literature but operationally novel |

---

## Claim 1 — Disease similarity from shared molecular features is a valid pathology concept

This is the central conceptual claim: that two diseases sharing molecular signals (genes, variants, interactions) reflect genuine pathobiological relatedness. Strongly established.

### Primary references

**Goh, Cusick, Valle, Childs, Vidal, Barabási (2007). "The human disease network." *PNAS* 104(21):8685–8690.** PMID: 17502601. DOI: [10.1073/pnas.0701361104](https://www.pnas.org/doi/10.1073/pnas.0701361104).

Quote from the abstract: *"Genes associated with similar disorders show both higher likelihood of physical interactions between their products and higher expression profiling similarity for their transcripts, supporting the existence of distinct disease-specific functional modules."* This is the canonical paper that founded "network medicine" and the diseasome concept. It is exactly the model we apply: diseases as nodes, shared molecular features as edges, communities emerge that match clinical disease classes.

**Menche, Sharma, Kitsak, Ghiassian, Vidal, Loscalzo, Barabási (2015). "Uncovering disease-disease relationships through the incomplete interactome." *Science* 347(6224):1257601.** PMID: 25700523. DOI: [10.1126/science.1257601](https://www.science.org/doi/10.1126/science.1257601).

The most important paper for our claim. The authors derived mathematical conditions for identifiability of disease modules in the human protein-protein interaction network and showed: *"diseases with overlapping network modules show significant coexpression patterns, symptom similarity, and comorbidity, whereas diseases residing in separated network neighborhoods are phenotypically distinct."*

This is the empirical backing for the central methodological claim of disease2vector: **distance in a molecular-similarity space predicts clinical relatedness.**

**Park, Lee, Christakis, Barabási (2009). "The impact of cellular networks on disease comorbidity." *Molecular Systems Biology* 5:262.** PMID: 19357641. DOI: [10.1038/msb.2009.16](https://www.embopress.org/doi/10.1038/msb.2009.16).

Combined cellular-interaction data with Medicare claims for 32M patients. Found *statistically significant correlations between the underlying structure of cellular networks and disease comorbidity patterns in the human population*. Closes the loop: shared cellular machinery → clinical comorbidity in real patients.

**Hidalgo, Blumm, Barabási, Christakis (2009). "A dynamic network approach for the study of human phenotypes." *PLoS Computational Biology* 5(4):e1000353.** PMID: 19360091. DOI: [10.1371/journal.pcbi.1000353](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000353).

Built a Phenotypic Disease Network (PDN) from MedPAR records on 30M+ elderly patients. The network's community structure recovers clinical disease groupings without supervision — i.e., the pathobiological signal in comorbidity data is strong enough to reconstruct disease taxonomy from co-occurrence alone. This is the most direct precedent for what disease2vector does, just with patient-comorbidity rather than variant co-occurrence as the link.

### How we use this claim

- Our PPMI matrix is mathematically a *molecular comorbidity network*: two diseases share mass if RNA-modification-disrupting variants have been independently associated with both. This is the variant-level analog of Hidalgo's patient-level PDN.
- Our anchor sets are operationally similar to Menche's disease modules — pre-defined disease groupings whose internal cohesion and external separation we test empirically.

---

## Claim 2 — Shared SNPs / variants carry pathology-relevant similarity

We use variant-level co-occurrence as our edge signal. The biological premise is that two diseases sharing many associated variants must share underlying mechanisms (pleiotropy, shared pathways).

### Primary reference

**Solovieff, Cotsapas, Lee, Purcell, Smoller (2013). "Pleiotropy in complex traits: challenges and strategies." *Nature Reviews Genetics* 14:483–495.** PMID: 23752797. DOI: [10.1038/nrg3461](https://www.nature.com/articles/nrg3461).

Authoritative review noting that *"genome-wide association studies have identified many variants that each affect multiple traits, particularly across autoimmune diseases, cancers and neuropsychiatric disorders, suggesting that pleiotropic effects on human complex traits may be widespread."* Establishes pleiotropy as a real, pervasive phenomenon — not noise.

### Cross-disease GWAS evidence

**Sharma et al. (2021). "Untangling the genetic link between type 1 and type 2 diabetes using functional genomics." *Scientific Reports* 11:13871.** DOI: [10.1038/s41598-021-93346-x](https://www.nature.com/articles/s41598-021-93346-x).

Quantitative finding directly relevant to our integration test: **T1D and T2D share 195 pleiotropic genes** (modulated by tissue-specific eQTLs associated with both), and 3 shared causal SNP regions (CTRB1/2, SH2B3, HLA). The pleiotropic genes are enriched in *inflammatory and metabolic pathways*.

### How we use this claim

- The CSVs in `data/raw/*.csv` are filtered to SNPs that disrupt RNA modification sites. When two diseases co-occur in the same `Disease_association` cell, the same SNP has been independently associated with both — a strict pleiotropic signal.
- **Honest caveat:** Sharma 2021 finds that T1D and T2D, while pleiotropic, are driven by *largely independent genetic signals* at the SNP level. Our integration test (`T1D-T2D cosine similarity > T1D-Alzheimer's cosine similarity`) should pass because Alzheimer's shares even less with T1D, but the margin may be small. Tasks 12's integration test is robust to this — it only requires that T1D-T2D be *closer than* T1D-AD, not that they be *very close*.

---

## Claim 3 — PMI / PPMI is a valid similarity measure for biomedical concepts

The methodology choice. We compute Positive Pointwise Mutual Information directly from co-occurrence counts. This is established practice in biomedical concept embedding.

### Theoretical foundation

**Levy & Goldberg (2014). "Neural word embedding as implicit matrix factorization." NIPS 27:2177–2185.** Paper: [neurips.cc paper 5477](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization).

The classic result: *"SGNS [Skip-gram with Negative Sampling] is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant."* Furthermore, *"using a sparse Shifted Positive PMI word-context matrix to represent words improves results on two word similarity tasks and one of two analogy tasks."*

This is the theoretical bridge from "Word2Vec on disease co-occurrence" to "directly compute PPMI on the disease co-occurrence matrix." We chose the latter for the same reasons Levy & Goldberg recommend: deterministic, debuggable, no training loop, often equal-or-better empirical performance.

### Biomedical applications

**Beam, Kompa, Schmaltz, Fried, Weber, Palmer, Shi, Cai, Kohane (2018). "Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data." *Pacific Symposium on Biocomputing* 25:295–306.** PMID: 31797605. PMC: [PMC6922053](https://pmc.ncbi.nlm.nih.gov/articles/PMC6922053/). arXiv: [1804.01486](https://arxiv.org/abs/1804.01486).

Combined a 60M-member insurance claims database, 20M clinical notes, and 1.7M biomedical journal articles to produce embeddings for 108,477 medical concepts (the largest set of biomedical embeddings at publication). Methodology: PMI-based concept co-occurrence. Validates that PMI on biomedical co-occurrence corpora yields embeddings that capture clinical similarity.

**Choi, Bahadori, Searles, Coffey, Thompson, Bost, Tejedor-Sojo, Sun (2016). "Multi-layer Representation Learning for Medical Concepts." *KDD 2016*:1495–1504.** Paper: [dl.acm.org/doi/10.1145/2939672.2939823](https://dl.acm.org/doi/10.1145/2939672.2939823). arXiv: [1602.05568](https://arxiv.org/abs/1602.05568).

Med2Vec — learned embeddings for medical codes and visits from EHR data using within-visit co-occurrence. Clinical experts confirmed the embedding similarity matched clinical relatedness. Same intuition as our work: diseases that co-occur (in EHR visits there, in shared SNP rows here) share underlying clinical meaning.

### How we use this claim

- Our PPMI computation in `disease2vector/ppmi.py` is the exact mathematical operation Levy & Goldberg identified as the implicit target of Word2Vec.
- We skip the neural-net training step because (a) Levy & Goldberg show direct PPMI is competitive, (b) determinism matters for benchmark scoring, (c) the corpus is small enough that training adds noise rather than removes it.

---

## Claim 4 — Aging diseases cluster by shared mechanisms (Hallmarks of Aging)

Our 8 aging anchor sets are organized along hallmark axes (proteostasis loss → neurodegeneration; senescence → fibrosis; mitochondrial decline → metabolic; etc.).

### Primary reference

**López-Otín, Blasco, Partridge, Serrano, Kroemer (2023). "Hallmarks of aging: An expanding universe." *Cell* 186(2):243–278.** PMID: 36599349. DOI: [10.1016/j.cell.2022.11.001](https://www.sciencedirect.com/science/article/pii/S0092867422013770).

Defines the 12 hallmarks: genomic instability, telomere attrition, epigenetic alterations, loss of proteostasis, disabled macroautophagy, deregulated nutrient-sensing, mitochondrial dysfunction, cellular senescence, stem cell exhaustion, altered intercellular communication, chronic inflammation, dysbiosis. Each hallmark fulfills three criteria: age-associated manifestation, experimental acceleration of aging, and therapeutic deceleration on intervention.

### How we use this claim

| Our anchor set | Primary hallmarks |
|---|---|
| `aging_neurodegenerative` | Loss of proteostasis, stem cell exhaustion |
| `aging_cardiovascular` | Altered intercellular communication, chronic inflammation |
| `aging_metabolic` | Deregulated nutrient sensing, mitochondrial dysfunction |
| `aging_musculoskeletal` | Stem cell exhaustion, cellular senescence |
| `aging_fibrosis_tissue` | Cellular senescence, chronic inflammation |
| `aging_cancer_solid` | Genomic instability, epigenetic alterations |
| `aging_cancer_hematologic` | Genomic instability (clonal hematopoiesis) |
| `aging_organ_decline` | Stem cell exhaustion, telomere attrition |

This mapping is *interpretive* — the anchor sets are chosen so that each one is dominated by a different combination of hallmarks. It is consistent with López-Otín 2023 but the assignment is a design decision, not a direct citation.

---

## Claim 5 — Source data is peer-reviewed and authoritative

Both data sources are from established epitranscriptomics groups, published in *Nucleic Acids Research* (the canonical journal for biological databases).

**Chen et al. (2023). "RMDisease V2.0: an updated database of genetic variants that affect RNA modifications with disease and trait implication." *Nucleic Acids Research* 51(D1):D1388–D1396.** DOI: [10.1093/nar/gkac750](https://academic.oup.com/nar/article/51/D1/D1388/6691863). Database: [rnamd.org/rmdisease2](http://www.rnamd.org/rmdisease2/index.html).

1,366,252 RNA-modification-associated variants across 16 modification types in 20 organisms; 14,749 disease-associated variants. The 13 `*_human_associatedSNPs.csv` files in `data/raw/` are direct exports of the disease-associated subset.

**Chen, Lin, Hu, Ye, Yao, Wu, Zhang, Wang, Deng, Guo, et al. (2023). "RNADisease v4.0: an updated resource of RNA-associated diseases." *Nucleic Acids Research* 51(D1):D1397–D1404.** DOI: [10.1093/nar/gkac814](https://academic.oup.com/nar/article/51/D1/D1397/6711138). PMC: [PMC9825423](https://pmc.ncbi.nlm.nih.gov/articles/PMC9825423/).

3,428,058 RNA-disease entries covering 18 RNA types, 117 species, 4,090 diseases. The source of EP-01 and EP-03 prompts.

---

## Our extensions beyond direct literature precedent

Honestly flagging where we go beyond what's published verbatim:

1. **PPMI applied specifically to RMDisease's SNP-disease bag co-occurrence.** No published paper has built a PPMI matrix from RMDisease's `Disease_association` column to our knowledge. This is novel application of established methods to a specific data source. The underlying logic (variant-level co-occurrence → disease similarity) is established (Park 2009, Menche 2015, Solovieff 2013); the operationalization is ours.

2. **The 12-anchor partition.** No published taxonomy of RNA-modification-related diseases uses exactly these 8+4 groupings. The aging side aligns with the hallmarks framework (López-Otín 2023) and with the EP-03 prompt categories (which are the benchmark's own design). The non-aging side draws from MeSH C-tree and Disease Ontology top-level categories.

3. **k-NN-to-anchor-set, not centroid, scoring.** Reading the closest k anchors instead of computing a centroid is uncommon for disease classification (most published work uses centroids). Our rationale (anchor sets are heterogeneous; centroids dilute signal) is justified by the heterogeneity of cancer types within `aging_cancer_solid` etc.

These extensions are defensible: each rests on multiple peer-reviewed papers and the design rationale is documented in the spec.

---

## Risks / known weak points

The reviewer's concern about pathology grounding has two specific weaknesses worth surfacing:

### Risk 1 — PPMI on a sparse corpus is noisy for rare diseases

Diseases appearing in fewer than ~10 rows will have very sparse PPMI vectors. Their k-NN distances to anchor sets are dominated by which 1–2 anchors happen to share their few co-occurring diseases — high variance.

**Mitigation:** `MIN_FREQ = 3` filter drops the very-rare tail. Diseases just above the threshold are flagged via the QC report's within-set tightness metric. This is also why we chose k=3 rather than k=1: k=1 amplifies noise; k=3 averages over multiple neighbors.

### Risk 2 — Integration test may be borderline

Sharma 2021 found T1D and T2D are driven by *largely independent* SNP-level signals despite sharing 195 pleiotropic genes. Our integration test asserts `cos(T1D, T2D) > cos(T1D, AD)`, which should hold (Alzheimer's shares even less with T1D), but the margin could be small.

**Mitigation:** The integration test (Task 12) checks several pairs, not just T1D-T2D. The `test_each_aging_set_owns_majority_of_its_anchors` test tolerates 30% misplacement — anchored in the empirical reality that disease similarity is fuzzy at the edges. If the T1D-T2D comparison fails, the QC report will identify the failure mode (sparse data vs. genuine biological surprise) and we can adjust by either adding stronger T1D-T2D-shared diseases as bridges or by raising `MIN_FREQ` to drop noisier diseases.

### Risk 3 — Anchor selection is human-curated

We chose ~107 anchors by hand. A reviewer can fairly ask "what if the anchor list is wrong?"

**Mitigation:**
- All anchors are drawn from canonical disease names listed in EP-03 prompts or in the hallmarks-of-aging review. None are idiosyncratic to us.
- The QC report (`anchor_qc.json`) makes any misplacement visible. Anchors that don't actually behave as anchors in the PPMI geometry are reported, not silently averaged.
- Anchor set membership can be revised post-build without rebuilding PPMI. The module's main artifact (PPMI matrix) is anchor-independent.

---

## One-paragraph defense for the reviewer

> **disease2vector** computes a sparse Positive Pointwise Mutual Information matrix over disease names that co-occur as associations of the same RNA-modification-disrupting variants in RMDisease v2.0 (Chen et al., *Nucleic Acids Research* 2023). The conceptual basis — that diseases sharing molecular signals cluster into clinically meaningful modules — is established by the network medicine literature (Goh et al., *PNAS* 2007; Menche et al., *Science* 2015; Park et al., *MSB* 2009; Hidalgo et al., *PLoS Comp Biol* 2009), which empirically demonstrated that disease-disease similarity computed from shared genes, interactions, or comorbidity predicts clinical relatedness, symptom overlap, and shared drug targets. PMI is the matrix that neural word embeddings (Word2Vec) implicitly factorize (Levy & Goldberg, *NIPS* 2014) and has been applied directly to biomedical concept embedding on 60M-patient claims data (Beam et al., *PSB* 2018) and EHR records (Choi et al., *KDD* 2016). Our anchor sets follow the hallmarks-of-aging framework (López-Otín et al., *Cell* 2023). The novelty is operational, not conceptual: we apply an established statistical primitive (PPMI) to an established data source (RMDisease) using an established disease-similarity-network framing (Goh/Menche/Park) and an established aging-disease taxonomy (hallmarks). Each component has been peer-reviewed; we combine them.

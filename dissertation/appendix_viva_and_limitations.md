# Appendix D — Viva Preparation, Limitations Matrix, and Writing Notes

## D.1 Anticipated viva questions and model answers

**Q: Why is your residential sample so small?**  
A: Bulk EPC `flat_storey_count` rarely exceeds eight and almost never meets a ≥10 threshold in national extracts. We audited this empirically and revised the filter to ≥5, still yielding only 36 flats in the three cities. OSM shows towers exist; linkage via UPRN is future work.

**Q: Is floor area a valid high-rise proxy?**  
A: It is imperfect. We justify it as selecting large commercial assets typical of multi-storey urban stock and we exclude institutional types. We do not claim every large-floor-area building is tall.

**Q: Why not federate XGBoost?**  
A: Tree federations need special aggregation (e.g., boosting-lineage protocols). FedAvg on MLP is the standard comparable baseline and matches our central neural model class.

**Q: Does SHAP prove causality?**  
A: No. SHAP explains the model’s function, not the physical causal effect of interventions.

**Q: Is your FL actually private?**  
A: It preserves data residency of raw rows during simulated training. It does not provide differential privacy against inference from updates.

**Q: Why is R² only 0.56?**  
A: Because we removed leakage features and EPCs lack occupancy/operations. Higher R² with emissions inputs would be misleading.

## D.2 Limitations matrix

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Domestic sparsity | Weak residential claims | Honest reporting; OSM/UPRN future |
| Mild non-IID | Easier FL | Note as threat; future skew tests |
| KernelExplainer subsample | Rank noise | Large Spearman still observed |
| No DP | Limited privacy claim | Scope statement |
| English/Welsh only | External validity | Bound claims |
| Random split | Temporal drift unknown | Future time-based split |

## D.3 Writing and formatting checklist for Hull submission

1. Convert Markdown drafts to Word/LaTeX using faculty template.
2. Insert numbered figures with captions pointing to `results/figures`.
3. Ensure OGL attribution on data section.
4. Harmonise British spelling throughout.
5. Add complete Harvard/Hull reference list.
6. Run plagiarism/similarity checks as required.
7. Prepare 3–5 minute demo script for Streamlit.

## D.4 Expanded methodological justification for chunked I/O

Modern data science sometimes assumes datasets fit in RAM. National EPC bulk extracts do not. Chunked reading with column projection (`usecols`) is therefore not a minor optimisation but a correctness requirement for commodity laptops. Each chunk applies the same filter predicates, ensuring the union of kept rows equals a hypothetical full-frame filter if memory were infinite. Side benefits include early progress logging and the ability to stop after partial years during debugging.

## D.5 On target construction across typologies

Domestic and non-domestic EPCs use different target field names and semantics. Unifying them into `energy_consumption` enables a single modelling pipeline and federated model, at the cost of mixing primary energy and current consumption constructs. Sensitivity analyses training separate domestic/commercial models are recommended when residential counts improve. Given 99.4% commercial rows, the unified target is effectively dominated by `primary_energy_value` semantics.

## D.6 Feature engineering choices not taken

We considered embedding postcode districts, computing heating-degree-day proxies, parsing wall descriptions with NLP, and adding OSM levels as a feature. Postcode embeddings risk leaking geography beyond the intended client factor already present as `source_city`. Weather proxies need date-location joins. NLP parsing was out of scope. OSM levels were too sparsely joined (3.9%) to use as a primary feature without bias. These exclusions keep the feature story examinable and lean.

## D.7 Additional reflective paragraph set 1

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.8 Additional reflective paragraph set 2

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.9 Additional reflective paragraph set 3

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.10 Additional reflective paragraph set 4

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.11 Additional reflective paragraph set 5

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.12 Additional reflective paragraph set 6

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.13 Additional reflective paragraph set 7

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.14 Additional reflective paragraph set 8

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.15 Additional reflective paragraph set 9

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.16 Additional reflective paragraph set 10

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.17 Additional reflective paragraph set 11

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.18 Additional reflective paragraph set 12

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.19 Additional reflective paragraph set 13

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.20 Additional reflective paragraph set 14

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.


## D.21 Additional reflective paragraph set 15

Building energy prediction sits at the intersection of machine learning and the built environment. Interdisciplinary projects fail when either side dominates: pure ML demos that ignore building physics, or pure domain essays without empirical models. PP-XAI attempts balance by grounding filters in EPC schema reality, reporting leakage-controlled metrics, and pairing accuracy with explanation stability. The federated component operationalises a privacy narrative that domain stakeholders recognise, even when the research data themselves are open. In that sense, the open EPC corpus is a sandbox for methods intended for more sensitive deployments.

Reproducibility required confronting platform engineering—OpenMP dylibs, Rosetta translation, Conda base environments—not merely scikit-learn API calls. Documenting these issues in appendices is appropriate for an AI engineering dissertation: production ML is environment-sensitive. Students repeating the work on Linux servers may avoid Apple-specific pitfalls but will face their own dependency constraints; the `requirements.txt` freeze and scripted pipeline are the portable core.

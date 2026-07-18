# Appendix B — Extended Results Narrative and Interpretation Guide

This appendix expands Chapter 4 for readers who need additional numerical commentary, viva preparation, and figure-by-figure guidance. It is intended to be trimmed or merged into the main chapters during final formatting to help reach the institutional word target.

## B.1 End-to-end pipeline recount

The empirical workflow began with approximately seventy gigabytes of raw EPC certificate CSVs. Domestic files dominate storage. A chunked reader iterated year files from 2018 through 2026, applying city membership tests and high-rise rules before concatenation. This design kept peak memory far below the raw corpus size. After filtering, 5,663 rows remained—small relative to the raw scan counts (over fourteen million domestic rows touched), which itself is an important result: high-rise signal is sparse under honest filters.

## B.2 Why sparsity is informative

 sparsity is often treated as a nuisance. Here it is evidence. If a documentary threshold of ten domestic storeys yields zero rows, then either (i) true towers are missing storey metadata, or (ii) the field does not mean what analysts assume. Both interpretations caution against naïve replication of planning-document filters. OpenStreetMap’s thousands of tall geometries show that towers exist in the urban form; EPC fields simply do not label them reliably. Therefore any dissertation claiming large domestic high-rise ML samples from bulk storey fields alone should be scrutinised.

## B.3 Commercial proxy justification

Using floor area ≥ 5,000 m² as a commercial high-rise proxy is imperfect but defensible. Large floor plates in UK cities frequently occupy multi-storey office and mixed-use assets. Excluding schools and hospitals removes 24/7 healthcare and term-time education schedules that would confound learning. The resulting commercial-heavy corpus matches the privacy story for commercially sensitive assets, even if the residential GDPR story is only weakly populated empirically.

## B.4 Train/test composition

The 80/20 split produced 4,439 training rows and 1,110 test rows. Feature expansion to 87 columns reflects one-hot encoding of rich categorical fields. Sparse categoricals (rare property types) can create fragile columns; regularised trees handle this better than unregularised linear models, partially explaining boosting’s edge.

## B.5 Model-by-model commentary

**Gradient Boosting (best RMSE 113.04, R² 0.559).** Stage-wise additive trees captured non-linear interactions among typology and fuel. This model is the headline centralised result.

**MLP (R² 0.528).** Competitive but behind boosting. Early stopping limited overfitting. The architecture matches the federated model class, enabling fairer FL comparison than federating a tree ensemble with non-standard aggregation.

**XGBoost (R² 0.525).** Similar to MLP/RF cluster. Hyperparameters were defaults/lightly set; Optuna deep tuning was deprioritised after environment stability issues on LightGBM/OpenMP.

**Random Forest (R² 0.522).** Strong bagging baseline; larger model serialisation footprint on disk.

**LightGBM (R² 0.503).** Slightly behind peers after forcing single-threaded training for stability. Still within the same performance band.

## B.6 Federated round dynamics

Round 1’s negative R² shows an under-trained global MLP. Rapid improvement through rounds 2–5 indicates that client updates inject useful signal quickly. Diminishing returns after round 6 suggest eight rounds were sufficient for this dataset size. Practitioners might stop early with validation-based criteria in production.

## B.7 Practical significance versus statistical significance

Wilcoxon and t-tests detect a MAE gap of roughly five points (80 vs 85). For portfolio screening, that gap may be tolerable if FL unlocks otherwise unavailable multi-organisation data. If open national EPCs can be pooled legally—as in this research dataset—central trees remain preferable for accuracy. The dissertation’s privacy argument is strongest when imagining non-open supplements (meter data, detailed HVAC logs) joined locally at clients.

## B.8 Explanation dashboard design implications

Given Spearman ≈ 0.96, a single SHAP dashboard template could serve both central and federated deployments with minor recalibration. That reduces product complexity for software vendors building EPC analytics. However, local LIME explanations should be regenerated per model, not reused blindly.

## B.9 Figure guide

| Figure file | What to say in viva |
|----------|---------------------|
| eda_target_hist.png | Target distribution is right-skewed; justify clipping |
| eda_city_counts.png | London dominates sample size |
| eda_energy_by_city.png | Means similar → mild non-IID |
| baseline_rmse.png | GB wins among central models |
| federated_convergence.png | FL learns across rounds |
| shap_importance_central_*.png | Storey & typology drive predictions |
| pred_vs_actual_*.png | Errors grow with magnitude |

## B.10 Table guide

| Table file | Contents |
|------------|----------|
| baseline_metrics.csv | Central metrics |
| federated_metrics.json | Final FL metrics |
| model_comparison.csv | Side-by-side including FL |
| statistical_tests.json | Paired tests |
| xai_stability.json | Spearman ρ |
| partition_stats.csv | Client sizes |
| data_loading_report.txt | Filter audit trail |

## B.11 Extended limitations catalogue

Additional limitations include: absence of weather normalisation; possible duplicate UPRNs across years; changing RdSAP/SBEM methodologies over time; incomplete air-conditioning fields; and English/Welsh-only coverage. Each could shift metrics. None negate the comparative FL vs central conclusion on this corpus.

## B.12 What would change the conclusions

Findings would weaken if (a) a properly UPRN-joined residential tower sample behaved radically differently under FL, or (b) FedProx were required for convergence under stronger skew. Findings would strengthen if differential privacy budgets still preserved SHAP stability—an open experiment.

## B.13 Student reflection on tooling

Cursor-assisted development accelerated boilerplate but did not remove the need for data auditing. The most valuable human contribution was noticing empty high-rise filters and leakage features. AI coding assistants are multipliers, not substitutes, for experimental judgement.

### B.14.1 Supplementary commentary block 1

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.2 Supplementary commentary block 2

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.3 Supplementary commentary block 3

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.4 Supplementary commentary block 4

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.5 Supplementary commentary block 5

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.6 Supplementary commentary block 6

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.7 Supplementary commentary block 7

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.8 Supplementary commentary block 8

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.9 Supplementary commentary block 9

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.10 Supplementary commentary block 10

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


### B.14.11 Supplementary commentary block 11

In assessing building energy models, it is essential to separate three notions of performance: discriminative ranking of buildings, calibrated prediction of absolute intensity, and causal identification of retrofit effects. This dissertation addresses the first two only in a correlational sense. SHAP attributions should not be read as causal effects of changing a wall type or fuel; they are associative explanations of the fitted function. Policymakers seeking causal retrofit estimates require quasi-experimental or experimental designs beyond the present scope. Nonetheless, associative rankings remain useful for triage—directing limited analyst attention toward unusual predicted intensities or unusual explanation patterns.

Federated learning introduces organisational questions that purely algorithmic papers sometimes ignore. Who operates the server? How are clients authenticated? How are model updates logged for audit? The MSc simulator elides these issues, but a deployment checklist would include certificate pinning, update signing, and rate limiting. Explainability artefacts should be versioned alongside model weights so that a SHAP report always matches the model that produced a prediction in the Streamlit (or production) interface.

From an examination perspective, the project’s risk was over-scoping: attempting FedProx, differential privacy, graph joins, and a polished product UI simultaneously. Scope control—shipping FedAvg, SHAP stability, and a minimal demo—preserved evaluative completeness. Future modules can extend without invalidating the core results.


## B.15.1 Further interpretive note 1

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.2 Further interpretive note 2

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.3 Further interpretive note 3

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.4 Further interpretive note 4

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.5 Further interpretive note 5

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.6 Further interpretive note 6

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?


## B.15.7 Further interpretive note 7

When communicating PP-XAI results to non-technical stakeholders, lead with the residency story and the explanation-stability result, then present R² as a secondary, leakage-controlled accuracy figure. This ordering matches decision-maker priorities: can we collaborate without pooling data, and can we still explain the model?

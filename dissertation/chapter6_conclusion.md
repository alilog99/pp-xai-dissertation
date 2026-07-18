# Chapter 6 — Conclusion and Future Work

## 6.1 Summary

This dissertation designed, implemented, and evaluated PP-XAI: a privacy-preserving explainable AI pipeline for predicting high-rise building energy performance from UK EPC certificates. The work spanned data auditing and high-rise filtering, centralised baselines, Federated Averaging across London, Manchester, and Birmingham clients, SHAP/LIME explainability, statistical comparison, and a Streamlit prototype.

## 6.2 Answers to research questions

1. **RQ1:** Federated Averaging produced an MLP with R² ≈ 0.52, closely matching the centralised MLP (≈ 0.53) and approaching the best centralised gradient boosting model (≈ 0.56), without pooling raw client CSVs during federated rounds.
2. **RQ2:** Global SHAP rankings were highly stable across centralised and federated models (Spearman ≈ 0.96).
3. **RQ3:** Storey count, property-type categories, and fuel type dominated attributions, consistent across settings.
4. **RQ4:** Stable federated explanations offer a practical pathway toward transparent energy analytics under data-residency constraints, subject to the limitations in Chapter 5.

## 6.3 Contributions restated

- Schema-aware, audit-based high-rise EPC filters and non-leaky feature design.
- Reproducible codebase (`src/`, `scripts/`, `results/`) for centralised vs federated comparison.
- Quantitative XAI stability evidence linking FL and explainability evaluation.
- Interactive demonstration artefact for viva and stakeholder communication.

## 6.4 Future work

1. **OS Open UPRN / coordinate join** to attach OSM `building:levels` to EPC rows at scale, recovering true residential towers.
2. **FedProx, more clients, and stronger non-IID** partitions; full Flower server deployment.
3. **Differential privacy** and secure aggregation for update-level protection.
4. **TreeExplainer** and grouped SHAP for cleaner typology narratives.
5. **Dedicated domestic models** if storey metadata improve or UPRN enrichment succeeds.
6. **Expand references and polish** toward the formal ~15,000-word submission (deadline 27 August 2026; hard stop 10 September 2026).

## 6.5 Closing remark

Privacy and explainability need not be traded away entirely for predictive performance. On this UK high-rise EPC corpus, federated neural training remained competitive with centralised neural baselines, and explanations remained aligned. That combination is the core empirical message of PP-XAI.


## 6.6 Reflection on learning outcomes

The project integrated data engineering (chunked I/O on ~70GB raw CSVs), classical ML, deep learning for FL, XAI tooling, statistical testing, and web prototyping. Failures—empty high-rise filters, OpenMP hangs, Rosetta/NumPy conflicts—were converted into documented constraints and fixes, which is itself an AI engineering competency.

## 6.7 Submission artefacts checklist

- [x] Code under `src/` and `scripts/`
- [x] Results tables/figures
- [x] Streamlit app
- [x] Chapter drafts + abstract
- [ ] Formal Word/LaTeX formatting per Hull guidelines
- [ ] Complete reference list in required style
- [ ] Supervisor review and video (if required)


## 6.8 Final sentence

PP-XAI demonstrates that privacy-preserving training and explainable outputs can coexist with competitive predictive performance on UK high-rise EPC analytics, provided researchers remain honest about data limitations and leakage risks.

# Live Speaker Notes — 15 Minutes

**Deck:** [`202440724_PPXAI_Presentation.pptx`](202440724_PPXAI_Presentation.pptx)  
**Student ID:** 202440724 (use ID only on slides / submission filenames)  
**Target:** ≤15 minutes talk + demo; then Q&A  

**Overlength reminder:** ≤10% over = no penalty; 10–20% over = 10% mark penalty; >50% over = unmarked.

---

## Timing overview

| Clock | Slides | Focus |
|------:|--------|--------|
| 0:00–1:00 | 1–2 | Title + problem hook |
| 1:00–2:30 | 3–4 | Aim / RQs + **options analysis** |
| 2:30–4:30 | 5–7 | Data, pipeline, FL architecture |
| 4:30–7:00 | 8–11 | Results + critical stats + XAI |
| 7:00–10:00 | 12 | **LIVE DEMO** (Streamlit) |
| 10:00–12:30 | 13–14 | Discussion, conclusions |
| 12:30–15:00 | — | Buffer / early Q&A invite |

Aim to finish content by ~12:30 so you are safely under 15:00.

---

## Slide-by-slide script

### 1 — Title (~45s)
“This dissertation project is **PP-XAI**: privacy-preserving explainable AI for building energy performance prediction. I combine **federated learning** with **SHAP and LIME** so we can predict energy intensity from UK EPCs **without pooling raw microdata**, and still explain the predictions.”

### 2 — The Problem (~70s)
“UK net-zero needs better insight into building energy. EPCs are a rich open source — but for high-rise residential and large commercial assets, councils and landlords often **cannot share** microdata. Centralised machine learning assumes pooling is allowed. That is the gap. The design triangle on the right is the response: **privacy, accuracy, and explainability** together — not accuracy alone.”

### 3 — Aim & RQs (~60s)
“**Aim:** build and evaluate a federated + XAI pipeline on UK EPC data for high-rise / large-asset prediction.  
**RQ1** asks whether FL can approach centralised accuracy with data kept local.  
**RQ2–3** ask whether explanations stay stable and consistent across cities.  
**RQ4** asks how this supports transparency expectations, including EU AI Act themes.  
Objectives run from baselines → FedAvg → SHAP/LIME → statistics → Streamlit prototype.”

### 4 — Options Analysis (~80s) — REQUIRED
“This slide is the deliberate design trail.  
I **compared** centralised and federated training — not federated alone.  
**FedAvg** is the primary FL algorithm; FedProx was evaluated and essentially tied.  
For FL I use an **MLP**, because FedAvg averages parameters — it does not natively aggregate tree ensembles. Trees remain strong **centralised** baselines.  
For XAI I use **SHAP plus LIME** — global rankings and local instance stories.  
**Differential privacy and secure aggregation** were considered and **deferred** as future hardening — out of MSc scope, but honestly flagged.”

### 5 — Data (~75s)
“I scanned roughly **13.8 million** domestic and **841 thousand** non-domestic certificates. After high-rise filters across London, Manchester, and Birmingham, the modelling corpus is **5,663** records.  
Domestic flats use storey count **≥ 5** — because **≥ 10** yields essentially zero rows in bulk exports. Non-domestic uses floor area **≥ 5,000 m²**.  
**Critical honesty:** the corpus is **99.4% commercial**; only **36** residential flats. That is a data-schema limitation, not a rhetorical one. I also excluded CO₂ and asset-rating fields that leaked the target and inflated R² above 0.95.”

### 6 — Pipeline (~60s)
“End-to-end: ingest open EPCs → filter → preprocess → **two training paths** — centralised baselines and federated MLP → SHAP/LIME → evaluation → Streamlit demonstrator.”

### 7 — FL Architecture (~60s)
“Three geographic clients. Each trains locally; the server aggregates **weights only** — raw CSVs never leave the partition. Shared model: **87 → 128 → 64 → 1**, eight FedAvg rounds, sample-size weighted aggregation.”

### 8 — Centralised Results (~50s)
“Among five centralised models, **gradient boosting** leads: RMSE about **113**, R² about **0.56**. That is moderate but **credible** — without leakage features.”

### 9 — Federated Results (~60s)
“Federated MLP converges from a poor round-1 start to **R² ≈ 0.52** by round 8 — essentially matching the **centralised MLP** (≈ 0.53). Hypothesis H1 is met: federated RMSE is within about **1%** of the central neural baseline.”

### 10 — Critical Stats (~70s)
“Wilcoxon and paired t-tests show the GB–Fed gap is **statistically significant**. But Cohen’s **d ≈ 0.06** is negligible, and bootstrap RMSE confidence intervals **overlap**. So: significant, but **practically small**. If pooling is allowed, central GB still wins; if pooling is restricted, FedAvg is the relevant benchmark.”

### 11 — XAI (~60s)
“SHAP mean absolute profiles between central GB and federated MLP correlate at Spearman **ρ ≈ 0.96**. Cross-city top-five Jaccard averages **0.78**. Dominant drivers include storey count, property type, and natural gas. Privacy-preserving training did **not** arbitrarily reshuffle explanations.”

### 12 — LIVE DEMO (~2–3 min)
**Pause slides. Switch to browser.**

```bash
conda deactivate   # if prompt shows (base)
/bin/bash scripts/run_streamlit.sh
# → http://localhost:8501
```

Talk track while demoing:
1. “This is the stakeholder prototype.”
2. Enter **building A** (e.g. London office-like inputs) → show prediction.
3. Point to **SHAP** importance / transparency note (EU AI Act Art. 13 / H4).
4. Enter **building B** with different city or typology — show the output changes (non-scripted feel).
5. Optional third input if time.

Return to slides.

### 13 — Discussion (~75s)
“The central trade-off is **accuracy versus data residency**. Explanation stability reduces the fear that FL produces an uninterpretable model. Limitations: commercial proxy dominance; simulator is not production secure FL; KernelSHAP depends on background samples. Strength: schema honesty and leakage ablation.”

### 14 — Conclusions (~50s)
“Federated neural accuracy approaches the central neural baseline; explanations remain stable; the Streamlit app makes this tangible. Contributions: the PP-XAI pipeline, FL–XAI stability evidence, and an honest high-rise schema audit. Future work: UPRN–OSM residential linkage, DP-SGD, secure aggregation. Thank you — happy to take questions.”

---

## Demo prep checklist (Technical Demonstration 10%)

- [ ] `source scripts/env.sh && streamlit run src/webapp/app.py` starts without errors
- [ ] Preprocessor + model load; prediction returns a number
- [ ] SHAP / importance panel visible
- [ ] Two or three ready input sets (different cities / types) — prefer live typing over a fixed script
- [ ] Close unrelated apps; disable notifications; zoom browser to ~110% for readability
- [ ] Backup: screenshots of a successful run if the network/env fails (use only as last resort)

## Q&A prompts you can offer

- Why MLP for FL rather than federated trees?
- Why is the residential count so low?
- What would DP change?
- How would local authorities deploy this?

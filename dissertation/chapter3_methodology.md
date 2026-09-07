# Chapter 3 — Methodology

## 3.1 Research design overview

The study follows a comparative experimental design. A single filtered EPC corpus is (a) trained centrally with multiple regressors and (b) trained federatively with an MLP under FedAvg across three geographic clients. Predictive metrics and SHAP-based explanation stability are compared on a held-out test set. A Streamlit prototype demonstrates stakeholder-facing use. The design privileges reproducibility: all scripts live under `scripts/`, core libraries under `src/`, and artefacts under `results/`.

Hypotheses:

- **H1:** Federated Averaging achieves test RMSE within 5% of the centralised MLP on the same held-out set.
- **H2:** Spearman rank correlation of mean |SHAP| profiles between the centralised Gradient Boosting model and the federated MLP exceeds 0.85.
- **H3:** Pairwise Jaccard similarity of per-client top-5 SHAP feature sets exceeds 0.60 on average (primary pass criterion). A secondary check requires that at least four of the five leading features appear in all three city clients.
- **H4:** The Streamlit prototype can present SHAP/LIME outputs with an EU AI Act Article 13 transparency note.

Figure 3.1 summarises the end-to-end pipeline from open data ingestion through filtering, centralised and federated training, XAI, evaluation, and the Streamlit demonstrator. Figure 3.2 details the FedAvg client–server architecture. Figure 3.3 shows the EnergyMLP layer stack used as the shared neural model.

## 3.2 Data sources, access, and licensing

All primary modelling data are **publicly downloadable**; no restricted or paid datasets are required to reproduce the pipeline.

| Dataset | Provider / portal | Licence | How obtained in this project | Local path (gitignored if large) |
|---------|-------------------|---------|------------------------------|----------------------------------|
| Domestic EPC certificates (England & Wales), annual bulk CSVs ≈2018–2026 | Ministry of Housing, Communities and Local Government (MHCLG) / successor via **Energy Performance of Buildings Data** portal | Open Government Licence v3.0 | Register at the official portal; download domestic certificate bulk files by year | `raw-data/domestic-csv/certificates-YYYY.csv` |
| Non-domestic EPC certificates, annual bulk CSVs | Same portal | OGL v3.0 | Download non-domestic certificate bulk files by year | `raw-data/non-domestic-csv/certificates-YYYY.csv` |
| OpenStreetMap building footprints with `building:levels ≥ 10` (London, Manchester, Birmingham) | OpenStreetMap contributors via Overpass API | Open Database Licence (ODbL) | `scripts/download_osm.py` (Overpass queries) | `data/raw/osm/*.geojson` |

**Canonical download URL (EPC):** https://get-energy-performance-data.communities.gov.uk/  
(The legacy `epc.opendatacommunities.org` bulk site was retired; use the current Communities portal above.)

**Attribution.** Analyses must acknowledge MHCLG (or successor) EPC open data under OGL v3.0 and OSM contributors under ODbL. Raw multi-gigabyte CSVs are **not** redistributed in the GitHub repository; collaborators must download them themselves (see `docs/DATA_LICENCE.md` and `docs/REPRODUCIBILITY.md`).

**What is not used.** EPC *recommendations* CSVs are excluded: prediction targets and modelling features reside on certificates only. Synthetic generators exist only for dry-run testing when raw files are absent and are flagged if used.

**Scope years.** Certificate years approximately 2018–2026 were scanned in chunked mode because domestic national files exceed several gigabytes per year.

## 3.3 High-rise filtering and targets

Chunked loading is implemented in `src/data/data_loader.py` with constants in `src/data/column_definitions.py` matching **underscore** bulk headers.

| Typology | Inclusion rules | Target variable |
|----------|-----------------|-----------------|
| Domestic | `property_type == Flat`; `flat_storey_count >= 5`; local authority in client mapping | `energy_consumption_current` |
| Non-domestic | `floor_area >= 5000`; exclude education/health/worship/sports keywords; client LA | `primary_energy_value` |

Unified modelling column: `energy_consumption`. Extreme or non-positive targets were removed. Client assignment uses `local_authority_label` mapped to London / Manchester / Birmingham authority lists.

**Leakage control:** `co2_per_area` and `asset_or_env_score` were excluded from features after early experiments showed unrealistically high R² when included.

## 3.4 Preprocessing

`EPCPreprocessor` (`src/data/preprocessor.py`) applies:

1. Frame preparation and outlier clipping of the target at the 1st–99th percentiles.
2. Median imputation for numeric features (`floor_area`, `storey_count`).
3. Most-frequent imputation and one-hot encoding for categoricals (typology, property type, fuel, heating, efficiencies, glazing, aircon, city).
4. Standard scaling of numeric columns.
5. 80/20 train–test split (city-stratified when feasible).

After clipping, 5,549 of 5,663 filtered records remain; the split yields 4,439 train / 1,110 test rows with 87 encoded features. The fitted `ColumnTransformer` is persisted to `data/processed/preprocessor.joblib`. Each client CSV is transformed with the same fitted pipeline for FL.

## 3.5 Federated partitioning

To simulate realistic FL conditions, the filtered corpus is partitioned by `source_city` into London, Manchester, and Birmingham clients (sample sizes ≈ 3,449 / 1,160 / 1,054). Mild label skew (city mean energies ≈ 234–243 kWh/m²/year) and strong sample-size imbalance exercise weighted FedAvg without adversarial non-IID extremes.

## 3.6 Centralised models

Models in `src/models/centralised_models.py`:

- Random Forest (200 trees)
- Gradient Boosting (sklearn)
- XGBoost
- LightGBM (`n_jobs=1` for macOS OpenMP stability)
- MLPRegressor (128–64 hidden units, early stopping)

Metrics: RMSE, MAE, R², MAPE. The best model by RMSE (Gradient Boosting) is used for the primary KernelSHAP comparison versus the federated MLP. A supplementary figure compares federated weighted SHAP with centralised Random Forest attributions.

## 3.7 Federated learning protocol and model architecture

An MLP (`EnergyMLP` in `src/federated/fl_client.py`) is trained with an in-process FedAvg simulator for eight rounds, four local epochs per round, Adam optimiser, and MSE loss. Client updates are averaged with weights proportional to local sample sizes. FedProx (μ = 0.01) is evaluated as a robustness variant. A Flower `NumPyClient` wrapper supports future full-server deployments; reported results use the reproducible simulator.

**Architecture (Figure 3.3).** Input dimension equals the encoded feature width (87). Hidden layers are Linear(87→128)–ReLU–Dropout(0.1), then Linear(128→64)–ReLU–Dropout(0.1), then Linear(64→1). The same topology is used for the centralised neural baseline to keep the FL comparison fair. Trees remain centralised competitors because native FedAvg averages parameters, not tree structures.

**Communication pattern (Figure 3.2).** Each round: (1) server broadcasts global weights; (2) each city client trains locally on its rows only; (3) clients return updated weights; (4) server aggregates. Raw EPC microdata never leave the client partition during federated rounds.

## 3.8 Explainability protocol

On a shared subsample (background ≈ 80 training rows; explain ≈ 60 test rows):

- SHAP KernelExplainer for the best centralised model (Gradient Boosting) and the federated MLP (`scripts/run_xai.py`)
- Mean |SHAP| ranking exported to CSV/PNG under `results/figures/`
- LIME tabular explanations for three instances per model
- Stability (H2): Spearman correlation between Gradient Boosting and federated MLP mean |SHAP| vectors (`results/tables/xai_stability.json`)
- Supplementary RF comparison: federated weighted SHAP versus centralised Random Forest (`scripts/run_federated_shap.py`; not the H2 test statistic)
- **H3:** per-client TreeExplainer SHAP on the centralised Gradient Boosting model applied separately to each city matrix; pairwise Jaccard similarity of top-5 feature sets (`scripts/run_h3_per_client_shap.py`)

## 3.9 Statistical tests

Paired Wilcoxon signed-rank and paired t-tests compare absolute errors of the best centralised model versus federated MLP predictions on the identical test set (`src/evaluation/metrics.py`). Cohen’s d on absolute errors and bootstrap percentile 95% confidence intervals for RMSE (2,000 resamples) are also reported (`scripts/run_bootstrap_ci.py`).

## 3.10 Prototype

`src/webapp/app.py` loads the preprocessor and best centralised model, accepts building inputs, returns a predicted energy intensity, and displays global SHAP importance. Launch:

```bash
arch -arm64 /bin/zsh -c 'source scripts/env.sh && streamlit run src/webapp/app.py'
```

## 3.11 Ethical considerations

Only open EPC certificate fields and OSM building tags were used. No attempts were made to re-identify individuals beyond data already published. Ethical approval was submitted under University of Hull Faculty processes for low-risk secondary open-data analysis (Appendix C).

## 3.12 Implementation environment and reproducibility

Development used Python 3.13 on macOS (Apple Silicon), with dependencies in `requirements.txt`. Key libraries: pandas, scikit-learn, XGBoost, LightGBM, PyTorch, Flower, SHAP, LIME, Streamlit, SciPy. Checklist: (1) place CSVs under `raw-data/`; (2) `source scripts/env.sh`; (3) run `scripts/run_all.sh` or staged scripts; (4) confirm `results/tables/model_comparison.csv` and `xai_stability.json`; (5) launch Streamlit. Full notes: `docs/REPRODUCIBILITY.md`.

## 3.13 Summary

Chapter 3 defined publicly accessible data sources, high-rise filters, leakage controls, centralised and federated model architectures (Figures 3.1–3.3), XAI protocol, and tests. Chapter 4 reports outcomes on the local EPC corpus.

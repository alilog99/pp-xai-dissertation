# Chapter 4 — Results

*All numeric results below correspond to the pipeline artefacts in `results/tables/` and `results/figures/` as of the final successful run.*

## 4.1 Dataset construction outcomes

Chunked scanning processed **13,794,680** domestic and **840,760** non-domestic certificate rows (2018–2026). After high-rise and geography filters:

| Partition | Records | Mean energy |
|-----------|---------|-------------|
| London | 3,449 | 234.0 |
| Manchester | 1,160 | 242.1 |
| Birmingham | 1,054 | 243.1 |
| **Combined** | **5,663** | **237.4** (std 194.7) |

Typology mix: commercial **5,627 (99.4%)**, residential **36 (0.6%)**. The domestic mid/high-rise proxy yields very few flats with recorded storey count ≥ 5 in the three cities—confirming the bulk-schema limitation discussed in Chapter 1. Non-IID ratio of city means ≈ **1.04** (mild heterogeneity).

OSM high-rise geometries: London **1,758**, Manchester **340**, Birmingham **159**. Postcode join to EPC matched **222 / 5,663 (3.9%)** rows—useful for documentation but insufficient as a primary residential filter without UPRN enrichment.

Figures: `results/figures/eda_target_hist.png`, `eda_city_counts.png`, `eda_energy_by_city.png`, `eda_area_vs_energy.png`.

## 4.2 Preprocessing summary

After cleaning and split: **4,439** training and **1,110** test samples with **87** one-hot-expanded features. Client matrices: London 3,379; Manchester 1,136; Birmingham 1,033 transformed rows.

## 4.3 Centralised baseline performance (RQ1 context)

| Model | RMSE | MAE | R² | MAPE (%) |
|-------|------|-----|-----|----------|
| **Gradient Boosting** | **113.04** | **80.02** | **0.559** | 47.30 |
| MLP | 117.04 | 83.56 | 0.528 | 46.76 |
| XGBoost | 117.36 | 80.95 | 0.525 | 46.46 |
| Random Forest | 117.75 | 82.66 | 0.522 | 47.30 |
| LightGBM | 120.12 | 84.13 | 0.503 | 47.79 |

Best centralised model by RMSE: **gradient_boosting**. Absolute accuracy is moderate (R² ≈ 0.56), which is plausible for cross-sectional EPC primary energy prediction without emissions leakage features, occupancy, or weather. Figures: `baseline_rmse.png`, `baseline_r2.png`, `pred_vs_actual_gradient_boosting.png`.

## 4.4 Federated Averaging performance (RQ1)

FedAvg MLP over eight rounds:

| Round | Test RMSE | Test R² |
|-------|-----------|---------|
| 1 | 206.1 | −0.47 |
| 2 | 143.8 | 0.29 |
| 3 | 129.6 | 0.42 |
| 4 | 122.6 | 0.48 |
| 5 | 119.8 | 0.51 |
| 6 | 119.2 | 0.51 |
| 7 | 118.7 | 0.51 |
| 8 | **118.3** | **0.517** |

Final federated metrics: RMSE **118.32**, MAE **84.92**, R² **0.517**, MAPE **49.57%**. The federated MLP nearly matches the centralised MLP (R² 0.528) and trails the best tree model by a modest margin. **RQ1 answer:** Yes—FedAvg can approach centralised neural performance on this geographic split without pooling raw client data during training. Convergence: `federated_convergence.png`.

## 4.5 Statistical comparison

Paired tests on absolute errors (gradient boosting vs federated MLP):

- Wilcoxon p ≈ **1.06×10⁻⁵**
- Paired t-test p ≈ **4.98×10⁻⁵**
- MAE: 80.02 (central GB) vs 84.92 (federated)

The gap is statistically significant but small relative to the privacy benefit narrative: federated training forgoes a few points of MAE to keep microdata local.

## 4.6 Explainability results (RQ2, RQ3)

SHAP mean |SHAP| Spearman correlation between centralised gradient boosting and federated MLP: **0.956** (p ≈ 5.7×10⁻⁴⁷). Top centralised drivers include `storey_count`, hotel/office/storage property-type indicators, and natural gas fuel—aligned with building-physics intuition that typology and vertical configuration matter for large buildings. LIME instance plots were generated for both models under `results/figures/lime_*`.

**RQ2:** Federated explanations are highly stable relative to centralised SHAP rankings on this corpus.  
**RQ3:** Dominant features are storey-related numerics and property/fuel categoricals; rankings remain consistent across training regimes.

## 4.7 Prototype demonstration (Objective 5)

The Streamlit app at `src/webapp/app.py` loads artefacts and serves interactive predictions with SHAP tables/figures. During verification it was launched successfully at `http://localhost:8501`.

## 4.8 Summary of empirical answers

| RQ | Empirical finding |
|----|-------------------|
| RQ1 | FedAvg MLP R² 0.52 ≈ central MLP; close to best GB 0.56 |
| RQ2 | SHAP Spearman 0.96 between central and federated |
| RQ3 | Storey count, property type, fuel dominate |
| RQ4 | Discussed in Chapter 5 as governance implication of stable XAI under FL |


## 4.9 Qualitative walkthrough of SHAP leaders

The leading mean |SHAP| features for centralised gradient boosting included numeric `storey_count` and categorical indicators for hotels, offices/workshops, storage/distribution, and natural gas. This pattern suggests the model separates large hospitality and office assets from other commercial uses and uses vertical configuration as a strong signal. Floor area’s mean |SHAP| ranked lower than storey count after encoding interactions—reminding us that importance is model- and encoding-dependent, not a pure physical ranking.

## 4.10 Error analysis notes

Residual plots (`pred_vs_actual_*.png`) show greater absolute errors at high energy intensities, typical of heteroscedastic energy data. Federated predictions exhibit slightly higher MAE, consistent with the statistical tests. No evidence was found of systematic city-only collapse (the model retains useful signal across clients), but per-city error tables remain a useful appendix for viva preparation.

## 4.11 Negative results and discarded paths

Using domestic `flat_storey_count >= 10` produced empty residential sets. Including emissions features produced spuriously high R²; those runs were discarded from the reported results. Full Flower multi-process deployment was stubbed but not required for the reported FedAvg metrics.


## 4.12 Chapter summary

Results show competitive federated neural performance, superior central boosting, and high SHAP stability, with a commercial-dominated sample and sparse domestic high-rise labels. Discussion in Chapter 5 interprets these findings against privacy and policy goals.

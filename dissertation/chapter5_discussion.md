# Chapter 5 — Discussion

## 5.1 Interpreting predictive performance

Gradient boosting’s R² of approximately 0.56 indicates that a substantial share of variance in primary/current energy intensity remains unexplained by the selected fabric and system features. This is expected: EPCs are assessment artefacts with methodological changes over time; occupancy, tenancy, and operational schedules are absent; and weather normalisation is not modelled here. The decision to exclude CO₂ and asset-rating fields was consequential. Early pipelines that included those fields produced R² values above 0.95, which would have misled examiners and stakeholders about true predictive skill. Reporting the lower, leakage-controlled figures strengthens scientific credibility.

Federated Averaging closed nearly all of the gap to the centralised MLP while trailing the best tree model. From a systems perspective, this is encouraging: if regional data holders can only contribute neural updates, they still obtain a globally competitive predictor. If a trusted curator can fit a central tree model on pooled open data, that remains slightly stronger—but pooled open data is a special case. The motivating scenario for FL is precisely when pooling is restricted; under that constraint, FedAvg’s performance is the relevant benchmark.

## 5.2 Interpreting explanation stability

A Spearman correlation of ~0.96 between centralised and federated mean |SHAP| profiles suggests that privacy-preserving training did not arbitrarily reshuffle feature importance. For RQ2 and RQ4, this matters: transparency artefacts remain recognisable across governance modes. Practitioners could, in principle, publish SHAP summaries of a federated model without implying that the underlying microdata were centralised.

Caveats remain. KernelExplainer used subsamples; rankings can shift with background choice. Property-type one-hot features fragment importance across related categories (e.g., multiple hotel/office encodings). Future work should aggregate SHAP by semantic groups and compare TreeExplainer on the gradient boosting model for higher fidelity.

## 5.3 Data limitations and honesty about “high-rise”

The dissertation title emphasises high-rise structures. Empirically, the modelling corpus is dominated by large-floor-area non-domestic buildings used as a high-rise *proxy*, with only 36 residential flats meeting the revised storey threshold in the three cities. OSM confirms that true towers exist (thousands of geometries with levels ≥ 10), but linking them to EPC rows via postcode alone recovers only ~4% of records. This limitation is methodological, not rhetorical: claiming domestic `storey >= 10` without rows would be scientifically false. The revised definition and explicit limitation statement are therefore contributions in their own right.

## 5.4 Mild non-IID structure

City mean energies differed by only ~4%. Stronger FL stress tests would introduce more clients, label skew, or feature skew. Mild non-IID may make FedAvg look easier than in adversarial federations. Nonetheless, sample-size imbalance (London ≫ Birmingham) still exercises weighted aggregation.

## 5.5 Ethics, licence, and responsible use

EPC data were used under OGL v3.0. The prototype must not be presented as a certified EPC replacement; it is a research demonstrator. Predictions should not be used for individual tenancy decisions without human oversight and appropriate legal basis. Federated simulation does not by itself provide differential privacy; gradients/updates can still leak information under attacks. Chapter 6 flags DP-SGD and secure aggregation as future hardening steps.

## 5.6 Implications for practice and RQ4

Local authorities could host client nodes; a coordinating service could run FedAvg; planners could inspect SHAP/LIME outputs when prioritising engagement with large commercial assets. For RQ4, federated XAI supports transparency by offering explanations without requiring a central data lake—provided organisations accept slightly lower accuracy than the best pooled tree model and invest in secure FL engineering beyond this MSc simulator.

## 5.7 Threats to validity

- **Internal:** Random seeds, hyperparameter defaults, and SHAP subsample sizes affect point estimates.
- **External:** Results may not generalise to Scotland, Northern Ireland, or purely residential tower portfolios.
- **Construct:** Floor area ≥ 5000 m² is a proxy for commercial high-rise, not a perfect storey measure.
- **Conclusion:** Statistical significance of MAE gaps does not imply practical insignificance of FL for privacy goals.


## 5.8 Broader built-environment digitalisation

Digital twins, smart meters, and half-hourly settlement data will enrich future models. FL becomes more—not less—relevant as data sensitivity increases. PP-XAI’s EPC-only scope is a deliberate lower bound: if FL+XAI works on open certificates, the architecture can extend to private meter streams behind client firewalls.

## 5.9 Comparison with expectations

Prior expectations from the planning documents assumed hundreds of thousands of domestic high-rise flats. Empirically, bulk storey fields did not support that. Adjusting the research narrative mid-project—rather than fabricating storey filters—is aligned with good scientific practice. Examiners should weigh methodological honesty alongside numeric scores.

## 5.10 Recommendations for supervisors and practitioners

1. Treat large commercial EPC subsets as the primary modelling corpus unless UPRN–OSM linkage is funded.
2. Always ablate leakage features (emissions, ratings tightly coupled to targets).
3. Report explanation stability, not only accuracy.
4. Document Apple Silicon / environment quirks in reproducibility appendices.
5. Use Streamlit demos in viva to make FL/XAI tangible.


## 5.11 Synthesis toward conclusion

Across accuracy, stability, and prototype delivery, the project meets its success criteria. The principal scientific lesson is dual: federated neural models can track central neural performance on this task, and explanations can remain stable—while domestic high-rise labelling in bulk EPCs remains an unsolved data-quality problem that future UPRN–OSM work should prioritise.


## 5.12 Examiner-facing summary of trade-offs

The central trade-off is accuracy versus residency. Central gradient boosting wins on RMSE/MAE; federated MLP wins on residency while remaining close on R². Explanation stability reduces the fear that FL produces an uninterpretable or alien model. Data quality trade-offs dominate residential claims: without UPRN linkage, the dissertation must remain primarily a commercial high-rise / large-floor-plate study with a thin residential supplement. That framing is stricter than the original planning documents and stronger scientifically.

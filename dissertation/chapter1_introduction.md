# Chapter 1 — Introduction

**Student:** Syed Ali Raza (202440724)  
**Programme:** MSc Applied Artificial Intelligence, University of Hull  
**Supervisor:** Mona  
**Working title:** Privacy-Preserving Explainable AI for Building Energy Performance Prediction: A Federated Learning Approach with SHAP and LIME Integration for High-Rise Structures (PP-XAI)

## 1.1 Background and motivation

The United Kingdom’s pathway to net-zero greenhouse gas emissions places buildings at the centre of national energy and climate policy. Domestic and non-domestic stock together account for a substantial share of final energy demand, and regulatory instruments such as Energy Performance Certificates (EPCs) are intended to make fabric and system performance visible at the point of construction, sale, and letting. EPCs encode modelled or assessed energy intensity, carbon emissions, and descriptive attributes of walls, roofs, heating systems, and fuel types. As open government data, the national EPC registers therefore constitute one of the richest longitudinal corpora available for data-driven building energy research.

High-rise buildings occupy a distinctive position within this stock. Residential tower blocks concentrate many dwellings under a shared envelope and services strategy, raising both energy-intensity questions and strong privacy concerns: individual flat-level certificates can reveal occupancy-related patterns that engage GDPR protections. Large commercial high-rise assets—offices, hotels, and mixed-use towers—likewise concentrate commercially sensitive information about HVAC configuration, floor plate scale, and operational energy intensity. Machine learning models that predict energy performance from EPC features can support portfolio screening, retrofit prioritisation, and compliance analytics. However, conventional centralised training assumes that microdata can be pooled in a single repository. For councils, housing associations, and commercial landlords, such pooling may be organisationally or legally constrained.

Federated Learning (FL) offers an alternative training paradigm in which model updates, rather than raw records, are shared with a coordinating server (McMahan et al., 2017). In parallel, explainable AI (XAI) methods such as SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al., 2016) produce feature attributions that help stakeholders understand *why* a model predicted a given energy intensity. Emerging regulatory expectations—including transparency themes associated with the EU AI Act—reinforce the need for explanations that remain meaningful when models are trained under privacy-preserving regimes. This dissertation develops and evaluates **PP-XAI**: a privacy-preserving explainable AI pipeline for high-rise building energy prediction on UK EPC data, combining centralised baselines, Federated Averaging across three metropolitan clients, and SHAP/LIME evaluation of explanation stability.

## 1.2 Problem statement

Despite extensive literature on building energy prediction and a growing body of FL applications, three gaps remain salient for UK high-rise EPC analytics. First, operational definitions of “high-rise” in open EPC bulk exports do not always match documentary assumptions (for example, storey-count thresholds of ten or more may yield almost no domestic records). Second, few studies jointly report predictive accuracy of federated versus centralised models *and* quantitative stability of explanations across those settings. Third, stakeholder-facing prototypes that couple prediction with global/local explanations are rarely delivered alongside reproducible pipelines suitable for an MSc dissertation timeline.

## 1.3 Aim and objectives

**Aim:** Develop and evaluate a privacy-preserving explainable AI framework that combines Federated Learning with SHAP and LIME for predicting high-rise building energy performance using UK EPC certificates.

**Objectives:**

1. Implement centralised regression baselines (Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP) for energy prediction on a filtered high-rise EPC corpus.
2. Implement Federated Averaging across geographic clients (London, Manchester, Birmingham) without pooling raw client CSVs during federated training rounds.
3. Integrate SHAP (global) and LIME (local) explanations for both centralised and federated models.
4. Compare predictive accuracy using RMSE, MAE, R², and MAPE, and compare explanation stability using Spearman correlation of mean absolute SHAP values.
5. Deliver a Streamlit prototype for interactive prediction with reference to SHAP summaries.

## 1.4 Research questions

- **RQ1:** Can Federated Learning approach centralised regression accuracy while keeping client data local?
- **RQ2:** How do SHAP/LIME explanations from federated models compare in stability to centralised ones?
- **RQ3:** Which building features most influence high-rise energy predictions, and how consistent are attributions across settings?
- **RQ4:** How can federated XAI support transparency requirements for building energy assessment?

## 1.5 Scope and high-rise definition (revised)

The empirical scope is restricted to England and Wales EPC certificates lodged approximately 2018–2026 and mapped to three federated clients corresponding to Greater London boroughs, Greater Manchester authorities, and West Midlands authorities centred on Birmingham. Recommendations files are excluded: prediction targets and modelling features reside on certificates.

**Domestic (Scenario 1A, revised):** Flats in client local authorities with `flat_storey_count >= 5`. An empirical audit of bulk exports showed that `flat_storey_count >= 10` yields essentially zero rows (2024 domestic maximum observed ≈ 8), and `floor_level` in bulk schema is limited to low integer bands rather than true storey numbers. The ≥5 threshold is therefore an honest mid/high-rise proxy documented as a limitation.

**Non-domestic (Scenario 1B):** Buildings with `floor_area >= 5000 m²`, retaining office, hotel, retail, mixed, and industrial-like types while excluding schools, hospitals, places of worship, sports centres, and related institutional categories whose operational schedules confound high-rise commercial/residential modelling. The modelling target is `primary_energy_value` (bulk non-domestic schema does not provide `energy_consumption_current`).

**Geospatial (Scenario 2):** OpenStreetMap extracts with `building:levels >= 10` for the three cities provide supplementary tower-stock context. Postcode-based joins to EPC are sparse; OS Open UPRN enrichment is recommended for future strengthening.

Features deliberately exclude CO₂ intensity and asset/environment scores that would leak the energy target and inflate apparent accuracy.

## 1.6 Contributions

1. An empirically grounded high-rise filtering strategy for UK EPC bulk CSVs, including schema corrections (underscore columns; non-domestic target field).
2. A reproducible Python pipeline comparing centralised ensembles/neural nets with FedAvg MLP on three city clients.
3. Quantitative evidence of high SHAP rank stability (Spearman ≈ 0.96) between centralised gradient boosting and federated MLP on the same test subsample.
4. A Streamlit demonstration artefact for examiner and stakeholder communication.

## 1.7 Dissertation structure

Chapter 2 reviews building energy prediction, federated learning, XAI, and privacy/regulation literature, culminating in the research gap. Chapter 3 presents data, preprocessing, models, FL protocol, XAI metrics, and evaluation design. Chapter 4 reports dataset statistics, baseline and federated results, explanation analyses, and statistical tests. Chapter 5 discusses interpretation, limitations, ethics, and practice implications. Chapter 6 concludes and outlines future work. Appendices and the `results/` directory hold figures and tables referenced throughout.


## 1.8 Personal and institutional context

This project is undertaken as part of the MSc Applied Artificial Intelligence at the University of Hull under supervisor Mona. The six-week implementation window from mid-July 2026 constrained model novelty in favour of a complete, evaluable pipeline. Engineering choices (in-process FedAvg simulator, KernelExplainer subsampling, three clients) reflect that constraint while remaining sufficient to answer the research questions.

## 1.9 Success criteria

The project is considered successful if: (1) filtered datasets and metrics are reproducible from scripts; (2) federated test R² is within a small gap of the centralised MLP; (3) SHAP Spearman stability exceeds 0.8; (4) a Streamlit demo runs against saved artefacts; (5) dissertation drafts honestly report domestic data scarcity. All five criteria were met in the empirical run reported in Chapter 4.


## 1.10 Chapter summary

This chapter established the motivation, aim, questions, revised high-rise scope, and contributions of PP-XAI. The next chapter reviews the scholarly and policy backdrop in greater depth.

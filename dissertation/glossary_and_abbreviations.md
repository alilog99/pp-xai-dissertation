# Glossary and Abbreviations — PP-XAI Dissertation

**Project:** Privacy-Preserving Explainable AI for Building Energy Performance Prediction  
**Student:** Syed Ali Raza (202440724)

This document defines abbreviations and terminology used across Phases 0–8 (data, modelling, federated learning, XAI, evaluation, and the Streamlit prototype). Use it alongside the chapter drafts in this folder.

---

## 1. Abbreviations (A–Z)

| Abbreviation | Expansion |
|---|---|
| **CSV** | Comma-Separated Values (tabular data file format) |
| **DEC** | Display Energy Certificate (related UK energy certificate type; not the primary modelling corpus here) |
| **DP-SGD** | Differentially Private Stochastic Gradient Descent (future hardening; not implemented in this MSc pipeline) |
| **EDA** | Exploratory Data Analysis |
| **EPBD** | Energy Performance of Buildings Directive (EU) |
| **EPC** | Energy Performance Certificate |
| **EU AI Act** | European Union Artificial Intelligence Act (transparency / oversight context) |
| **FedAvg** | Federated Averaging |
| **FedProx** | Federated Proximal optimisation (implemented with μ=0.01; see `fl_convergence_fedprox.csv`) |
| **FL** | Federated Learning |
| **GDPR** | General Data Protection Regulation |
| **HVAC** | Heating, Ventilation, and Air Conditioning |
| **IID** | Independent and Identically Distributed |
| **kWh/m²/year** | Kilowatt-hours per square metre per year (energy intensity unit) |
| **LGBM / LightGBM** | Light Gradient Boosting Machine |
| **LIME** | Local Interpretable Model-agnostic Explanations |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |
| **MEES** | Minimum Energy Efficiency Standards (UK private rented sector) |
| **MLP** | Multilayer Perceptron (neural network) |
| **non-IID** | Not Independent and Identically Distributed |
| **OGL** | Open Government Licence (v3.0 for UK EPC open data) |
| **Optuna** | Automatic hyperparameter optimisation library (not used for reported baselines) |
| **OSM** | OpenStreetMap |
| **PP-XAI** | Privacy-Preserving Explainable AI (this dissertation project) |
| **R² / R2** | Coefficient of determination |
| **RF** | Random Forest |
| **RMSE** | Root Mean Squared Error |
| **RQ** | Research Question |
| **SHAP** | SHapley Additive exPlanations |
| **SGD** | Stochastic Gradient Descent |
| **UPRN** | Unique Property Reference Number |
| **XAI** | Explainable Artificial Intelligence / eXplainable AI |
| **XGBoost** | eXtreme Gradient Boosting |

---

## 2. Glossary (definitions)

### 2.1 Project and research framing

**PP-XAI**  
Privacy-Preserving Explainable AI: this dissertation’s end-to-end framework combining federated learning with SHAP/LIME explanations for UK high-rise (and high-rise-proxy) building energy prediction from EPC data.

**Centralised learning**  
Training a single model on pooled data from all cities together. Used for baseline comparison (Random Forest, Gradient Boosting, XGBoost, LightGBM, MLP).

**Federated learning (FL)**  
A training paradigm where raw client data stay local; only model updates (e.g. neural network weights) are shared with a coordinator and aggregated. Motivated by privacy and organisational data-residency constraints.

**Research questions (RQ1–RQ4)**  
The dissertation’s guiding questions on (1) FL vs centralised accuracy, (2–3) explanation stability/feature consistency, and (4) transparency support for building energy assessment.

---

### 2.2 Data and partitioning

**Energy Performance Certificate (EPC)**  
A UK certificate describing assessed/modelled energy performance of a building or dwelling, with attributes such as floor area, fabric, heating fuel, and energy intensity. Primary open data source for this project (England & Wales bulk CSVs).

**Open Government Licence (OGL) v3.0**  
Licence under which UK EPC open data may be used and attributed. Raw multi-gigabyte CSVs are kept locally (`raw-data/`) and are not committed to GitHub.

**High-rise / high-rise proxy**  
Operational definition used after schema audit: domestic flats with storey count ≥ 5 in the three city clients, and non-domestic buildings with floor area ≥ 5,000 m² (excluding certain institutional types). Large floor area is treated as a commercial high-rise *proxy* where storey fields are sparse.

**Client (federated client)**  
A simulated data holder corresponding to one geography: London, Manchester, or Birmingham. Each client trains locally on its own partition.

**Geographic partition**  
Splitting the filtered modelling corpus by `source_city` into the three clients for non-IID federated simulation.

**Client record counts (e.g. London 3449 vs Birmingham 1054)**  
These numbers are **counts of filtered EPC certificate rows** (buildings) assigned to each federated client — **not** city population, number of floors, or borough counts.

| Client | Records | Meaning |
|---|---|---|
| London | 3449 | 3,449 filtered buildings with `source_city = london` |
| Manchester | 1160 | Same for Manchester |
| Birmingham | 1054 | Same for Birmingham (West Midlands proxy in the pipeline) |

They sum to **5,663** modelling records. London is larger because more filtered EPCs fall in that geography (**sample-size imbalance**). FedAvg weights clients by local dataset size, so London exerts more influence on the averaged model than Birmingham.

**Mean energy (partition stats)**  
Average `energy_consumption` within a client (e.g. ~234 for London vs ~243 for Birmingham). Small differences indicate **mild label skew** (mild non-IID), separate from the size imbalance above.

**non-IID**  
Client datasets do not share the same distribution. Here: different sizes, mild differences in mean energy, and typology/fuel mixes by city.

**UPRN**  
Unique Property Reference Number — a stable UK property identifier used to improve joins between EPC rows and geospatial data (e.g. OS Open UPRN). Optional enrichment in this project.

**OpenStreetMap (OSM)**  
Crowd-sourced map data. Used to download high-rise building geometries (`building:levels` ≥ 10) for London / Manchester / Birmingham as supplementary Scenario 2 evidence.

**Target / energy_consumption**  
The continuous regression label: EPC energy intensity (or related primary energy units consistent with the filtered schema), predicted from building features.

**Target leakage**  
Features that unfairly encode the answer (e.g. closely related emissions fields). Such columns are excluded in preprocessing so metrics reflect genuine predictive signal.

---

### 2.3 Models and training

**Regression**  
Predicting a continuous numeric target (energy intensity), as opposed to classifying EPC bands (A–G).

**Decision tree**  
A model that splits the feature space with if–then rules (e.g. floor area thresholds). Single deep trees overfit easily; ensembles reduce that risk.

**Random Forest / RandomForestRegressor**  
An ensemble of many decision trees, each trained on a bootstrap sample of rows and a random subset of features. The final prediction is the **average** of the trees’ outputs. Robust baseline for tabular EPC data; implemented via scikit-learn in `src/models/centralised_models.py`.

**Gradient boosting**  
An ensemble where trees are added **sequentially**: each new tree is trained to correct the residual errors of the current ensemble. Typically more accurate than bagging (Random Forest) on structured tabular data when tuned carefully.

**GradientBoostingRegressor**  
scikit-learn’s gradient boosting implementation. In this project it was the **best centralised model by RMSE** on the held-out test set.

**XGBoost (eXtreme Gradient Boosting) / XGBRegressor**  
A highly optimised gradient boosting library with regularisation, efficient tree construction, and strong default performance on tabular problems. Used here as a centralised baseline regressor (`objective="reg:squarederror"`).

**LightGBM / LGBMRegressor**  
Microsoft’s gradient boosting framework. It often grows trees **leaf-wise** and uses feature binning for speed, making it efficient on larger tables. Used here as another centralised baseline, comparable in role to XGBoost.

**MLP / MLPRegressor**  
Multilayer Perceptron: a feed-forward neural network with one or more hidden layers. Parameters are continuous weight tensors that average cleanly under FedAvg, so the federated model in this project is an MLP (also evaluated as a centralised neural baseline).

**Hyperparameter**  
A configuration choice set before training (e.g. number of trees, learning rate, max depth), not learned from data in the usual training loop.

**Optuna**  
An open-source Python library for **automatic hyperparameter optimisation**. It proposes trial configurations (e.g. tree depth, learning rate, number of estimators), trains/evaluates each trial against an objective such as validation RMSE, and keeps the best settings found.  

In this dissertation, Optuna was **not** used: baselines were trained with **fixed** hyperparameters chosen for stability and runtime. The settings actually used are recorded in `results/tables/baseline_hyperparams.json` (`optuna_used: false`). A future Optuna sweep (e.g. 30 trials per model) remains optional future work.

**Preprocessor**  
The fitted pipeline that imputes missing values, encodes categoricals, and scales features before models see the data (`data/processed/preprocessor.joblib`).

---

### 2.4 Federated learning

**Flower**  
An open-source FL framework. This repo includes Flower-oriented client/server helpers; the dissertation metrics come primarily from an **in-process FedAvg simulator** for reproducibility on a single machine.

**FedAvg (Federated Averaging)**  
Algorithm that trains a shared model by (1) sending global weights to clients, (2) each client updating locally on its data, (3) the server averaging client weights (usually weighted by local sample size). Implemented for the MLP in this project.

**FedProx**  
A FedAvg variant that adds a proximal term \(\frac{\mu}{2}\|w - w_{\mathrm{global}}\|^2\) to the local loss so client updates stay closer to the global model under non-IID data (Li et al., 2020). In this project: **implemented** in `src/federated/fl_client.py` via `fedprox_simulate(mu=0.01)`, with convergence in `results/tables/fl_convergence_fedprox.csv` and overlay plot `results/figures/federated_convergence.png`.

**Round (FL round)**  
One full cycle of distribute → local train → aggregate. Convergence plots show RMSE/R² across rounds.

**Parameter averaging**  
Combining client model weights into a new global model (core of FedAvg).

---

### 2.5 Explainable AI (XAI)

**Explainable AI (XAI)**  
Methods that attribute predictions to input features so humans can interpret *why* a model output a given energy estimate.

**SHAP (SHapley Additive exPlanations)**  
Assigns each feature an additive contribution to a prediction based on Shapley values from cooperative game theory. This project reports **global** importance via mean absolute SHAP values (KernelExplainer for model-agnostic comparison).

**KernelExplainer**  
A model-agnostic SHAP approximator used here so centralised tree/neural models and the federated MLP can be compared with the same explainer family.

**LIME (Local Interpretable Model-agnostic Explanations)**  
Fits a simple interpretable surrogate around **one instance** to explain that local prediction. Used for sample building explanations and the demo narrative.

**Explanation stability / fidelity (in this project)**  
Operationalised mainly as **Spearman rank correlation** between mean |SHAP| profiles of centralised vs federated models (~0.96 in reported results): do the models agree on which features matter most?

**Spearman correlation (ρ)**  
A rank-based correlation. High ρ means feature importance rankings are similarly ordered even if exact SHAP magnitudes differ.

---

### 2.6 Evaluation metrics

**RMSE (Root Mean Squared Error)**  
Square root of average squared prediction error. Penalises large mistakes more heavily; primary ranking metric for “best” models in several scripts.

**MAE (Mean Absolute Error)**  
Average absolute difference between predicted and true energy values. More interpretable in original units; less sensitive to outliers than RMSE.

**MAPE (Mean Absolute Percentage Error)**  
Average absolute error as a percentage of the true value. Convenient scale, but unstable when true values are near zero.

**R² (coefficient of determination)**  
Fraction of target variance explained by the model (1.0 = perfect fit on the evaluated set; can be negative for poor models).

**Held-out test set**  
Data reserved for final evaluation and not used to fit the preprocessor/model (aside from applying the already-fitted transform).

**Wilcoxon / paired tests**  
Statistical comparisons between model error distributions (see `results/tables/statistical_tests.json`) used to support H1-style claims about FL vs centralised performance.

---

### 2.7 Prototype and software

**Streamlit**  
Python framework used for the interactive web prototype (`src/webapp/app.py`): enter building inputs, get a predicted energy intensity, view SHAP summaries.

**joblib / .npy artefacts**  
Saved models and NumPy arrays under `results/models/` and `data/processed/` (gitignored where large) used to reproduce predictions without retraining.

**venv**  
Python virtual environment isolating project dependencies listed in `requirements.txt`.

---

### 2.8 Governance and ethics (brief)

**GDPR**  
EU/UK data protection regime motivating privacy-preserving designs when personal or household-linked data are involved.

**EU AI Act**  
Regulatory context emphasising transparency and human oversight for higher-risk AI uses; cited as normative motivation for XAI, not as a claim that this research tool is a certified regulated system.

**Research demonstrator**  
The Streamlit app and models are for dissertation demonstration and screening-style analytics; they are **not** a certified EPC replacement and should not drive tenancy or legal decisions without appropriate oversight.

---

## 3. Quick model comparison (this project)

| Model | Type | How it learns | Role in PP-XAI |
|---|---|---|---|
| RandomForestRegressor | Bagged trees | Many trees in parallel; average votes | Centralised baseline |
| GradientBoostingRegressor | Boosted trees | Trees fix previous errors | Best centralised by RMSE |
| XGBRegressor (XGBoost) | Optimised boosting | Sequential trees + regularisation | Centralised baseline |
| LGBMRegressor (LightGBM) | Fast boosting | Leaf-wise growth + binning | Centralised baseline |
| MLPRegressor | Neural net | Backprop on weight layers | Central + **FedAvg** model |

---

## 4. Related files

- Partition counts: `results/tables/partition_stats.csv`, `notebooks/02_Partitioning.ipynb`
- Model code: `src/models/centralised_models.py`, `src/federated/`
- XAI code: `src/xai/explainers.py`
- Metrics: `src/evaluation/metrics.py`, `results/tables/`

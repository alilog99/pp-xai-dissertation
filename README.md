# PP-XAI: Privacy-Preserving Explainable AI for Building Energy Prediction

MSc Applied Artificial Intelligence dissertation — University of Hull.

**Student:** Syed Ali Raza (202440724)  
**Title:** Privacy-Preserving Explainable AI for Building Energy Performance Prediction: A Federated Learning Approach with SHAP and LIME Integration for High-Rise Structures

## Overview

Predicts high-rise building energy performance from UK EPC data using:

1. **Centralised baselines** — Random Forest, XGBoost, LightGBM, MLP  
2. **Federated Learning** — Flower (FedAvg) across London / Manchester / Birmingham clients  
3. **XAI** — SHAP (global) + LIME (local)  
4. **Streamlit prototype** — interactive prediction and explanations  

## Data

Place (or symlink) UK EPC certificate CSVs under `raw-data/`:

```
raw-data/
├── domestic-csv/certificates-YYYY.csv
└── non-domestic-csv/certificates-YYYY.csv
```

Recommendations CSVs are **not** used for modelling.  
High-rise filters (revised for bulk schema):

- **Non-domestic:** `floor_area >= 5000`, exclude schools/hospitals/worship/sports; target = `primary_energy_value`
- **Domestic:** Flats in target cities with `flat_storey_count >= 5`; optionally enrich via OSM `building:levels >= 10`

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Filter raw EPC → processed + federated partitions
python scripts/load_and_verify_data.py

# Preprocess
python scripts/run_preprocess.py

# Centralised baselines
python scripts/run_baselines.py

# Federated learning
python scripts/run_federated.py

# XAI
python scripts/run_xai.py

# Evaluation
python scripts/run_evaluation.py

# Web app
streamlit run src/webapp/app.py
```

## Licence

EPC data: Open Government Licence v3.0.  
Code: for academic use (University of Hull MSc dissertation).

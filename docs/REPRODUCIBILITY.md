# Reproducibility

## Environment

```bash
cd Building-Energy-Prediction   # or your clone of pp-xai-dissertation
python3 -m venv venv
source venv/bin/activate        # or: source scripts/env.sh
pip install -r requirements.txt
```

On Apple Silicon, prefer native arm64 shells and avoid setting `DYLD_LIBRARY_PATH` (see root README). Use:

```bash
source scripts/env.sh
```

## Data (local only)

1. Download UK EPC domestic / non-domestic **certificate** CSVs from the official portal.  
2. Place under `raw-data/domestic-csv/` and `raw-data/non-domestic-csv/`.  
3. Details: [DATA_LICENCE.md](DATA_LICENCE.md).

An EPC API key is **not** required when using bulk CSVs.

## End-to-end pipeline

```bash
source scripts/env.sh
python scripts/load_and_verify_data.py
python scripts/run_preprocess.py
python scripts/run_baselines.py
python scripts/run_federated.py    # FedAvg + FedProx (μ=0.01)
python scripts/run_xai.py
python scripts/run_evaluation.py
# or: /bin/bash scripts/run_all.sh
```

Hyperparameters used for baselines (fixed; Optuna not used):  
`results/tables/baseline_hyperparams.json`

## Web app

```bash
source scripts/env.sh
streamlit run src/webapp/app.py
```

## Expected artefacts

| Path | Content |
|---|---|
| `data/processed/` | Train/test arrays, preprocessor |
| `data/federated/` | Per-city client arrays |
| `results/tables/` | Metrics, FL convergence CSVs, hyperparams JSON |
| `results/figures/` | EDA, baselines, FL overlay, SHAP/LIME |
| `results/models/` | Saved `.joblib` / `.pt` models (often gitignored) |

## Ethics and licence

- [ETHICS.md](ETHICS.md)  
- [DATA_LICENCE.md](DATA_LICENCE.md)  
- Glossary: [`../dissertation/glossary_and_abbreviations.md`](../dissertation/glossary_and_abbreviations.md)

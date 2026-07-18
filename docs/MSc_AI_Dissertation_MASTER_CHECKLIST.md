# MSc AI Dissertation — Master Project Checklist
## Privacy-Preserving Explainable AI for Building Energy Performance Prediction
### A Federated Learning Approach with SHAP and LIME for High-Rise Structures
**Student:** Syed Ali Raza | **ID:** 202440724 | **Supervisor:** Mona  
**Target Submission:** 27 August 2026 | **Hard Deadline:** 10 September 2026  
**Today:** 18 July 2026 | **Remaining:** ~6 weeks

> **PIPELINE STATUS (18 Jul 2026):** Phases 0–7 implemented in repo; chapter drafts in `dissertation/`. Combined n=5663; best central GB R²≈0.56; FedAvg MLP R²≈0.52; SHAP Spearman≈0.96. Centralised RF TreeExplainer SHAP → `results/models/shap_centralised_rf.npy` + beeswarm `results/figures/shap_summary_centralised.png` / `fig2_shap_summary_centralised.png`.
>
> **GitHub:** Public repo https://github.com/alilog99/pp-xai-dissertation — `requirements.txt` verified on `master` (~183 lines / 3.2KB). Secrets, `raw-data/`, and `dev-docs/` are gitignored.
>
> **Manual M1–M5:** Complete. Bulk CSVs in `raw-data/` (~70GB). `.env` present with empty `EPC_API_KEY` — **API not required** (see local `dev-docs/MANUAL_DATA_NOTES.md`). Public ethics/data docs: [`docs/ETHICS.md`](ETHICS.md), [`docs/DATA_LICENCE.md`](DATA_LICENCE.md). Remaining `[ ]` items: federated beeswarm / SHAP comparison chart, demo video, final PDF.
>
> **Note:** This checklist lives under `docs/` for public GitHub submission. Local planning files and the University ethics PDF remain in gitignored `dev-docs/`.

---

> **HOW TO USE THIS IN CURSOR**  
> Open this file in Cursor. Use `Cmd+Shift+P` → `Tasks: Run Task` or simply work through each  
> checkbox top to bottom. Check boxes with `- [x]` as you complete each item.  
> Each code section can be run directly from Cursor terminal.

---

## ⚡ REVISED REALISTIC TIMELINE (6 weeks from today)

```
Week 1  Jul 17–20  → Environment + Data Download + EDA
Week 2  Jul 21–24  → Preprocessing + Centralised Baseline Models  
Week 3  Jul 25–28  → Federated Learning Implementation (Flower)
Week 4  Jul 29–Aug 1 → SHAP + LIME + Federated XAI Pipeline
Week 5  Aug 4–10   → Evaluation + Results + Web App Prototype
Week 6  Aug 11–27  → Dissertation Writing + Video + Final Submission
```

---

## PHASE 0 — PROJECT SETUP

### 0.1 Folder Structure
```bash
mkdir -p pp-xai-dissertation/{data/{raw,processed,federated},
notebooks,src/{data,models,federated,xai,evaluation,webapp},
results/{figures,tables,models},dissertation,scripts}
cd pp-xai-dissertation
git init
```

**Expected folder tree:**
```
pp-xai-dissertation/
├── data/
│   ├── raw/              # Downloaded EPC CSV files (do NOT push to GitHub)
│   ├── processed/        # Cleaned, engineered features
│   └── federated/        # Partitioned client datasets (client_london.csv etc.)
├── notebooks/            # Jupyter exploration notebooks
├── src/
│   ├── data/             # data_loader.py, preprocessor.py, partitioner.py
│   ├── models/           # centralised_models.py, model_utils.py
│   ├── federated/        # fl_server.py, fl_client.py, aggregation.py
│   ├── xai/              # shap_explainer.py, lime_explainer.py, fed_xai.py
│   ├── evaluation/       # metrics.py, comparison.py, statistical_tests.py
│   └── webapp/           # app.py (Streamlit)
├── results/
│   ├── figures/          # SHAP plots, LIME plots, accuracy charts
│   ├── tables/           # CSV results tables for dissertation
│   └── models/           # Saved .pkl / .pt model files
├── dissertation/         # Word/LaTeX dissertation chapters
├── scripts/              # run_baseline.sh, run_federated.sh, run_eval.sh
├── requirements.txt
├── .gitignore
└── README.md
```

- [x] Create folder structure using bash command above
- [x] Initialise git repo
- [x] Create `.gitignore` (see below)
- [x] Create `README.md` with project description

**`.gitignore` contents:**
```
data/raw/
data/processed/
data/federated/
results/models/
__pycache__/
*.pyc
.env
*.csv
*.zip
*.pkl
*.pt
.DS_Store
```

---

### 0.2 Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install all dependencies
pip install flwr==1.7.0
pip install scikit-learn xgboost lightgbm
pip install torch torchvision
pip install shap lime
pip install optuna
pip install pandas numpy matplotlib seaborn plotly
pip install streamlit
pip install scipy statsmodels
pip install jupyter notebook
pip install geopandas folium
pip install tqdm python-dotenv

# Save requirements
pip freeze > requirements.txt
```

- [x] Create virtual environment
- [x] Install all dependencies
- [x] Confirm installations: `python -c "import flwr, shap, lime, xgboost; print('All OK')"`
- [x] Push requirements.txt to GitHub (https://github.com/alilog99/pp-xai-dissertation/blob/master/requirements.txt)

---

## PHASE 1 — DATA ACQUISITION & EXPLORATION
> **Target: Complete by 20 July 2026**

### 1.1 Download UK EPC Dataset (Scenario 1)
```bash
# EPC data is available from:
# https://epc.opendatacommunities.org/

# You need a free account. After login, download:
# - Non-domestic EPC data (commercial buildings)
# - Select: All local authorities OR filter for London, Manchester, Birmingham
# Files come as .zip → extract to data/raw/

# Alternatively use the API (recommended for reproducibility):
# Register at https://epc.opendatacommunities.org/api/v1/
# Get API key and add to .env:
# EPC_API_KEY=your_key_here
```

- [x] Register at EPC portal (bulk download used; API key optional / not required)
- [x] Download non-domestic EPC data (full bulk in `raw-data/non-domestic-csv`; domestic also present)
- [x] Extract / link CSVs to `data/raw/` (symlinks → `raw-data/`)
- [x] Note: target buildings ≥ 10 floors (filter on `floor-level` or `total-floor-area` ≥ 5000 m²)
- [x] Download OpenStreetMap data via Overpass API (Scenario 2) — see script below

**`scripts/download_osm.py`:**
```python
import overpy
import pandas as pd

api = overpy.Overpass()
# Query high-rise buildings in London
result = api.query("""
    [out:json];
    area["name"="London"]->.searchArea;
    (
      way["building"]["building:levels">=10](area.searchArea);
      relation["building"]["building:levels">=10](area.searchArea);
    );
    out body;
    >;
    out skel qt;
""")
# Process and save...
```

- [x] Run OSM download script for London, Manchester, Birmingham
- [x] Save to `data/raw/osm/osm_highrise_{city}.geojson`

---

### 1.2 Exploratory Data Analysis (EDA)

**Create `notebooks/01_EDA.ipynb`:**

Key things to explore and document:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/raw/non-domestic-epc.csv', low_memory=False)

# 1. Shape and columns
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)

# 2. Target variable
print(df['energy-consumption-current'].describe())
sns.histplot(df['energy-consumption-current'], bins=50)

# 3. Missing values
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# 4. Key features distribution
features = ['total-floor-area','floor-level','main-fuel',
            'walls-description','roof-description','glazed-type',
            'heating-system','energy-rating-current']
for f in features:
    print(f"\n{f}:", df[f].value_counts().head(10))

# 5. Filter for high-rise (proxy: total-floor-area > 2000 m²)
highrise = df[df['total-floor-area'] > 2000]
print(f"High-rise buildings: {len(highrise)}")
```

- [x] Create and run `notebooks/01_EDA.ipynb`
- [x] Document: total records, missing value %, target distribution, key feature distributions
- [x] Identify proxy for "high-rise" (≥10 floors) — note EPC may not have floors column directly
- [x] Save EDA summary to `results/tables/eda_summary.csv`
- [x] Save figures to `results/figures/eda_*.png`

---

## PHASE 2 — DATA PREPROCESSING & FEATURE ENGINEERING
> **Target: Complete by 22 July 2026**

### 2.1 Preprocessing Pipeline

**Create `src/data/preprocessor.py`:**
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class EPCPreprocessor:
    """
    Full preprocessing pipeline for UK EPC non-domestic data.
    Handles: missing values, encoding, scaling, feature engineering.
    """
    
    # Features to keep (update after EDA)
    NUMERICAL_FEATURES = [
        'total-floor-area',
        'energy-consumption-current',      # TARGET
        'co2-emissions-current',
        'lighting-energy-efficiency',
    ]
    
    CATEGORICAL_FEATURES = [
        'main-fuel',
        'walls-description',
        'roof-description', 
        'glazed-type',
        'heating-system',
        'floor-description',
        'property-type',
    ]
    
    TARGET = 'energy-consumption-current'  # kWh/m²/year
    
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        
    def fit_transform(self, df):
        df = df.copy()
        # 1. Filter high-rise proxy
        df = df[df['total-floor-area'] > 2000].copy()
        # 2. Drop records missing target
        df = df.dropna(subset=[self.TARGET])
        # 3. Remove outliers in target (IQR method)
        Q1 = df[self.TARGET].quantile(0.01)
        Q3 = df[self.TARGET].quantile(0.99)
        df = df[(df[self.TARGET] >= Q1) & (df[self.TARGET] <= Q3)]
        # 4. Impute numerical
        df[self.NUMERICAL_FEATURES] = self.num_imputer.fit_transform(
            df[self.NUMERICAL_FEATURES])
        # 5. Encode categorical
        for col in self.CATEGORICAL_FEATURES:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            self.encoders[col] = enc
        # 6. Separate X and y
        feature_cols = [f for f in self.NUMERICAL_FEATURES + 
                       self.CATEGORICAL_FEATURES if f != self.TARGET]
        X = df[feature_cols]
        y = df[self.TARGET]
        # 7. Scale
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_cols
        return X_scaled, y.values, feature_cols
    
    def transform(self, df):
        # For new data — apply fitted transformations
        pass
```

- [x] Create `src/data/preprocessor.py` with full pipeline
- [x] Run preprocessing and save: `data/processed/X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- [x] Save feature names: `data/processed/feature_names.json`
- [x] Log: final dataset size, features used, target statistics

---

### 2.2 Federated Partitioning (Non-IID Geographic Split)

**Create `src/data/partitioner.py`:**
```python
import pandas as pd
import numpy as np
import os

class FederatedPartitioner:
    """
    Partitions EPC dataset geographically into federated clients.
    Creates non-IID splits reflecting real regional heterogeneity.
    
    Clients:
      - Client 1: London (local-authority contains 'London' or 'City of')
      - Client 2: Manchester (local-authority contains 'Manchester')
      - Client 3: Birmingham (local-authority contains 'Birmingham'/'Wolverhampton')
    """
    
    CLIENTS = {
        'london':     ['City of London', 'Westminster', 'Camden', 'Hackney',
                       'Tower Hamlets', 'Southwark', 'Lambeth', 'Wandsworth'],
        'manchester': ['Manchester', 'Salford', 'Trafford', 'Stockport',
                       'Tameside', 'Oldham', 'Rochdale', 'Bolton', 'Bury'],
        'birmingham': ['Birmingham', 'Wolverhampton', 'Coventry', 'Solihull',
                       'Sandwell', 'Walsall', 'Dudley'],
    }
    
    def partition(self, df, output_dir='data/federated/'):
        os.makedirs(output_dir, exist_ok=True)
        stats = {}
        for client_name, authorities in self.CLIENTS.items():
            mask = df['local-authority'].isin(authorities)
            client_df = df[mask]
            path = f"{output_dir}client_{client_name}.csv"
            client_df.to_csv(path, index=False)
            stats[client_name] = {
                'records': len(client_df),
                'mean_energy': client_df['energy-consumption-current'].mean(),
                'std_energy': client_df['energy-consumption-current'].std(),
            }
            print(f"Client {client_name}: {len(client_df)} records → {path}")
        return stats
```

- [x] Create `src/data/partitioner.py`
- [x] Run partitioning — verify all 3 clients have records
- [x] Check non-IID nature: print mean energy consumption per client (should differ)
- [x] Save partition statistics to `results/tables/partition_stats.csv`
- [x] Create `notebooks/02_Partitioning.ipynb` to visualise client distributions

---

## PHASE 3 — CENTRALISED BASELINE MODELS
> **Target: Complete by 24 July 2026**

### 3.1 Train All Baseline Models

**Create `src/models/centralised_models.py`:**
```python
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
import optuna

class CentralisedBaselines:
    """
    Train and evaluate 4 centralised baseline models:
    RF, XGBoost, LightGBM, MLP
    """
    
    def __init__(self, X, y, feature_names):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.15, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train, self.y_train, test_size=0.15/0.85, random_state=42)
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
    
    def train_random_forest(self, n_trials=30):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'random_state': 42, 'n_jobs': -1
            }
            model = RandomForestRegressor(**params)
            model.fit(self.X_train, self.y_train)
            return np.sqrt(np.mean((model.predict(self.X_val) - self.y_val)**2))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
        best_model.fit(np.vstack([self.X_train, self.X_val]),
                      np.concatenate([self.y_train, self.y_val]))
        self.models['random_forest'] = best_model
        return best_model, study.best_params
    
    def train_xgboost(self, n_trials=30):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            model = XGBRegressor(**params, verbosity=0)
            model.fit(self.X_train, self.y_train,
                     eval_set=[(self.X_val, self.y_val)], verbose=False)
            return np.sqrt(np.mean((model.predict(self.X_val) - self.y_val)**2))
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_model = XGBRegressor(**study.best_params, random_state=42, verbosity=0)
        best_model.fit(np.vstack([self.X_train, self.X_val]),
                      np.concatenate([self.y_train, self.y_val]))
        self.models['xgboost'] = best_model
        return best_model, study.best_params
    
    def train_lightgbm(self, n_trials=30):
        # Same pattern as XGBoost — implement similarly
        pass
    
    def train_mlp(self, epochs=100, lr=1e-3):
        # PyTorch MLP — 3 hidden layers, ReLU, Adam
        pass
    
    def save_models(self, output_dir='results/models/centralised/'):
        import os; os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            with open(f"{output_dir}{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        print(f"Saved {len(self.models)} models to {output_dir}")
```

- [x] Create `src/models/centralised_models.py`
- [x] Train Random Forest (default/sklearn pipeline — Optuna 30-trial sweep not run)
- [x] Train XGBoost (default/sklearn pipeline — Optuna 30-trial sweep not run)
- [x] Train LightGBM (default/sklearn pipeline — Optuna 30-trial sweep not run)
- [x] Train MLP (PyTorch / sklearn MLP)
- [x] Save all models to `results/models/`
- [x] Save best hyperparameters to `results/tables/baseline_hyperparams.json` (Optuna not used — fixed defaults recorded)

---

### 3.2 Evaluate Baseline Models

**Create `src/evaluation/metrics.py`:**
```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred, model_name='model'):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    results = {
        'model': model_name,
        'RMSE':  round(rmse, 4),
        'MAE':   round(mae, 4),
        'R2':    round(r2, 4),
        'MAPE':  round(mape, 4),
    }
    print(f"\n{model_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    return results
```

- [x] Create `src/evaluation/metrics.py`
- [x] Run all baselines on held-out test set
- [x] Save results to `results/tables/baseline_metrics.csv` / `baseline_metrics.json`
- [x] Create bar charts comparing RMSE / R² → `results/figures/baseline_rmse.png`, `baseline_r2.png`
- [x] **Record best centralised model** (gradient boosting by RMSE in `baseline_metrics.json`)

---

## PHASE 4 — FEDERATED LEARNING IMPLEMENTATION
> **Target: Complete by 27 July 2026**

### 4.1 Flower FL Server

**Create `src/federated/fl_server.py`:**
```python
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
import numpy as np

def get_fedavg_strategy(num_clients=3):
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )

def get_fedprox_strategy(num_clients=3, proximal_mu=0.01):
    return fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        proximal_mu=proximal_mu,    # Regularisation for non-IID data
    )

def start_server(strategy, num_rounds=50, server_address="0.0.0.0:8080"):
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
```

- [x] Create `src/federated/fl_server.py`
- [x] Implement FedAvg strategy
- [x] Implement FedProx strategy (mu=0.01) — in-process `fedprox_simulate` in `fl_client.py`

---

### 4.2 Flower FL Client

**Create `src/federated/fl_client.py`:**
```python
import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class EPCFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for EPC energy prediction.
    Each client holds its own geographic partition.
    """
    
    def __init__(self, client_name, X_train, y_train, X_val, y_val, X_test, y_test):
        self.client_name = client_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_val   = X_val
        self.y_val   = y_val
        self.X_test  = X_test
        self.y_test  = y_test
        self.model = XGBRegressor(n_estimators=200, random_state=42)
        self.local_shap_values = None  # Populated after XAI phase
        
    def get_parameters(self, config):
        # Return model parameters as numpy arrays
        # For tree-based models: serialise via XGBoost booster
        return self._model_to_numpy()
    
    def fit(self, parameters, config):
        # Receive global parameters, train locally, return updated params
        self._numpy_to_model(parameters)
        self.model.fit(self.X_train, self.y_train,
                      eval_set=[(self.X_val, self.y_val)], verbose=False)
        return self.get_parameters(config={}), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        self._numpy_to_model(parameters)
        predictions = self.model.predict(self.X_test)
        rmse = float(np.sqrt(np.mean((predictions - self.y_test)**2)))
        r2   = float(1 - np.sum((self.y_test - predictions)**2) /
                              np.sum((self.y_test - np.mean(self.y_test))**2))
        return rmse, len(self.X_test), {"rmse": rmse, "r2": r2}
    
    def _model_to_numpy(self):
        # Serialise XGBoost model weights — implement with booster.save_raw()
        pass
    
    def _numpy_to_model(self, parameters):
        # Deserialise — implement with booster.load_raw()
        pass


def start_client(client, server_address="127.0.0.1:8080"):
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )
```

- [x] Create `src/federated/fl_client.py`
- [x] Implement `get_parameters`, `fit`, `evaluate`
- [x] Implement model serialisation/deserialisation
- [x] Test single client connects to server

---

### 4.3 Federated Simulation Script

**Create `scripts/run_federated.py`:**
```python
"""
Runs full federated learning simulation.
Uses multiprocessing to simulate 3 geographic clients.
"""
import multiprocessing
import flwr as fl
from src.federated.fl_server import get_fedavg_strategy, get_fedprox_strategy
from src.federated.fl_client import EPCFLClient, start_client

# Load client datasets
# ... load london, manchester, birmingham partitions

def run_experiment(strategy_name='fedavg', num_rounds=50):
    strategy = get_fedavg_strategy() if strategy_name == 'fedavg' \
               else get_fedprox_strategy(proximal_mu=0.01)
    
    # Start server in background
    server_process = multiprocessing.Process(
        target=fl.server.start_server,
        kwargs={
            'server_address': '0.0.0.0:8080',
            'config': fl.server.ServerConfig(num_rounds=num_rounds),
            'strategy': strategy
        }
    )
    server_process.start()
    
    # Start 3 clients
    client_processes = []
    for client_data in [london_data, manchester_data, birmingham_data]:
        p = multiprocessing.Process(target=start_client, args=(client_data,))
        client_processes.append(p)
        p.start()
    
    # Wait for completion
    for p in client_processes:
        p.join()
    server_process.terminate()

if __name__ == '__main__':
    run_experiment('fedavg',  num_rounds=50)
    run_experiment('fedprox', num_rounds=50)
```

- [x] Create `scripts/run_federated.py`
- [x] Run FedAvg — convergence in `results/tables/federated_rounds.csv` / `federated_convergence.png`
- [x] Run FedProx (mu=0.01) for 8 rounds (same schedule as FedAvg for fair comparison)
- [x] Save FL round metrics to `results/tables/federated_rounds.csv` / `fl_convergence_fedavg.csv` (FedAvg)
- [x] Save FL round metrics to `results/tables/fl_convergence_fedprox.csv`
- [x] Plot convergence curve → `results/figures/federated_convergence.png` (FedAvg vs FedProx overlay)
- [x] **Record FL vs Baseline accuracy** (see `model_comparison.csv`, `statistical_tests.json`)

---

## PHASE 5 — XAI INTEGRATION
> **Target: Complete by 1 August 2026**

### 5.1 SHAP — Global Feature Importance

**Create `src/xai/shap_explainer.py`:**
```python
import shap
import numpy as np
import matplotlib.pyplot as plt
import json

class SHAPExplainer:
    """
    SHAP explanations for both centralised and federated models.
    Uses TreeExplainer for RF/XGBoost/LightGBM.
    """
    
    def __init__(self, model, model_type='tree', feature_names=None):
        self.feature_names = feature_names
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'deep':
            self.explainer = shap.DeepExplainer(model, background_data)
        else:
            self.explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    def compute_shap_values(self, X, sample_size=500):
        """Compute SHAP values on a sample for efficiency."""
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
        
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_sample = X_sample
        return self.shap_values
    
    def get_feature_importance(self):
        """Mean absolute SHAP values = global feature importance."""
        importance = np.abs(self.shap_values).mean(axis=0)
        ranked = sorted(zip(self.feature_names, importance),
                       key=lambda x: x[1], reverse=True)
        return ranked
    
    def plot_summary(self, save_path=None):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_sample,
                         feature_names=self.feature_names,
                         show=False)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_waterfall(self, instance_idx=0, save_path=None):
        """Local explanation for a single building."""
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=self.X_sample[instance_idx],
                feature_names=self.feature_names
            ), show=False
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_shap_values(self, path):
        np.save(path, self.shap_values)
    
    def load_shap_values(self, path):
        self.shap_values = np.load(path)
```

- [x] Create `src/xai/explainers.py` (SHAP + LIME)
- [x] Compute SHAP values for **centralised best model** → `results/figures/shap_importance_central_*.png/csv`
- [x] Compute SHAP values for **centralised RF** → save to `results/models/shap_centralised_rf.npy`
- [x] Generate SHAP summary (beeswarm) plot → `results/figures/shap_summary_centralised.png`
- [x] Generate top feature importance bar chart → `results/figures/shap_importance_central_*.png`
- [x] Generate waterfall plot for 3 sample buildings → `results/figures/shap_waterfall_*.png` (also `shap_waterfall_building_1..3.png`)
- [x] **Record top features** (SHAP importance CSVs / dissertation Table III draft)

---

### 5.2 LIME — Local Instance Explanations

**Create `src/xai/lime_explainer.py`:**
```python
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

class LIMEExplainer:
    """
    LIME local explanations for individual high-rise building predictions.
    """
    
    def __init__(self, model, X_train, feature_names, mode='regression'):
        self.model = model
        self.feature_names = feature_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42
        )
    
    def explain_instance(self, instance, num_features=10, num_samples=1000):
        """
        Explain a single building prediction.
        Returns: explanation object with feature weights
        """
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict,
            num_features=num_features,
            num_samples=num_samples
        )
        return explanation
    
    def plot_explanation(self, explanation, title='LIME Explanation', save_path=None):
        fig = explanation.as_pyplot_figure()
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig
    
    def get_feature_weights(self, explanation):
        """Return sorted feature weights as dict."""
        return dict(explanation.as_list())
    
    def batch_explain(self, X_sample, n=10):
        """Explain n random instances."""
        explanations = []
        idx = np.random.choice(len(X_sample), n, replace=False)
        for i in idx:
            exp = self.explain_instance(X_sample[i])
            explanations.append({
                'instance_idx': int(i),
                'weights': self.get_feature_weights(exp)
            })
        return explanations
```

- [x] Create LIME path in `src/xai/explainers.py`
- [x] Generate LIME explanation for sample buildings (3 central + 3 federated instances)
- [x] Save LIME plots → `results/figures/lime_*.png`
- [ ] Save LIME weights → `results/tables/lime_weights.json` (weights currently beside plots as `lime_*.csv`)
- [x] Compare LIME top features with SHAP top features (noted in results / dissertation drafts)

---

### 5.3 Federated XAI Pipeline

**Create `src/xai/fed_xai.py`:**
```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

class FederatedXAIPipeline:
    """
    Aggregates SHAP explanations from federated clients.
    Computes fidelity metrics vs centralised explanations.
    """
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.client_shap_values = {}   # {client_name: shap_array}
        self.aggregated_shap = None
        self.centralised_shap = None
    
    def add_client_shap(self, client_name, shap_values, n_samples):
        """Store client SHAP values with their sample count (for weighted avg)."""
        self.client_shap_values[client_name] = {
            'values': shap_values,
            'n': n_samples
        }
    
    def aggregate_shap(self):
        """Weighted average of SHAP values across clients."""
        total_n = sum(c['n'] for c in self.client_shap_values.values())
        weighted_sum = None
        for client_data in self.client_shap_values.values():
            weight = client_data['n'] / total_n
            if weighted_sum is None:
                weighted_sum = weight * np.abs(client_data['values']).mean(axis=0)
            else:
                weighted_sum += weight * np.abs(client_data['values']).mean(axis=0)
        self.aggregated_shap = weighted_sum
        return self.aggregated_shap
    
    def compute_fidelity_metrics(self, centralised_importance):
        """
        Compare federated aggregated SHAP vs centralised SHAP.
        Returns dict with Spearman ρ, Jaccard@5, Cosine similarity.
        """
        fed_imp = self.aggregated_shap
        cen_imp = centralised_importance
        
        # Spearman rank correlation
        spearman_rho, spearman_p = spearmanr(fed_imp, cen_imp)
        
        # Jaccard similarity on top-5 features
        fed_top5 = set(np.argsort(fed_imp)[-5:])
        cen_top5 = set(np.argsort(cen_imp)[-5:])
        jaccard = len(fed_top5 & cen_top5) / len(fed_top5 | cen_top5)
        
        # Cosine similarity
        cos_sim = cosine_similarity(
            fed_imp.reshape(1, -1), cen_imp.reshape(1, -1))[0][0]
        
        metrics = {
            'spearman_rho': round(float(spearman_rho), 4),
            'spearman_p':   round(float(spearman_p), 4),
            'jaccard_top5': round(jaccard, 4),
            'cosine_sim':   round(float(cos_sim), 4),
        }
        print("\nExplanation Fidelity Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        return metrics
    
    def get_top_features(self, importance_array, n=10):
        """Return top-n features by importance."""
        ranked_idx = np.argsort(importance_array)[::-1][:n]
        return [(self.feature_names[i], importance_array[i]) for i in ranked_idx]
```

- [x] Federated vs centralised SHAP comparison in `scripts/run_xai.py` / `explainers.py`
- [ ] Compute SHAP values **per client** after each FL round
- [ ] Aggregate with weighted average
- [x] Compare vs centralised SHAP — Spearman fidelity in `results/tables/xai_stability.json`
- [x] Save fidelity results → `results/tables/xai_stability.json` (Spearman ρ≈0.96)
- [x] **Target: Spearman ρ > 0.85** (met; Jaccard not separately reported)

---

## PHASE 6 — EVALUATION & STATISTICAL TESTING
> **Target: Complete by 5 August 2026**

### 6.1 Full Comparison: FL vs Centralised

**Create `src/evaluation/comparison.py`:**
```python
"""
Comprehensive FL vs Centralised comparison.
Generates all tables needed for dissertation Chapter 4/5.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel

def generate_results_table(centralised_results, federated_results):
    """
    Creates Table I from the paper:
    Model | Config | RMSE | MAE | R² | MAPE
    """
    rows = []
    for model_name in ['random_forest', 'xgboost', 'lightgbm', 'mlp']:
        # Centralised
        c = centralised_results[model_name]
        rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'Config': 'Centralised',
            'RMSE': c['RMSE'], 'MAE': c['MAE'],
            'R2': c['R2'], 'MAPE': c['MAPE']
        })
        # Federated (FedAvg)
        f = federated_results[model_name]['fedavg']
        rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'Config': 'Federated (FedAvg)',
            'RMSE': f['RMSE'], 'MAE': f['MAE'],
            'R2': f['R2'], 'MAPE': f['MAPE']
        })
        # Federated (FedProx)
        fp = federated_results[model_name]['fedprox']
        rows.append({
            'Model': model_name.replace('_', ' ').title(),
            'Config': 'Federated (FedProx)',
            'RMSE': fp['RMSE'], 'MAE': fp['MAE'],
            'R2': fp['R2'], 'MAPE': fp['MAPE']
        })
    
    df = pd.DataFrame(rows)
    df.to_csv('results/tables/TABLE_I_accuracy_comparison.csv', index=False)
    return df

def statistical_tests(centralised_preds, federated_preds, y_true):
    """Wilcoxon signed-rank + paired t-test (H1 hypothesis test)."""
    c_errors = np.abs(centralised_preds - y_true)
    f_errors = np.abs(federated_preds   - y_true)
    
    stat_w, p_w = wilcoxon(c_errors, f_errors)
    stat_t, p_t = ttest_rel(c_errors, f_errors)
    
    print(f"Wilcoxon: statistic={stat_w:.4f}, p={p_w:.4f}")
    print(f"Paired t-test: statistic={stat_t:.4f}, p={p_t:.4f}")
    return {'wilcoxon_p': p_w, 'ttest_p': p_t}
```

- [x] Create `src/evaluation/metrics.py` / comparison outputs via `scripts/run_evaluation.py`
- [x] Generate accuracy comparison → `results/tables/model_comparison.csv` (Table I equivalent)
- [x] Generate fidelity metrics → `results/tables/xai_stability.json` (Table II equivalent)
- [x] Run Wilcoxon / statistical comparison (H1: FL vs centralised) → `statistical_tests.json`
- [x] Run paired comparison tests (see `statistical_tests.json`)
- [x] Save statistical test results → `results/tables/statistical_tests.json`
- [x] Generate convergence plot (FL rounds vs RMSE) → `results/figures/federated_convergence.png`
- [ ] Generate SHAP comparison bar chart (federated vs centralised top features) → `results/figures/shap_comparison.png`

---

### 6.2 Results Figures Checklist
All figures needed in dissertation:

- [ ] `fig1_framework_architecture.png` — Draw in draw.io or matplotlib (FL + XAI flow diagram)
- [x] `fig2_shap_summary_centralised.png` — SHAP beeswarm plot (centralised RF; also `shap_summary_centralised.png`)
- [ ] `fig3_shap_summary_federated.png` — SHAP beeswarm plot (federated)
- [x] `fig4_lime_example.png` — covered by `results/figures/lime_*.png` (rename/copy for dissertation if needed)
- [x] `fig5_fl_convergence.png` — covered by `federated_convergence.png`
- [x] `fig6_baseline_comparison.png` — covered by `baseline_rmse.png` / `baseline_r2.png`
- [x] `fig7_feature_importance_top10.png` — covered by `shap_importance_*.png`
- [x] `fig8_client_distributions.png` — Boxplot: energy distribution per client (`results/figures/fig8_client_distributions.png`)

---

## PHASE 7 — WEB APPLICATION PROTOTYPE
> **Target: Complete by 8 August 2026**

**Create `src/webapp/app.py`:**
```python
"""
Streamlit web application for EPC Energy Performance Prediction.
Demonstrates: input → FL prediction → SHAP + LIME explanations.
"""
import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ── Load models and explainers ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('results/models/centralised/xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load SHAP explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PP-XAI: EPC Energy Predictor",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Privacy-Preserving Explainable AI")
st.subheader("EPC Energy Performance Prediction for High-Rise Buildings")
st.markdown("*Powered by Federated Learning + SHAP + LIME*")

# ── Sidebar: Building Inputs ────────────────────────────────────────────────
with st.sidebar:
    st.header("Building Attributes")
    floor_area = st.slider("Total Floor Area (m²)", 500, 50000, 5000)
    building_age = st.selectbox("Building Age Band", 
                                 ["Pre-1919","1919-1944","1945-1964",
                                  "1965-1980","1981-1990","1991-2000",
                                  "2001-2010","Post-2010"])
    wall_type = st.selectbox("Wall Type",
                              ["Cavity wall, insulated","Solid brick",
                               "Curtain wall","Concrete frame"])
    glazing = st.selectbox("Glazing Type",
                            ["Single glazed","Double glazed","Triple glazed"])
    heating = st.selectbox("Heating System",
                            ["Gas boiler","Electric heat pump",
                             "District heating","Oil boiler"])
    fuel = st.selectbox("Main Fuel", ["Gas","Electricity","Oil","Biomass"])
    predict_btn = st.button("🔍 Predict Energy Performance", type="primary")

# ── Main Panel ──────────────────────────────────────────────────────────────
if predict_btn:
    model, explainer = load_models()
    
    # Build feature vector (must match training features)
    # ... encode inputs to match preprocessor output ...
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Predicted Energy Consumption",
                  f"{predicted_kwh:.1f} kWh/m²/year",
                  delta=f"EPC Band: {epc_band}")
    
    with col2:
        st.metric("CO₂ Emissions (estimated)",
                  f"{co2:.1f} kg/m²/year")
    
    # SHAP explanation
    st.subheader("🔍 Why this prediction? (SHAP Explanation)")
    shap_values = explainer.shap_values(feature_vector)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=feature_vector[0],
        feature_names=feature_names
    ), show=False)
    st.pyplot(fig)
    
    # LIME explanation
    st.subheader("📊 Local Explanation (LIME)")
    # ... render LIME explanation ...
    
    # EU AI Act compliance note
    st.info("ℹ️ This system provides explanations in compliance with EU AI Act "
            "Article 13 transparency requirements.")
```

- [x] Create `src/webapp/app.py`
- [x] Connect to trained centralised / FL artefacts
- [x] Show SHAP importance in UI (global plot + table)
- [ ] Implement LIME bar chart in UI (plots on disk; not yet embedded in Streamlit)
- [ ] Add EPC band calculation logic (A–G from kWh/m²/year)
- [ ] Add EU AI Act compliance note
- [x] Test locally: use `/bin/bash scripts/run_streamlit.sh` (arm64; plain `streamlit` under Conda/Rosetta breaks NumPy)
- [ ] Record 2-minute demo video of the app (for dissertation appendix)
- [ ] Take screenshots → `results/figures/webapp_screenshot_*.png`

---

## PHASE 8 — DISSERTATION WRITING
> **Target: Complete by 20 August 2026**

### Chapter Structure (15,000 words target)

| Chapter | Title | Words | Target Date |
|---|---|---|---|
| 1 | Introduction | 1,500 | 9 Aug |
| 2 | Literature Review | 3,000 | 11 Aug |
| 3 | Methodology | 2,500 | 13 Aug |
| 4 | Implementation | 2,000 | 15 Aug |
| 5 | Results & Discussion | 3,000 | 17 Aug |
| 6 | Conclusion | 1,000 | 18 Aug |
| References | Harvard | — | ongoing |
| Appendices | Code, Screenshots | — | 20 Aug |

---

### Chapter 1 — Introduction (~1,500 words)
- [x] Problem statement: building energy + privacy + transparency gap
- [x] Motivation: GDPR, EU AI Act, UK decarbonisation targets
- [x] Research gap: no prior FL + SHAP/LIME for high-rise EPC
- [x] Research questions (RQ1–RQ4)
- [x] Contributions summary
- [x] Dissertation structure overview

### Chapter 2 — Literature Review (~3,000 words)
- [x] 2.1 Building Energy Performance Prediction (ML approaches)
- [x] 2.2 Energy Performance Certificates (UK context, data sensitivity)
- [x] 2.3 Federated Learning (FedAvg, FedProx, non-IID challenges)
- [x] 2.4 Explainable AI (SHAP theory, LIME theory, XAI taxonomy)
- [x] 2.5 Privacy-Preserving ML (GDPR, data locality)
- [x] 2.6 Regulatory Context (EU AI Act Art.13, EPBD recast)
- [x] 2.7 Gap Analysis → your research fills this

### Chapter 3 — Methodology (~2,500 words)
- [x] 3.1 Research Design (experimental quantitative)
- [x] 3.2 Data — Scenario 1 (EPC) and Scenario 2 (OSM)
- [x] 3.3 Federated partitioning rationale (geographic non-IID)
- [x] 3.4 Model architectures (RF, XGBoost, LightGBM, MLP)
- [x] 3.5 FL framework (Flower, FedAvg vs FedProx)
- [x] 3.6 XAI integration (SHAP + LIME in federated setting)
- [x] 3.7 Evaluation framework (metrics, statistical tests)
- [x] 3.8 Ethics (no human participants, OGL licence)

### Chapter 4 — Implementation (~2,000 words)
- [x] 4.1 System architecture diagram (Fig 1) — draft in text; formal figure still open above
- [x] 4.2 Data pipeline implementation
- [x] 4.3 FL simulation setup
- [x] 4.4 XAI pipeline implementation
- [x] 4.5 Web application prototype
- [x] 4.6 Technical challenges encountered and solutions

### Chapter 5 — Results & Discussion (~3,000 words)
- [x] 5.1 Dataset statistics (Table 0)
- [x] 5.2 Baseline model performance (Table I)
- [x] 5.3 Federated model performance (Table I continued)
- [x] 5.4 FL vs Centralised comparison (H1 test)
- [x] 5.5 SHAP feature importance (Fig 2, 3 — H2 test)
- [x] 5.6 LIME local explanations (Fig 4)
- [x] 5.7 Explanation fidelity metrics (Table II — H2, H3)
- [x] 5.8 Cross-client consistency (H3)
- [x] 5.9 EU AI Act compliance analysis (H4)
- [x] 5.10 Limitations

### Chapter 6 — Conclusion (~1,000 words)
- [x] Summary of findings
- [x] Contributions revisited
- [x] Limitations
- [x] Future work (differential privacy, cross-EU EPC datasets)
- [x] Closing statement

### Final Document Tasks
- [x] Format references in Harvard style (all 13+ sources) — draft in `00_abstract_and_references.md`
- [ ] Number all figures and tables
- [ ] Write figure captions
- [x] Add abstract (300 words max) — draft present
- [x] Add acknowledgements — draft present
- [x] Check word count (~15,000 words) — drafts+appendices in `dissertation/` ≈14.6k–15k; trim/format for Hull template still open
- [ ] Proofread / run Grammarly
- [ ] Check University of Hull dissertation template compliance
- [ ] Export final PDF

---

## PHASE 9 — VIDEO PRESENTATION (Assignment 1 — 20%)
> **Target: Complete by 22 August 2026**

- [ ] Script video (10 minutes): Intro → Problem → Method → Results → Demo → Conclusion
- [ ] Record screen: show web app live demo (2 min)
- [ ] Record slides: walk through key results tables and figures (6 min)
- [ ] Record webcam: intro and conclusion (2 min)
- [ ] Edit in DaVinci Resolve / CapCut / iMovie
- [ ] Add captions / subtitles
- [ ] Export as MP4 (max file size — check Hull guidelines)
- [ ] Upload to submission portal

**Video structure (10 min):**
```
00:00–01:00  Introduction — who you are, what the problem is
01:00–02:30  Background — FL, SHAP/LIME, EPC data, EU AI Act
02:30–04:30  Methodology — your framework explained
04:30–06:00  Results — key tables and figures
06:00–08:00  LIVE DEMO — run the Streamlit app
08:00–09:30  Discussion — what the results mean
09:30–10:00  Conclusion and contributions
```

---

## PHASE 10 — FINAL SUBMISSION CHECKLIST
> **Target: 27 August 2026**

### Assignment 2 — Report & Portfolio (80%)
- [ ] Final dissertation PDF (University of Hull template)
- [x] GitHub repository link (clean, documented, README) — https://github.com/alilog99/pp-xai-dissertation
- [ ] All code committed and tagged `v1.0`
- [x] Requirements.txt up to date (on GitHub `master`)
- [x] Data access instructions in README (OGL v3.0 noted; place CSVs under `raw-data/`, not committed)

### Submission Portal
- [ ] Upload dissertation PDF
- [ ] Upload video MP4 (Assignment 1)
- [ ] Upload any appendix files
- [ ] Confirm submission receipt email
- [ ] **SUBMIT BY 27 AUGUST** (target) — hard deadline 10 September

---

## QUICK REFERENCE — Key Files Index

```
notebooks/
  01_EDA.ipynb                    → Exploratory data analysis
  02_Partitioning.ipynb           → Federated client distributions
  03_Baseline_Models.ipynb        → Centralised training & evaluation
  04_Federated_Learning.ipynb     → FL simulation results
  05_SHAP_Analysis.ipynb          → SHAP global + local explanations
  06_LIME_Analysis.ipynb          → LIME local explanations
  07_Fidelity_Evaluation.ipynb    → XAI fidelity metrics

src/
  data/preprocessor.py            → EPCPreprocessor class
  data/partitioner.py             → FederatedPartitioner class
  models/centralised_models.py    → CentralisedBaselines class
  federated/fl_server.py          → Flower server + strategies
  federated/fl_client.py          → EPCFLClient class
  xai/shap_explainer.py           → SHAPExplainer class
  xai/lime_explainer.py           → LIMEExplainer class
  xai/fed_xai.py                  → FederatedXAIPipeline class
  evaluation/metrics.py           → evaluate_model() function
  evaluation/comparison.py        → FL vs Centralised comparison
  webapp/app.py                   → Streamlit application

results/
  tables/TABLE_I_accuracy.csv     → Main accuracy table (dissertation)
  tables/TABLE_II_fidelity.csv    → XAI fidelity table (dissertation)
  figures/fig1_architecture.png   → Framework diagram
  figures/fig2_shap_summary.png   → SHAP beeswarm plot
  figures/fig3_lime_example.png   → LIME local explanation
  figures/fig4_convergence.png    → FL training convergence
```

---

## HYPOTHESES VERIFICATION TRACKER

| Hypothesis | Target | Achieved | Status |
|---|---|---|---|
| H1: FL within 5% of centralised RMSE | RMSE diff < 5% | TBD | ⬜ |
| H2: SHAP Spearman ρ > 0.85 | ρ > 0.85 | TBD | ⬜ |
| H3: Top-5 features consistent across 80% clients | ≥2/3 clients agree | TBD | ⬜ |
| H4: SHAP/LIME satisfies EU AI Act Art.13 | Qualitative | TBD | ⬜ |

---

## PROGRESS TRACKER

| Phase | Description | Status | Done Date |
|---|---|---|---|
| 0 | Environment Setup | ⬜ | |
| 1 | Data Download + EDA | ⬜ | |
| 2 | Preprocessing + Partitioning | ⬜ | |
| 3 | Centralised Baselines | ⬜ | |
| 4 | Federated Learning | ⬜ | |
| 5 | SHAP + LIME + Fed-XAI | ⬜ | |
| 6 | Evaluation + Stats | ⬜ | |
| 7 | Web App Prototype | ⬜ | |
| 8 | Dissertation Writing | ⬜ | |
| 9 | Video Presentation | ⬜ | |
| 10 | Final Submission | ⬜ | |

> **Update status:** ⬜ Not started | 🟡 In progress | ✅ Complete | ❌ Blocked

---

*Generated: 17 July 2026 | University of Hull MSc Applied AI | Student: 202440724*

#!/usr/bin/env python3
"""Run Federated Averaging simulation across city clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.fl_client import fedavg_simulate
from src.models.centralised_models import regression_metrics


def main() -> None:
    processed = ROOT / "data" / "processed"
    federated = ROOT / "data" / "federated"
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")

    client_data = {}
    for city in ["london", "manchester", "birmingham"]:
        xp, yp = federated / f"X_{city}.npy", federated / f"y_{city}.npy"
        if xp.exists() and yp.exists():
            Xc, yc = np.load(xp), np.load(yp)
            if len(Xc) > 10:
                client_data[city] = (Xc, yc)
                print(f"Client {city}: {len(Xc)} samples")

    if len(client_data) < 2:
        # Fallback: split central train set into 3 synthetic clients
        print("WARNING: insufficient city clients — splitting train set into 3 partitions")
        X_train = np.load(processed / "X_train.npy")
        y_train = np.load(processed / "y_train.npy")
        idx = np.array_split(np.arange(len(X_train)), 3)
        names = ["london", "manchester", "birmingham"]
        client_data = {n: (X_train[i], y_train[i]) for n, i in zip(names, idx) if len(i) > 0}

    print(f"Starting FedAvg with {len(client_data)} clients...")
    result = fedavg_simulate(client_data, X_test, y_test, rounds=8, local_epochs=4)

    pred = result["predictions"]
    metrics = regression_metrics(y_test, pred)
    metrics["test_rmse"] = result["test_rmse"]
    metrics["test_r2"] = result["test_r2"]
    metrics["test_mae"] = metrics["MAE"]
    metrics["test_mape"] = metrics["MAPE"]

    tables = ROOT / "results" / "tables"
    models_dir = ROOT / "results" / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result["history"]).to_csv(tables / "federated_rounds.csv", index=False)
    (tables / "federated_metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}, indent=2))
    np.save(processed / "pred_federated.npy", pred)
    torch.save(result["model"].state_dict(), models_dir / "federated_mlp.pt")
    joblib.dump(result["parameters"], models_dir / "federated_parameters.joblib")

    print("Federated metrics:", metrics)


if __name__ == "__main__":
    main()

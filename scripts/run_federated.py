#!/usr/bin/env python3
"""Run Federated Averaging and FedProx simulations across city clients."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.fl_client import fedavg_simulate, fedprox_simulate
from src.models.centralised_models import regression_metrics

ROUNDS = 8
LOCAL_EPOCHS = 4
FEDPROX_MU = 0.01


def _load_client_data(
    processed: Path, federated: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    client_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for city in ["london", "manchester", "birmingham"]:
        xp, yp = federated / f"X_{city}.npy", federated / f"y_{city}.npy"
        if xp.exists() and yp.exists():
            Xc, yc = np.load(xp), np.load(yp)
            if len(Xc) > 10:
                client_data[city] = (Xc, yc)
                print(f"Client {city}: {len(Xc)} samples")

    if len(client_data) < 2:
        print("WARNING: insufficient city clients — splitting train set into 3 partitions")
        X_train = np.load(processed / "X_train.npy")
        y_train = np.load(processed / "y_train.npy")
        idx = np.array_split(np.arange(len(X_train)), 3)
        names = ["london", "manchester", "birmingham"]
        client_data = {n: (X_train[i], y_train[i]) for n, i in zip(names, idx) if len(i) > 0}
    return client_data


def _metrics_from_result(y_test: np.ndarray, result: dict) -> dict:
    pred = result["predictions"]
    metrics = regression_metrics(y_test, pred)
    metrics["test_rmse"] = result["test_rmse"]
    metrics["test_r2"] = result["test_r2"]
    metrics["test_mae"] = metrics["MAE"]
    metrics["test_mape"] = metrics["MAPE"]
    metrics["strategy"] = result.get("strategy", "FedAvg")
    metrics["mu"] = float(result.get("mu", 0.0))
    return metrics


def _save_json(path: Path, metrics: dict) -> None:
    path.write_text(
        json.dumps(
            {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
            indent=2,
        )
    )


def _plot_convergence(hist_avg: pd.DataFrame, hist_prox: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(hist_avg["round"], hist_avg["rmse"], marker="o", label="FedAvg")
    plt.plot(hist_prox["round"], hist_prox["rmse"], marker="s", label=f"FedProx (μ={FEDPROX_MU})")
    plt.xlabel("Federated round")
    plt.ylabel("Test RMSE")
    plt.title("Federated convergence: FedAvg vs FedProx")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.with_name("fl_convergence.png"), dpi=150)
    plt.close()


def main() -> None:
    processed = ROOT / "data" / "processed"
    federated = ROOT / "data" / "federated"
    tables = ROOT / "results" / "tables"
    models_dir = ROOT / "results" / "models"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")
    client_data = _load_client_data(processed, federated)

    print(f"Starting FedAvg with {len(client_data)} clients ({ROUNDS} rounds)...")
    result_avg = fedavg_simulate(
        client_data, X_test, y_test, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS
    )
    metrics_avg = _metrics_from_result(y_test, result_avg)
    hist_avg = pd.DataFrame(result_avg["history"])
    hist_avg.to_csv(tables / "federated_rounds.csv", index=False)
    hist_avg.to_csv(tables / "fl_convergence_fedavg.csv", index=False)
    _save_json(tables / "federated_metrics.json", metrics_avg)
    np.save(processed / "pred_federated.npy", result_avg["predictions"])
    torch.save(result_avg["model"].state_dict(), models_dir / "federated_mlp.pt")
    joblib.dump(result_avg["parameters"], models_dir / "federated_parameters.joblib")
    print("FedAvg metrics:", metrics_avg)

    print(f"\nStarting FedProx (mu={FEDPROX_MU}) with {len(client_data)} clients...")
    result_prox = fedprox_simulate(
        client_data,
        X_test,
        y_test,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        mu=FEDPROX_MU,
    )
    metrics_prox = _metrics_from_result(y_test, result_prox)
    hist_prox = pd.DataFrame(result_prox["history"])
    hist_prox.to_csv(tables / "fl_convergence_fedprox.csv", index=False)
    _save_json(tables / "federated_metrics_fedprox.json", metrics_prox)
    np.save(processed / "pred_federated_fedprox.npy", result_prox["predictions"])
    torch.save(result_prox["model"].state_dict(), models_dir / "federated_mlp_fedprox.pt")
    joblib.dump(result_prox["parameters"], models_dir / "federated_parameters_fedprox.joblib")
    print("FedProx metrics:", metrics_prox)

    _plot_convergence(hist_avg, hist_prox, figures / "federated_convergence.png")
    print("Wrote convergence overlay → results/figures/federated_convergence.png")


if __name__ == "__main__":
    main()

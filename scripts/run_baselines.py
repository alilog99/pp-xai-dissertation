#!/usr/bin/env python3
"""Train centralised baseline models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_table, plot_metrics_bar, plot_pred_vs_actual
from src.models.centralised_models import save_models, train_and_evaluate


def main() -> None:
    processed = ROOT / "data" / "processed"
    X_train = np.load(processed / "X_train.npy")
    X_test = np.load(processed / "X_test.npy")
    y_train = np.load(processed / "y_train.npy")
    y_test = np.load(processed / "y_test.npy")

    print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
    metrics, models, preds = train_and_evaluate(X_train, y_train, X_test, y_test)

    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    models_dir = ROOT / "results" / "models"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    df = metrics_table(metrics)
    df.to_csv(tables / "baseline_metrics.csv", index=False)
    (tables / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Record fixed hyperparameters used (Optuna not run)
    hyperparams_src = ROOT / "results" / "tables" / "baseline_hyperparams.json"
    if not hyperparams_src.exists():
        # Fallback write if file was deleted — mirrors build_models() defaults
        from src.models.centralised_models import build_models

        built = build_models(random_state=42)
        models_meta = {}
        for name, model in built.items():
            params = model.get_params() if hasattr(model, "get_params") else {}
            # Keep JSON-serialisable scalars/lists only
            clean = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    clean[k] = v
                elif isinstance(v, tuple):
                    clean[k] = list(v)
            models_meta[name] = {"class": type(model).__module__ + "." + type(model).__name__, **clean}
        payload = {
            "tuning_method": "fixed_defaults",
            "optuna_used": False,
            "random_state": 42,
            "models": models_meta,
        }
        hyperparams_src.write_text(json.dumps(payload, indent=2))
    else:
        # Refresh timestamp note that training used the documented fixed settings
        pass

    plot_metrics_bar(df, figures / "baseline_rmse.png", "RMSE")
    plot_metrics_bar(df, figures / "baseline_r2.png", "R2")

    best = min(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0]
    # All pred-vs-actual plots use the held-out test set (never training predictions).
    for name, pred in preds.items():
        plot_pred_vs_actual(
            y_test,
            pred,
            figures / f"pred_vs_actual_{name}.png",
            title=f"{name} — pred vs actual (held-out test set)",
        )
        np.save(processed / f"pred_{name}.npy", pred)

    save_models(models, models_dir)
    print(f"Best model by RMSE: {best}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

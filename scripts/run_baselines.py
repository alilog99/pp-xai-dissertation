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
    plot_metrics_bar(df, figures / "baseline_rmse.png", "RMSE")
    plot_metrics_bar(df, figures / "baseline_r2.png", "R2")

    best = min(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0]
    plot_pred_vs_actual(y_test, preds[best], figures / f"pred_vs_actual_{best}.png", title=f"{best} — pred vs actual")
    np.save(processed / f"pred_{best}.npy", preds[best])
    for name, pred in preds.items():
        np.save(processed / f"pred_{name}.npy", pred)

    save_models(models, models_dir)
    print(f"Best model by RMSE: {best}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

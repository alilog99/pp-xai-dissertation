#!/usr/bin/env python3
"""Aggregate evaluation tables, statistical tests, and comparison figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    paired_stat_tests,
    plot_metrics_bar,
    plot_pred_vs_actual,
    save_comparison,
)
from src.models.centralised_models import regression_metrics


def main() -> None:
    processed = ROOT / "data" / "processed"
    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    y_test = np.load(processed / "y_test.npy")
    baseline = json.loads((tables / "baseline_metrics.json").read_text())

    fed_metrics = {}
    if (tables / "federated_metrics.json").exists():
        fed_metrics = json.loads((tables / "federated_metrics.json").read_text())
    elif (processed / "pred_federated.npy").exists():
        pred_f = np.load(processed / "pred_federated.npy")
        m = regression_metrics(y_test, pred_f)
        fed_metrics = {
            "test_rmse": m["RMSE"],
            "test_mae": m["MAE"],
            "test_r2": m["R2"],
            "test_mape": m["MAPE"],
            **m,
        }

    df = save_comparison(baseline, fed_metrics, tables)
    plot_metrics_bar(df.rename(columns={"model": "model"}), figures / "all_models_rmse.png", "RMSE")

    # Statistical test: best central vs federated
    best = min(baseline.items(), key=lambda kv: kv[1]["RMSE"])[0]
    pred_c_path = processed / f"pred_{best}.npy"
    pred_f_path = processed / "pred_federated.npy"
    if pred_c_path.exists() and pred_f_path.exists():
        pred_c = np.load(pred_c_path)
        pred_f = np.load(pred_f_path)
        tests = paired_stat_tests(y_test, pred_c, pred_f, name_a=best, name_b="federated")
        (tables / "statistical_tests.json").write_text(json.dumps(tests, indent=2))
        print("Statistical tests:", tests)
        plot_pred_vs_actual(y_test, pred_f, figures / "pred_vs_actual_federated.png", title="Federated MLP")

    # Round history plot if available
    hist_path = tables / "federated_rounds.csv"
    if hist_path.exists():
        import matplotlib.pyplot as plt

        hist = pd.read_csv(hist_path)
        plt.figure(figsize=(7, 4))
        plt.plot(hist["round"], hist["rmse"], marker="o")
        plt.xlabel("Federated round")
        plt.ylabel("Test RMSE")
        plt.title("FedAvg convergence")
        plt.tight_layout()
        plt.savefig(figures / "federated_convergence.png", dpi=150)
        plt.close()

    print(df.to_string(index=False))
    print("Evaluation artefacts written to results/")


if __name__ == "__main__":
    main()

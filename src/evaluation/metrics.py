"""Evaluation metrics, statistical tests, and comparison tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.models.centralised_models import regression_metrics


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "model"})
    return df


def paired_stat_tests(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> dict[str, float]:
    err_a = np.abs(y_true - pred_a)
    err_b = np.abs(y_true - pred_b)
    # Wilcoxon signed-rank on absolute errors
    try:
        w_stat, w_p = stats.wilcoxon(err_a, err_b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    t_stat, t_p = stats.ttest_rel(err_a, err_b)
    return {
        f"{name_a}_MAE": float(err_a.mean()),
        f"{name_b}_MAE": float(err_b.mean()),
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else np.nan,
        "wilcoxon_p": float(w_p) if w_p == w_p else np.nan,
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
    }


def plot_metrics_bar(df: pd.DataFrame, out_path: str | Path, metric: str = "RMSE") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[metric])
    plt.ylabel(metric)
    plt.title(f"Model comparison — {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path,
    title: str = "Predicted vs Actual",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_comparison(
    central_metrics: dict[str, dict[str, float]],
    federated_metrics: dict[str, float],
    out_dir: str | Path,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in central_metrics.items():
        rows.append({"setting": "centralised", "model": name, **m})
    rows.append(
        {
            "setting": "federated",
            "model": "fedavg_mlp",
            "RMSE": federated_metrics.get("test_rmse"),
            "MAE": federated_metrics.get("test_mae"),
            "R2": federated_metrics.get("test_r2"),
            "MAPE": federated_metrics.get("test_mape"),
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    return df

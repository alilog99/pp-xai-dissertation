"""SHAP and LIME explainers for centralised and federated models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except Exception:  # noqa: BLE001
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # noqa: BLE001
    LimeTabularExplainer = None  # type: ignore


def _predict_fn_from_model(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(model, "predict"):
        return lambda X: np.asarray(model.predict(X), dtype=float)

    # Torch MLP
    import torch

    def _predict(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            return model(t).cpu().numpy()

    return _predict


def run_shap(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_dir: str | Path,
    label: str = "model",
    max_samples: int = 200,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if shap is None:
        print("SHAP not available — skipping")
        return {}

    predict = _predict_fn_from_model(model)
    bg = X_background[: min(100, len(X_background))]
    xe = X_explain[: min(max_samples, len(X_explain))]

    # KernelExplainer works for any model (tree or neural)
    explainer = shap.KernelExplainer(predict, bg)
    shap_values = explainer.shap_values(xe, nsamples=100)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names[: len(mean_abs)], "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(out_dir / f"shap_importance_{label}.csv", index=False)

    plt.figure(figsize=(10, 6))
    top = importance.head(15)
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP|")
    plt.title(f"SHAP feature importance — {label}")
    plt.tight_layout()
    fig_path = out_dir / f"shap_importance_{label}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    return {"importance": importance, "shap_values": shap_values, "figure": str(fig_path)}


def run_lime(
    model: Any,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_dir: str | Path,
    label: str = "model",
    n_instances: int = 5,
) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if LimeTabularExplainer is None:
        print("LIME not available — skipping")
        return []

    predict = _predict_fn_from_model(model)
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
    )
    results = []
    for i in range(min(n_instances, len(X_explain))):
        exp = explainer.explain_instance(X_explain[i], predict, num_features=10)
        fig = exp.as_pyplot_figure()
        fig_path = out_dir / f"lime_{label}_instance_{i}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        weights = exp.as_list()
        results.append({"instance": i, "weights": weights, "figure": str(fig_path)})
        pd.DataFrame(weights, columns=["feature", "weight"]).to_csv(
            out_dir / f"lime_{label}_instance_{i}.csv", index=False
        )
    return results


def explanation_stability(
    shap_a: np.ndarray,
    shap_b: np.ndarray,
) -> dict[str, float]:
    """Compare two SHAP matrices (e.g. centralised vs federated) via rank correlation."""
    from scipy.stats import spearmanr

    mean_a = np.mean(np.abs(shap_a), axis=0)
    mean_b = np.mean(np.abs(shap_b), axis=0)
    n = min(len(mean_a), len(mean_b))
    corr, p = spearmanr(mean_a[:n], mean_b[:n])
    return {"spearman_corr": float(corr), "p_value": float(p)}

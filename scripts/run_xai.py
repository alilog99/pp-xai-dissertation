#!/usr/bin/env python3
"""Run SHAP and LIME on best centralised model and federated MLP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.fl_client import EnergyMLP, set_parameters
from src.xai.explainers import explanation_stability, run_lime, run_shap


def _best_central_model(models_dir: Path):
    metrics_path = ROOT / "results" / "tables" / "baseline_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        best = min(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0]
    else:
        best = "random_forest"
    path = models_dir / f"{best}.joblib"
    if not path.exists():
        # fallback any available
        cands = list(models_dir.glob("*.joblib"))
        cands = [c for c in cands if "federated" not in c.name and "parameters" not in c.name]
        if not cands:
            raise SystemExit("No centralised models found — run baselines first")
        path = cands[0]
        best = path.stem
    return best, joblib.load(path)


def main() -> None:
    processed = ROOT / "data" / "processed"
    models_dir = ROOT / "results" / "models"
    fig_dir = ROOT / "results" / "figures"
    table_dir = ROOT / "results" / "tables"

    X_train = np.load(processed / "X_train.npy")
    X_test = np.load(processed / "X_test.npy")
    feature_names = json.loads((processed / "feature_names.json").read_text())

    # Subsample for KernelExplainer speed
    n_bg, n_ex = min(80, len(X_train)), min(60, len(X_test))
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=n_bg, replace=False)
    ex_idx = rng.choice(len(X_test), size=n_ex, replace=False)

    name, central = _best_central_model(models_dir)
    print(f"Explaining centralised model: {name}")
    shap_c = run_shap(
        central,
        X_train[bg_idx],
        X_test[ex_idx],
        feature_names,
        fig_dir,
        label=f"central_{name}",
        max_samples=n_ex,
    )
    run_lime(central, X_train[bg_idx], X_test[ex_idx], feature_names, fig_dir, label=f"central_{name}", n_instances=3)

    # Federated MLP
    fed_path = models_dir / "federated_mlp.pt"
    params_path = models_dir / "federated_parameters.joblib"
    shap_f = {}
    if fed_path.exists() or params_path.exists():
        fed_model = EnergyMLP(X_train.shape[1])
        if params_path.exists():
            set_parameters(fed_model, joblib.load(params_path))
        else:
            fed_model.load_state_dict(torch.load(fed_path, map_location="cpu"))
        print("Explaining federated MLP...")
        shap_f = run_shap(
            fed_model,
            X_train[bg_idx],
            X_test[ex_idx],
            feature_names,
            fig_dir,
            label="federated_mlp",
            max_samples=n_ex,
        )
        run_lime(fed_model, X_train[bg_idx], X_test[ex_idx], feature_names, fig_dir, label="federated_mlp", n_instances=3)

        if shap_c.get("shap_values") is not None and shap_f.get("shap_values") is not None:
            stab = explanation_stability(shap_c["shap_values"], shap_f["shap_values"])
            (table_dir / "xai_stability.json").write_text(json.dumps(stab, indent=2))
            print("Explanation stability (central vs federated):", stab)

    print("XAI complete.")


if __name__ == "__main__":
    main()

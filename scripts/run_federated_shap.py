#!/usr/bin/env python3
"""Per-client SHAP after each FedAvg round + weighted aggregation.

Outputs:
  results/tables/federated_shap_rounds.json
  results/tables/federated_shap_aggregated.csv
  results/models/shap_federated_per_round.npz
  results/figures/shap_comparison.png
  results/figures/shap_summary_federated.png (bar of final aggregated importance)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.fl_client import EnergyMLP, get_parameters, set_parameters, train_local, evaluate_local
from src.xai.fed_xai import FederatedXAIPipeline
from src.xai.explainers import _predict_fn_from_model


def _client_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    federated = ROOT / "data" / "federated"
    out = {}
    for city in ["london", "manchester", "birmingham"]:
        xp, yp = federated / f"X_{city}.npy", federated / f"y_{city}.npy"
        if xp.exists() and yp.exists():
            Xc, yc = np.load(xp), np.load(yp)
            if len(Xc) > 10:
                out[city] = (Xc, yc)
    return out


def _shap_mean_abs(
    model: EnergyMLP,
    X_bg: np.ndarray,
    X_ex: np.ndarray,
    nsamples: int = 50,
) -> np.ndarray:
    predict = _predict_fn_from_model(model)
    explainer = shap.KernelExplainer(predict, X_bg)
    values = explainer.shap_values(X_ex, nsamples=nsamples, silent=True)
    return np.asarray(values, dtype=float)


def main() -> None:
    processed = ROOT / "data" / "processed"
    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    models_dir = ROOT / "results" / "models"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_names = json.loads((processed / "feature_names.json").read_text())
    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")
    client_data = _client_data()
    if len(client_data) < 2:
        raise SystemExit("Need at least 2 federated client arrays under data/federated/")

    # Centralised RF mean |SHAP| for fidelity (reuse saved if present)
    cen_imp = None
    rf_shap_path = models_dir / "shap_centralised_rf.npy"
    if rf_shap_path.exists():
        cen_imp = np.mean(np.abs(np.load(rf_shap_path)), axis=0)
        print(f"Loaded centralised RF SHAP from {rf_shap_path}")
    else:
        # fallback: GB importance CSV if RF missing
        csvs = list(figures.glob("shap_importance_central_*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0])
            # align by feature name order
            m = dict(zip(df["feature"], df["mean_abs_shap"]))
            cen_imp = np.array([m.get(f, 0.0) for f in feature_names], dtype=float)

    rounds = 8
    local_epochs = 3
    n_bg, n_ex = 25, 30
    nsamples = 50
    rng = np.random.default_rng(42)

    n_features = next(iter(client_data.values()))[0].shape[1]
    global_model = EnergyMLP(n_features)
    pipeline = FederatedXAIPipeline(feature_names)

    # Fixed explain indices per client for comparability across rounds
    client_explain: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for city, (Xc, yc) in client_data.items():
        bg_idx = rng.choice(len(Xc), size=min(n_bg, len(Xc)), replace=False)
        ex_idx = rng.choice(len(Xc), size=min(n_ex, len(Xc)), replace=False)
        client_explain[city] = (Xc[bg_idx], Xc[ex_idx])

    round_arrays: dict[str, list] = {"rounds": [], "aggregated": []}
    for city in client_data:
        round_arrays[city] = []

    print(f"Running FedAvg ({rounds} rounds) with per-client SHAP each round...")
    for rnd in range(1, rounds + 1):
        client_weights = []
        client_sizes = []
        global_params = get_parameters(global_model)
        local_models: dict[str, EnergyMLP] = {}

        for city, (X_c, y_c) in client_data.items():
            local = EnergyMLP(n_features)
            set_parameters(local, global_params)
            train_local(local, X_c, y_c, epochs=local_epochs)
            client_weights.append(get_parameters(local))
            client_sizes.append(len(X_c))
            local_models[city] = local

        total = float(sum(client_sizes))
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            stacked = np.stack(
                [w[layer_idx] * (client_sizes[i] / total) for i, w in enumerate(client_weights)],
                axis=0,
            )
            aggregated.append(np.sum(stacked, axis=0))
        set_parameters(global_model, aggregated)

        metrics = evaluate_local(global_model, X_test, y_test)
        print(f"  Round {rnd}: RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f} — computing client SHAP...")

        # Per-client SHAP on each client's local model after local training
        # (captures client-specific attributions before aggregation)
        pipeline.clear_clients()
        for city, local in local_models.items():
            X_bg, X_ex = client_explain[city]
            shap_vals = _shap_mean_abs(local, X_bg, X_ex, nsamples=nsamples)
            pipeline.add_client_shap(city, shap_vals, n_samples=len(client_data[city][0]))
            round_arrays[city].append(np.mean(np.abs(shap_vals), axis=0))

        entry = pipeline.record_round(rnd)
        round_arrays["rounds"].append(rnd)
        round_arrays["aggregated"].append(np.array(entry["aggregated_mean_abs_shap"]))
        top3 = ", ".join(f"{n.split('__')[-1][:24]}={v:.2f}" for n, v in entry["top_features"][:3])
        print(f"    aggregated top3: {top3}")

    # Final fidelity vs centralised
    fidelity = {}
    if cen_imp is not None:
        fidelity = pipeline.compute_fidelity_metrics(cen_imp)

    # Persist
    history_path = tables / "federated_shap_rounds.json"
    serialisable = []
    for e in pipeline.round_history:
        serialisable.append(
            {
                "round": e["round"],
                "client_sizes": e["client_sizes"],
                "top_features": [{"feature": a, "mean_abs_shap": b} for a, b in e["top_features"]],
                # omit full vectors from JSON to keep file smaller — stored in npz
            }
        )
    payload = {
        "rounds": rounds,
        "clients": list(client_data.keys()),
        "client_sizes": {k: len(v[0]) for k, v in client_data.items()},
        "aggregation": "weighted_average_by_client_n",
        "shap_config": {"n_background": n_bg, "n_explain": n_ex, "nsamples": nsamples},
        "history_summary": serialisable,
        "final_top_features": [
            {"feature": a, "mean_abs_shap": b} for a, b in pipeline.get_top_features(pipeline.aggregated_shap, 15)
        ],
        "fidelity_vs_centralised_rf": fidelity,
    }
    history_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {history_path}")

    agg_final = pipeline.aggregated_shap
    assert agg_final is not None
    pd.DataFrame(
        {"feature": feature_names[: len(agg_final)], "mean_abs_shap_weighted": agg_final}
    ).sort_values("mean_abs_shap_weighted", ascending=False).to_csv(
        tables / "federated_shap_aggregated.csv", index=False
    )

    np.savez_compressed(
        models_dir / "shap_federated_per_round.npz",
        rounds=np.array(round_arrays["rounds"]),
        aggregated=np.stack(round_arrays["aggregated"], axis=0),
        **{f"client_{c}": np.stack(round_arrays[c], axis=0) for c in client_data},
        feature_names=np.array(feature_names[: len(agg_final)]),
    )
    print(f"Saved {models_dir / 'shap_federated_per_round.npz'}")

    # Comparison bar chart: centralised RF vs final weighted federated
    if cen_imp is not None:
        n = min(len(cen_imp), len(agg_final), len(feature_names))
        # top-10 by federated importance for readable chart
        top_idx = np.argsort(agg_final[:n])[::-1][:10]
        labels = [feature_names[i].replace("cat__", "").replace("num__", "")[:40] for i in top_idx]
        x = np.arange(len(top_idx))
        width = 0.38
        plt.figure(figsize=(11, 6))
        plt.bar(x - width / 2, cen_imp[top_idx], width, label="Centralised RF")
        plt.bar(x + width / 2, agg_final[top_idx], width, label="Federated weighted avg")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.ylabel("Mean |SHAP|")
        plt.title("SHAP comparison — centralised RF vs federated (final round)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "shap_comparison.png", dpi=150)
        plt.close()
        print(f"Saved {figures / 'shap_comparison.png'}")

    # Federated summary bar (proxy beeswarm for MLP KernelSHAP)
    top = pd.DataFrame(
        {"feature": feature_names[: len(agg_final)], "mean_abs_shap": agg_final}
    ).sort_values("mean_abs_shap", ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
    plt.xlabel("Weighted mean |SHAP|")
    plt.title("Federated SHAP summary (weighted client aggregation, final round)")
    plt.tight_layout()
    plt.savefig(figures / "shap_summary_federated.png", dpi=150)
    plt.savefig(figures / "fig3_shap_summary_federated.png", dpi=150)
    plt.close()
    print(f"Saved {figures / 'shap_summary_federated.png'}")

    # Persist final global model params for consistency
    torch.save(global_model.state_dict(), models_dir / "federated_mlp_shaprun.pt")
    joblib.dump(get_parameters(global_model), models_dir / "federated_parameters_shaprun.joblib")
    print("DONE")


if __name__ == "__main__":
    main()

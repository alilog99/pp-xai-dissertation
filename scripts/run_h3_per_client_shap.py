#!/usr/bin/env python3
"""Per-client SHAP + Jaccard similarity to test Hypothesis H3.

H3: Top-5 influential building features are consistent across at least 80%
of federated clients.

Uses the centralised Gradient Boosting model on each city's transformed
client matrix separately (not federated weights).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import shap
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"SHAP required for H3 analysis: {exc}") from exc

CITIES = ["london", "manchester", "birmingham"]
N_BACKGROUND = 50
N_EXPLAIN = 40
TOP_K = 5
RNG = np.random.default_rng(42)


def _load_best_central(models_dir: Path):
    metrics_path = ROOT / "results" / "tables" / "baseline_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        best = min(metrics.items(), key=lambda kv: kv[1]["RMSE"])[0]
    else:
        best = "gradient_boosting"
    path = models_dir / f"{best}.joblib"
    if not path.exists():
        raise SystemExit(f"Missing model: {path}")
    return best, joblib.load(path)


def _client_mean_abs_shap(model, X: np.ndarray, feature_names: list[str]) -> pd.Series:
    """Mean |SHAP| on a within-client subsample (TreeExplainer for GB)."""
    n = len(X)
    if n < 20:
        raise SystemExit(f"Client too small for SHAP (n={n})")

    n_bg = min(N_BACKGROUND, max(10, n // 3))
    n_ex = min(N_EXPLAIN, max(10, n // 4))
    # Non-overlapping background / explain draws where possible
    perm = RNG.permutation(n)
    if n_bg + n_ex <= n:
        bg_idx = perm[:n_bg]
        ex_idx = perm[n_bg : n_bg + n_ex]
    else:
        bg_idx = perm[:n_bg]
        ex_idx = perm[:n_ex]

    X_bg = X[bg_idx]
    X_ex = X[ex_idx]

    # Exact tree SHAP for GradientBoosting; fallback to KernelExplainer
    if hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_ex)
    else:
        explainer = shap.KernelExplainer(model.predict, X_bg)
        values = explainer.shap_values(X_ex, nsamples=100)

    values = np.asarray(values)
    if values.ndim == 3:  # safety for multi-output
        values = values[:, :, 0]
    mean_abs = np.mean(np.abs(values), axis=0)
    return pd.Series(mean_abs, index=feature_names[: len(mean_abs)], name="mean_abs_shap")


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def main() -> None:
    federated = ROOT / "data" / "federated"
    models_dir = ROOT / "results" / "models"
    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    feature_names = json.loads((ROOT / "data" / "processed" / "feature_names.json").read_text())
    model_name, model = _load_best_central(models_dir)
    print(f"H3 analysis using centralised model: {model_name}")

    per_client: dict[str, pd.Series] = {}
    top5: dict[str, list[str]] = {}
    for city in CITIES:
        X = np.load(federated / f"X_{city}.npy")
        print(f"  SHAP for {city} (n={len(X)})...")
        series = _client_mean_abs_shap(model, X, feature_names)
        per_client[city] = series
        top5[city] = list(series.sort_values(ascending=False).head(TOP_K).index)
        print(f"    Top-5: {top5[city]}")

    # Wide table: Rank | Feature | London_SHAP | ...
    # Union of top-10 features across clients for readability, ranked by mean across cities
    all_imp = pd.DataFrame({c.title(): s for c, s in per_client.items()})
    all_imp["mean_across_clients"] = all_imp.mean(axis=1)
    all_imp = all_imp.sort_values("mean_across_clients", ascending=False)
    top_union = all_imp.head(max(15, TOP_K * 3)).copy()
    top_union.insert(0, "Rank", np.arange(1, len(top_union) + 1))
    top_union = top_union.reset_index().rename(columns={"index": "Feature"})
    # Rename columns to match requested schema
    out_cols = ["Rank", "Feature"] + [f"{c.title()}_SHAP" for c in CITIES]
    rename_map = {c.title(): f"{c.title()}_SHAP" for c in CITIES}
    shap_table = top_union.rename(columns=rename_map)[out_cols]
    shap_table.to_csv(tables / "per_client_shap.csv", index=False)

    # Pairwise Jaccard on top-5
    pairs = []
    jaccard_scores = []
    for a, b in combinations(CITIES, 2):
        sa, sb = set(top5[a]), set(top5[b])
        score = _jaccard(sa, sb)
        shared = sorted(sa & sb)
        pairs.append(
            {
                "Pair": f"{a.title()} vs {b.title()}",
                "Jaccard_score": round(score, 4),
                "Shared_top5_features": "; ".join(shared) if shared else "(none)",
                "n_shared": len(shared),
            }
        )
        jaccard_scores.append(score)
        print(f"  Jaccard {a} vs {b}: {score:.4f}  shared={shared}")

    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    jaccard_df = pd.DataFrame(pairs)
    jaccard_df.loc[len(jaccard_df)] = {
        "Pair": "Average (all pairs)",
        "Jaccard_score": round(avg_jaccard, 4),
        "Shared_top5_features": "",
        "n_shared": np.nan,
    }
    jaccard_df.to_csv(tables / "jaccard_similarity.csv", index=False)

    # Cross-client consistency of top-5 membership
    sets = {c: set(top5[c]) for c in CITIES}
    in_all_3 = sorted(sets["london"] & sets["manchester"] & sets["birmingham"])
    in_at_least_2 = sorted(
        f
        for f in set.union(*sets.values())
        if sum(f in sets[c] for c in CITIES) >= 2
    )
    # With 3 clients, ≥80% of clients ⇒ all 3 (ceil(0.8*3)=3)
    consistency_all3 = len(in_all_3) / TOP_K
    consistency_ge2 = len(in_at_least_2) / TOP_K

    h3_pass_jaccard = avg_jaccard > 0.60
    h3_pass_80pct = consistency_all3 >= 0.80  # ≥4/5 features in all clients, or use all-3 count
    # Operationalise stated H3 ("≥80% of clients"): feature must appear in 3/3 clients.
    # Also report checklist-style ≥2/3 agreement.
    status = "PASS" if h3_pass_jaccard else "FAIL"

    verdict = {
        "hypothesis": "H3",
        "statement": (
            "Top-5 influential building features are consistent across "
            "at least 80% of federated clients"
        ),
        "model_used": model_name,
        "top5_per_client": top5,
        "features_in_all_3_clients": in_all_3,
        "features_in_at_least_2_clients": in_at_least_2,
        "cross_client_consistency_all3": round(consistency_all3, 4),
        "cross_client_consistency_ge2of3": round(consistency_ge2, 4),
        "average_jaccard": round(avg_jaccard, 4),
        "pairwise_jaccard": pairs,
        "pass_criterion": "average Jaccard > 0.60",
        "h3_status": status,
        "note_80pct_clients": (
            "With 3 clients, ≥80% requires membership in all 3 clients "
            f"(ceil(0.8×3)=3). Features in all 3: {len(in_all_3)}/5."
        ),
    }
    (tables / "h3_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n" + "=" * 60)
    print(f"H3 RESULT: Average Jaccard = {avg_jaccard:.4f}")
    print(f"Cross-client consistency = {len(in_all_3)}/5 features in all clients")
    print(f"Features in ≥2/3 clients = {len(in_at_least_2)}/5 → {in_at_least_2}")
    print(f"Features in all 3 clients = {in_all_3}")
    print(f"H3 STATUS: {status}" + ("" if h3_pass_jaccard else " (average Jaccard ≤ 0.60)"))
    print("=" * 60)

    # Grouped bar chart: top-10 by mean |SHAP| across clients
    plot_df = all_imp.head(10).copy()
    x = np.arange(len(plot_df))
    width = 0.25
    colours = {"london": "#1f77b4", "manchester": "#ff7f0e", "birmingham": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, city in enumerate(CITIES):
        ax.bar(
            x + (i - 1) * width,
            plot_df[city.title()].values,
            width,
            label=city.title(),
            color=colours[city],
        )
    ax.set_xticks(x)
    labels = [f if len(f) < 28 else f[:25] + "…" for f in plot_df.index]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Per-client SHAP importance (centralised GB on each city)")
    ax.legend()
    fig.tight_layout()
    out_fig = figures / "per_client_shap_comparison.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_fig}")
    print(f"Wrote {tables / 'per_client_shap.csv'}")
    print(f"Wrote {tables / 'jaccard_similarity.csv'}")
    print(f"Wrote {tables / 'h3_verdict.json'}")


if __name__ == "__main__":
    main()

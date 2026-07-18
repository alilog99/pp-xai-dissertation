"""Federated SHAP aggregation and fidelity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class FederatedXAIPipeline:
    """
    Aggregates SHAP explanations from federated clients.
    Computes fidelity metrics vs centralised explanations.
    """

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = feature_names
        self.client_shap_values: dict[str, dict[str, Any]] = {}
        self.aggregated_shap: np.ndarray | None = None
        self.centralised_shap: np.ndarray | None = None
        # round -> {client: mean_abs_shap, aggregated: ...}
        self.round_history: list[dict[str, Any]] = []

    def add_client_shap(self, client_name: str, shap_values: np.ndarray, n_samples: int) -> None:
        """Store client SHAP values with their sample count (for weighted avg)."""
        self.client_shap_values[client_name] = {
            "values": np.asarray(shap_values, dtype=float),
            "n": int(n_samples),
            "mean_abs": np.mean(np.abs(shap_values), axis=0),
        }

    def clear_clients(self) -> None:
        self.client_shap_values = {}

    def aggregate_shap(self) -> np.ndarray:
        """Weighted average of mean |SHAP| across clients (weights = local n)."""
        if not self.client_shap_values:
            raise RuntimeError("No client SHAP values to aggregate")
        total_n = sum(c["n"] for c in self.client_shap_values.values())
        weighted_sum = None
        for client_data in self.client_shap_values.values():
            weight = client_data["n"] / total_n
            contrib = weight * client_data["mean_abs"]
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        self.aggregated_shap = np.asarray(weighted_sum, dtype=float)
        return self.aggregated_shap

    def record_round(self, round_idx: int) -> dict[str, Any]:
        """Snapshot per-client + aggregated importances for one FL round."""
        agg = self.aggregate_shap()
        entry = {
            "round": round_idx,
            "client_sizes": {k: v["n"] for k, v in self.client_shap_values.items()},
            "client_mean_abs_shap": {
                k: v["mean_abs"].tolist() for k, v in self.client_shap_values.items()
            },
            "aggregated_mean_abs_shap": agg.tolist(),
            "top_features": self.get_top_features(agg, n=10),
        }
        self.round_history.append(entry)
        return entry

    def compute_fidelity_metrics(self, centralised_importance: np.ndarray) -> dict[str, float]:
        """
        Compare federated aggregated SHAP vs centralised SHAP.
        Returns Spearman ρ, Jaccard@5, cosine similarity.
        """
        if self.aggregated_shap is None:
            self.aggregate_shap()
        fed_imp = np.asarray(self.aggregated_shap, dtype=float)
        cen_imp = np.asarray(centralised_importance, dtype=float)
        n = min(len(fed_imp), len(cen_imp))
        fed_imp, cen_imp = fed_imp[:n], cen_imp[:n]
        self.centralised_shap = cen_imp

        spearman_rho, spearman_p = spearmanr(fed_imp, cen_imp)
        fed_top5 = set(np.argsort(fed_imp)[-5:])
        cen_top5 = set(np.argsort(cen_imp)[-5:])
        jaccard = len(fed_top5 & cen_top5) / max(len(fed_top5 | cen_top5), 1)
        cos_sim = float(cosine_similarity(fed_imp.reshape(1, -1), cen_imp.reshape(1, -1))[0][0])

        metrics = {
            "spearman_rho": round(float(spearman_rho), 4),
            "spearman_p": round(float(spearman_p), 4),
            "jaccard_top5": round(float(jaccard), 4),
            "cosine_sim": round(cos_sim, 4),
        }
        print("\nExplanation Fidelity Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return metrics

    def get_top_features(self, importance_array: np.ndarray, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n features by importance."""
        importance_array = np.asarray(importance_array, dtype=float)
        ranked_idx = np.argsort(importance_array)[::-1][:n]
        return [
            (self.feature_names[i] if i < len(self.feature_names) else f"f{i}", float(importance_array[i]))
            for i in ranked_idx
        ]

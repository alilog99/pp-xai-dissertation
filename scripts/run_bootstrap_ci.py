#!/usr/bin/env python3
"""Bootstrap 95% CIs for RMSE + Cohen's d (strengthens H1 statistical reporting)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired absolute-error distributions (fed - central)."""
    diff = a - b
    # Use pooled SD of the two absolute-error samples (common reporting style)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _cohen_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def main() -> None:
    processed = ROOT / "data" / "processed"
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    y_test = np.load(processed / "y_test.npy")
    pred_gb = np.load(processed / "pred_gradient_boosting.npy")
    pred_fed = np.load(processed / "pred_federated.npy")
    assert len(y_test) == len(pred_gb) == len(pred_fed)

    n = len(y_test)
    rmse_gb_boot = np.empty(N_BOOTSTRAP)
    rmse_fed_boot = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n, size=n)
        rmse_gb_boot[i] = _rmse(y_test[idx], pred_gb[idx])
        rmse_fed_boot[i] = _rmse(y_test[idx], pred_fed[idx])

    point_gb = _rmse(y_test, pred_gb)
    point_fed = _rmse(y_test, pred_fed)
    ci_gb = np.percentile(rmse_gb_boot, [2.5, 97.5])
    ci_fed = np.percentile(rmse_fed_boot, [2.5, 97.5])
    overlap = not (ci_gb[1] < ci_fed[0] or ci_fed[1] < ci_gb[0])

    err_gb = np.abs(y_test - pred_gb)
    err_fed = np.abs(y_test - pred_fed)
    d = _cohen_d(err_fed, err_gb)
    d_label = _cohen_label(d)

    # Existing paired tests (recompute for complete table)
    from src.evaluation.metrics import paired_stat_tests

    paired = paired_stat_tests(y_test, pred_gb, pred_fed, "gradient_boosting", "federated")

    print("\n" + "=" * 60)
    print(f"Centralised GB:    RMSE = {point_gb:.2f}  [95% CI: {ci_gb[0]:.2f} – {ci_gb[1]:.2f}]")
    print(f"Federated MLP:     RMSE = {point_fed:.2f}  [95% CI: {ci_fed[0]:.2f} – {ci_fed[1]:.2f}]")
    print(f"CIs overlap? {'YES' if overlap else 'NO'} → gap is {'within noise band' if overlap else 'robust'}")
    print(f"Cohen's d (abs error fed − central) = {d:.3f} ({d_label})")
    print("=" * 60)

    bootstrap_df = pd.DataFrame(
        [
            {
                "model": "gradient_boosting",
                "setting": "centralised",
                "RMSE": round(point_gb, 4),
                "CI_low_95": round(float(ci_gb[0]), 4),
                "CI_high_95": round(float(ci_gb[1]), 4),
                "n_bootstrap": N_BOOTSTRAP,
            },
            {
                "model": "federated_mlp",
                "setting": "federated",
                "RMSE": round(point_fed, 4),
                "CI_low_95": round(float(ci_fed[0]), 4),
                "CI_high_95": round(float(ci_fed[1]), 4),
                "n_bootstrap": N_BOOTSTRAP,
            },
        ]
    )
    bootstrap_df.to_csv(tables / "bootstrap_ci.csv", index=False)

    complete = {
        **paired,
        "central_gb_rmse": point_gb,
        "federated_rmse": point_fed,
        "central_gb_rmse_ci95_low": float(ci_gb[0]),
        "central_gb_rmse_ci95_high": float(ci_gb[1]),
        "federated_rmse_ci95_low": float(ci_fed[0]),
        "federated_rmse_ci95_high": float(ci_fed[1]),
        "rmse_cis_overlap": overlap,
        "cohens_d_abs_error": d,
        "cohens_d_interpretation": d_label,
        "n_bootstrap": N_BOOTSTRAP,
        "n_test": n,
    }
    (tables / "statistical_tests_complete.json").write_text(json.dumps(complete, indent=2))
    pd.DataFrame([complete]).to_csv(tables / "statistical_tests_complete.csv", index=False)

    # Keep legacy statistical_tests.json in sync with new fields (non-breaking add)
    legacy = json.loads((tables / "statistical_tests.json").read_text()) if (
        tables / "statistical_tests.json"
    ).exists() else {}
    legacy.update(
        {
            "central_gb_rmse_ci95_low": float(ci_gb[0]),
            "central_gb_rmse_ci95_high": float(ci_gb[1]),
            "federated_rmse_ci95_low": float(ci_fed[0]),
            "federated_rmse_ci95_high": float(ci_fed[1]),
            "rmse_cis_overlap": overlap,
            "cohens_d_abs_error": d,
            "cohens_d_interpretation": d_label,
        }
    )
    (tables / "statistical_tests.json").write_text(json.dumps(legacy, indent=2))

    print(f"Wrote {tables / 'bootstrap_ci.csv'}")
    print(f"Wrote {tables / 'statistical_tests_complete.csv'}")
    print(f"Wrote {tables / 'statistical_tests_complete.json'}")


if __name__ == "__main__":
    main()

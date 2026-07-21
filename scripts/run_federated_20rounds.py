#!/usr/bin/env python3
"""Extend FedAvg / FedProx to 20 rounds and produce checkpoint comparison (RQ1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.federated.fl_client import fedavg_simulate, fedprox_simulate  # noqa: E402

ROUNDS = 20
LOCAL_EPOCHS = 4
FEDPROX_MU = 0.01
CHECKPOINTS = [8, 12, 16, 20]


def _load_client_data(
    processed: Path, federated: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    client_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for city in ["london", "manchester", "birmingham"]:
        xp, yp = federated / f"X_{city}.npy", federated / f"y_{city}.npy"
        if xp.exists() and yp.exists():
            Xc, yc = np.load(xp), np.load(yp)
            if len(Xc) > 10:
                client_data[city] = (Xc, yc)
                print(f"Client {city}: {len(Xc)} samples")
    if len(client_data) < 2:
        raise SystemExit("Need at least 2 city clients in data/federated/")
    return client_data


def main() -> None:
    processed = ROOT / "data" / "processed"
    federated = ROOT / "data" / "federated"
    tables = ROOT / "results" / "tables"
    figures = ROOT / "results" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    X_test = np.load(processed / "X_test.npy")
    y_test = np.load(processed / "y_test.npy")
    client_data = _load_client_data(processed, federated)

    # Centralised reference lines from published baseline metrics
    baseline = json.loads((tables / "baseline_metrics.json").read_text())
    central_mlp_rmse = float(baseline["mlp"]["RMSE"])
    central_gb_rmse = float(baseline["gradient_boosting"]["RMSE"])

    print(f"FedAvg — {ROUNDS} rounds...")
    result_avg = fedavg_simulate(
        client_data, X_test, y_test, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS
    )
    hist_avg = pd.DataFrame(result_avg["history"])
    hist_avg.to_csv(tables / "fl_fedavg_20rounds.csv", index=False)

    print(f"\nFedProx (mu={FEDPROX_MU}) — {ROUNDS} rounds...")
    result_prox = fedprox_simulate(
        client_data,
        X_test,
        y_test,
        rounds=ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        mu=FEDPROX_MU,
    )
    hist_prox = pd.DataFrame(result_prox["history"])
    hist_prox.to_csv(tables / "fl_fedprox_20rounds.csv", index=False)

    # Checkpoint comparison table
    rows = []
    for name, hist, final_r2 in [
        ("FedAvg", hist_avg, float(result_avg["test_r2"])),
        (f"FedProx(mu={FEDPROX_MU})", hist_prox, float(result_prox["test_r2"])),
    ]:
        row = {"Strategy": name, "Final_R2": round(final_r2, 4)}
        for r in CHECKPOINTS:
            val = float(hist.loc[hist["round"] == r, "rmse"].iloc[0])
            row[f"Round_{r}_RMSE"] = round(val, 4)
        rows.append(row)
    cmp_df = pd.DataFrame(rows)
    # Column order matching prompt
    cols = ["Strategy"] + [f"Round_{r}_RMSE" for r in CHECKPOINTS] + ["Final_R2"]
    cmp_df = cmp_df[cols]
    cmp_df.to_csv(tables / "fl_strategy_comparison.csv", index=False)

    # Convergence figure with centralised reference lines
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist_avg["round"], hist_avg["rmse"], marker="o", label="FedAvg")
    ax.plot(
        hist_prox["round"],
        hist_prox["rmse"],
        marker="s",
        label=f"FedProx (μ={FEDPROX_MU})",
    )
    ax.axhline(
        central_mlp_rmse,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Central MLP RMSE ({central_mlp_rmse:.2f})",
    )
    ax.axhline(
        central_gb_rmse,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label=f"Central GB RMSE ({central_gb_rmse:.2f})",
    )
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Federated convergence (20 rounds): FedAvg vs FedProx")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(range(1, ROUNDS + 1, 2))
    fig.tight_layout()
    out_fig = figures / "federated_convergence_20rounds.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    # Analysis printout
    rmse8_avg = float(hist_avg.loc[hist_avg["round"] == 8, "rmse"].iloc[0])
    rmse20_avg = float(hist_avg.loc[hist_avg["round"] == 20, "rmse"].iloc[0])
    improved = rmse20_avg < rmse8_avg - 0.05  # material improvement threshold

    target = central_mlp_rmse * 1.05  # within 5% of central MLP
    min_rounds = None
    for _, row in hist_avg.iterrows():
        if row["rmse"] <= target:
            min_rounds = int(row["round"])
            break

    print("\n" + "=" * 60)
    print(cmp_df.to_string(index=False))
    print("-" * 60)
    print(
        f"More rounds beyond 8 improve FedAvg RMSE? "
        f"{'YES' if improved else 'NO / marginal'} "
        f"(R8={rmse8_avg:.2f} → R20={rmse20_avg:.2f})"
    )
    if min_rounds is not None:
        print(
            f"Minimum FedAvg rounds to reach within 5% of central MLP "
            f"(RMSE ≤ {target:.2f}): {min_rounds}"
        )
    else:
        print(
            f"FedAvg never reached within 5% of central MLP "
            f"(target RMSE ≤ {target:.2f}) within {ROUNDS} rounds"
        )
    print(f"Wrote {out_fig}")
    print(f"Wrote {tables / 'fl_fedavg_20rounds.csv'}")
    print(f"Wrote {tables / 'fl_fedprox_20rounds.csv'}")
    print(f"Wrote {tables / 'fl_strategy_comparison.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

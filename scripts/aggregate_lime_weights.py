#!/usr/bin/env python3
"""Aggregate results/figures/lime_*.csv into results/tables/lime_weights.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    fig = ROOT / "results" / "figures"
    out = ROOT / "results" / "tables" / "lime_weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    explanations = []
    for path in sorted(fig.glob("lime_*.csv")):
        df = pd.read_csv(path)
        feat_col = "feature" if "feature" in df.columns else df.columns[0]
        weight_col = "weight" if "weight" in df.columns else df.columns[1]
        weights = [
            {"feature": str(row[feat_col]), "weight": float(row[weight_col])}
            for _, row in df.iterrows()
        ]
        stem = path.stem
        instance = None
        if "_instance_" in stem:
            instance = int(stem.rsplit("_instance_", 1)[1])
            model_label = stem.replace("lime_", "").rsplit("_instance_", 1)[0]
        else:
            model_label = stem.replace("lime_", "")
        explanations.append(
            {
                "file": path.name,
                "model": model_label,
                "instance": instance,
                "weights": weights,
                "top_feature": weights[0]["feature"] if weights else None,
            }
        )

    payload = {
        "source": "aggregated from results/figures/lime_*.csv",
        "n_explanations": len(explanations),
        "explanations": explanations,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out} ({len(explanations)} explanations)")


if __name__ == "__main__":
    main()

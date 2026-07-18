#!/usr/bin/env python3
"""Fit preprocessor and save train/test arrays."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import EPCPreprocessor


def main() -> None:
    processed = ROOT / "data" / "processed"
    combined_path = processed / "highrise_combined.csv"
    if not combined_path.exists():
        raise SystemExit("Run scripts/load_and_verify_data.py first")

    df = pd.read_csv(combined_path)
    print(f"Loaded {len(df):,} rows from {combined_path}")

    pre = EPCPreprocessor()
    X_train, X_test, y_train, y_test, meta = pre.fit_transform(df)

    np.save(processed / "X_train.npy", X_train)
    np.save(processed / "X_test.npy", X_test)
    np.save(processed / "y_train.npy", y_train)
    np.save(processed / "y_test.npy", y_test)
    meta.to_csv(processed / "test_meta.csv", index=False)
    pre.save(processed)

    # Also save client-level transformed features for FL
    federated_dir = ROOT / "data" / "federated"
    for city in ["london", "manchester", "birmingham"]:
        cpath = federated_dir / f"client_{city}.csv"
        if not cpath.exists():
            continue
        cdf = pd.read_csv(cpath)
        if cdf.empty:
            continue
        try:
            Xc, yc = pre.transform(cdf)
            np.save(federated_dir / f"X_{city}.npy", Xc)
            np.save(federated_dir / f"y_{city}.npy", yc)
            print(f"  Client {city}: X={Xc.shape}")
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING client {city}: {exc}")

    print(f"Train: {X_train.shape}, Test: {X_test.shape}, features: {len(pre.feature_names)}")
    print("Preprocessor saved.")


if __name__ == "__main__":
    main()

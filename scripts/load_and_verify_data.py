#!/usr/bin/env python3
"""Load, filter, partition, and verify EPC high-rise datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.column_definitions import FEDERATED_CLIENTS, FEDERATED_DIR, PROCESSED_DIR
from src.data.data_loader import EPCDataLoader


def main() -> None:
    processed = ROOT / PROCESSED_DIR
    federated = ROOT / FEDERATED_DIR
    processed.mkdir(parents=True, exist_ok=True)
    federated.mkdir(parents=True, exist_ok=True)

    loader = EPCDataLoader()
    files_dom = list(loader.domestic_dir.glob("certificates-*.csv"))
    files_nd = list(loader.nondomestic_dir.glob("certificates-*.csv"))

    if not files_dom and not files_nd:
        print("WARNING: No raw certificate CSVs found — generating synthetic data.")
        combined = loader.generate_synthetic(12_000)
        df_dom = combined[combined["building_typology"] == "residential"]
        df_nondom = combined[combined["building_typology"] == "commercial"]
    else:
        combined, df_dom, df_nondom = loader.load_and_prepare(use_recent_years_only=True)

    if combined.empty:
        print("Filtered dataset empty — falling back to synthetic data.")
        combined = loader.generate_synthetic(12_000)

    report = loader.verification_report(combined)
    print(report)

    combined_path = processed / "highrise_combined.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined → {combined_path}")

    if not df_dom.empty:
        df_dom.to_csv(processed / "highrise_domestic.csv", index=False)
    if not df_nondom.empty:
        df_nondom.to_csv(processed / "highrise_nondomestic.csv", index=False)

    # Geographic federated partitions
    partition_stats = []
    for city in FEDERATED_CLIENTS:
        client_df = combined[combined["source_city"] == city].copy()
        out = federated / f"client_{city}.csv"
        client_df.to_csv(out, index=False)
        mean_e = client_df["energy_consumption"].mean() if len(client_df) else float("nan")
        partition_stats.append({"client": city, "records": len(client_df), "mean_energy": mean_e})
        print(f"Client {city}: {len(client_df):,} → {out}")

    pd.DataFrame(partition_stats).to_csv(ROOT / "results" / "tables" / "partition_stats.csv", index=False)
    (ROOT / "results" / "tables" / "data_loading_report.txt").write_text(report + "\n")
    print("Done.")


if __name__ == "__main__":
    main()

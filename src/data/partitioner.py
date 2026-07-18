"""Geographic federated partitioning helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.column_definitions import FEDERATED_CLIENTS


class FederatedPartitioner:
    """Split a combined EPC frame into city clients and report non-IID stats."""

    def __init__(self, clients: dict[str, list[str]] | None = None) -> None:
        self.clients = clients or FEDERATED_CLIENTS

    def partition(self, df: pd.DataFrame, output_dir: str | Path) -> dict:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats: dict = {}
        for client_name in self.clients:
            client_df = df[df["source_city"] == client_name].copy()
            path = output_dir / f"client_{client_name}.csv"
            client_df.to_csv(path, index=False)
            stats[client_name] = {
                "records": len(client_df),
                "mean_energy": float(client_df["energy_consumption"].mean()) if len(client_df) else None,
                "std_energy": float(client_df["energy_consumption"].std()) if len(client_df) else None,
                "path": str(path),
            }
        return stats

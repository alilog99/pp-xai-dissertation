"""Flower server entrypoint (optional full Flower deployment).

The dissertation pipeline uses an in-process FedAvg simulator in
`fl_client.fedavg_simulate` for reproducibility without opening ports.
Use this module when you want a real Flower server:

    flower-supernode / flwr run  (see Flower docs)
    or: python -m src.federated.fl_server
"""

from __future__ import annotations

try:
    import flwr as fl
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"flwr required: {exc}") from exc


def main() -> None:
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=8),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

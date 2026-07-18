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
    # Optional Flower entrypoint — in-process FedAvg/FedProx live in fl_client.py
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    # Note: proximal FedProx for this dissertation is implemented in
    # `fl_client.fedprox_simulate(mu=0.01)` (in-process). Flower's FedProx
    # strategy can be swapped here for a full server deployment if needed.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=8),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

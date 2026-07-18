"""Flower federated learning client/server utilities for EPC regression."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class EnergyMLP(nn.Module):
    """Simple MLP used as the shared federated model."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state, strict=True)


def train_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(xb)
            n += len(xb)
    return {"train_mse": total_loss / max(n, 1)}


def evaluate_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
) -> dict[str, float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred))
    return {"rmse": rmse, "r2": r2, "predictions": pred}  # type: ignore[return-value]


def fedavg_simulate(
    client_data: dict[str, tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test: np.ndarray,
    rounds: int = 10,
    local_epochs: int = 3,
) -> dict[str, Any]:
    """
    Lightweight FedAvg simulation without requiring a Flower server process.

    Still uses the same MLP architecture intended for Flower deployment.
    """
    n_features = next(iter(client_data.values()))[0].shape[1]
    global_model = EnergyMLP(n_features)
    history = []

    for rnd in range(1, rounds + 1):
        client_weights = []
        client_sizes = []
        for _city, (X_c, y_c) in client_data.items():
            if len(X_c) == 0:
                continue
            local = EnergyMLP(n_features)
            set_parameters(local, get_parameters(global_model))
            train_local(local, X_c, y_c, epochs=local_epochs)
            client_weights.append(get_parameters(local))
            client_sizes.append(len(X_c))

        if not client_weights:
            break

        total = float(sum(client_sizes))
        aggregated = []
        for layer_idx in range(len(client_weights[0])):
            stacked = np.stack(
                [w[layer_idx] * (client_sizes[i] / total) for i, w in enumerate(client_weights)],
                axis=0,
            )
            aggregated.append(np.sum(stacked, axis=0))
        set_parameters(global_model, aggregated)

        metrics = evaluate_local(global_model, X_test, y_test)
        history.append({"round": rnd, "rmse": metrics["rmse"], "r2": metrics["r2"]})
        print(f"  Round {rnd}: RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f}")

    final = evaluate_local(global_model, X_test, y_test)
    return {
        "model": global_model,
        "history": history,
        "test_rmse": final["rmse"],
        "test_r2": final["r2"],
        "predictions": final["predictions"],
        "parameters": get_parameters(global_model),
    }


# Optional Flower NumPyClient wrapper for full Flower runs
try:
    import flwr as fl

    class FlowerEnergyClient(fl.client.NumPyClient):
        def __init__(self, model: EnergyMLP, X: np.ndarray, y: np.ndarray) -> None:
            self.model = model
            self.X = X
            self.y = y

        def get_parameters(self, config):  # noqa: ANN001
            return get_parameters(self.model)

        def fit(self, parameters, config):  # noqa: ANN001
            set_parameters(self.model, parameters)
            train_local(self.model, self.X, self.y, epochs=int(config.get("local_epochs", 3)))
            return get_parameters(self.model), len(self.X), {}

        def evaluate(self, parameters, config):  # noqa: ANN001
            set_parameters(self.model, parameters)
            metrics = evaluate_local(self.model, self.X, self.y)
            return float(metrics["rmse"] ** 2), len(self.X), {"rmse": metrics["rmse"], "r2": metrics["r2"]}

except Exception:  # noqa: BLE001
    FlowerEnergyClient = None  # type: ignore

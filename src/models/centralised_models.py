"""Centralised baseline regression models for EPC energy prediction."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # noqa: BLE001
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:  # noqa: BLE001
    LGBMRegressor = None  # type: ignore


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def build_models(random_state: int = 42) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor

    models: dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            early_stopping=True,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
            force_col_wise=True,
            verbose=-1,
        )
    return models


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], dict[str, np.ndarray]]:
    models = models or build_models()
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics[name] = regression_metrics(y_test, pred)
        predictions[name] = pred
        print(f"    {metrics[name]}")
    return metrics, models, predictions


def save_models(models: dict[str, Any], directory: str) -> None:
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, path / f"{name}.joblib")


def load_model(path: str):
    return joblib.load(path)

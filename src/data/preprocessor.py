"""Preprocessing pipeline for unified high-rise EPC features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.column_definitions import (
    CATEGORICAL_FEATURES,
    FINAL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    UNIFIED_TARGET,
)


class EPCPreprocessor:
    """Impute, encode, and scale features; persist transformers."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.pipeline: Pipeline | None = None
        self.numeric_features = list(NUMERICAL_FEATURES)
        self.categorical_features = list(CATEGORICAL_FEATURES)

    def _build_pipeline(self) -> ColumnTransformer:
        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric, self.numeric_features),
                ("cat", categorical, self.categorical_features),
            ]
        )

    def prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Ensure required columns exist
        for col in FINAL_FEATURE_COLUMNS + [UNIFIED_TARGET]:
            if col not in out.columns:
                out[col] = np.nan
        out = out.dropna(subset=[UNIFIED_TARGET])
        out = out[out[UNIFIED_TARGET] > 0]
        # Clip extreme outliers
        lo, hi = out[UNIFIED_TARGET].quantile(0.01), out[UNIFIED_TARGET].quantile(0.99)
        out = out[(out[UNIFIED_TARGET] >= lo) & (out[UNIFIED_TARGET] <= hi)]
        for col in self.categorical_features:
            out[col] = out[col].astype(str).replace({"nan": "unknown", "None": "unknown"})
        for col in self.numeric_features:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.reset_index(drop=True)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        clean = self.prepare_frame(df)
        X = clean[FINAL_FEATURE_COLUMNS]
        y = clean[UNIFIED_TARGET].values.astype(float)

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            clean.index,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Stratify by city when feasible
        try:
            if clean["source_city"].nunique() > 1 and clean["source_city"].value_counts().min() >= 2:
                X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                    X,
                    y,
                    clean.index,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=clean["source_city"],
                )
        except ValueError:
            pass

        self.pipeline = self._build_pipeline()
        X_train_t = self.pipeline.fit_transform(X_train)
        X_test_t = self.pipeline.transform(X_test)
        self.feature_names = list(self.pipeline.get_feature_names_out())
        meta = clean.loc[idx_test, ["source_city", "building_typology", "certificate_number"]].reset_index(drop=True)
        return X_train_t, X_test_t, y_train, y_test, meta

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.pipeline is None:
            raise RuntimeError("Preprocessor not fitted")
        clean = self.prepare_frame(df)
        X = self.pipeline.transform(clean[FINAL_FEATURE_COLUMNS])
        y = clean[UNIFIED_TARGET].values.astype(float)
        return X, y

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "preprocessor.joblib")
        (directory / "feature_names.json").write_text(json.dumps(self.feature_names, indent=2))
        meta = {
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target": UNIFIED_TARGET,
        }
        (directory / "preprocessor_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "EPCPreprocessor":
        directory = Path(directory)
        obj = cls()
        obj.pipeline = joblib.load(directory / "preprocessor.joblib")
        obj.feature_names = json.loads((directory / "feature_names.json").read_text())
        return obj

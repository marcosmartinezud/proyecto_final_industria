# Pipeline de features y preprocesamiento.
# Uso: creo features derivadas y evito fuga de datos con un Pipeline de sklearn.

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


RAW_NUMERIC_COLS: Sequence[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

RAW_CATEGORICAL_COLS: Sequence[str] = [
    "Type",
]

DERIVED_NUMERIC_COLS: Sequence[str] = [
    "Temp_diff",
    "Torque_per_rpm",
]

DERIVED_CATEGORICAL_COLS: Sequence[str] = [
    "Tool_state",
]

MODEL_NUMERIC_COLS: Sequence[str] = [*RAW_NUMERIC_COLS, *DERIVED_NUMERIC_COLS]
MODEL_CATEGORICAL_COLS: Sequence[str] = [*RAW_CATEGORICAL_COLS, *DERIVED_CATEGORICAL_COLS]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # Agrega las features derivadas que usa el modelo.

    required = set(RAW_NUMERIC_COLS) | set(RAW_CATEGORICAL_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    out = df.copy()
    out["Temp_diff"] = out["Process temperature [K]"] - out["Air temperature [K]"]

    rpm = out["Rotational speed [rpm]"].replace(0, np.nan)
    out["Torque_per_rpm"] = out["Torque [Nm]"] / rpm

    wear = out["Tool wear [min]"]
    out["Tool_state"] = pd.cut(
        wear,
        bins=[0, 50, 150, np.inf],
        labels=["Nuevo", "Medio", "Viejo"],
        right=False,
        include_lowest=True,
    )
    return out


def build_preprocessor() -> Pipeline:
    # Devuelve un pipeline de preprocesado (derivadas + imputar + escalar/one-hot).

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(MODEL_NUMERIC_COLS)),
            ("cat", categorical_pipe, list(MODEL_CATEGORICAL_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("derived", FunctionTransformer(add_derived_features, validate=False)),
            ("preprocess", column_transformer),
        ]
    )

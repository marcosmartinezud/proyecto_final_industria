"""Feature engineering helpers."""

from typing import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES: Sequence[str] = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Temp_diff',
    'Torque_per_rpm'
]


CATEGORICAL_FEATURES: Sequence[str] = ['Type', 'Tool_state']


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe ready for modeling with scaled numeric and encoded categorical features."""

    df_copy = df.copy()

    scaler = StandardScaler()
    df_copy[NUMERIC_FEATURES] = scaler.fit_transform(df_copy[NUMERIC_FEATURES])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    cat_array = ohe.fit_transform(df_copy[CATEGORICAL_FEATURES])
    cat_cols = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
    cat_df = pd.DataFrame(cat_array, columns=cat_cols, index=df_copy.index)

    df_prepared = pd.concat([df_copy.drop(columns=CATEGORICAL_FEATURES), cat_df], axis=1)
    return df_prepared
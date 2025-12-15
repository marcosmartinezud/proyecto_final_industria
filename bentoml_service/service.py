from __future__ import annotations

from pathlib import Path

import bentoml
import joblib
from bentoml.io import JSON, PandasDataFrame


def _default_model_path() -> Path:
	return Path(__file__).resolve().parents[1] / "models" / "model_binary_rf.joblib"


_artifact = joblib.load(_default_model_path())
_pipeline = _artifact["pipeline"] if isinstance(_artifact, dict) else _artifact

svc = bentoml.Service("ai4i_failure")


@svc.api(input=PandasDataFrame(), output=JSON())
def predict(df):
	preds = _pipeline.predict(df)
	return {"predictions": preds.tolist()}


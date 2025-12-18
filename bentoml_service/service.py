from __future__ import annotations

# Servicio BentoML local para exponer el modelo de mantenimiento predictivo AI4I.

import os
from pathlib import Path

import bentoml
import joblib
from bentoml.io import JSON, PandasDataFrame


MODEL_ENV_VAR = "BENTO_MODEL_PATH"


def _default_model_path() -> Path:
	# Ruta por defecto al modelo binario entrenado desde scripts/Streamlit.
	return Path(__file__).resolve().parents[1] / "models" / "model_binary_rf.joblib"


def _resolve_model_path() -> Path:
	# Obtiene la ruta al modelo desde la variable de entorno o usa el predeterminado.
	env_value = os.getenv(MODEL_ENV_VAR)
	if env_value:
		path = Path(env_value)
		if not path.is_absolute():
			path = Path(__file__).resolve().parents[1] / path
	else:
		path = _default_model_path()

	if not path.exists():
		raise FileNotFoundError(
			f"Modelo no encontrado en {path}. Define {MODEL_ENV_VAR} o genera el modelo con src.train/Streamlit."
		)
	return path


_artifact = joblib.load(_resolve_model_path())
_pipeline = _artifact["pipeline"] if isinstance(_artifact, dict) else _artifact

svc = bentoml.Service("ai4i_failure")


@svc.api(input=PandasDataFrame(), output=JSON())
def predict(df):
	preds = _pipeline.predict(df)
	return {"predictions": preds.tolist()}

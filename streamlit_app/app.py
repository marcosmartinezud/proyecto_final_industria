from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def _default_model_path() -> Path:
	return Path(__file__).resolve().parents[1] / "models" / "model_binary_rf.joblib"


@st.cache_resource
def _load_pipeline(model_path: Path):
	artifact = joblib.load(model_path)
	return artifact["pipeline"] if isinstance(artifact, dict) else artifact


def main() -> None:
	st.set_page_config(page_title="AI4I Predictive Maintenance", layout="wide")
	st.title("AI4I – Predicción de fallo")

	model_path = _default_model_path()
	if not model_path.exists():
		st.warning(
			"No encuentro el modelo entrenado. Entrena primero con: `python -m src.train` (desde la raíz del repo)."
		)
		st.stop()

	st.caption(f"Modelo: {model_path.name}")
	pipeline = _load_pipeline(model_path)

	uploaded = st.file_uploader("Sube un CSV con columnas del dataset AI4I", type=["csv"])
	if uploaded is None:
		st.info("Sube un CSV para obtener predicciones.")
		st.stop()

	df = pd.read_csv(uploaded)
	st.subheader("Vista previa")
	st.dataframe(df.head(20), use_container_width=True)

	if st.button("Predecir", type="primary"):
		preds = pipeline.predict(df)
		out = df.copy()
		out["prediction"] = preds
		st.subheader("Resultados")
		st.dataframe(out.head(50), use_container_width=True)
		st.write("Distribución de predicciones")
		st.dataframe(out["prediction"].value_counts().to_frame("count"))


if __name__ == "__main__":
	main()


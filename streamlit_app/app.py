from __future__ import annotations

# App Streamlit para AI4I: EDA, entrenamiento rápido y consumo del servicio BentoML.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
	accuracy_score,
	auc,
	classification_report,
	confusion_matrix,
	roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor
from src.train import _build_model, _default_drop_columns, _select_target


plt.style.use("seaborn-v0_8-whitegrid")

_MODEL_LABELS = {
	"rf": "Random Forest",
	"dt": "Árbol de decisión",
	"logreg": "Regresión logística",
}


def _model_label(key: str) -> str:
	return _MODEL_LABELS.get(key, key.upper())


def _default_model_path() -> Path:
	return Path(__file__).resolve().parents[1] / "models" / "model_binary_rf.joblib"


def _default_data_path() -> Path:
	return Path(__file__).resolve().parents[1] / "data" / "ai4i_clean.csv"


@st.cache_resource
def _load_pipeline(model_path: Path):
	artifact = joblib.load(model_path)
	return artifact["pipeline"] if isinstance(artifact, dict) else artifact


@st.cache_data
def _load_df_from_path(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)


def _split_xy(df: pd.DataFrame, target: str):
	y = _select_target(df, target=target)  # usa la misma lógica del script de entrenamiento
	X = df.drop(columns=_default_drop_columns(df))
	return X, y


def _train_once(
	df: pd.DataFrame,
	target: str,
	model_type: str,
	test_size: float,
	random_state: int,
	hyperparams: dict | None = None,
	calc_importance: bool = False,
	max_importance_features: int = 10,
):
	# Entrena una vez y devuelve métricas y figuras para mostrarlas en la UI.
	X, y = _split_xy(df, target)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, stratify=y, random_state=random_state
	)

	prep = build_preprocessor()
	model = _build_model(
		model_type=model_type,
		random_state=random_state,
		hyperparams=hyperparams,
	)
	pipeline = Pipeline(steps=[("prep", prep), ("model", model)])
	pipeline.fit(X_train, y_train)

	preds = pipeline.predict(X_test)
	acc = accuracy_score(y_test, preds)
	report = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).T
	conf = confusion_matrix(y_test, preds)

	fig_conf, ax_conf = plt.subplots(figsize=(4, 3))
	sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", ax=ax_conf)
	ax_conf.set_title("Matriz de confusión")
	ax_conf.set_xlabel("Predicción")
	ax_conf.set_ylabel("Real")
	plt.tight_layout()

	roc_auc = None
	fig_roc = None
	if target == "binary" and hasattr(model, "predict_proba"):
		proba = pipeline.predict_proba(X_test)[:, 1]
		fpr, tpr, _ = roc_curve(y_test, proba)
		roc_auc = auc(fpr, tpr)
		fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
		ax_roc.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
		ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
		ax_roc.set_xlabel("FPR")
		ax_roc.set_ylabel("TPR")
		ax_roc.legend()
		ax_roc.set_title("ROC (binario)")
		plt.tight_layout()

	return {
		"pipeline": pipeline,
		"accuracy": acc,
		"report": report,
		"conf_fig": fig_conf,
		"roc_fig": fig_roc,
		"roc_auc": roc_auc,
		"hyperparams": hyperparams or {},
		"importance_fig": _importance_plot(pipeline, X_test, y_test, random_state, max_importance_features) if calc_importance else None
	}


def _importance_plot(pipeline, X_test, y_test, random_state: int, max_features: int):
	# Calcula importancia por permutación y devuelve la figura.
	if pipeline is None:
		return None
	try:
		# Hago subsample para que la UI siga rápida
		sample = X_test
		y_sample = y_test
		if len(X_test) > 500:
			sample = X_test.sample(500, random_state=random_state)
			y_sample = y_test.loc[sample.index]

		result = permutation_importance(
			pipeline, sample, y_sample, n_repeats=5, random_state=random_state, n_jobs=1
		)

		feature_names = None
		prep = pipeline.named_steps.get("prep")
		if prep and hasattr(prep, "named_steps"):
			preprocess = prep.named_steps.get("preprocess")
			if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
				feature_names = preprocess.get_feature_names_out()

		importances = result.importances_mean
		if feature_names is None or len(feature_names) != len(importances):
			# Si no calzan las longitudes (por one-hot o derivadas), uso columnas originales.
			feature_names = list(sample.columns[: len(importances)])

		imp_df = (
			pd.DataFrame({"feature": feature_names, "importance": importances})
			.sort_values("importance", ascending=False)
			.head(max_features)
		)

		fig, ax = plt.subplots(figsize=(5, 4))
		sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue", ax=ax)
		ax.set_title("Impacto de variables (importancia por permutación)")
		ax.set_xlabel("Disminución media en accuracy")
		plt.tight_layout()
		return fig
	except Exception as exc:  # noqa: BLE001
		st.warning(f"No se pudo calcular la importancia de las variables: {exc}")
		return None


def _hyperparams_ui(models_to_run: list[str]) -> dict[str, dict]:
	# Pide al usuario hiperparámetros básicos para cada modelo.

	params: dict[str, dict] = {}
	for model_key in models_to_run:
		if model_key == "rf":
			with st.expander("Random Forest | ajustes rápidos", expanded=False):
				n_estimators = st.slider(
					"Número de árboles (n_estimators)", 50, 500, value=300, step=50, key="rf_n_estimators"
				)
				max_depth_opt = st.selectbox(
					"Profundidad máxima",
					options=["Auto (None)", "10", "20", "40"],
					index=0,
					key="rf_max_depth",
				)
				max_depth = None if max_depth_opt.startswith("Auto") else int(max_depth_opt)
				params[model_key] = {"n_estimators": n_estimators, "max_depth": max_depth}

		elif model_key == "dt":
			with st.expander("Árbol de decisión | ajustes rápidos", expanded=False):
				max_depth_opt = st.selectbox(
					"Profundidad máxima",
					options=["Auto (None)", "5", "10", "20", "40"],
					index=0,
					key="dt_max_depth",
				)
				max_depth = None if max_depth_opt.startswith("Auto") else int(max_depth_opt)
				params[model_key] = {"max_depth": max_depth}

		elif model_key == "logreg":
			with st.expander("Regresión logística | ajustes rápidos", expanded=False):
				c_value = st.select_slider(
					"C (fuerza de regularización)", options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0, key="logreg_c"
				)
				params[model_key] = {"C": float(c_value)}

	return params


def _eda_section(df: pd.DataFrame) -> None:
	st.subheader("Exploración del conjunto de datos")
	col1, col2, col3 = st.columns(3)
	col1.metric("Filas", len(df))
	col2.metric("Columnas", df.shape[1])
	col3.metric("Valores nulos", int(df.isna().sum().sum()))

	st.write("Vista previa del conjunto de datos")
	st.dataframe(df.head(15), use_container_width=True)

	numeric_cols = list(df.select_dtypes(include=["number"]).columns)
	categorical_cols = list(df.select_dtypes(exclude=["number"]).columns)

	with st.expander("Estadísticas descriptivas", expanded=False):
		st.dataframe(df.describe(include="all").T, use_container_width=True)

	if numeric_cols:
		st.markdown("### Distribución de variables numéricas")
		col_sel = st.selectbox("Variable numérica a explorar", numeric_cols)
		fig, ax = plt.subplots(figsize=(6, 3))
		sns.histplot(df[col_sel].dropna(), bins=30, kde=True, color="steelblue", ax=ax)
		ax.set_title(col_sel)
		st.pyplot(fig, clear_figure=True)

	if len(numeric_cols) > 1:
		st.markdown("### Matriz de correlación")
		corr = df[numeric_cols].corr()
		fig, ax = plt.subplots(figsize=(6, 4))
		sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
		ax.set_title("Matriz de correlación")
		st.pyplot(fig, clear_figure=True)

	if categorical_cols:
		st.markdown("### Distribución por categoría")
		cat_col = st.selectbox("Variable categórica a revisar", categorical_cols)
		st.bar_chart(df[cat_col].value_counts())


def _training_section(df: pd.DataFrame) -> Pipeline | None:
	st.subheader("Entrenamiento y evaluación ágil")
	target = st.selectbox("Tipo de objetivo", ["binary", "multiclass"], format_func=lambda x: "Binario" if x == "binary" else "Multiclase")
	models_to_run = st.multiselect(
		"Modelos a probar", ["rf", "dt", "logreg"], default=["rf", "logreg"], format_func=_model_label
	)
	test_size = st.slider("Proporción para prueba", 0.1, 0.3, value=0.2, step=0.05)
	random_state = st.number_input("Semilla aleatoria", min_value=0, value=42, step=1)
	calc_importance = st.checkbox("Calcular importancia de variables (permutación)", value=False)
	max_features = st.slider("N.º de variables a mostrar", 5, 15, value=10) if calc_importance else 10

	hyperparams = _hyperparams_ui(models_to_run)

	if st.button("Entrenar y evaluar modelos", type="primary"):
		results = []
		summary_rows = []
		for model_key in models_to_run:
			with st.spinner(f"Entrenando {_model_label(model_key)}..."):
				model_params = hyperparams.get(model_key)
				res = _train_once(
					df,
					target,
					model_key,
					test_size,
					random_state,
					model_params,
					calc_importance=calc_importance,
					max_importance_features=max_features,
				)
				results.append((model_key, res))
				summary_rows.append(
					{
						"Modelo": _model_label(model_key),
						"Accuracy": round(res["accuracy"], 3),
						"ROC AUC (binario)": round(res["roc_auc"], 3) if res["roc_auc"] is not None else None,
					}
				)
		for model_key, res in results:
			st.markdown(f"#### {_model_label(model_key)} — Accuracy {res['accuracy']:.3f}")
			st.dataframe(res["report"], use_container_width=True)
			col_a, col_b = st.columns(2)
			with col_a:
				st.pyplot(res["conf_fig"], clear_figure=True)
			with col_b:
				if res["roc_fig"] is not None:
					st.pyplot(res["roc_fig"], clear_figure=True)
				else:
					st.info("La curva ROC solo se muestra para objetivo binario con `predict_proba`.")
			if calc_importance and res["importance_fig"] is not None:
				st.pyplot(res["importance_fig"], clear_figure=True)
		if summary_rows:
			comp_df = pd.DataFrame(summary_rows).sort_values(by="Accuracy", ascending=False)
			st.markdown("#### Resumen comparativo")
			st.dataframe(comp_df.set_index("Modelo"), use_container_width=True)
		if results:
			best_model_key, best_res = max(results, key=lambda x: x[1]["accuracy"])
			st.success(f"Mejor accuracy: {_model_label(best_model_key)}")

			default_save = Path(__file__).resolve().parents[1] / "models" / f"model_{target}_{best_model_key}_tuned.joblib"
			save_path = st.text_input(
				"Ruta para guardar el mejor modelo (para exponer con BentoML)",
				value=str(default_save),
				key="best_model_save_path",
			)
			if st.button("Guardar pipeline ganador", type="secondary"):
				out_path = Path(save_path)
				out_path.parent.mkdir(parents=True, exist_ok=True)
				joblib.dump(
					{
						"pipeline": best_res["pipeline"],
						"target": target,
						"model_type": best_model_key,
						"random_state": random_state,
						"hyperparams": best_res.get("hyperparams", {}),
					},
					out_path,
				)
				st.success(f"Modelo guardado en {out_path}")
			return best_res["pipeline"]
		return None
	return None


def _predict_with_local_model(pipeline, df: pd.DataFrame) -> None:
	st.subheader("Predicción rápida con el modelo cargado")
	if pipeline is None:
		st.info("Entrena un modelo en la sección anterior o carga uno ya preparado.")
		return
	preds = pipeline.predict(df)
	out = df.copy()
	out["prediction"] = preds
	st.dataframe(out.head(30), use_container_width=True)
	st.write("Distribución de clases predichas")
	st.dataframe(out["prediction"].value_counts().to_frame("conteo"))


def _bento_section(df: pd.DataFrame) -> None:
	st.subheader("Predicciones vía API BentoML (servicio local)")
	st.caption("Asegúrate de tener el servicio en marcha: `bentoml serve bentoml_service.service:svc --reload`")
	base_url = st.text_input("URL del servicio BentoML", "http://127.0.0.1:3000/predict")

	if df is None or df.empty:
		st.info("Carga datos en la pestaña de exploración para enviar muestras.")
		return

	n_rows = st.slider("Número de filas a enviar", 1, min(10, len(df)), value=min(5, len(df)))
	sample = df.head(n_rows)
	st.dataframe(sample, use_container_width=True)

	if st.button("Enviar muestra al servicio BentoML"):
		try:
			resp = requests.post(base_url, json=sample.to_dict(orient="records"), timeout=10)
		except requests.RequestException as exc:
			st.error(f"No se pudo conectar con el servicio BentoML: {exc}")
			return

		if resp.status_code != 200:
			st.error(f"Respuesta {resp.status_code}: {resp.text}")
			return
		try:
			data = resp.json()
			preds = data.get("predictions")
		except ValueError:
			st.error("No se pudo parsear la respuesta de la API.")
			return

		if preds is None:
			st.warning(f"Respuesta de la API inesperada: {data}")
			return
		out = sample.copy()
		out["prediction_api"] = preds
		st.success("Predicciones recibidas desde la API BentoML")
		st.dataframe(out, use_container_width=True)
		st.write("Distribución de clases predichas (API)")
		st.dataframe(out["prediction_api"].value_counts().to_frame("conteo"))


def main() -> None:
	st.set_page_config(page_title="AI4I Predictive Maintenance", layout="wide")
	st.title("AI4I · Mantenimiento predictivo en una sola vista")
	st.caption("Explora los datos, ajusta modelos y consume la API sin salir de este panel.")

	default_data_path = _default_data_path()
	loaded_df = None

	with st.sidebar:
		st.header("Datos de entrada")
		use_default = st.checkbox(
			f"Usar conjunto de datos de ejemplo ({default_data_path.name})", value=default_data_path.exists()
		)
		uploaded = st.file_uploader("O sube un CSV propio", type=["csv"])
		if use_default and default_data_path.exists():
			loaded_df = _load_df_from_path(default_data_path)
		elif uploaded is not None:
			loaded_df = pd.read_csv(uploaded)

	if loaded_df is None:
		st.info("Carga el conjunto de datos de ejemplo (`data/ai4i_clean.csv`) o sube un CSV para comenzar.")
		return

	tab_eda, tab_train, tab_bento = st.tabs(["Explorar datos", "Entrenar y evaluar", "Consumir API BentoML"])

	with tab_eda:
		_eda_section(loaded_df)

	with tab_train:
		pipeline = _training_section(loaded_df)
		if pipeline is None:
			# Intento cargar un pipeline entrenado para predecir rápido
			model_path = _default_model_path()
			if model_path.exists():
				pipeline = _load_pipeline(model_path)
		_predict_with_local_model(pipeline, loaded_df)

	with tab_bento:
		_bento_section(loaded_df)


if __name__ == "__main__":
	main()

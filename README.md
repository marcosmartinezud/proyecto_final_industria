# Proyecto final (Industria) – AI4I Predictive Maintenance

Aplicación completa para mantenimiento predictivo usando el dataset AI4I: limpieza de datos, análisis interactivo en Streamlit, entrenamiento/evaluación de modelos y despliegue local con BentoML.

## Objetivo del trabajo final
Demostrar un flujo industrial: preparar datos (notebook), explorar y modelar de forma interactiva (Streamlit) y publicar un servicio de predicción (BentoML) que pueda consumirse en tiempo real desde la propia app.

## Dataset
- Fuente: AI4I 2020 (fallos en maquinaria). Crudo en `data/ai4i2020.csv`, limpio en `data/ai4i_clean.csv`.
- Columnas clave: temperaturas, velocidad, torque, desgaste, tipo de producto y etiquetas de fallo binario/multiclase.
- Notebook `notebooks/01_data_prep.ipynb` documenta duplicados, imputación de nulos por mediana, creación de variables derivadas y guardado del CSV limpio.

## Estructura rápida
- `notebooks/01_data_prep.ipynb` y `02_exploratory.ipynb`: preparación y EDA guiada.
- `streamlit_app/app.py`: panel con tres pestañas (Explorar, Entrenar/Evaluar, API BentoML).
- `src/`: scripts de preparación (`data_prep.py`), features, entrenamiento (`train.py`) y predicción batch (`predict.py`).
- `bentoml_service/service.py`: servicio BentoML que carga el modelo indicado en `BENTO_MODEL_PATH`.
- `models/`: modelos `.joblib` guardados desde scripts o Streamlit.

## Requisitos
Instala dependencias:
```bash
pip install -r requirements.txt
```

## Levantar todo (API BentoML + app Streamlit)
0) (Opcional) Crea y activa un entorno: `python -m venv .venv && source .venv/bin/activate`  
1) Instala dependencias: `pip install -r requirements.txt`  
2) Prepara datos (deja `data/ai4i_clean.csv`): `python -m src.data_prep`  
3) Entrena un modelo (binario por defecto) y guárdalo en `models/`:  
```bash
python -m src.train --target binary --model rf --test-size 0.2 --seed 42
```
4) Terminal 1 — lanza la API local de predicción con BentoML:  
```bash
export BENTO_MODEL_PATH=models/model_binary_rf.joblib   # en Windows: set BENTO_MODEL_PATH=...
bentoml serve bentoml_service.service:svc --reload
```
El endpoint expuesto es `POST http://127.0.0.1:3000/predict` (orient="records").
5) Terminal 2 — arranca la app Streamlit y consume la API:  
```bash
streamlit run streamlit_app/app.py
```
En la pestaña **API BentoML**, deja la URL por defecto o cámbiala si modificaste host/puerto.

## Flujo paso a paso
1) **Preparar datos**  
```bash
python -m src.data_prep               # genera data/ai4i_clean.csv
```
O usa el notebook `01_data_prep.ipynb` si quieres ver cada decisión de limpieza.

2) **Análisis exploratorio**  
Notebook `02_exploratory.ipynb` (histogramas, correlaciones, boxplots por tipo de fallo). La pestaña **Explorar datos** en Streamlit replica lo básico con tablas, histogramas dinámicos, correlación y conteos categóricos.

3) **Entrenar y evaluar modelos**  
- Script directo:
```bash
python -m src.train --target binary --model rf --test-size 0.2 --seed 42
```
- Interactivo en Streamlit (pestaña **Entrenar/Evaluar**): selecciona target binario/multiclase, escoge modelos (RF, DT, LogReg), ajusta hiperparámetros simples, compara accuracy, matriz de confusión y ROC (binario). Permite guardar el mejor modelo en `models/` listo para BentoML.
- Optimización/selección: activa el checkbox **“Calcular importancia de características (perm. importance)”** para ver el impacto de variables (incluidas las derivadas) y usa el comparativo de métricas para elegir el mejor modelo. El slider **“Top features a mostrar”** permite inspeccionar la aportación de las más relevantes.

4) **Predicción por lotes**  
```bash
python -m src.predict --model models/model_binary_rf.joblib --in data/ai4i2020.csv
```
Genera un CSV con columna `prediction`.

5) **Panel Streamlit**  
```bash
streamlit run streamlit_app/app.py
```
- Usa el CSV limpio por defecto o sube uno propio.  
- Incluye pestaña **API BentoML** que envía muestras al servicio local y muestra la distribución de predicciones.

6) **Servicio BentoML (tiempo real local)**  
```bash
export BENTO_MODEL_PATH=models/model_binary_rf.joblib   # o el modelo guardado desde Streamlit
bentoml serve bentoml_service.service:svc --reload
```
- Endpoint `POST /predict` recibe JSON orient="records".  
- `bentofile.yaml` permite empaquetar:  
```bash
python -m src.train --target multiclass --model rf --seed 42
export BENTO_MODEL_PATH=models/model_multiclass_rf.joblib
bentoml build -t ai4i:latest
bentoml serve ai4i:latest
```

## Checklist de cumplimiento (entrega)
- Preparación de datos documentada en notebook + script reproducible (`src.data_prep`).
- EDA con estadísticas y gráficas en notebook y panel interactivo en Streamlit.
- Ingeniería de características incluida (features derivadas y preprocesamiento en `src/features.py`).
- Entrenamiento/evaluación interactiva: selección de modelos, métricas (accuracy, report, matriz de confusión, ROC), tabla comparativa de resultados.
- Selección/optimización de características: importancia por permutación para cuantificar el efecto de variables derivadas y originales.
- Panel con controles (selectores, sliders, uploads) para experiencia dinámica.
- Despliegue local con BentoML: API `POST /predict`, configurable vía `BENTO_MODEL_PATH`, consumida desde la pestaña correspondiente.

## Comandos rápidos
- Limpiar datos: `python -m src.data_prep`
- Entrenar RF binario: `python -m src.train --target binary --model rf`
- Lanzar panel: `streamlit run streamlit_app/app.py`
- Servir modelo: `export BENTO_MODEL_PATH=models/model_binary_rf.joblib && bentoml serve bentoml_service.service:svc --reload`

## Notas de optimización teórica
- Reducir dimensionalidad/colinealidad: probar PCA o selección por importancia acumulada y simplificar la matriz de one-hot si el rendimiento en producción es crítico.
- Ajuste de umbrales: para el problema binario, mover el threshold de decisión según la curva ROC para equilibrar falsa alarma vs. fallo no detectado.
- Imbalance / costes: añadir ponderación de clases en LogReg/árboles o aplicar técnicas de subsampling si se prioriza recall de fallos.
- Búsqueda más amplia: usar `RandomizedSearchCV` o `GridSearchCV` sobre `n_estimators`, `max_depth`, `C` y balanceo para ganar robustez sin depender solo de sliders rápidos en la UI.

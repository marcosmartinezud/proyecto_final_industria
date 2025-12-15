# Proyecto final (Industria) – AI4I Predictive Maintenance

Estructura mínima para:
- preparar datos para análisis (CSV limpio en `data/`)
- entrenar un modelo reproducible (pipeline sklearn + modelo)
- predecir por batch (CSV → CSV)
- servir el modelo por Streamlit y por BentoML

## Requisitos

Instala dependencias:

```bash
pip install -r requirements.txt
```

## Scripts (desde la raíz del repo)

### 1) Preparar CSV limpio (para notebooks/EDA)

```bash
python -m src.data_prep
```

Genera `data/ai4i_clean.csv`.

### 2) Entrenar

```bash
python -m src.train --target binary --model rf
```

Guarda el modelo en `models/`.

### 3) Predecir (batch)

```bash
python -m src.predict --model models/model_binary_rf.joblib --in data/ai4i2020.csv
```

## Streamlit

```bash
streamlit run streamlit_app/app.py
```

## BentoML

Desde la raíz del repo:

```bash
bentoml serve bentoml_service.service:svc --reload
```


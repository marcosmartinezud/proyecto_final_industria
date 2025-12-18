from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import build_preprocessor
from src.utils import data_dir, ensure_dir, models_dir


TargetType = Literal["binary", "multiclass"]
ModelType = Literal["rf", "dt", "logreg"]


def _make_multiclass_target(df: pd.DataFrame) -> pd.Series:
	failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
	for col in failure_cols:
		if col not in df.columns:
			raise ValueError(f"Missing failure column: {col}")

	y = pd.Series("No failure", index=df.index, dtype="object")
	for col in failure_cols:
		y = y.mask(df[col].astype(int) == 1, col)
	return y


def _select_target(df: pd.DataFrame, target: TargetType) -> pd.Series:
	if target == "binary":
		if "Machine failure" not in df.columns:
			raise ValueError("Missing target column: Machine failure")
		return df["Machine failure"].astype(int)
	if target == "multiclass":
		return _make_multiclass_target(df)
	raise ValueError(f"Unknown target type: {target}")


def _default_drop_columns(df: pd.DataFrame) -> list[str]:
	drop = [
		"UDI",
		"Product ID",
		"Machine failure",
		"TWF",
		"HDF",
		"PWF",
		"OSF",
		"RNF",
	]
	return [c for c in drop if c in df.columns]


def _build_model(model_type: ModelType, random_state: int, hyperparams: dict | None = None) -> object:
	# Arma el modelo con hiperparámetros opcionales si se pasan.

	hyperparams = hyperparams or {}
	if model_type == "rf":
		return RandomForestClassifier(
			n_estimators=hyperparams.get("n_estimators", 300),
			max_depth=hyperparams.get("max_depth"),
			random_state=random_state,
		)
	if model_type == "dt":
		return DecisionTreeClassifier(
			max_depth=hyperparams.get("max_depth"),
			random_state=random_state,
		)
	if model_type == "logreg":
		return LogisticRegression(
			max_iter=2000,
			solver="lbfgs",
			C=hyperparams.get("C", 1.0),
			random_state=random_state,
		)
	raise ValueError(f"Unknown model type: {model_type}")


def train(
	in_path: str | Path | None = None,
	target: TargetType = "binary",
	model_type: ModelType = "rf",
	test_size: float = 0.2,
	random_state: int = 42,
	hyperparams: dict | None = None,
) -> Path:
	if in_path is None:
		in_path = data_dir() / "ai4i2020.csv"
	in_path = Path(in_path)

	df = pd.read_csv(in_path)
	y = _select_target(df, target=target)
	X = df.drop(columns=_default_drop_columns(df))

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		stratify=y,
		random_state=random_state,
	)

	preprocessor = build_preprocessor()
	model = _build_model(model_type=model_type, random_state=random_state, hyperparams=hyperparams)
	pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

	pipeline.fit(X_train, y_train)
	preds = pipeline.predict(X_test)

	print(f"Target: {target} | Model: {model_type}")
	print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
	print(classification_report(y_test, preds, zero_division=0))

	out_dir = ensure_dir(models_dir())
	model_path = out_dir / f"model_{target}_{model_type}.joblib"
	joblib.dump(
		{
			"pipeline": pipeline,
			"target": target,
			"model_type": model_type,
			"random_state": random_state,
		},
		model_path,
	)
	print("Saved model:", model_path)
	return model_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Train AI4I predictive maintenance model")
	parser.add_argument("--in", dest="in_path", default=None, help="Input CSV path (raw ai4i2020.csv)")
	parser.add_argument("--target", choices=["binary", "multiclass"], default="binary")
	parser.add_argument("--model", choices=["rf", "dt", "logreg"], default="rf")
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	train(
		in_path=args.in_path,
		target=args.target,
		model_type=args.model,
		test_size=args.test_size,
		random_state=args.seed,
	)


if __name__ == "__main__":
	main()

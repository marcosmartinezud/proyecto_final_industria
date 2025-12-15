from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.utils import data_dir, models_dir


def predict_csv(
	model_path: str | Path,
	in_csv: str | Path,
	out_csv: str | Path | None = None,
) -> Path:
	model_path = Path(model_path)
	in_csv = Path(in_csv)
	if out_csv is None:
		out_csv = in_csv.with_name(in_csv.stem + "_pred.csv")
	out_csv = Path(out_csv)

	artifact = joblib.load(model_path)
	pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact

	df = pd.read_csv(in_csv)
	preds = pipeline.predict(df)
	df_out = df.copy()
	df_out["prediction"] = preds
	df_out.to_csv(out_csv, index=False)
	print("Saved predictions:", out_csv)
	return out_csv


def main() -> None:
	parser = argparse.ArgumentParser(description="Run predictions using a trained model")
	parser.add_argument("--model", dest="model_path", default=None, help="Path to .joblib model")
	parser.add_argument("--in", dest="in_csv", default=None, help="Input CSV to score")
	parser.add_argument("--out", dest="out_csv", default=None, help="Output CSV with predictions")
	args = parser.parse_args()

	model_path = args.model_path
	if model_path is None:
		model_path = models_dir() / "model_binary_rf.joblib"

	in_csv = args.in_csv
	if in_csv is None:
		in_csv = data_dir() / "ai4i2020.csv"

	predict_csv(model_path=model_path, in_csv=in_csv, out_csv=args.out_csv)


if __name__ == "__main__":
	main()


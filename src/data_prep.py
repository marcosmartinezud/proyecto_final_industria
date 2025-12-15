from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

from src.features import add_derived_features
from src.utils import data_dir


def main(in_path: str | Path | None = None, out_path: str | Path | None = None) -> Path:
	"""Clean raw AI4I dataset for analysis/EDA."""

	if in_path is None:
		in_path = data_dir() / 'ai4i2020.csv'
	if out_path is None:
		out_path = data_dir() / 'ai4i_clean.csv'

	in_path = Path(in_path)
	out_path = Path(out_path)

	df = pd.read_csv(in_path)
	df = df.drop_duplicates()

	num_cols = df.select_dtypes(include='number').columns
	imputer = SimpleImputer(strategy='median')
	df[num_cols] = imputer.fit_transform(df[num_cols])

	df = add_derived_features(df)

	df.to_csv(out_path, index=False)
	print("Saved", out_path)
	return out_path

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Preprocess AI4I dataset')
	parser.add_argument('in_path', nargs='?', default='data/ai4i2020.csv', help='Input CSV path')
	parser.add_argument('out_path', nargs='?', default='data/ai4i_clean.csv', help='Output CSV path')
	args = parser.parse_args()
	main(args.in_path, args.out_path)



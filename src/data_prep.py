import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def main(in_path='data/ai4i2020.csv', out_path='data/ai4i_clean.csv'):
	df = pd.read_csv(in_path)

	df = df.drop_duplicates()

	num_cols = df.select_dtypes(include='number').columns
	imputer = SimpleImputer(strategy='median')
	df[num_cols] = imputer.fit_transform(df[num_cols])

	scaler = StandardScaler()
	df[num_cols] = scaler.fit_transform(df[num_cols])
	df.to_csv(out_path, index=False)
	print("Saved", out_path)

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Preprocess AI4I dataset')
	parser.add_argument('in_path', nargs='?', default='data/ai4i2020.csv', help='Input CSV path')
	parser.add_argument('out_path', nargs='?', default='data/ai4i_clean.csv', help='Output CSV path')
	args = parser.parse_args()
	main(args.in_path, args.out_path)


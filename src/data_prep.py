import pandas as pd
from sklearn.impute import SimpleImputer

def main(in_path='data/ai4i2020.csv', out_path='data/ai4i_clean.csv'):
	df = pd.read_csv(in_path)

	df = df.drop_duplicates()

	num_cols = df.select_dtypes(include='number').columns
	imputer = SimpleImputer(strategy='median')
	df[num_cols] = imputer.fit_transform(df[num_cols])

	# columnas derivadas
	df['Temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
	df['Torque_per_rpm'] = df['Torque [Nm]'] / df['Rotational speed [rpm]'].replace(0, 1)
	df['Tool_state'] = pd.cut(df['Tool wear [min]'], [0, 50, 150, df['Tool wear [min]'].max()+1], labels=['Nuevo', 'Medio', 'Viejo'], right=False)

	df.to_csv(out_path, index=False)
	print("Saved", out_path)
	print(df.head())

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Preprocess AI4I dataset')
	parser.add_argument('in_path', nargs='?', default='data/ai4i2020.csv', help='Input CSV path')
	parser.add_argument('out_path', nargs='?', default='data/ai4i_clean.csv', help='Output CSV path')
	args = parser.parse_args()
	main(args.in_path, args.out_path)


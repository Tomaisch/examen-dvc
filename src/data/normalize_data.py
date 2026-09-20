import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

input_path = Path("data/processed_data")
output_path = Path("data/processed_data")
X_train = pd.read_csv(input_path / "X_train.csv")
X_test = pd.read_csv(input_path / "X_test.csv")

col_names = X_train.columns
normalizer = MinMaxScaler()
X_train_scaled = normalizer.fit_transform(X_train)
X_test_scaled = normalizer.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=col_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=col_names)

X_train_scaled.to_csv(output_path / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(output_path / "X_test_scaled.csv", index=False)

joblib.dump(normalizer, "data/scaling_model.pkl")
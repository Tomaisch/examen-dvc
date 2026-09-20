import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import joblib

input_path = Path("data/processed_data")
models_path = Path("models")

X_train_scaled = pd.read_csv(input_path / "X_train_scaled.csv")
y_train = pd.read_csv(input_path / "y_train.csv")
y_train = y_train.squeeze()

best_params = joblib.load(models_path / "best_params.pkl")
gbr = GradientBoostingRegressor(**best_params)
gbr.fit(X_train_scaled, y_train)

joblib.dump(gbr, models_path / "gbrt_model.pkl")
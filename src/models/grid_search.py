import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
import joblib

input_path = Path("data/processed_data")
output_path = Path("models")
output_path.mkdir(parents=True, exist_ok=True)

X_train_scaled = pd.read_csv(input_path / "X_train_scaled.csv")
y_train = pd.read_csv(input_path / "y_train.csv")
y_train = y_train.squeeze()


gbr = GradientBoostingRegressor()

params = {"n_estimators": [100, 150, 200],
          "max_depth": [2, 3, 4,],
          "learning_rate": [0.001, 0.01 , 0.1]}

clf = GridSearchCV(estimator=gbr,
                   param_grid=params,
                   cv=3)

clf.fit(X_train_scaled, y_train)
best_params = clf.best_params_
joblib.dump(best_params, output_path / "best_params.pkl")
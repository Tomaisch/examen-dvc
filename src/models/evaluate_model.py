import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import joblib
import json

data_path = Path("data")
models_path = Path("models")
metrics_path = Path("metrics")
metrics_path.mkdir(parents=True, exist_ok=True)

X_test_scaled = pd.read_csv(data_path / "processed_data/X_test_scaled.csv")
y_test = pd.read_csv(data_path / "processed_data/y_test.csv")
y_test = y_test.squeeze()

gbrt_model = joblib.load(models_path / "gbrt_model.pkl")

y_predict = gbrt_model.predict(X_test_scaled)
y_predict = pd.DataFrame(y_predict)
y_predict.to_csv(data_path / "prediction.csv", index=False)

mse = mean_squared_error(y_test, y_predict)
r_2 = r2_score(y_test, y_predict)
metrics = {"mse": mse,
           "r_2": r_2}
scores_json = json.dumps(metrics)
(metrics_path / "scores.json").write_text(scores_json)
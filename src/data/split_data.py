import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

input_path = Path("data/raw_data/raw.csv")
df = pd.read_csv(input_path)
X = df.drop(columns = ["silica_concentrate", "date"])
y = df["silica_concentrate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=161)

output_dir = Path("data/processed_data")
output_dir.mkdir(parents=True, exist_ok=True)
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)
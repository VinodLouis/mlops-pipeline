import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/raw/iris.csv")
PROCESSED_PATH = Path("data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_csv(DATA_PATH)

# Split features and labels
features = df.drop(columns=["Id", "Species"])
labels = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Save processed features with headers
pd.DataFrame(X_train, columns=features.columns).to_csv(PROCESSED_PATH / "X_train.csv", index=False)
pd.DataFrame(X_test, columns=features.columns).to_csv(PROCESSED_PATH / "X_test.csv", index=False)

# Save labels with proper wrapping
pd.DataFrame({"label": y_train}).to_csv(PROCESSED_PATH / "y_train.csv", index=False)
pd.DataFrame({"label": y_test}).to_csv(PROCESSED_PATH / "y_test.csv", index=False)

print("Preprocessing complete. Files saved in:", PROCESSED_PATH)

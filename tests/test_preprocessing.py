import pandas as pd
import pytest
from pathlib import Path

RAW_PATH = Path("data/raw/iris.csv")
PROCESSED_PATH = Path("data/processed")

@pytest.fixture(scope="module")
def processed_data():
    # Load processed files
    X_train = pd.read_csv(PROCESSED_PATH / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_PATH / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_PATH / "y_train.csv")
    y_test = pd.read_csv(PROCESSED_PATH / "y_test.csv")
    return X_train, X_test, y_train, y_test

def test_raw_data_exists():
    assert RAW_PATH.exists(), f"Raw data not found at {RAW_PATH}"

def test_processed_files_exist():
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        path = PROCESSED_PATH / fname
        assert path.exists(), f"Missing processed file: {path}"

def test_feature_shape_consistency(processed_data):
    X_train, X_test, _, _ = processed_data
    assert X_train.shape[1] == X_test.shape[1], "Feature column mismatch between train and test"

def test_label_column_exists(processed_data):
    _, _, y_train, y_test = processed_data
    assert "label" in y_train.columns
    assert "label" in y_test.columns

def test_stratified_split_balance():
    df = pd.read_csv(RAW_PATH)
    original_dist = df["Species"].value_counts(normalize=True)

    y_train = pd.read_csv(PROCESSED_PATH / "y_train.csv")["label"]
    y_test = pd.read_csv(PROCESSED_PATH / "y_test.csv")["label"]

    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)

    for cls in original_dist.index:
        assert abs(train_dist[cls] - original_dist[cls]) < 0.1, f"Train class imbalance for {cls}"
        assert abs(test_dist[cls] - original_dist[cls]) < 0.1, f"Test class imbalance for {cls}"

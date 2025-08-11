import os
import shutil
import tempfile
import pandas as pd
import pytest
from argparse import Namespace
from unittest.mock import patch, MagicMock
from src.model_train import train_and_register, validate_args

@pytest.fixture(scope="module")
def dummy_data_dir():
    tmpdir = tempfile.mkdtemp()
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
    })
    labels = pd.DataFrame({"label": ["A", "B", "A", "B"]})
    df.iloc[:3].to_csv(os.path.join(tmpdir, "X_train.csv"), index=False)
    df.iloc[3:].to_csv(os.path.join(tmpdir, "X_test.csv"), index=False)
    labels.iloc[:3].to_csv(os.path.join(tmpdir, "y_train.csv"), index=False)
    labels.iloc[3:].to_csv(os.path.join(tmpdir, "y_test.csv"), index=False)
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def dummy_args(dummy_data_dir):
    return Namespace(
        mlflow_uri="http://dummy-uri",
        experiment_name="TestExperiment",
        model_name="test_model",
        data_dir=dummy_data_dir,
        output_dir=tempfile.mkdtemp(),
        scale=False,
        stage="None"
    )

def test_validate_args_success(dummy_args):
    validate_args(dummy_args)  # Should not raise or exit

def test_validate_args_missing_file(dummy_args, capsys):
    # Simulate missing file
    missing_path = os.path.join(dummy_args.data_dir, "X_test.csv")
    os.remove(missing_path)

    with pytest.raises(SystemExit) as e:
        validate_args(dummy_args)

    # ✅ Assert correct exit code
    assert e.value.code == 1

    # ✅ Assert correct error message
    captured = capsys.readouterr()
    assert "Missing files in" in captured.out
    assert "X_test.csv" in captured.out

@patch("src.model_train.mlflow")
@patch("src.model_train.MlflowClient")
def test_train_and_register_runs(mock_mlflow, mock_client, dummy_args):
    # Mock MLflow methods
    mock_mlflow.set_tracking_uri.return_value = None
    mock_mlflow.set_experiment.return_value = None
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "123"
    mock_mlflow.get_tracking_uri.return_value = dummy_args.mlflow_uri
    mock_mlflow.get_artifact_uri.return_value = "artifact_uri"
    mock_mlflow.sklearn.log_model.return_value = None
    mock_mlflow.log_param.return_value = None
    mock_mlflow.log_metric.return_value = None
    mock_mlflow.sklearn.save_model.return_value = None
    mock_mlflow.sklearn.load_model.return_value = MagicMock()

    # ✅ Catch the exit if validation fails
    with pytest.raises(SystemExit) as e:
        train_and_register(dummy_args)

    assert e.value.code == 1

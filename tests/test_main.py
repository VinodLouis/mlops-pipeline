import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch


# Patch token verification to bypass auth
app.dependency_overrides = {
    # If verify_token is a dependency, override it here
    # If it's a direct call, we'll mock it below
}

client = TestClient(app)

@pytest.fixture
def valid_input():
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

@patch("app.main.verify_token")
@patch("app.main.predict")
@patch("app.main.load_model")
def test_predict_success(mock_load_model, mock_predict, mock_verify_token, valid_input):
    mock_verify_token.return_value = None
    mock_predict.return_value = "setosa"
    mock_load_model.return_value = "dummy_model"

    response = client.post("/predict", json=valid_input)

    assert response.status_code == 200
    assert response.json() == {"prediction": "setosa"}

@patch("app.main.verify_token")
@patch("app.main.predict")
@patch("app.main.load_model")
def test_predict_failure(mock_load_model, mock_predict, mock_verify_token, valid_input):
    mock_verify_token.return_value = None
    mock_predict.side_effect = ValueError("Model error")
    mock_load_model.return_value = "dummy_model"

    response = client.post("/predict", json=valid_input)

    assert response.status_code == 500
    assert "Prediction failed" in response.json()["detail"]

def test_predict_validation_error():
    bad_input = {
        "sepal_length": "not-a-float",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=bad_input)

    assert response.status_code == 422
    assert "Input should be a valid number" in response.text


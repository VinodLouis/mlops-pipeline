import pytest
from fastapi.testclient import TestClient
from app.health import health_router
from unittest.mock import patch


client = TestClient(health_router)


@pytest.fixture
def dummy_input():
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


@patch("app.health.sqlite3.connect")
@patch("app.health.predict")
def test_health_ok(mock_predict, mock_connect, dummy_input):
    # Mock DB connection and table check
    mock_conn = mock_connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = ("logs",)

    # Mock model prediction
    mock_predict.return_value = "setosa"

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "components": ["sqlite", "model"]
    }


@patch("app.health.sqlite3.connect")
def test_health_sqlite_failure(mock_connect):
    # Simulate missing table
    mock_conn = mock_connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = None

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["component"] == "sqlite"
    assert "Table 'logs' does not exist" in body["detail"]


@patch("app.health.sqlite3.connect")
@patch("app.health.predict")
def test_health_model_failure(mock_predict, mock_connect):
    # DB is fine
    mock_conn = mock_connect.return_value
    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.fetchone.return_value = ("logs",)

    # Model prediction fails
    mock_predict.side_effect = ValueError("Model crashed")

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["component"] == "model"
    assert "Model crashed" in body["detail"]

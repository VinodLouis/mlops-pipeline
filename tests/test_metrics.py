import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.metrics import metrics_router


client = TestClient(metrics_router)


def mock_cursor_sequence():
    """Returns a cursor with sequential fetchone/fetchall behavior."""
    cursor = MagicMock()
    cursor.execute = MagicMock()

    # Simulate sequential results
    cursor.fetchone.side_effect = [
        [100],  # total
        [80],   # success
        [20],   # error
    ]
    cursor.fetchall.return_value = [
        {
            "id": 1,
            "timestamp": "2025-08-11T12:00:00",
            "method": "POST",
            "url": "/predict",
            "status": "success",
            "source": "api",
        }
    ]
    return cursor


@patch("app.metrics.conn")
def test_metrics_no_filters(mock_conn):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        [100],  # total
        [80],   # success
        [20],   # error
    ]
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "timestamp": "2025-08-11T12:00:00",
            "method": "POST",
            "url": "/predict",
            "status": "success",
            "source": "api",
        }
    ]
    mock_conn.execute.return_value = mock_cursor

    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()

    assert body["total_requests"] == 100
    assert body["success_count"] == 80
    assert body["error_count"] == 20
    assert body["limit"] == 25
    assert body["offset"] == 0
    assert body["status_filter"] is None
    assert body["source_filter"] is None
    assert len(body["logs"]) == 1
    assert body["logs"][0]["status"] == "success"


@patch("app.metrics.conn")
def test_metrics_with_filters(mock_conn):
    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [
        [50],  # total
        [30],  # success
        [20],  # error
    ]
    mock_cursor.fetchall.return_value = [
        {
            "id": 2,
            "timestamp": "2025-08-11T13:00:00",
            "method": "POST",
            "url": "/predict",
            "status": "error",
            "source": "cli",
        }
    ]
    mock_conn.execute.return_value = mock_cursor

    response = client.get("/metrics?status=error&source=cli&limit=10&offset=5")
    assert response.status_code == 200
    body = response.json()

    assert body["total_requests"] == 50
    assert body["success_count"] == 30
    assert body["error_count"] == 20
    assert body["limit"] == 10
    assert body["offset"] == 5
    assert body["status_filter"] == "error"
    assert body["source_filter"] == "cli"
    assert len(body["logs"]) == 1
    assert body["logs"][0]["source"] == "cli"
    assert body["logs"][0]["status"] == "error"


import json
import pytest
from unittest.mock import patch, MagicMock
from app.logger import log_request


class DummyRequest:
    def __init__(self, method="POST", url="http://localhost/predict"):
        self.method = method
        self.url = url


@patch("app.logger.sqlite3.connect")
def test_log_request_success(mock_connect):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    request = DummyRequest()
    input_data = {"sepal_length": 5.1}
    prediction = {"class": "setosa"}

    log_request(
        request=request,
        input_data=input_data,
        prediction=prediction,
        status="success",
        error=None,
        source="prediction",
        details=None,
    )

    mock_connect.assert_called_once_with("logs/logs.db")
    mock_cursor.execute.assert_called_once()

    # Validate SQL args
    args = mock_cursor.execute.call_args[0][1]
    assert args[0] == "POST"
    assert args[1] == "http://localhost/predict"
    assert json.loads(args[2]) == input_data
    assert json.loads(args[3]) == prediction
    assert args[4] == "success"
    assert args[5] is None
    assert args[6] == "prediction"
    assert args[7] is None

    # Config values
    from app.config import MODEL_STAGE, MODEL_NAME, MODEL_SOURCE, MODEL_VERSION
    assert args[8] == MODEL_STAGE
    assert args[9] == MODEL_NAME
    assert args[10] == MODEL_SOURCE
    assert args[11] == MODEL_VERSION

    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()

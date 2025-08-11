import pytest
import json
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
from unittest.mock import patch
from app.exceptions import handle_validation_error, handle_http_exception


def make_post_request(json_body: dict) -> StarletteRequest:
    scope = {
        "type": "http",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
    }
    request = StarletteRequest(scope)
    request._body = json.dumps(json_body).encode("utf-8")  # simulate body
    return request


@pytest.mark.asyncio
@patch("app.exceptions.log_request")
async def test_handle_validation_error(mock_log):
    request = make_post_request({"sepal_length": "not-a-float"})
    exc = RequestValidationError(errors=[{"loc": ["body", "sepal_length"], "msg": "Invalid float"}])

    response = await handle_validation_error(request, exc)

    assert response.status_code == 422
    assert json.loads(response.body) == {"detail": exc.errors()}
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert kwargs["status"] == "error"
    assert kwargs["source"] == "validation"
    assert "sepal_length" in kwargs["error"]


@pytest.mark.asyncio
@patch("app.exceptions.log_request")
async def test_handle_http_exception_auth(mock_log):
    request = make_post_request({"some": "input"})
    exc = HTTPException(status_code=403, detail="Unauthorized")

    response = await handle_http_exception(request, exc)

    assert response.status_code == 403
    assert json.loads(response.body) == {"detail": "Unauthorized"}
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert kwargs["status"] == "error"
    assert kwargs["source"] == "auth"
    assert "Unauthorized" in kwargs["error"]


@pytest.mark.asyncio
@patch("app.exceptions.log_request")
async def test_handle_http_exception_generic(mock_log):
    request = make_post_request({"some": "input"})
    exc = HTTPException(status_code=404, detail="Not found")

    response = await handle_http_exception(request, exc)

    assert response.status_code == 404
    assert json.loads(response.body) == {"detail": "Not found"}
    mock_log.assert_called_once()
    args, kwargs = mock_log.call_args
    assert kwargs["source"] == "http"
    assert "Not found" in kwargs["error"]

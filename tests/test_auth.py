import pytest
from fastapi import Request, HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request as StarletteRequest
from app.auth import verify_token
from app.config import API_TOKEN


def make_request_with_auth(auth_header: str) -> Request:
    scope = {
        "type": "http",
        "headers": [(b"authorization", auth_header.encode())],
    }
    return StarletteRequest(scope)


def test_verify_token_success():
    request = make_request_with_auth(f"Bearer {API_TOKEN}")
    # Should not raise
    verify_token(request)


def test_verify_token_missing_header():
    request = make_request_with_auth("")
    with pytest.raises(HTTPException) as exc:
        verify_token(request)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Missing or invalid token"


def test_verify_token_invalid_token():
    request = make_request_with_auth("Bearer wrong-token")
    with pytest.raises(HTTPException) as exc:
        verify_token(request)
    assert exc.value.status_code == 403
    assert exc.value.detail == "Unauthorized"

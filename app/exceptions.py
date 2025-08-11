from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import json
from app.logger import log_request

async def handle_validation_error(request: Request, exc: RequestValidationError):
    input_data = await request.json() if request.method == "POST" else {}
    error_detail = exc.errors()

    log_request(
        request=request,
        input_data=input_data,
        prediction=None,
        status="error",
        error=json.dumps(error_detail),
        source="validation",
        details=None
    )

    return JSONResponse(status_code=422, content={"detail": error_detail})


async def handle_http_exception(request: Request, exc: HTTPException):
    input_data = await request.json() if request.method == "POST" else {}
    error_detail = {"status_code": exc.status_code, "detail": exc.detail}

    log_request(
        request=request,
        input_data=input_data,
        prediction=None,
        status="error",
        error=json.dumps(error_detail),
        source="auth" if exc.status_code in [401, 403] else "http",
        details=None
    )

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

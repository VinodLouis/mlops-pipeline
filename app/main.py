import json
import traceback
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.model import load_model, predict
from app.logger import log_request
from app.metrics import metrics_router
from app.health import health_router
from app.auth import verify_token
from app.exceptions import handle_validation_error, handle_http_exception
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException


app = FastAPI()
app.add_exception_handler(RequestValidationError, handle_validation_error)
app.add_exception_handler(HTTPException, handle_http_exception)
app.include_router(metrics_router)
app.include_router(health_router)

model = load_model()

class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict_endpoint(data: Input, request: Request):
    verify_token(request)
    input_dict = data.dict()

    prediction = None
    status = "success"
    error = None
    details = None

    try:
        prediction = predict(model, input_dict)
    except Exception as e:
        status = "error"
        error = str(e)
        details = json.dumps({
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        })

    # Always log the request
    log_request(
        request=request,
        input_data=input_dict,
        prediction=prediction,
        status=status,
        error=error,
        source="prediction",
        details=details
    )

    if status == "error":
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

    return {"prediction": prediction}

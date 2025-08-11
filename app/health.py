from app.model import predict
from fastapi import APIRouter
import sqlite3


health_router = APIRouter()


@health_router.get("/health")
def health_check():
    # Check SQLite DB and logs table
    try:
        conn = sqlite3.connect("logs/logs.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='logs'"
        )
        if not cursor.fetchone():
            raise Exception("Table 'logs' does not exist")
    except Exception as e:
        return {"status": "error", "component": "sqlite", "detail": str(e)}

    # Check model prediction
    try:
        dummy_input = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
        _ = predict(None, dummy_input)
    except Exception as e:
        return {"status": "error", "component": "model", "detail": str(e)}

    return {"status": "ok", "components": ["sqlite", "model"]}

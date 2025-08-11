import json
import sqlite3
from datetime import datetime
from app.config import MODEL_STAGE, MODEL_NAME, MODEL_SOURCE, MODEL_VERSION

conn = sqlite3.connect("logs/logs.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    method TEXT,
    url TEXT,
    input TEXT,
    prediction TEXT,
    status TEXT,
    error TEXT,
    source TEXT,
    details TEXT,
    model_stage TEXT, 
    model_name TEXT, 
    model_source TEXT, 
    model_version TEXT
);
""")

def log_request(request, input_data, prediction, status="success", error=None, source=None, details=None):
    conn = sqlite3.connect("logs/logs.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs (timestamp, method, url, input, prediction, status, error, source, details, model_stage, model_name, model_source, model_version)
        VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        request.method,
        str(request.url),
        json.dumps(input_data),
        json.dumps(prediction),
        status,
        error,
        source,
        details,
        MODEL_STAGE,
        MODEL_NAME,
        MODEL_SOURCE,
        MODEL_VERSION
    ))

    conn.commit()
    conn.close()


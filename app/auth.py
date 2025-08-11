# app/auth.py
from fastapi import Request, HTTPException

from app.config import API_TOKEN


# Dummy Auth
def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth.split("Bearer ")[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Unauthorized")

import sys
from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", None)
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")
MODEL_SOURCE = os.getenv("MODEL_SOURCE", None)
MODEL_VERSION = os.getenv("MODEL_VERSION", None)

# token
API_TOKEN = os.getenv("API_TOKEN", "supersecret")


# Validate MODEL_SOURCE
if MODEL_SOURCE not in ("LOCAL", "REMOTE"):
    print("Invalid MODEL_SOURCE. Must be 'LOCAL' or 'REMOTE'.")
    sys.exit(1)

# Validate MODEL_SOURCE
if MODEL_NAME is None:
    print("Invalid MODEL_NAME. A value is required!")
    sys.exit(1)


# Validate MODEL_VERSION only if source is REMOTE
if MODEL_SOURCE == "REMOTE" and MODEL_VERSION is None:
    print("MODEL_VERSION is required when MODEL_SOURCE is 'REMOTE'.")
    sys.exit(1)

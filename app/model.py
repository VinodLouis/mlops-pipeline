import os
import mlflow.pyfunc
import traceback
import mlflow

from app.config import (
    MLFLOW_TRACKING_URI,
    MODEL_STAGE,
    MODEL_NAME,
    MODEL_SOURCE,
    MODEL_VERSION,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model():
    print("Calling load_model")
    print("MLFLOW_TRACKING_URI:", mlflow.get_tracking_uri())
    print("MODEL_SOURCE:", MODEL_SOURCE)

    if MODEL_SOURCE == "LOCAL":
        return load_local_model()

    # Try remote first
    try:
        print("Searching for model versions...")
        versions = mlflow.search_model_versions(filter_string=f"name='{MODEL_NAME}'")
        print(versions)

        if MODEL_VERSION:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
            print(f"Loading specific version: {MODEL_VERSION}")
        else:
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            print(f"Loading from stage: {MODEL_STAGE}")

        return mlflow.pyfunc.load_model(model_uri)

    except Exception:
        print("Failed to load from MLflow registry")
        traceback.print_exc()
        print("Falling back to local model...")
        return load_local_model()


def load_local_model():
    path = f"models/{MODEL_NAME}"
    print(f"Trying to load local model from: {path}")

    try:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "MLmodel")):
            print("Detected MLflow model directory")
            return mlflow.pyfunc.load_model(path)
        elif os.path.isfile(path):
            print("Detected raw pickle file")
            import pickle

            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Model path not found: {path}")

    except FileNotFoundError:
        print(f"File not found: {path}")
    except PermissionError:
        print(f"Permission denied for: {path}")
    except Exception as e:
        print(f"Unexpected error loading local model: {e}")
        traceback.print_exc()

    return None


def predict(model, features):
    if model is None:
        return "dummy-class"
    try:
        return model.predict([list(features.values())])[0]
    except Exception:
        print("Failed to load local model")
        traceback.print_exc()
        return "error"

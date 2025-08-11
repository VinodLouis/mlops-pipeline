import os
import shutil
import sys
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import requests
import re


# Save model locally
def save_model_locally(model, path):
    if os.path.exists(path):
        print(f"🧹 Removing existing model directory: {path}")
        shutil.rmtree(path)

    mlflow.sklearn.save_model(model, path=path)
    print(f"Saved model locally to: {path}")


# Validate inputs
def validate_args(args):
    required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [
        f for f in required_files if not os.path.isfile(os.path.join(args.data_dir, f))
    ]
    if missing:
        print(f"Missing files in {args.data_dir}: {', '.join(missing)}")
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except Exception as e:
            print(f"Cannot create output directory: {args.output_dir}\n{e}")
            sys.exit(2)

    if args.mlflow_uri.startswith("http"):
        try:
            response = requests.get(args.mlflow_uri)
            if response.status_code != 200:
                print(
                    f"Warning: MLflow URI responded with status {response.status_code}"
                )
        except Exception:
            print(f"Warning: Cannot reach MLflow URI: {args.mlflow_uri}")

    if not re.match(r"^[A-Za-z0-9_\-]+$", args.model_name):
        print(
            f"Invalid model name: '{args.model_name}'. "
            "Use only letters, numbers, underscores, or hyphens."
        )
        sys.exit(3)


# Register model only if using remote MLflow URI
def register_model_if_remote(
    model_uri, args, mlflow_client, local_model_path, best_model_instance
):
    if args.mlflow_uri.startswith("http"):
        try:
            try:
                mlflow_client.get_registered_model(args.model_name)
                print(f"Model '{args.model_name}' exists. Registering new version.")
            except RestException:
                print(f"Creating new registered model '{args.model_name}'")

            model_version = mlflow.register_model(
                model_uri=model_uri, name=args.model_name
            )
            print(
                f"Registered model: {model_version.name} "
                f"version {model_version.version}"
            )

            if args.stage and args.stage != "None":
                mlflow_client.transition_model_version_stage(
                    name=args.model_name,
                    version=model_version.version,
                    stage=args.stage,
                )
                print(f"Transitioned model to stage: {args.stage}")

            best_model_loaded = mlflow.sklearn.load_model(model_uri)
            save_model_locally(best_model_loaded, local_model_path)

        except Exception as e:
            print(f"Error registering or transitioning model: {e}")
            print("You can register manually via MLflow UI.")
            save_model_locally(best_model_instance, local_model_path)
    else:
        print("ℹSkipping model registration — not using remote MLflow URI.")
        save_model_locally(best_model_instance, local_model_path)


# Main training logic
def train_and_register(args):
    validate_args(args)

    output_dir = os.path.abspath(args.output_dir)

    if not args.mlflow_uri or args.mlflow_uri.strip() == "":
        print(
            "MLflow URI not provided. "
            "Please specify --mlflow-uri pointing to your MLflow server."
        )
        sys.exit(4)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    mlflow_client = MlflowClient()

    # Load pre-split data
    X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(args.data_dir, "y_test.csv")).squeeze()

    # Optional scaling
    if args.scale:
        print("Scaling features with StandardScaler")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model_configs = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=100, random_state=42, min_samples_leaf=1, max_features="sqrt"
        ),
    }

    run_infos = []

    for model_name, model_instance in model_configs.items():
        with mlflow.start_run(run_name=model_name) as run:
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"{model_name} Accuracy: {acc:.4f}")

            print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

            print("Artifact URI:", mlflow.get_artifact_uri())

            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model_instance, artifact_path="model")

            print(f"Logged model to run: {run.info.run_id}")
            run_infos.append((model_name, acc, run.info.run_id, model_instance))

            local_model_path = os.path.join(output_dir, f"{model_name}_classifier")
            save_model_locally(model_instance, local_model_path)
            print(f"Saved local model: {local_model_path}")

    best_model_name, best_accuracy, best_run_id, best_model_instance = sorted(
        run_infos, key=lambda x: x[1], reverse=True
    )[0]
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    print(f"Model URI: {model_uri}")

    local_model_path = os.path.join(output_dir, args.model_name)

    register_model_if_remote(
        model_uri=model_uri,
        args=args,
        mlflow_client=mlflow_client,
        local_model_path=local_model_path,
        best_model_instance=best_model_instance,
    )


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and register Iris classifier with MLflow (pre-split version)"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="",
        help="MLflow tracking URI (default: local mlruns folder)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Iris_Classification",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="iris_classifier",
        help="Registered model name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing X_train.csv, X_test.csv, y_train.csv, y_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Directory to save local model",
    )
    parser.add_argument(
        "--scale", action="store_true", help="Apply StandardScaler to features"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="None",
        choices=["None", "Staging", "Production"],
        help="Optional stage to transition model",
    )

    args = parser.parse_args()
    train_and_register(args)

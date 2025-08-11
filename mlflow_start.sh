#!/bin/bash
set -euo pipefail

# --- CONFIG ---
AWS_REGION="eu-north-1"
BUCKET_NAME="vinod-mlop-data"
ARTIFACT_PREFIX="mlflow-artifacts"
HOST_PORT=5002
CONTAINER_PORT=5000
CONTAINER_NAME="mlflow-server"
MLFLOW_IMAGE="ghcr.io/mlflow/mlflow:v2.9.2"

# === OPTIONAL: GHCR LOGIN ===
CR_PAT="<token to connect GHCR>"
GHCR_USER="<user>"    

# Paths on host for persistence
HOST_DB_DIR="$(pwd)/mlflow_db"
mkdir -p "$HOST_DB_DIR"

# Optional: create an S3 prefix marker (not required)
aws s3api put-object --bucket "$BUCKET_NAME" --key "$ARTIFACT_PREFIX/" || true

# Cleanup
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "🧹 Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
fi


# === LOGIN (Optional) ===
if [ -n "$CR_PAT" ]; then
  echo "Logging into GHCR..."
  echo "$CR_PAT" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
fi

echo "Pulling MLflow image: $MLFLOW_IMAGE"
docker pull "$MLFLOW_IMAGE"

echo "$HOME"

echo "Starting MLflow server..."
docker run -d --rm \
  --name "$CONTAINER_NAME" \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -v "$HOST_DB_DIR:/mlflow/db" \
  -v "$HOME/.aws:/root/.aws:ro" \
  -e AWS_REGION="$AWS_REGION" \
  "$MLFLOW_IMAGE" \
  mlflow server \
    --backend-store-uri "sqlite:////mlflow/db/mlflow.db" \
    --default-artifact-root "s3://$BUCKET_NAME/$ARTIFACT_PREFIX" \
    --host 0.0.0.0 \
    --port "$CONTAINER_PORT"

echo "Waiting 2–3s..."
sleep 15
echo "Logs:"
docker logs "$CONTAINER_NAME" --tail 200
echo "UI: http://localhost:$HOST_PORT"

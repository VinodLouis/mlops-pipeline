# MLOps Pipeline

A modular, reproducible MLOps pipeline for training, tracking, registering, and serving ML models using MLflow, Docker Compose, and FastAPI. Built with robust error handling, environment validation, and CI/CD integration.

## ⚙️ Setup Instructions

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate iris-mlops-env
```

### 2. Data

The data folder contains two sub folders

```
raw : Contains whole dataset in csv format
preprocessed : Contains the preprocessed data USES DVC for storage
```

### 3. Steps to preprocess data

```
The data preprocess here is a simple python file which cleans the data and creates the preprocessed files
```

To clean data & create the preprocessed file. One can directly run the following command from root

> python src/preprocess_data.py

### 4. Run MLflow

There are two ways to run MLflow in your local

#### 4.1. Using Standalone mlflow docker

> bash mlflow_start.sh

The script contains several important details

1. To fetch the [mlflow official docker image](https://mlflow.org/docs/latest/ml/docker/) we need GHCR token which needs to be populated

2. mlflow db is mounted to host folder

3. mlflow artifacts are stored directly on s3. To ensure this works you need to configure `.aws/credentails` in your home

If all goes well your browser http://localhost:5002 should point to mlflow ui

#### 4.2 Using docker compose

There is a docker compose file which you can use. It contains scripts to run both mlflow and the app service to test endpoints which serves model. This is useful for `dev` environment

To run docker compose

> docker compose up

Should bring both app endpoint and mlflow up and running

### 5. Train the model

To train the model you can directly RUN the bash script which abstracts certain details

> bash model_train.sh

It passes some arguments and invokes `src/model_train.py`. The arguments are explain in the `model_train.py` file. It trains the model and saves all details into the mlflow which we started in above steps. As the DB and artifacts are persisted mount, data is preserved.

All the artifacts are saved into s3. Meanwhile we compare and save the best model locally as well as a fallback plan. Also this best model is registered in mlflow for future use

### 6. Test the model

If you are running the docker compose version the app should be up and liseting on http://localhost:8080. It certainly exposes API to test the model and logging for each request. When it starts

1. It first tries to load the remote model from mlflow matching with name, stage and version based on the setting `MODEL_SOURCE=REMOTE` in .env

2. If it fails it tries to load the locally saved model which we saved in train considering the best model

3. If this also fails it just fall backs to a default DUMMY response

A healt check API is exposed

```
curl --location 'http://localhost:8000/health'
```

responds with

```
{
    "status": "ok",
    "components": [
        "sqlite",
        "model"
    ]
}
```

If any component is down will notify

You can test the API with

```
curl --location 'http://localhost:8000/predict' \
--header 'Authorization: Bearer supersecret123' \
--header 'Content-Type: application/json' \
--data '{"sepal_length":51.1,"sepal_width":32.5,"petal_length":1.4,"petal_width":0.2}'
```

It will run the model and return the result mostly like:

```
{
    "prediction": "Iris-versicolor"
}
```

To see the metrics you can access via

```
curl --location 'http://localhost:8000/metrics'
```

It retuns the metrics captured ovee time

```
{
    "total_requests": 7,
    "success_count": 3,
    "error_count": 4,
    "limit": 25,
    "offset": 0,
    "status_filter": null,
    "source_filter": null,
    "logs": [
        {
            "id": 7,
            "timestamp": "2025-08-11 08:14:20",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_length\": 51.1, \"sepal_width\": 32.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "\"Iris-versicolor\"",
            "status": "success",
            "error": null,
            "source": "prediction",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "LOCAL",
            "model_version": "2"
        },
        {
            "id": 6,
            "timestamp": "2025-08-11 08:14:09",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_lengthsdfsd\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "null",
            "status": "error",
            "error": "[{\"type\": \"missing\", \"loc\": [\"body\", \"sepal_length\"], \"msg\": \"Field required\", \"input\": {\"sepal_lengthsdfsd\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}}]",
            "source": "validation",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "LOCAL",
            "model_version": "2"
        },
        {
            "id": 5,
            "timestamp": "2025-08-11 08:13:59",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "\"Iris-versicolor\"",
            "status": "success",
            "error": null,
            "source": "prediction",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "LOCAL",
            "model_version": "2"
        },
        {
            "id": 4,
            "timestamp": "2025-08-11 08:13:53",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "null",
            "status": "error",
            "error": "{\"status_code\": 403, \"detail\": \"Unauthorized\"}",
            "source": "auth",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "LOCAL",
            "model_version": "2"
        },
        {
            "id": 3,
            "timestamp": "2025-08-11 08:12:40",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "null",
            "status": "error",
            "error": "{\"status_code\": 403, \"detail\": \"Unauthorized\"}",
            "source": "auth",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "REMOTE",
            "model_version": "2"
        },
        {
            "id": 2,
            "timestamp": "2025-08-11 08:12:33",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_lengthsdfsdf\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "null",
            "status": "error",
            "error": "[{\"type\": \"missing\", \"loc\": [\"body\", \"sepal_length\"], \"msg\": \"Field required\", \"input\": {\"sepal_lengthsdfsdf\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}}]",
            "source": "validation",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "REMOTE",
            "model_version": "2"
        },
        {
            "id": 1,
            "timestamp": "2025-08-11 08:12:27",
            "method": "POST",
            "url": "http://localhost:8000/predict",
            "input": "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}",
            "prediction": "\"Iris-versicolor\"",
            "status": "success",
            "error": null,
            "source": "prediction",
            "details": null,
            "model_stage": "Production",
            "model_name": "iris_classifier",
            "model_source": "REMOTE",
            "model_version": "2"
        }
    ]
}
```

This API is pretty flexible paginated and allows params as

```
limit  : integer as limit default is 25
offset : integer as offset for pagination
status : string Filter by 'success' or 'error'
source : filter by source
```

# MLOps Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-VinodLouis%2Fmlops--pipeline-blue?logo=github)](https://github.com/VinodLouis/mlops-pipeline)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://www.docker.com/)

A production-ready MLOps pipeline for training, tracking, registering, and serving machine learning models. This project demonstrates best practices in ML operations using MLflow for experiment tracking, Docker for containerization, and FastAPI for model serving.

## 🎯 Overview

This MLOps pipeline provides a complete, end-to-end workflow for machine learning model development and deployment. Built as a demonstration of production-ready ML engineering practices, it uses the classic **Iris dataset** for classification as an example use case.

**Why This Project?**

This pipeline showcases how to build a robust, scalable ML system that goes beyond just training models. It addresses real-world challenges like:

- How do you track and compare different model experiments?
- How do you version and deploy models safely to production?
- How do you monitor model performance and API usage?
- How do you handle failures gracefully with fallback mechanisms?

While the example uses Iris classification, the **modular architecture** makes it easy to adapt for your own ML projects—whether you're working with time series forecasting, NLP, computer vision, or any other domain.

**Key Technologies:**

- **MLflow**: Experiment tracking, model registry, and versioning
- **Docker & Docker Compose**: Containerization and orchestration
- **FastAPI**: High-performance API framework for model serving
- **DVC**: Data version control for preprocessing
- **Conda**: Environment management
- **AWS S3**: Artifact storage

### 🔗 Quick Links

- [GitHub Repository](https://github.com/VinodLouis/mlops-pipeline)
- [Report Issues](https://github.com/VinodLouis/mlops-pipeline/issues)
- [Request Features](https://github.com/VinodLouis/mlops-pipeline/issues/new)
- [MLflow UI](http://localhost:5002) (after setup)
- [API Documentation](http://localhost:8000/docs) (after setup)

---

## ✨ Features

- **Modular Design**: Easily adaptable components for different ML workflows
- **Reproducible Experiments**: Track all experiments with MLflow
- **Model Versioning**: Automatic model registration and version management
- **Robust Error Handling**: Multiple fallback mechanisms for model loading
- **Request Logging**: Comprehensive logging of all API requests and predictions
- **Health Monitoring**: Built-in health checks for system components
- **Flexible Deployment**: Supports both local and remote model serving
- **CI/CD Ready**: Designed for integration with continuous deployment pipelines

### MLflow Tracking UI

Track experiments, compare runs, and manage model versions through an intuitive web interface.

### API Endpoints

RESTful API with automatic interactive documentation powered by FastAPI.

### Real-time Metrics

Monitor prediction requests, success rates, and error logs in real-time.

> **Note:** Screenshots and detailed demos will be added soon. Check the [Issues](https://github.com/VinodLouis/mlops-pipeline/issues) for progress.

## 🗺️ Features Done

- [x] Basic MLOps pipeline with MLflow integration
- [x] FastAPI model serving with authentication
- [x] Request logging and metrics tracking
- [x] Docker containerization
- [X ] CI/CD pipeline with GitHub Actions

## 🏗️ Architecture

```
┌─────────────────┐
│   Data Sources  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │ ◄── DVC
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │ ──► MLflow (Tracking)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │ ──► S3 (Artifacts)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Server │ ──► SQLite (Logs)
└─────────────────┘
```

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Conda** (Miniconda or Anaconda) - [Installation Guide](https://docs.conda.io/en/latest/miniconda.html)
- **Docker** (20.10+) and **Docker Compose** (v2.0+) - [Installation Guide](https://docs.docker.com/get-docker/)
- **Git** (2.0+)
- **AWS CLI** (configured with credentials for S3 access) - [Installation Guide](https://aws.amazon.com/cli/)
- **GitHub Personal Access Token** (for MLflow Docker image) - [Create Token](https://github.com/settings/tokens)

### System Requirements

- **OS**: Linux, macOS, or Windows with WSL2
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 10GB free space
- **Network**: Internet connection for pulling Docker images and downloading packages

### Compatibility Matrix

| Component    | Tested Versions      |
| ------------ | -------------------- |
| Python       | 3.8, 3.9, 3.10, 3.11 |
| Docker       | 20.10+, 24.0+        |
| MLflow       | 2.x                  |
| FastAPI      | 0.100+               |
| scikit-learn | 1.2+                 |

✅ **Recommended Setup**: Ubuntu 22.04 or macOS with Docker Desktop

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VinodLouis/mlops-pipeline.git
cd mlops-pipeline
```

### 2. Create Conda Environment

Create and activate the Conda environment with all required dependencies:

```bash
conda env create -f environment.yml
conda activate iris-mlops-env
```

This will install all necessary Python packages including:

- scikit-learn
- mlflow
- fastapi
- pandas
- numpy
- and other dependencies

### 3. Configure AWS Credentials

Ensure your AWS credentials are properly configured for S3 access:

```bash
aws configure
```

You'll need:

- AWS Access Key ID
- AWS Secret Access Key
- Default region name
- Default output format

⚠️ **Important**:

- Never commit AWS credentials to version control
- Use IAM roles with minimal required permissions
- In production, use instance roles or managed identities instead of hardcoded credentials

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5002
MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Model Configuration
MODEL_NAME=iris_classifier
MODEL_STAGE=Production
MODEL_VERSION=latest
MODEL_SOURCE=REMOTE  # Options: REMOTE, LOCAL

# API Configuration
API_KEY=supersecret123
PORT=8000
```

🔒 **Security Note**: Add `.env` to your `.gitignore` file to prevent committing sensitive credentials:

```bash
echo ".env" >> .gitignore
```

### 5. Set Up GitHub Token for MLflow

Export your GitHub token to pull the official MLflow Docker image:

```bash
export GHCR_TOKEN=<your-github-token>
```

## 🆘 First Time Here?

If you're new to MLOps or this project, follow this recommended path:

1. **Start Simple**: Use Docker Compose to get everything running quickly
2. **Explore MLflow**: Open http://localhost:5002 and familiarize yourself with the UI
3. **Train a Model**: Run `bash model_train.sh` and watch it appear in MLflow
4. **Test the API**: Use the `/predict` endpoint with the example curl command
5. **Check Metrics**: Visit `/metrics` to see logged requests
6. **Read the Code**: Start with `src/model_train.py` to understand the workflow

💡 **Tip**: Keep the MLflow UI and API docs open in separate browser tabs while experimenting!

## 📁 Project Structure

```
mlops-pipeline/
├── data/
│   ├── raw/                    # Original datasets (CSV format)
│   └── processed/              # Cleaned data (DVC tracked)
├── src/
│   ├── preprocess_data.py      # Data cleaning and preprocessing
│   ├── model_train.py          # Model training script
│   app/                        # FastAPI application
│   ├── main.py                 # API endpoints
│   ├── models.py               # Model loader
│   ├── auth.py                 # Auth interceptor
│   ├── exceptions.py           # Global exceptions
│   ├── health.py               # Global exceptions
│   ├── metrics.py              # Inference metrics
│   └── logger.py               # Logging database
├── artifacts/                  # Locally saved best models
├── mlflow_db/                  # MLflow database files
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # App container definition
├── environment.yml             # Conda environment specs
├── mlflow_start.sh             # MLflow standalone script
├── model_train.sh              # Training wrapper script
└── README.md                   # This file
```

## 🎬 Quick Start

### Option 1: Using Docker Compose (Recommended for Development)

This brings up both MLflow and the FastAPI server:

```bash
# Start all services
docker compose up

# Run in detached mode
docker compose up -d

# Stop all services
docker compose down
```

Services will be available at:

- MLflow UI: http://localhost:5002
- FastAPI Server: http://localhost:8000
- API Docs: http://localhost:8000/docs

✅ **Verify Setup**:

```bash
# Check if containers are running
docker ps

# Check logs
docker compose logs -f

# Test health endpoint
curl http://localhost:8000/health
```

### Option 2: Standalone MLflow

If you only want to run MLflow for experiment tracking:

```bash
bash mlflow_start.sh
```

Access MLflow UI at: http://localhost:5002

## 📖 Detailed Usage

### Step 1: Data Preprocessing

The preprocessing step cleans the raw data and creates preprocessed files tracked by DVC.

```bash
python src/preprocess_data.py
```

**What it does:**

- Loads raw data from `data/raw/`
- Performs data cleaning and validation
- Saves preprocessed data to `data/preprocessed/`
- Versions the preprocessed data with DVC

**Input:** `data/raw/*.csv`  
**Output:** `data/preprocessed/*.csv`

### Step 2: Model Training

Train the model and log all metrics, parameters, and artifacts to MLflow:

```bash
bash model_train.sh
```

**What it does:**

- Loads preprocessed data
- Trains the model with configurable hyperparameters
- Logs all experiments to MLflow
- Saves artifacts to S3
- Compares with previous models
- Registers the best model locally and in MLflow registry

**Arguments (configured in `model_train.sh`):**

- `--data-path`: Path to preprocessed data
- `--experiment-name`: MLflow experiment name
- `--model-name`: Name for model registration
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth
- `--random-state`: Random seed for reproducibility

**Example with custom parameters:**

```bash
python src/model_train.py \
  --data-path data/preprocessed/iris.csv \
  --experiment-name iris-classification \
  --model-name iris_classifier \
  --n-estimators 200 \
  --max-depth 10 \
  --random-state 42
```

### Step 3: Model Serving

The FastAPI server provides a REST API for model predictions with robust fallback mechanisms.

#### Model Loading Strategy

The server attempts to load models in the following order:

1. **Remote Model** (if `MODEL_SOURCE=REMOTE` in `.env`):
   - Fetches from MLflow Model Registry
   - Uses specified `MODEL_NAME`, `MODEL_STAGE`, and `MODEL_VERSION`

2. **Local Model** (fallback):
   - Loads the best model saved locally during training
   - Located in `models/` directory

3. **Dummy Model** (last resort):
   - Returns default predictions if all else fails
   - Logs warnings for investigation

#### Starting the Server

If using Docker Compose, the server starts automatically. Otherwise:

```bash
# Run locally with uvicorn
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔌 API Documentation

### Health Check

Check the status of system components:

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "components": ["sqlite", "model"],
  "model_info": {
    "name": "iris_classifier",
    "version": "2",
    "stage": "Production",
    "source": "REMOTE"
  }
}
```

**Component Status:**

- `ok`: All components are functioning
- `degraded`: Some components are down
- Components checked: database, model

### Make Predictions

Send prediction requests with proper authentication:

```bash
curl --location 'http://localhost:8000/predict' \
  --header 'Authorization: Bearer supersecret123' \
  --header 'Content-Type: application/json' \
  --data '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

**Request Body:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "prediction": "Iris-versicolor",
  "model_version": "2",
  "model_stage": "Production",
  "timestamp": "2025-02-10T10:30:45Z"
}
```

**Error Responses:**

Authentication Error (403):

```json
{
  "detail": "Unauthorized"
}
```

Validation Error (422):

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "sepal_length"],
      "msg": "Field required"
    }
  ]
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📊 Monitoring and Metrics

### Request Logging

All API requests are logged to a SQLite database with detailed information.

#### Retrieve Metrics

```bash
curl http://localhost:8000/metrics
```

**Response:**

```json
{
  "total_requests": 150,
  "success_count": 142,
  "error_count": 8,
  "success_rate": 94.67,
  "limit": 25,
  "offset": 0,
  "logs": [
    {
      "id": 150,
      "timestamp": "2025-02-10 10:30:45",
      "method": "POST",
      "url": "http://localhost:8000/predict",
      "input": "{\"sepal_length\": 5.1, ...}",
      "prediction": "\"Iris-versicolor\"",
      "status": "success",
      "error": null,
      "source": "prediction",
      "model_name": "iris_classifier",
      "model_version": "2",
      "model_stage": "Production",
      "model_source": "REMOTE"
    }
  ]
}
```

#### Query Parameters

- `limit` (default: 25): Number of records to return
- `offset` (default: 0): Pagination offset
- `status`: Filter by 'success' or 'error'
- `source`: Filter by source ('prediction', 'auth', 'validation', etc.)

**Examples:**

```bash
# Get last 50 requests
curl 'http://localhost:8000/metrics?limit=50'

# Get only errors
curl 'http://localhost:8000/metrics?status=error'

# Get authentication failures
curl 'http://localhost:8000/metrics?source=auth'

# Pagination
curl 'http://localhost:8000/metrics?limit=25&offset=50'
```

### Log Categories

Requests are categorized by source:

- **prediction**: Successful predictions
- **auth**: Authentication failures
- **validation**: Input validation errors
- **model**: Model loading/inference errors
- **system**: System-level errors

### MLflow Tracking

Monitor experiments and model performance:

1. Open MLflow UI: http://localhost:5002
2. Navigate to your experiment
3. View metrics, parameters, and artifacts
4. Compare different runs
5. Promote models to different stages (Staging, Production, Archived)

## 🔧 Troubleshooting

### MLflow Not Starting

**Issue:** MLflow container fails to start

**Solutions:**

1. Verify GitHub token is set: `echo $GHCR_TOKEN`
2. Check Docker is running: `docker ps`
3. Review logs: `docker logs mlflow`
4. Ensure port 5002 is not in use: `lsof -i :5002`

### AWS S3 Connection Issues

**Issue:** Cannot save/load artifacts from S3

**Solutions:**

1. Verify AWS credentials: `aws s3 ls`
2. Check `.aws/credentials` file exists
3. Verify S3 bucket permissions
4. Check environment variable: `echo $MLFLOW_S3_ENDPOINT_URL`

### Model Loading Failures

**Issue:** Server cannot load models

**Check:**

1. MLflow is running and accessible
2. Model is registered in MLflow registry
3. Local model exists in `models/` directory
4. Check server logs for detailed error messages

### Authentication Errors

**Issue:** Getting 403 Unauthorized

**Solutions:**

1. Verify API key in `.env` file
2. Include correct header: `Authorization: Bearer <API_KEY>`
3. Check for trailing spaces in the token

### Database Connection Issues

**Issue:** Cannot connect to SQLite database

**Solutions:**

1. Check file permissions on database file
2. Ensure directory exists for database
3. Verify SQLite is installed
4. Check logs for specific error messages

### Port Conflicts

**Issue:** Port already in use

**Solutions:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in .env or docker-compose.yml
```

## 🧪 Testing

### Test Preprocessing

```bash
python -m pytest tests/test_preprocessing.py
```

### Test Model Training

```bash
python -m pytest tests/test_training.py
```

### Test API Endpoints

```bash
python -m pytest tests/test_api.py
```

### Load Testing

Use tools like Apache Bench or Locust:

```bash
# Apache Bench
ab -n 1000 -c 10 -H "Authorization: Bearer supersecret123" \
  -p data/sample_request.json -T application/json \
  http://localhost:8000/predict
```

## ❓ Frequently Asked Questions (FAQ)

### General Questions

**Q: Can I use this with my own dataset?**  
A: Absolutely! Replace the Iris dataset with your own data, update the preprocessing logic in `src/preprocess_data.py`, and modify the model training script to match your use case.

**Q: Does this work with other ML frameworks like TensorFlow or PyTorch?**  
A: Currently, it's built with scikit-learn, but MLflow supports TensorFlow, PyTorch, and many other frameworks. You'll need to update the model training code accordingly.

**Q: Can I deploy this to AWS/GCP/Azure?**  
A: Yes! The Docker containers can be deployed to any cloud platform. You may want to use managed services like AWS SageMaker, Azure ML, or Google AI Platform for production deployments.

**Q: Is this production-ready?**  
A: This project demonstrates production-grade practices, but you should add additional security, scaling, and monitoring before using it in a production environment with real users.

### Technical Questions

**Q: Why use both local and remote model storage?**  
A: It's a fallback strategy. If the remote model (from MLflow) can't be loaded, the system falls back to a locally saved model to ensure availability.

**Q: How do I change the model algorithm?**  
A: Edit `src/model_train.py` and replace the RandomForestClassifier with your preferred algorithm. Update the hyperparameters accordingly.

**Q: Can I use a different database instead of SQLite?**  
A: Yes, you can configure FastAPI to use PostgreSQL, MySQL, or any other database by updating the database connection in `src/app/database.py`.

**Q: How do I secure the API in production?**  
A: Use proper authentication (OAuth2, JWT), HTTPS, rate limiting, and store secrets in a secure vault (AWS Secrets Manager, HashiCorp Vault, etc.).

**Q: What's the cost of running this?**  
A: Locally, it's free. For cloud deployment, costs depend on compute resources, S3 storage, and data transfer. Start with free tiers and scale as needed.

### Troubleshooting

**Q: I'm getting "Permission denied" errors with Docker**  
A: Add your user to the docker group: `sudo usermod -aG docker $USER` and log out/in.

**Q: Models are taking too long to load**  
A: Large models from S3 can be slow. Consider using a CDN, or switch to `MODEL_SOURCE=LOCAL` for faster startup times.

**Q: Can I run this without AWS S3?**  
A: Yes! MLflow supports local artifact storage. Update the MLflow configuration to use a local directory instead of S3.

## 🔐 Security Considerations

1. **API Keys**: Change default API key in production
2. **Environment Variables**: Never commit `.env` to version control
3. **AWS Credentials**: Use IAM roles in production
4. **Network Security**: Use HTTPS in production
5. **Input Validation**: All inputs are validated using Pydantic

## 🚢 Production Deployment

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### Environment-Specific Configurations

- **Development**: Use Docker Compose
- **Staging**: Deploy to Kubernetes cluster with scaled resources
- **Production**: Use managed services (ECS, EKS, or similar)

## 📈 Performance Optimization

1. **Model Caching**: Models are loaded once at startup
2. **Database Indexing**: Logs table is indexed on timestamp
3. **Async Operations**: FastAPI handles requests asynchronously
4. **Connection Pooling**: Reuse database connections

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Vinod Louis**

- Software Architect | Fullstack Engineer | AI/ML Enthusiast
- GitHub: [@VinodLouis](https://github.com/VinodLouis)
- LinkedIn: [linkedin.com/in/vinodlouis](https://linkedin.com/in/vinodlouis)
- Website: [vinodlouis.com](https://vinodlouis.com)
- Medium: [@louis.vinod14](https://medium.com/@louis.vinod14)

## 📧 Support

For issues and questions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/VinodLouis/mlops-pipeline/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/VinodLouis/mlops-pipeline/discussions)
- **Documentation**: Check this README and code comments
- **External Resources**:
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
  - [FastAPI Documentation](https://fastapi.tiangolo.com/)
  - [Docker Documentation](https://docs.docker.com/)

---

## ⭐ Star the Project

If you find this MLOps pipeline helpful, please consider giving it a star on GitHub! It helps others discover the project and motivates further development.

[![GitHub stars](https://img.shields.io/github/stars/VinodLouis/mlops-pipeline?style=social)](https://github.com/VinodLouis/mlops-pipeline/stargazers)

---

**Note:** This README assumes you're working with the Iris classification example. Adapt the commands and configurations based on your specific ML project requirements.

FROM python:3.10-slim

# Force unbuffered output
ENV PYTHONUNBUFFERED=1

# Optional: Set UTF-8 encoding for consistent logs
ENV PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies (optional but useful for ML packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages directly
RUN pip install --no-cache-dir \
    fastapi==0.116.1 \
    uvicorn==0.35.0 \
    scikit-learn==1.6.1 \
    mlflow==2.9.2 \
    pydantic==2.11.7 \
    python-dotenv==1.1.1 \
    boto3==1.39.11

# Copy your app code
COPY . .

# Optional: copy .env if needed
COPY .env .

EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

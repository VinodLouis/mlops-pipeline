#!/bin/bash


  python src/model_train.py \
  --mlflow-uri http://localhost:5002 \
  --experiment-name Iris_Classification \
  --model-name iris_classifier \
  --data-dir data/processed \
  --output-dir artifacts \
  --scale \
  --stage Staging

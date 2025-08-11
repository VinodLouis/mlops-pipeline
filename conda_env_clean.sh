#!/bin/bash
echo "Exporting clean environment..."
conda env export --name iris-mlops-env | grep -v "^prefix:" > environment.yml
echo "Environment.yml updated."

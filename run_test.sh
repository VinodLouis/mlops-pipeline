#!/bin/bash

# Set project root as PYTHONPATH
export PYTHONPATH=$(pwd)

# Optional: fail fast on first error
FAIL_FAST="--exitfirst"

# Optional: enable coverage reporting
COVERAGE="--cov=app --cov-report=term-missing"

# Optional: treat warnings as errors
WARNINGS="--strict-markers --disable-warnings"

# Run pytest
echo "Running tests with PYTHONPATH=$PYTHONPATH"
pytest tests/ $FAIL_FAST $COVERAGE $WARNINGS

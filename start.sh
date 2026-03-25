#!/bin/bash

set -e  # stop on error

echo "PORT is: $PORT"
echo "Starting app..."

python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000} --log-level debug

#!/bin/bash

IMAGE_NAME="modeltrack-tests"

echo "Building Docker image..."
if ! docker build -t $IMAGE_NAME .; then
  echo "Docker build failed!"
  exit 1
fi

echo "Running tests with coverage inside Docker container..."
if ! docker run --rm -e PYTHONPATH=/app/src $IMAGE_NAME; then
  echo "Tests failed or coverage less than 80%."
  exit 1
fi

echo "All tests passed with coverage >= 80%."
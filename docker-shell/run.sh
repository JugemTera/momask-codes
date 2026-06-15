#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="momask"
CONTAINER_NAME="momask"

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}"
fi

echo "Starting container: ${CONTAINER_NAME}"
docker run -dit \
    --gpus all \
    --name "${CONTAINER_NAME}" \
    --shm-size=8g \
    -v "${PROJECT_DIR}:/workspace" \
    "${IMAGE_NAME}" \
    /bin/bash

echo "Container started. Use exec.sh to enter."

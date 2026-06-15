#!/bin/bash
set -e

CONTAINER_NAME="momask"

echo "Entering container: ${CONTAINER_NAME}"
docker exec -it "${CONTAINER_NAME}" /bin/bash

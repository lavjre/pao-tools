#!/bin/bash

# Build script for Pao Tools Docker image

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Pao Tools Docker image...${NC}"

# Build the Docker image
docker build \
    --platform linux/amd64 \
    -t pao-tools:latest \
    -t pao-tools:$(git rev-parse --short HEAD 2>/dev/null || echo "dev") \
    .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo -e "${YELLOW}To run the container:${NC}"
    echo "  docker run -p 127.0.0.1:5500:5500 --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined pao-tools:latest"
    echo -e "${YELLOW}Or use docker-compose:${NC}"
    echo "  docker-compose up -d"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi 

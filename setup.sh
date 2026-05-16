#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] Installing host Python test dependencies ..."
pip3 install httpx

echo "[2/4] Building Docker image alo1:1.0 ..."
docker build -f "$ROOT/docker/Dockerfile" -t alo1:1.0 "$ROOT"

echo "[3/4] Installing Node dependencies into docker/node_modules ..."
docker run --rm \
    -v "$ROOT/docker:/work" \
    -w /work \
    node:20 npm ci --no-audit --no-fund

echo "[4/4] Done. Run the pipeline with:"
echo "  docker run --rm --network host --ipc=host \\"
echo "    -v \"\$(pwd)/pipeline_service:/workspace\" \\"
echo "    -v \"\$(pwd)/docker/node_modules:/workspace/node_modules\" \\"
echo "    -v \"\$(pwd)/configuration.yaml:/workspace/configuration.yaml\" \\"
echo "    alo1:1.0"

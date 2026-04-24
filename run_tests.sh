#!/bin/bash
set -euo pipefail

echo "Building .ea kernels..."
./build_kernels.sh

echo "Running integration tests..."
python3 -m pytest tests/ -v

#!/usr/bin/env bash

set -euo pipefail

LOG_FILE="kmeans_sweep.log"

for k in {2..8}; do
    echo "========================================"
    echo "K-Means experiment: k=$k"
    echo "========================================"

    python AT_models/at_kmeans_base.py "$k"

    echo
done 2>&1 | tee "$LOG_FILE"


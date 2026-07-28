#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${SCRIPT_DIR}/output/formal_all_wells"
LOG_PATH="${OUTPUT_ROOT}/gr_resistivity_build.log"

python3 "${SCRIPT_DIR}/build_gr_resistivity_samples.py" \
  --output-root "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_PATH}"

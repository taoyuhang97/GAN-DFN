#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_mine_base_density_v1"
CONFIG="${SCRIPT_DIR}/configs/${VERSION}.json"
OUTPUT_DIR="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1/step6_base_density"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[mine-base-density] started_at=$(date --iso-8601=seconds)"
echo "[mine-base-density] version=${VERSION}"
echo "[mine-base-density] log=${LOG_FILE}"

if [ -e "${OUTPUT_DIR}/predicted_fracture_density.sgy" ] || [ -e "${OUTPUT_DIR}/prediction_summary.json" ]; then
  echo "[mine-base-density] output already exists; refusing to overwrite: ${OUTPUT_DIR}"
  exit 0
fi

if [ -e "${OUTPUT_DIR}/predicted_fracture_density.sgy.partial" ]; then
  echo "[mine-base-density] partial output exists (previous run interrupted): ${OUTPUT_DIR}/predicted_fracture_density.sgy.partial"
  echo "[mine-base-density] confirm the old process is stopped, then remove the .partial file and rerun."
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/predict_step6_two_stage_density_volume.py" \
  --config "${CONFIG}"

echo "[mine-base-density] completed_at=$(date --iso-8601=seconds)"

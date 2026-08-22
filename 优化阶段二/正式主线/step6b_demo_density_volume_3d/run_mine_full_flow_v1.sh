#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_mine_full_flow_v1"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_${RUN_STAMP}.log"

HORIZON_CONFIG="${FORMAL_ROOT}/common/horizon_trace_table/configs/formal_horizon_trace_table_v2_mine.json"
HORIZON_OUTPUT="${FORMAL_ROOT}/common/horizon_trace_table/output/formal_horizon_trace_table_v2_mine"
MASTER_CONFIG="${SCRIPT_DIR}/configs/formal_mine_multiscale_flow_v1.json"
OUTPUT_ROOT="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1"
BASE_DENSITY_DIR="${OUTPUT_ROOT}/step6_base_density"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[mine-full-flow] started_at=$(date --iso-8601=seconds)"
echo "[mine-full-flow] log=${LOG_FILE}"

if [ ! -f "${HORIZON_OUTPUT}/horizon_trace_table.npy" ] || [ ! -f "${HORIZON_OUTPUT}/horizon_windows_2ms.npz" ]; then
  echo "[mine-full-flow] building mine horizon contract v2_mine"
  "${PYTHON_BIN}" "${FORMAL_ROOT}/common/horizon_trace_table/build_horizon_contract_v2.py" --config "${HORIZON_CONFIG}"
else
  echo "[mine-full-flow] horizon contract already present; skipping"
fi

if [ -e "${BASE_DENSITY_DIR}/predicted_fracture_density.sgy.partial" ]; then
  echo "[mine-full-flow] base density prediction is incomplete (.partial exists); refusing to relaunch"
  echo "[mine-full-flow] confirm the old process is stopped, remove the .partial file, and rerun."
  exit 1
fi

if [ ! -f "${BASE_DENSITY_DIR}/prediction_summary.json" ]; then
  echo "[mine-full-flow] building mine base density volume"
  bash "${SCRIPT_DIR}/run_mine_step6_base_density_v1.sh"
else
  echo "[mine-full-flow] base density already present; skipping"
fi

echo "[mine-full-flow] running Step6A-D -> Step9 mine flow"
bash "${SCRIPT_DIR}/run_mine_step6_to_step9_flow_v1.sh"

echo "[mine-full-flow] completed_at=$(date --iso-8601=seconds)"

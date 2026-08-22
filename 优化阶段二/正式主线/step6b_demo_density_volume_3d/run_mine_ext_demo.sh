#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_mine_ext_demo_v1"
MASTER_CONFIG="${SCRIPT_DIR}/configs/${VERSION}.json"
OUTPUT_ROOT="$(python3 -c "import json; print(json.load(open('${MASTER_CONFIG}'))['output_root'])")"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_${RUN_STAMP}.log"
RUN_STATE="${OUTPUT_ROOT}/demo_run_state.json"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="preflight"
write_state() {
  local status="$1"
  "${PYTHON_BIN}" - "${RUN_STATE}" "${status}" "${CURRENT_STAGE}" "${LOG_FILE}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "log_file": sys.argv[4],
    "updated_at": datetime.now().astimezone().isoformat(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}
trap 'code=$?; if [ "${CURRENT_STAGE:-}" != "complete" ]; then write_state "failed"; echo "[mine-ext-demo] failed stage=${CURRENT_STAGE} exit=${code}"; fi; exit ${code}' EXIT

echo "[mine-ext-demo] started_at=$(date --iso-8601=seconds)"
echo "[mine-ext-demo] version=${VERSION}"
echo "[mine-ext-demo] log=${LOG_FILE}"
write_state "running"

CURRENT_STAGE="scan_attribute_bottom"
if [ -f "${OUTPUT_ROOT}/attribute_bottom/attribute_bottom_stats.json" ]; then
  echo "[mine-ext-demo] skip: attribute_bottom already exists"
else
  write_state "running"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scan_attribute_bottom.py" --config "${MASTER_CONFIG}"
fi

CURRENT_STAGE="build_extension_contract"
if [ -f "${OUTPUT_ROOT}/horizon_contract_ext/horizon_contract_ext_metadata.json" ]; then
  echo "[mine-ext-demo] skip: extension contract already exists"
else
  write_state "running"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_extension_horizon_contract.py" --config "${MASTER_CONFIG}"
fi

CURRENT_STAGE="extend_base_density"
if [ -f "${OUTPUT_ROOT}/base_density_ext/density_extension_qc.json" ]; then
  echo "[mine-ext-demo] skip: extended base density already exists"
else
  write_state "running"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/extend_base_density.py" --config "${MASTER_CONFIG}"
fi

CURRENT_STAGE="step6_to_step9_flow"
write_state "running"
bash "${SCRIPT_DIR}/run_mine_step6_to_step9_flow_v1.sh" \
  "${VERSION}" "${MASTER_CONFIG}" "${MASTER_CONFIG}" --resume

CURRENT_STAGE="complete"
write_state "pass"
trap - EXIT
echo "[mine-ext-demo] completed_at=$(date --iso-8601=seconds)"
echo "[mine-ext-demo] status=pass"

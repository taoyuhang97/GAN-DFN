#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_mine_ext_demo_v1"
MASTER_CONFIG="${SCRIPT_DIR}/configs/${VERSION}.json"
STEP9_CONFIG="${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"
OUTPUT_ROOT="$(python3 -c "import json; print(json.load(open('${MASTER_CONFIG}'))['output_root'])")"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_step9_rerun_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[step9-rerun] started_at=$(date --iso-8601=seconds)"
echo "[step9-rerun] log=${LOG_FILE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" --master-config "${MASTER_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" --config "${STEP9_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_single_scale_diagnostic_sections.py" --config "${STEP9_CONFIG}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_mine_flow_v1.py" \
  --formal-root "${FORMAL_ROOT}" --step6-root "${OUTPUT_ROOT}" --version "${VERSION}"

echo "[step9-rerun] completed_at=$(date --iso-8601=seconds)"

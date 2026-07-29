#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="${VERSION:-formal_demo_10km_multiscale_flow_v2}"
STEP6_ROOT="${SCRIPT_DIR}/output/${VERSION}"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_step8_to_step9_geometry_fix_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[resume] started_at=$(date --iso-8601=seconds)"
echo "[resume] version=${VERSION}"
echo "[resume] log=${LOG_FILE}"

"${PYTHON_BIN}" -m py_compile \
  "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" \
  "${SCRIPT_DIR}/validate_demo_10km_flow_v2.py"

echo "[resume] stage=step8"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  --config "${FORMAL_ROOT}/step8_dfn_well_correction/configs/${VERSION}.json"

echo "[resume] stage=step9"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" \
  --config "${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"

echo "[resume] stage=acceptance"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_demo_10km_flow_v2.py" \
  --formal-root "${FORMAL_ROOT}" \
  --step6-root "${STEP6_ROOT}" \
  --version "${VERSION}"

echo "[resume] completed_at=$(date --iso-8601=seconds)"
echo "[resume] status=pass"

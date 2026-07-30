#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_demo_10km_multiscale_flow_v2"
RESUME_STEP6="${RESUME_STEP6:-0}"
STEP6_CONFIG="${SCRIPT_DIR}/configs/${VERSION}_expanded.json"
STEP6_ROOT="${SCRIPT_DIR}/output/${VERSION}"
STEP7B_CONFIG="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/configs/${VERSION}.json"
STEP7C_CONFIG="${FORMAL_ROOT}/step7c_large_fault_dfn/configs/${VERSION}.json"
STEP7D_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/${VERSION}.json"
STEP8_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/${VERSION}.json"
STEP9_CONFIG="${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
LOG_FILE="${LOG_DIR}/lowcoh_blank_recovery_${RUN_STAMP}.log"
STATE_FILE="${STEP6_ROOT}/lowcoh_blank_recovery_run_state.json"
AUDIT_DIR="${STEP6_ROOT}/lowcoh_blank_recovery_audit"
BASELINE_DIR="${AUDIT_DIR}/baseline_${RUN_STAMP}"
FINAL_AUDIT="${AUDIT_DIR}/lowcoh_blank_recovery_${RUN_STAMP}.json"

mkdir -p "${LOG_DIR}" "${STEP6_ROOT}" "${BASELINE_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE="preflight"
write_state() {
  local status="$1"
  "${PYTHON_BIN}" - "${STATE_FILE}" "${status}" "${CURRENT_STAGE}" "${LOG_FILE}" "${FINAL_AUDIT}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2], "stage": sys.argv[3], "log_file": sys.argv[4],
    "audit_file": sys.argv[5], "updated_at": datetime.now().astimezone().isoformat(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}
trap 'code=$?; write_state "failed"; echo "[lowcoh-recovery] failed stage=${CURRENT_STAGE} exit=${code}"; exit ${code}' ERR

echo "[lowcoh-recovery] started_at=$(date --iso-8601=seconds)"
echo "[lowcoh-recovery] log=${LOG_FILE}"
write_state "running"

CURRENT_STAGE="syntax_and_config"
"${PYTHON_BIN}" -m py_compile \
  "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
  "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
  "${SCRIPT_DIR}/audit_lowcoh_blank_recovery_v2.py" \
  "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py"
"${PYTHON_BIN}" - "${STEP6_CONFIG}" "${STEP7B_CONFIG}" "${STEP7C_CONFIG}" "${STEP7D_CONFIG}" "${STEP8_CONFIG}" "${STEP9_CONFIG}" <<'PY'
import json, sys
from pathlib import Path
for text in sys.argv[1:]:
    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(path)
    json.loads(path.read_text(encoding="utf-8"))
print("[lowcoh-recovery] config_parse=pass")
PY

CURRENT_STAGE="baseline_and_cleanup"
for source in \
  "${STEP6_ROOT}/step6b_medium/medium_corridor_qc.json" \
  "${STEP6_ROOT}/step6c_large/large_fault_qc.json" \
  "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/output/${VERSION}/medium_dfn_summary.json" \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/output/${VERSION}/large_fault_dfn_summary.json" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/output/${VERSION}/fused_multiscale_summary.json" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/output/${VERSION}/well_corrected_dfn_summary.json"; do
  if [[ -f "${source}" ]]; then cp "${source}" "${BASELINE_DIR}/$(basename "${source}")"; fi
done
if [[ "${RESUME_STEP6}" != "1" ]]; then
  rm -rf "${STEP6_ROOT}/step6b_medium" "${STEP6_ROOT}/step6c_large"
else
  test -f "${STEP6_ROOT}/step6b_medium/medium_corridor_qc.json"
  test -f "${STEP6_ROOT}/step6c_large/large_fault_qc.json"
  echo "[lowcoh-recovery] reusing completed Step6B/Step6C outputs"
fi
rm -rf \
  "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/output/${VERSION}" \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/output/${VERSION}" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/output/${VERSION}" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/output/${VERSION}" \
  "${FORMAL_ROOT}/step9_section_visualize/output/${VERSION}"

if [[ "${RESUME_STEP6}" != "1" ]]; then
  CURRENT_STAGE="step6b"
  write_state "running"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
    --config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/step6b_medium" \
    --anttrack-weight 0.70 --lowcoh-weight 0.20 --curvature-weight 0.10 \
    --seed-medium-score-threshold 0.62 --growth-anttrack-floor 0.28 \
    --growth-medium-score-threshold 0.52 \
    --min-component-voxels 120 --max-component-voxels-before-split 12000 --split-time-samples 8

  CURRENT_STAGE="step6c"
  write_state "running"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
    --config "${STEP6_CONFIG}" --input-qc-dir "${STEP6_ROOT}/input_qc" \
    --output-dir "${STEP6_ROOT}/step6c_large" --inferred-extraction-mode surface_ransac \
    --lowcoh-weight 0.75 --anttrack-weight 0.15 --curvature-weight 0.10 \
    --lowcoh-candidate-floor 0.55 --large-score-threshold 0.58 \
    --surface-support-score-threshold 0.35 --surface-min-support-fraction 0.18 \
    --faultlike-min-vertical-extent-ms 80 \
    --surface-ransac-min-inlier-voxels 300 --surface-ransac-min-inlier-fraction 0.025 \
    --surface-ransac-max-raw-components 128 \
    --surface-ransac-iterations 120 --surface-ransac-max-points 15000 \
    --surface-ransac-max-surfaces-per-component 4 --surface-panel-target-length-m 350 \
    --surface-min-panel-length-m 150 --surface-max-panel-length-m 500
fi

CURRENT_STAGE="step7b"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" --config "${STEP7B_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/export_existing_dfn_vtk.py" --config "${STEP7B_CONFIG}"

CURRENT_STAGE="step7c"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" --config "${STEP7C_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/export_existing_dfn_vtk.py" --config "${STEP7C_CONFIG}"

CURRENT_STAGE="step7d"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" --config "${STEP7D_CONFIG}"

CURRENT_STAGE="step8"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" --config "${STEP8_CONFIG}"

CURRENT_STAGE="step9"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" --config "${STEP9_CONFIG}"

CURRENT_STAGE="target_window_audit"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_lowcoh_blank_recovery_v2.py" \
  --formal-root "${FORMAL_ROOT}" --version "${VERSION}" --output "${FINAL_AUDIT}"

CURRENT_STAGE="complete"
write_state "pass"
trap - ERR
echo "[lowcoh-recovery] completed_at=$(date --iso-8601=seconds)"
echo "[lowcoh-recovery] status=pass audit=${FINAL_AUDIT}"

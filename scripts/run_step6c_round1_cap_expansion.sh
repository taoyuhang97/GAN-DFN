#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
FORMAL_ROOT="${REPO_ROOT}/优化阶段二/正式主线"
TAIGU_ROOT="${REPO_ROOT}/太古界"
LOG_ROOT="${FORMAL_ROOT}/logs/step6c_round1_cap_expansion"
STATE_FILE="${LOG_ROOT}/round1_state.json"
LOG_FILE="${LOG_ROOT}/round1.log"

SAND_INPUT_QC="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1/input_qc"
SAND_OUTPUT="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1_step6c_round1_cap512"
TAIGU_CONFIG="${TAIGU_ROOT}/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_quanqu1_round1_cap384.json"
TAIGU_OUTPUT="${TAIGU_ROOT}/step6c_large_fault/output/taigu_step6c_large_v3_attribute_v3_quanqu1_round1_cap384"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

write_state() {
  python3 - "$STATE_FILE" "$1" "$2" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "status": sys.argv[2],
            "stage": sys.argv[3],
            "updated_at": datetime.now().astimezone().isoformat(),
            "log_file": str(path.with_name("round1.log")),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

on_exit() {
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    write_state "failed" "${CURRENT_STAGE:-unknown}"
    echo "[step6c-round1] failed stage=${CURRENT_STAGE:-unknown} exit=${exit_code}"
  fi
  exit ${exit_code}
}
trap on_exit EXIT

CURRENT_STAGE="preflight"
write_state "running" "${CURRENT_STAGE}"
echo "[step6c-round1] started_at=$(date --iso-8601=seconds)"
echo "[step6c-round1] mode=component_count_calibration"
echo "[step6c-round1] sandstone_cap=512 taigu_cap=384"
echo "[step6c-round1] large_fault_prior_sgy=skipped"

for required in \
  "${SAND_INPUT_QC}/input_qc_summary.json" \
  "${FORMAL_ROOT}/step6b_demo_density_volume_3d/configs/formal_mine_multiscale_flow_v1.json" \
  "${TAIGU_CONFIG}"; do
  [[ -e "${required}" ]] || {
    echo "[step6c-round1] missing required input: ${required}" >&2
    exit 1
  }
done

if [[ -e "${SAND_OUTPUT}/large_fault_qc.json" || -e "${TAIGU_OUTPUT}/large_fault_qc.json" ]]; then
  echo "[step6c-round1] refusing to overwrite existing round1 output" >&2
  exit 1
fi

CURRENT_STAGE="sandstone_step6c_cap512"
write_state "running" "${CURRENT_STAGE}"
echo "[step6c-round1] sandstone start"
python3 "${FORMAL_ROOT}/step6b_demo_density_volume_3d/build_step6c_large_fault_prior.py" \
  --config "${FORMAL_ROOT}/step6b_demo_density_volume_3d/configs/formal_mine_multiscale_flow_v1.json" \
  --input-qc-dir "${SAND_INPUT_QC}" \
  --output-dir "${SAND_OUTPUT}" \
  --inferred-extraction-mode surface_ransac \
  --lowcoh-weight 0.75 \
  --anttrack-weight 0.15 \
  --curvature-weight 0.10 \
  --lowcoh-candidate-floor 0.55 \
  --large-score-threshold 0.58 \
  --surface-support-score-threshold 0.35 \
  --surface-min-support-fraction 0.18 \
  --faultlike-min-vertical-extent-ms 80 \
  --surface-ransac-min-inlier-voxels 300 \
  --surface-ransac-min-inlier-fraction 0.025 \
  --surface-ransac-max-raw-components 512 \
  --surface-ransac-iterations 120 \
  --surface-ransac-max-points 15000 \
  --surface-ransac-max-surfaces-per-component 4 \
  --surface-panel-target-length-m 350 \
  --surface-min-panel-length-m 150 \
  --surface-max-panel-length-m 500 \
  --skip-large-prior-sgy
echo "[step6c-round1] sandstone complete"

CURRENT_STAGE="taigu_step6c_cap384"
write_state "running" "${CURRENT_STAGE}"
echo "[step6c-round1] taigu start"
python3 "${TAIGU_ROOT}/step6c_large_fault/code/build_step6c_large_fault_prior.py" \
  --config "${TAIGU_CONFIG}" \
  --surface-ransac-max-raw-components 384 \
  --skip-large-prior-sgy
echo "[step6c-round1] taigu complete"

CURRENT_STAGE="complete"
write_state "pass" "${CURRENT_STAGE}"
echo "[step6c-round1] completed_at=$(date --iso-8601=seconds)"

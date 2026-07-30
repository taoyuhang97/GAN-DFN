#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="formal_demo_10km_multiscale_flow_v2"
MASTER_CONFIG="${SCRIPT_DIR}/configs/${VERSION}.json"
BASE_CONFIG="${SCRIPT_DIR}/configs/formal_demo_10km_curvature_led_2ms_v1.json"
STEP6_CONFIG="${SCRIPT_DIR}/configs/${VERSION}_expanded.json"
STEP6_ROOT="${SCRIPT_DIR}/output/${VERSION}"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_${RUN_STAMP}.log"
RUN_STATE="${STEP6_ROOT}/run_state.json"

STEP7A_CONFIG="${FORMAL_ROOT}/step7a_small_scale_dfn/configs/${VERSION}.json"
STEP7B_CONFIG="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/configs/${VERSION}.json"
STEP7C_CONFIG="${FORMAL_ROOT}/step7c_large_fault_dfn/configs/${VERSION}.json"
STEP7D_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/${VERSION}.json"
STEP8_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/${VERSION}.json"
STEP9_CONFIG="${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"

mkdir -p "${LOG_DIR}" "${STEP6_ROOT}"
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
trap 'code=$?; write_state "failed"; echo "[flow-v2] failed stage=${CURRENT_STAGE} exit=${code}"; exit ${code}' ERR

echo "[flow-v2] started_at=$(date --iso-8601=seconds)"
echo "[flow-v2] version=${VERSION}"
echo "[flow-v2] log=${LOG_FILE}"
write_state "running"

CURRENT_STAGE="syntax_and_config"
"${PYTHON_BIN}" -m py_compile \
  "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" \
  "${SCRIPT_DIR}/validate_demo_10km_flow_v2.py" \
  "${SCRIPT_DIR}/build_step6a_small_background.py" \
  "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
  "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
  "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
  "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" \
  "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_coherence_sections.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" --master-config "${MASTER_CONFIG}"

CURRENT_STAGE="preflight"
"${PYTHON_BIN}" - "${MASTER_CONFIG}" "${STEP6_ROOT}" "${FORMAL_ROOT}" "${VERSION}" <<'PY'
import json, shutil, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step6 = Path(sys.argv[2]); formal = Path(sys.argv[3]); version = sys.argv[4]
required = [Path(config["input_density_sgy"]), Path(config["trace_mapping_npz"]), *(Path(v) for v in config["volume_paths"].values())]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
markers = [
    step6 / "input_qc/input_qc_summary.json",
    step6 / "step6a_small/small_background_qc.json",
    step6 / "step6b_medium/medium_corridor_qc.json",
    step6 / "step6c_large/large_fault_qc.json",
    step6 / "step6d_bundle/multiscale_bundle_summary.json",
    formal / f"step7a_small_scale_dfn/output/{version}/small_dfn_summary.json",
    formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_summary.json",
]
existing = [str(path) for path in markers if path.exists()]
if existing:
    raise FileExistsError("v2 stage outputs already exist; refusing mixed run:\n" + "\n".join(existing))
free_gib = shutil.disk_usage(step6).free / 1024**3
if free_gib < 80.0:
    raise RuntimeError(f"insufficient free space: {free_gib:.1f} GiB")
small = config["small_evidence"]
if abs(float(small["density_weight"]) - 0.5) > 1e-12 or abs(float(small["curvature_weight"]) - 0.5) > 1e-12:
    raise ValueError(f"unexpected Step6A weights: {small}")
print(f"[flow-v2] preflight=pass free_space_gib={free_gib:.1f}")
PY

CURRENT_STAGE="input_qc"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/qc_multiscale_rebalance_inputs.py" \
  --base-config "${BASE_CONFIG}" --multiscale-config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/input_qc"

CURRENT_STAGE="step6a"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6a_small_background.py" \
  --config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/step6a_small" --candidate-q 0.88 --core-q 0.95

CURRENT_STAGE="step6b"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
  --config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/step6b_medium" \
  --anttrack-weight 0.70 --lowcoh-weight 0.20 --curvature-weight 0.10 \
  --seed-medium-score-threshold 0.62 --growth-anttrack-floor 0.28 \
  --growth-medium-score-threshold 0.52 --min-component-voxels 120 \
  --max-component-voxels-before-split 12000 --split-time-samples 8

CURRENT_STAGE="step6c"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
  --config "${STEP6_CONFIG}" --input-qc-dir "${STEP6_ROOT}/input_qc" \
  --output-dir "${STEP6_ROOT}/step6c_large" --inferred-extraction-mode surface_ransac \
  --lowcoh-weight 0.75 --anttrack-weight 0.15 --curvature-weight 0.10 \
  --lowcoh-candidate-floor 0.55 --large-score-threshold 0.58 \
  --surface-support-score-threshold 0.35 --surface-min-support-fraction 0.18 \
  --faultlike-min-vertical-extent-ms 80 --surface-ransac-min-inlier-voxels 300 \
  --surface-ransac-min-inlier-fraction 0.025 --surface-ransac-max-raw-components 128 \
  --surface-ransac-iterations 120 --surface-ransac-max-points 15000 \
  --surface-ransac-max-surfaces-per-component 4 --surface-panel-target-length-m 350 \
  --surface-min-panel-length-m 150 --surface-max-panel-length-m 500

CURRENT_STAGE="step6d"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
  --config "${STEP6_CONFIG}" --rebalance-root "${STEP6_ROOT}" --output-dir "${STEP6_ROOT}/step6d_bundle" \
  --medium-damage-outer-decay 0.45 --large-damage-outer-decay 0.35

CURRENT_STAGE="step7a"
write_state "running"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" --config "${STEP7A_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7a_small_scale_dfn/export_existing_dfn_vtk.py" --config "${STEP7A_CONFIG}"

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

CURRENT_STAGE="acceptance"
write_state "running"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_demo_10km_flow_v2.py" \
  --formal-root "${FORMAL_ROOT}" --step6-root "${STEP6_ROOT}" --version "${VERSION}"

CURRENT_STAGE="complete"
write_state "pass"
trap - ERR
echo "[flow-v2] completed_at=$(date --iso-8601=seconds)"
echo "[flow-v2] status=pass"

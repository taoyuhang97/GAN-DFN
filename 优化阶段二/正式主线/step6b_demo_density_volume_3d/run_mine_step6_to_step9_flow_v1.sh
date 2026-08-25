#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESUME=0
POS_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --resume|-r) RESUME=1 ;;
    *) POS_ARGS+=("${arg}") ;;
  esac
done
VERSION="${POS_ARGS[0]:-formal_mine_multiscale_flow_v1}"
MASTER_CONFIG="${POS_ARGS[1]:-${SCRIPT_DIR}/configs/${VERSION}.json}"
BASE_CONFIG="${POS_ARGS[2]:-${SCRIPT_DIR}/configs/formal_mine_base_density_v1.json}"
STEP6_CONFIG="${SCRIPT_DIR}/configs/${VERSION}_expanded.json"
STEP6_ROOT="$(python3 -c "import json; print(json.load(open('${MASTER_CONFIG}'))['output_dir'])")"
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
  "${PYTHON_BIN}" - "${RUN_STATE}" "${status}" "${CURRENT_STAGE}" "${LOG_FILE}" "${RESUME}" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "log_file": sys.argv[4],
    "resumed": sys.argv[5] == "1",
    "updated_at": datetime.now().astimezone().isoformat(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}
trap 'code=$?; if [ "${CURRENT_STAGE:-}" != "complete" ]; then write_state "failed"; echo "[flow-mine] failed stage=${CURRENT_STAGE} exit=${code}"; fi; exit ${code}' EXIT

echo "[flow-mine] started_at=$(date --iso-8601=seconds)"
echo "[flow-mine] version=${VERSION}"
echo "[flow-mine] resume=${RESUME}"
echo "[flow-mine] log=${LOG_FILE}"
write_state "running"

# resume_skip returns 0 only when RESUME=1 and every marker exists with status=pass.
resume_skip() {
  [ "${RESUME}" = "1" ] || return 1
  local p
  for p in "$@"; do
    [ -f "${p}" ] || return 1
  done
  "${PYTHON_BIN}" - "$@" <<'PY' || return 1
import json, sys
from pathlib import Path
for path in sys.argv[1:]:
    candidate = Path(path)
    if candidate.suffix.lower() == ".json":
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("status", "pass") != "pass":
            sys.exit(1)
sys.exit(0)
PY
}

stage_syntax_and_config() {
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" \
    "${SCRIPT_DIR}/validate_mine_flow_v1.py" \
    "${SCRIPT_DIR}/build_step6a_small_background.py" \
    "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
    "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
    "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
    "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" \
    "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" \
    "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" \
    "${FORMAL_ROOT}/step7c_large_fault_dfn/export_existing_dfn_vtk.py" \
    "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
    "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
    "${FORMAL_ROOT}/step8_dfn_well_correction/export_single_scale_dfn.py" \
    "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_coherence_sections.py" \
    "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" \
    "${FORMAL_ROOT}/step9_section_visualize/build_single_scale_diagnostic_sections.py"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" --master-config "${MASTER_CONFIG}"
}

stage_wp1_horizon_qc() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/common/horizon_trace_table/qc_wp1_horizon_integration.py" \
    --master-config "${MASTER_CONFIG}" --output "${STEP6_ROOT}/wp1_horizon_integration_qc.json"
}

stage_preflight() {
  "${PYTHON_BIN}" - "${MASTER_CONFIG}" "${STEP6_ROOT}" "${FORMAL_ROOT}" "${VERSION}" "${RESUME}" "${STEP6_ROOT}" <<'PY'
import json, shutil, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
step6 = Path(sys.argv[2]); formal = Path(sys.argv[3]); version = sys.argv[4]; resume = sys.argv[5] == "1"
output_root = Path(sys.argv[6])
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
    output_root / "step7a_small_scale/small_dfn_summary.json",
    output_root / "step8_well_correction/well_corrected_dfn_summary.json",
]
existing = [str(path) for path in markers if path.exists()]
if existing and not resume:
    raise FileExistsError("stage outputs already exist; refusing mixed run (use --resume to continue):\n" + "\n".join(existing))
free_gib = shutil.disk_usage(step6).free / 1024**3
if free_gib < 40.0:
    raise RuntimeError(f"insufficient free space: {free_gib:.1f} GiB")
small = config["small_evidence"]
if abs(float(small["density_weight"]) - 0.5) > 1e-12 or abs(float(small["curvature_weight"]) - 0.5) > 1e-12:
    raise ValueError(f"unexpected Step6A weights: {small}")
print(f"[flow-mine] preflight=pass free_space_gib={free_gib:.1f} resume={resume}")
PY
}

stage_input_qc() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/qc_multiscale_rebalance_inputs.py" \
    --base-config "${BASE_CONFIG}" --multiscale-config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/input_qc"
}

stage_step6a() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6a_small_background.py" \
    --config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/step6a_small" --candidate-q 0.88 --core-q 0.95
}

stage_step6b() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
    --config "${STEP6_CONFIG}" --output-dir "${STEP6_ROOT}/step6b_medium" \
    --anttrack-weight 0.70 --lowcoh-weight 0.20 --curvature-weight 0.10 \
    --seed-medium-score-threshold 0.62 --growth-anttrack-floor 0.28 \
    --growth-medium-score-threshold 0.52 --min-component-voxels 120 \
    --max-component-voxels-before-split 12000 --split-time-samples 8
}

stage_step6c() {
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
}

stage_step6d() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
    --config "${STEP6_CONFIG}" --rebalance-root "${STEP6_ROOT}" --output-dir "${STEP6_ROOT}/step6d_bundle" \
    --medium-damage-outer-decay 0.45 --large-damage-outer-decay 0.35
}

stage_step7a() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" --config "${STEP7A_CONFIG}"
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7a_small_scale_dfn/export_existing_dfn_vtk.py" --config "${STEP7A_CONFIG}"
}

stage_step7b() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" --config "${STEP7B_CONFIG}"
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/export_existing_dfn_vtk.py" --config "${STEP7B_CONFIG}"
}

stage_step7c() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" --config "${STEP7C_CONFIG}"
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/export_existing_dfn_vtk.py" --config "${STEP7C_CONFIG}"
}

stage_step7d() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" --config "${STEP7D_CONFIG}"
}

stage_step8() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" --config "${STEP8_CONFIG}"
  read XMIN XMAX YMIN YMAX <<<"$(python3 -c "import json; b=json.load(open('${MASTER_CONFIG}'))['target_block']; print(f'{b[\"x_min\"]} {b[\"x_max\"]} {b[\"y_min\"]} {b[\"y_max\"]}')")"
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/export_well_trajectories_vtk.py" \
    --wells-root "${FORMAL_ROOT}/step2_real_well_t4_t7_samples/output/formal_all_wells" \
    --output-dir "${STEP6_ROOT}/step8_well_correction" \
    --x-min "${XMIN}" --x-max "${XMAX}" --y-min "${YMIN}" --y-max "${YMAX}"
}

stage_single_scale_export() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/export_single_scale_dfn.py" --config "${STEP8_CONFIG}"
}

stage_step9() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" --config "${STEP9_CONFIG}"
}

stage_single_scale_diag() {
  "${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_single_scale_diagnostic_sections.py" --config "${STEP9_CONFIG}"
}

stage_acceptance() {
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_mine_flow_v1.py" \
    --formal-root "${FORMAL_ROOT}" --step6-root "${STEP6_ROOT}" --version "${VERSION}"
}

run_stage() {
  local name="$1"
  shift
  CURRENT_STAGE="${name}"
  if resume_skip "$@"; then
    echo "[flow-mine] resume: skipping completed stage=${name}"
    return 0
  fi
  write_state "running"
  "stage_${name}"
}

CURRENT_STAGE="syntax_and_config"
write_state "running"
stage_syntax_and_config

CURRENT_STAGE="wp1_horizon_qc"
write_state "running"
stage_wp1_horizon_qc

CURRENT_STAGE="preflight"
write_state "running"
stage_preflight
run_stage input_qc "${STEP6_ROOT}/input_qc/input_qc_summary.json"
run_stage step6a "${STEP6_ROOT}/step6a_small/small_background_qc.json"
run_stage step6b "${STEP6_ROOT}/step6b_medium/medium_corridor_qc.json"
run_stage step6c "${STEP6_ROOT}/step6c_large/large_fault_qc.json"
run_stage step6d "${STEP6_ROOT}/step6d_bundle/multiscale_bundle_summary.json"
run_stage step7a \
  "${STEP6_ROOT}/step7a_small_scale/small_dfn_summary.json" \
  "${STEP6_ROOT}/step7a_small_scale/small_dfn_raw_time.vtk"
run_stage step7b \
  "${STEP6_ROOT}/step7b_medium_scale/medium_dfn_summary.json" \
  "${STEP6_ROOT}/step7b_medium_scale/medium_dfn_raw_time.vtk"
run_stage step7c \
  "${STEP6_ROOT}/step7c_large_fault/large_fault_dfn_summary.json" \
  "${STEP6_ROOT}/step7c_large_fault/large_fault_dfn_raw_time.vtk"
run_stage step7d "${STEP6_ROOT}/step7d_fused/fused_multiscale_summary.json"
run_stage step8 "${STEP6_ROOT}/step8_well_correction/well_corrected_dfn_summary.json"
run_stage single_scale_export \
  "${STEP6_ROOT}/step8_well_correction/single_scale_dfn/single_scale_dfn_export_summary.json"
run_stage step9 \
  "${STEP6_ROOT}/step9_sections/cheye1_dfn_multibackground_sections/section_summary.json"
run_stage single_scale_diag \
  "${STEP6_ROOT}/step9_sections/single_scale_diagnostic_sections/single_scale_diagnostic_summary.json"
run_stage acceptance "${STEP6_ROOT}/flow_acceptance_mine_v1.json"

CURRENT_STAGE="complete"
write_state "pass"
trap - EXIT
echo "[flow-mine] completed_at=$(date --iso-8601=seconds)"
echo "[flow-mine] status=pass"

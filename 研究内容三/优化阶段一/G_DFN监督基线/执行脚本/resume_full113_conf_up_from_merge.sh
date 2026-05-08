#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-gan-dfn}"
RUN_NAME="${RUN_NAME:-full113_conf_up_traininfer_20260506_142956}"

PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-/home/tyh/data/project-oil/砂砾岩}"
RUN_ROOT="${PRIVATE_DATA_ROOT}/优化阶段一/研究内容三/G_DFN监督基线/后台执行/${RUN_NAME}"
LOG_ROOT="${RUN_ROOT}/logs"

DOCX_PATH="${DOCX_PATH:-${PRIVATE_DATA_ROOT}/优化阶段一/实验记录/实验记录20260328.docx}"
TRACE_HEADER_CSV="${TRACE_HEADER_CSV:-/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv}"
SGY_FILE="${SGY_FILE:-/data/shared/project-oil/wx数据/砂砾岩/psdm_final_time.sgy}"
FAULT_PATCHES_ROOT="${FAULT_PATCHES_ROOT:-/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out/patches}"

BLOCK_X_START="${BLOCK_X_START:-0}"
BLOCK_X_END="${BLOCK_X_END:-84}"
BLOCK_Y_START="${BLOCK_Y_START:-0}"
BLOCK_Y_END="${BLOCK_Y_END:-60}"

POSTPROCESS_PHASE="${POSTPROCESS_PHASE:-2}"
POSTPROCESS_BACKEND="${POSTPROCESS_BACKEND:-gpu}"
POSTPROCESS_GPU_TILE_POINTS="${POSTPROCESS_GPU_TILE_POINTS:-2048}"
POSTPROCESS_MAX_CPU_THREADS="${POSTPROCESS_MAX_CPU_THREADS:-24}"
POSTPROCESS_GMM_BIC_SAMPLE_CAP="${POSTPROCESS_GMM_BIC_SAMPLE_CAP:-250000}"
POSTPROCESS_GMM_FIT_SAMPLE_CAP="${POSTPROCESS_GMM_FIT_SAMPLE_CAP:-400000}"
POSTPROCESS_PHASE2_CHUNK_ROW_THRESHOLD="${POSTPROCESS_PHASE2_CHUNK_ROW_THRESHOLD:-250000}"
POSTPROCESS_PHASE2_CHUNK_UNIT_WIDTH="${POSTPROCESS_PHASE2_CHUNK_UNIT_WIDTH:-12}"
POSTPROCESS_PHASE2_CHUNK_UNIT_HEIGHT="${POSTPROCESS_PHASE2_CHUNK_UNIT_HEIGHT:-12}"
POSTPROCESS_PHASE2_CHUNK_OVERLAP_UNITS="${POSTPROCESS_PHASE2_CHUNK_OVERLAP_UNITS:-1}"

FAULT_HALF_BAND_MS="${FAULT_HALF_BAND_MS:-100.0}"
FAULT_REMOVE_MS="${FAULT_REMOVE_MS:-50.0}"
FAULT_TRANSITION_MS="${FAULT_TRANSITION_MS:-100.0}"
FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO="${FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO:-1500.0}"

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
IFS=$'\n\t'
FULL_SHARD_COUNT="${FULL_SHARD_COUNT:-${#GPU_IDS[@]}}"

PRODUCTION_ROOT="${RUN_ROOT}/production"
MERGE_ROOT="${RUN_ROOT}/merge"
POSTPROCESS_ROOT="${RUN_ROOT}/postprocess"
FAULT_ROOT="${RUN_ROOT}/fault_postfusion"

TRAIN_REGISTRY_PY="${RUN_ROOT}/train_layerwise/layer_model_registry.py"
FULL_INFER_RUN_DIR="${PRODUCTION_ROOT}/full_infer"
MERGE_RUN_DIR="${MERGE_ROOT}/merge_full"
MERGED_VTK="${MERGE_RUN_DIR}/merged_full_infer_predicted_patches_raw_time.vtk"
POSTPROCESS_RUN_DIR="${POSTPROCESS_ROOT}/postprocess_full"
POSTPROCESS_VTK="${POSTPROCESS_RUN_DIR}/regional_dfn_postprocessed.vtk"
FAULT_SUMMARY_JSON="${FAULT_ROOT}/fault_postfusion_full/regional_fault_postfusion_pipeline_summary.json"

VALIDATE_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/validate_full_infer_aggregated.py"
MERGE_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/merge_unit_dfn_vtks.py"
POSTPROCESS_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/单元DFN融合/run_regional_postprocess_multiscale.py"
FAULT_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/单元DFN融合/区域断层后融合/run_regional_fault_postfusion_pipeline_v2.py"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "Required directory not found: ${path}" >&2
    exit 1
  fi
}

run_logged() {
  local stage_name="$1"
  shift
  log "START ${stage_name}"
  "$@" 2>&1 | tee "${LOG_ROOT}/${stage_name}.log"
  log "END ${stage_name}"
}

require_dir "${RUN_ROOT}"
mkdir -p "${LOG_ROOT}"

require_dir "${FULL_INFER_RUN_DIR}"
require_dir "${FULL_INFER_RUN_DIR}/aggregated"
require_dir "${FULL_INFER_RUN_DIR}/units"
require_dir "${FAULT_PATCHES_ROOT}"
require_file "${TRACE_HEADER_CSV}"
require_file "${SGY_FILE}"
require_file "${TRAIN_REGISTRY_PY}"
require_file "${VALIDATE_SCRIPT}"
require_file "${MERGE_SCRIPT}"
require_file "${POSTPROCESS_SCRIPT}"
require_file "${FAULT_SCRIPT}"

log "Resume from merge for RUN_ROOT=${RUN_ROOT}"
log "FULL_INFER_RUN_DIR=${FULL_INFER_RUN_DIR}"
log "GPU_IDS=${GPU_IDS_CSV}"

VALIDATE_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${VALIDATE_SCRIPT}"
  --aggregated-dir "${FULL_INFER_RUN_DIR}/aggregated"
  --expected-shards "${FULL_SHARD_COUNT}"
)
run_logged validate_full_infer_resume "${VALIDATE_CMD[@]}"

MERGE_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${MERGE_SCRIPT}"
  --units-root "${FULL_INFER_RUN_DIR}/units"
  --block-x-start "${BLOCK_X_START}"
  --block-x-end "${BLOCK_X_END}"
  --block-y-start "${BLOCK_Y_START}"
  --block-y-end "${BLOCK_Y_END}"
  --vtk-name predicted_patches_raw_time.vtk
  --output-root "${MERGE_ROOT}"
  --run-name merge_full
  --output-vtk-name merged_full_infer_predicted_patches_raw_time.vtk
  --docx-path "${DOCX_PATH}"
)
run_logged merge_units "${MERGE_CMD[@]}"
require_file "${MERGED_VTK}"

POSTPROCESS_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${POSTPROCESS_SCRIPT}"
  --input-vtk "${MERGED_VTK}"
  --output-root "${POSTPROCESS_ROOT}"
  --run-name postprocess_full
  --output-vtk-name regional_dfn_postprocessed.vtk
  --phase "${POSTPROCESS_PHASE}"
  --compute-backend "${POSTPROCESS_BACKEND}"
  --max-cpu-threads "${POSTPROCESS_MAX_CPU_THREADS}"
  --gpu-tile-points "${POSTPROCESS_GPU_TILE_POINTS}"
  --gmm-bic-sample-cap "${POSTPROCESS_GMM_BIC_SAMPLE_CAP}"
  --gmm-fit-sample-cap "${POSTPROCESS_GMM_FIT_SAMPLE_CAP}"
  --phase2-chunk-row-threshold "${POSTPROCESS_PHASE2_CHUNK_ROW_THRESHOLD}"
  --phase2-chunk-unit-width "${POSTPROCESS_PHASE2_CHUNK_UNIT_WIDTH}"
  --phase2-chunk-unit-height "${POSTPROCESS_PHASE2_CHUNK_UNIT_HEIGHT}"
  --phase2-chunk-overlap-units "${POSTPROCESS_PHASE2_CHUNK_OVERLAP_UNITS}"
  --docx-path "${DOCX_PATH}"
  --overwrite
)
run_logged postprocess "${POSTPROCESS_CMD[@]}"
require_file "${POSTPROCESS_VTK}"

FAULT_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${FAULT_SCRIPT}"
  --input-vtk "${POSTPROCESS_VTK}"
  --fault-patches-root "${FAULT_PATCHES_ROOT}"
  --block-x-start "${BLOCK_X_START}"
  --block-x-end "${BLOCK_X_END}"
  --block-y-start "${BLOCK_Y_START}"
  --block-y-end "${BLOCK_Y_END}"
  --output-root "${FAULT_ROOT}"
  --run-name fault_postfusion_full
  --fault-half-band-ms "${FAULT_HALF_BAND_MS}"
  --fault-remove-ms "${FAULT_REMOVE_MS}"
  --fault-transition-ms "${FAULT_TRANSITION_MS}"
  --surface-max-fragment-area-ratio "${FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO}"
  --docx-path "${DOCX_PATH}"
  --overwrite
)
run_logged fault_postfusion "${FAULT_CMD[@]}"
require_file "${FAULT_SUMMARY_JSON}"

log "Resume pipeline finished."
log "Merged VTK: ${MERGED_VTK}"
log "Postprocess VTK: ${POSTPROCESS_VTK}"
log "Fault summary: ${FAULT_SUMMARY_JSON}"

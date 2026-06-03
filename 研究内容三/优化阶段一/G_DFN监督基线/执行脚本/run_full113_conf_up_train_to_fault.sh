#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-gan-dfn}"
RUN_NAME="${RUN_NAME:-full113_conf_up_traininfer_$(date +%Y%m%d_%H%M%S)}"

PRIVATE_DATA_ROOT="${PRIVATE_DATA_ROOT:-/home/tyh/data/project-oil/砂砾岩}"
RUN_ROOT="${PRIVATE_DATA_ROOT}/优化阶段一/研究内容三/G_DFN监督基线/后台执行/${RUN_NAME}"
LOG_ROOT="${RUN_ROOT}/logs"

DOCX_PATH="${DOCX_PATH:-${PRIVATE_DATA_ROOT}/优化阶段一/实验记录/实验记录20260328.docx}"
SPLIT_RUN_DIR="${SPLIT_RUN_DIR:-${PRIVATE_DATA_ROOT}/优化阶段一/研究内容三/G_DFN监督基线/验证_新划分/full113_conf_up_fixed_split_79_17_17_20260506}"
DATASET_RUN_DIR="${DATASET_RUN_DIR:-${PRIVATE_DATA_ROOT}/优化阶段一/研究内容三/GAN训练准备/训练样本打包/dataset_113units_conf_up_20260506}"

UNIT_DFN_ROOT="${UNIT_DFN_ROOT:-/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分}"
TRACE_HEADER_CSV="${TRACE_HEADER_CSV:-/data/shared/project-oil/wx数据/砂砾岩/研究内容一/trace_header_xy.csv}"
SGY_FILE="${SGY_FILE:-/data/shared/project-oil/wx数据/砂砾岩/psdm_final_time.sgy}"
SURFACE_DIR="${SURFACE_DIR:-/data/shared/project-oil/wx数据/砂砾岩/层位}"
FAULT_PATCHES_ROOT="${FAULT_PATCHES_ROOT:-/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验/fault_patches_out/patches}"

BLOCK_X_START="${BLOCK_X_START:-0}"
BLOCK_X_END="${BLOCK_X_END:-84}"
BLOCK_Y_START="${BLOCK_Y_START:-0}"
BLOCK_Y_END="${BLOCK_Y_END:-60}"
BLOCK_X_MIN=$(( BLOCK_X_START < BLOCK_X_END ? BLOCK_X_START : BLOCK_X_END ))
BLOCK_X_MAX=$(( BLOCK_X_START > BLOCK_X_END ? BLOCK_X_START : BLOCK_X_END ))
BLOCK_Y_MIN=$(( BLOCK_Y_START < BLOCK_Y_END ? BLOCK_Y_START : BLOCK_Y_END ))
BLOCK_Y_MAX=$(( BLOCK_Y_START > BLOCK_Y_END ? BLOCK_Y_START : BLOCK_Y_END ))
FULL_INFER_TOTAL_UNITS=$(( (BLOCK_X_MAX - BLOCK_X_MIN + 1) * (BLOCK_Y_MAX - BLOCK_Y_MIN + 1) ))

LAYER_TOP_BOUNDARY_MS="${LAYER_TOP_BOUNDARY_MS:-1100.0}"
LAYER_BOTTOM_BOUNDARY_MS="${LAYER_BOTTOM_BOUNDARY_MS:-3800.0}"

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
FAULT_INDUCED_COUNT_SCALE="${FAULT_INDUCED_COUNT_SCALE:-5.5}"
FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO="${FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO:-1500.0}"

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
IFS=$'\n\t'
FULL_SHARD_COUNT="${FULL_SHARD_COUNT:-${#GPU_IDS[@]}}"
if [[ "${FULL_SHARD_COUNT}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "FULL_SHARD_COUNT (${FULL_SHARD_COUNT}) must equal GPU count (${#GPU_IDS[@]})." >&2
  exit 1
fi

TRAIN_ROOT="${RUN_ROOT}/train_layerwise"
EVAL_ROOT="${RUN_ROOT}/eval"
PRODUCTION_ROOT="${RUN_ROOT}/production"
MERGE_ROOT="${RUN_ROOT}/merge"
POSTPROCESS_ROOT="${RUN_ROOT}/postprocess"
FAULT_ROOT="${RUN_ROOT}/fault_postfusion"

TRAIN_REGISTRY_PY="${TRAIN_ROOT}/layer_model_registry.py"
EVAL_RUN_DIR="${EVAL_ROOT}/eval_test"
FULL_INFER_RUN_DIR="${PRODUCTION_ROOT}/full_infer"
MERGE_RUN_DIR="${MERGE_ROOT}/merge_full"
MERGED_VTK="${MERGE_RUN_DIR}/merged_full_infer_predicted_patches_raw_time.vtk"
POSTPROCESS_RUN_DIR="${POSTPROCESS_ROOT}/postprocess_full"
POSTPROCESS_VTK="${POSTPROCESS_RUN_DIR}/regional_dfn_postprocessed.vtk"
FAULT_RUN_DIR="${FAULT_ROOT}/fault_postfusion_full"
FAULT_SUMMARY_JSON="${FAULT_ROOT}/fault_postfusion_full/regional_fault_postfusion_pipeline_summary.json"

TRAIN_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/train_layerwise_models.py"
EVAL_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/infer_and_evaluate_baseline.py"
PRODUCTION_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/infer_production_units_baseline.py"
MERGE_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/merge_unit_dfn_vtks.py"
POSTPROCESS_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/单元DFN融合/run_regional_postprocess_multiscale.py"
FAULT_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/单元DFN融合/区域断层后融合/run_regional_fault_postfusion_pipeline_v2.py"
LAYER_TRAINING_CONFIG_PY="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/layerwise_training_plan_demo_v1.py"
VALIDATE_SCRIPT="${REPO_ROOT}/研究内容三/优化阶段一/G_DFN监督基线/validate_full_infer_aggregated.py"

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

csv_data_row_count() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  local line_count
  line_count=$(wc -l < "${path}")
  if [[ "${line_count}" -le 0 ]]; then
    echo 0
  else
    echo $((line_count - 1))
  fi
}

if [[ -e "${RUN_ROOT}" ]]; then
  echo "RUN_ROOT already exists: ${RUN_ROOT}" >&2
  echo "Please set a new RUN_NAME or remove the old run dir first." >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"

require_dir "${SPLIT_RUN_DIR}"
require_dir "${DATASET_RUN_DIR}"
require_dir "${UNIT_DFN_ROOT}"
require_dir "${SURFACE_DIR}"
require_dir "${FAULT_PATCHES_ROOT}"
require_file "${TRACE_HEADER_CSV}"
require_file "${SGY_FILE}"
require_file "${TRAIN_SCRIPT}"
require_file "${EVAL_SCRIPT}"
require_file "${PRODUCTION_SCRIPT}"
require_file "${MERGE_SCRIPT}"
require_file "${POSTPROCESS_SCRIPT}"
require_file "${FAULT_SCRIPT}"
require_file "${LAYER_TRAINING_CONFIG_PY}"
require_file "${VALIDATE_SCRIPT}"

log "RUN_ROOT=${RUN_ROOT}"
log "SPLIT_RUN_DIR=${SPLIT_RUN_DIR}"
log "DATASET_RUN_DIR=${DATASET_RUN_DIR}"
log "GPU_IDS=${GPU_IDS_CSV}"

TRAIN_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${TRAIN_SCRIPT}"
  --split-run-dir "${SPLIT_RUN_DIR}"
  --unit-dfn-root "${UNIT_DFN_ROOT}"
  --output-root "${TRAIN_ROOT}"
  --run-name-prefix layerwise_baseline
  --registry-output-py "${TRAIN_REGISTRY_PY}"
  --docx-path "${DOCX_PATH}"
  --layer-training-config-py "${LAYER_TRAINING_CONFIG_PY}"
  --gpu-ids "${GPU_IDS[@]}"
  --max-concurrent-jobs "${#GPU_IDS[@]}"
  --batch-size 1
  --num-workers 2
  --torch-num-threads 2
  --torch-num-interop-threads 1
  --no-cache-raw-packages
  --seed 20260506
)
run_logged train_layerwise "${TRAIN_CMD[@]}"
require_file "${TRAIN_REGISTRY_PY}"

EVAL_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${EVAL_SCRIPT}"
  --split-run-dir "${SPLIT_RUN_DIR}"
  --layer-model-registry-py "${TRAIN_REGISTRY_PY}"
  --split-name test
  --output-root "${EVAL_ROOT}"
  --run-name eval_test
  --docx-path "${DOCX_PATH}"
  --unit-dfn-root "${UNIT_DFN_ROOT}"
  --device cuda
  --layer-density-source robust
  --layer-density-scale 1.0
  --layer-density-calibration-min-scale 0.25
  --layer-density-calibration-max-scale 4.0
)
run_logged eval_test "${EVAL_CMD[@]}"
require_file "${EVAL_RUN_DIR}/aggregated/evaluation_summary.json"

log "START full_infer_parallel"
PIDS=()
for shard_idx in "${!GPU_IDS[@]}"; do
  gpu_id="${GPU_IDS[$shard_idx]}"
  shard_log="${LOG_ROOT}/full_infer_shard_${shard_idx}.log"
  SHARD_CMD=(
    conda run --no-capture-output -n "${CONDA_ENV}" python "${PRODUCTION_SCRIPT}"
    --layer-model-registry-py "${TRAIN_REGISTRY_PY}"
    --unit-dfn-root "${UNIT_DFN_ROOT}"
    --surface-dir "${SURFACE_DIR}"
    --trace-header-csv "${TRACE_HEADER_CSV}"
    --sgy-file "${SGY_FILE}"
    --output-root "${PRODUCTION_ROOT}"
    --run-name full_infer
    --docx-path "${DOCX_PATH}"
    --block-x-start "${BLOCK_X_START}"
    --block-x-end "${BLOCK_X_END}"
    --block-y-start "${BLOCK_Y_START}"
    --block-y-end "${BLOCK_Y_END}"
    --unit-shard-index "${shard_idx}"
    --unit-shard-count "${FULL_SHARD_COUNT}"
    --window-size 128
    --z-step-ms 0.2
    --overlap-ratio 0.5
    --center-threshold 0.7
    --decode-mode strict
    --relaxed-min-count 1
    --count-activation-threshold 0.5
    --min-count-if-active 1
    --max-slots-per-voxel 16
    --max-total-patches-per-window 256
    --layer-density-source robust
    --layer-density-scale 1.0
    --layer-density-calibration-min-scale 0.25
    --layer-density-calibration-max-scale 4.0
    --layer-top-boundary-ms "${LAYER_TOP_BOUNDARY_MS}"
    --layer-bottom-boundary-ms "${LAYER_BOTTOM_BOUNDARY_MS}"
    --artifact-profile compact
    --device cuda
  )
  log "Launch full_infer shard=${shard_idx} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${SHARD_CMD[@]}" > "${shard_log}" 2>&1 &
  PIDS+=("$!")
done

FULL_INFER_UNITS_DIR="${FULL_INFER_RUN_DIR}/units"
FULL_INFER_AGGREGATED_DIR="${FULL_INFER_RUN_DIR}/aggregated"
mkdir -p "${FULL_INFER_UNITS_DIR}" "${FULL_INFER_AGGREGATED_DIR}"
last_full_infer_signature=""
while true; do
  alive_count=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive_count=$((alive_count + 1))
    fi
  done

  completed_units=$(find "${FULL_INFER_UNITS_DIR}" -name unit_prediction_summary.json 2>/dev/null | wc -l | tr -d ' ')
  shard_parts=()
  for shard_idx in "${!GPU_IDS[@]}"; do
    shard_suffix=$(printf "%02dof%02d" "${shard_idx}" "${FULL_SHARD_COUNT}")
    selected_csv="${FULL_INFER_AGGREGATED_DIR}/selected_units_shard_${shard_suffix}.csv"
    unit_csv="${FULL_INFER_AGGREGATED_DIR}/unit_prediction_shard_${shard_suffix}.csv"
    selected_count=$(csv_data_row_count "${selected_csv}")
    produced_count=$(csv_data_row_count "${unit_csv}")
    shard_parts+=("s${shard_idx}=${produced_count}/${selected_count}")
  done
  shard_status=$(printf '%s ' "${shard_parts[@]}")
  shard_status="${shard_status% }"
  current_signature="${completed_units}|${alive_count}|${shard_status}"
  if [[ "${current_signature}" != "${last_full_infer_signature}" ]]; then
    log "full_infer progress units=${completed_units}/${FULL_INFER_TOTAL_UNITS} alive_shards=${alive_count}/${FULL_SHARD_COUNT} ${shard_status}"
    last_full_infer_signature="${current_signature}"
  fi
  if [[ "${alive_count}" -eq 0 ]]; then
    break
  fi
  sleep 60
done
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done
log "END full_infer_parallel"

VALIDATE_CMD=(
  conda run --no-capture-output -n "${CONDA_ENV}" python "${VALIDATE_SCRIPT}"
  --aggregated-dir "${FULL_INFER_RUN_DIR}/aggregated"
  --expected-shards "${FULL_SHARD_COUNT}"
)
run_logged validate_full_infer "${VALIDATE_CMD[@]}"

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
  --fault-induced-count-scale "${FAULT_INDUCED_COUNT_SCALE}"
  --surface-max-fragment-area-ratio "${FAULT_SURFACE_MAX_FRAGMENT_AREA_RATIO}"
  --docx-path "${DOCX_PATH}"
  --overwrite
)
run_logged fault_postfusion "${FAULT_CMD[@]}"
require_file "${FAULT_SUMMARY_JSON}"

log "Pipeline finished."
log "Train registry: ${TRAIN_REGISTRY_PY}"
log "Eval summary: ${EVAL_RUN_DIR}/aggregated/evaluation_summary.json"
log "Full infer dir: ${FULL_INFER_RUN_DIR}"
log "Merged VTK: ${MERGED_VTK}"
log "Postprocess VTK: ${POSTPROCESS_VTK}"
log "Fault summary: ${FAULT_SUMMARY_JSON}"

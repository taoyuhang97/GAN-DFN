#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAND_CONFIG="优化阶段二/正式主线/step8_dfn_well_correction/configs/formal_mine_multiscale_flow_v1_seg1_density_balance_v1.json"
SAND_OUTPUT="优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_v1"
SAND_NEG_OUTPUT="${SAND_OUTPUT}_z_negative"
TAIGU_MAIN_CONFIG="太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1_seg1.json"
TAIGU_MAIN_OUTPUT="太古界/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1_seg1_density_balance_v1"
TAIGU_MAIN_NEG_OUTPUT="${TAIGU_MAIN_OUTPUT}_z_negative"
TAIGU_SUPP_ROOT="太古界/supp_layer/output/taigu_supp_step8_density_balance_v1"
TAIGU_SUPP_CONFIG="${TAIGU_SUPP_ROOT}/supp_8_config.json"
TAIGU_SUPP_OUTPUT="${TAIGU_SUPP_ROOT}/step8_supp"
TAIGU_MERGED_OUTPUT="太古界/supp_layer/output/taigu_supp_merged_density_balance_v1"
TAIGU_MERGED_NEG_OUTPUT="${TAIGU_MERGED_OUTPUT}_z_negative"

echo "[step8-balance] sandstone Step8"
"${PYTHON_BIN}" 优化阶段二/正式主线/step8_dfn_well_correction/correct_dfn_with_well_controls.py \
  --config "${SAND_CONFIG}"

echo "[step8-balance] prepare Taigu supplemental Step8 inputs"
"${PYTHON_BIN}" 太古界/supp_layer/build_supp_step8_inputs.py \
  --output-dir "${TAIGU_SUPP_ROOT}"

echo "[step8-balance] Taigu main Step8"
"${PYTHON_BIN}" 太古界/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config "${TAIGU_MAIN_CONFIG}" \
  --output-dir "${TAIGU_MAIN_OUTPUT}"

echo "[step8-balance] Taigu supplemental Step8"
"${PYTHON_BIN}" 太古界/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config "${TAIGU_SUPP_CONFIG}"

echo "[step8-balance] relabel Taigu supplemental layer"
"${PYTHON_BIN}" 太古界/supp_layer/relabel_supp_dfn.py \
  --input-csv "${TAIGU_SUPP_OUTPUT}/well_corrected_dfn_fracture_patches.csv" \
  --output-csv "${TAIGU_SUPP_OUTPUT}/well_corrected_dfn_supp_patches.csv"

echo "[step8-balance] merge Taigu expanded DFN"
"${PYTHON_BIN}" 太古界/supp_layer/merge_supp_dfn.py \
  --full-csv "${TAIGU_MAIN_OUTPUT}/well_corrected_dfn_fracture_patches.csv" \
  --full-vtk "${TAIGU_MAIN_OUTPUT}/well_corrected_dfn_raw_time.vtk" \
  --supp-csv "${TAIGU_SUPP_OUTPUT}/well_corrected_dfn_supp_patches.csv" \
  --supp-vtk "${TAIGU_SUPP_OUTPUT}/well_corrected_dfn_raw_time.vtk" \
  --output-dir "${TAIGU_MERGED_OUTPUT}"

echo "[step8-balance] create underground-coordinate (negative Z) copies"
"${PYTHON_BIN}" scripts/negate_dfn_z.py \
  --input-dir "${SAND_OUTPUT}" \
  --output-dir "${SAND_NEG_OUTPUT}" \
  --csv well_corrected_dfn_fracture_patches.csv \
  --vtk well_corrected_dfn_raw_time.vtk
"${PYTHON_BIN}" scripts/negate_dfn_z.py \
  --input-dir "${TAIGU_MAIN_OUTPUT}" \
  --output-dir "${TAIGU_MAIN_NEG_OUTPUT}" \
  --csv well_corrected_dfn_fracture_patches.csv \
  --vtk well_corrected_dfn_raw_time.vtk
"${PYTHON_BIN}" scripts/negate_dfn_z.py \
  --input-dir "${TAIGU_SUPP_OUTPUT}" \
  --output-dir "太古界/supp_layer/output/taigu_supp_step8_density_balance_v1_z_negative/step8_supp" \
  --csv well_corrected_dfn_fracture_patches.csv \
  --csv well_corrected_dfn_supp_patches.csv \
  --vtk well_corrected_dfn_raw_time.vtk
"${PYTHON_BIN}" scripts/negate_dfn_z.py \
  --input-dir "${TAIGU_MERGED_OUTPUT}" \
  --output-dir "${TAIGU_MERGED_NEG_OUTPUT}" \
  --csv merged_dfn_patches.csv \
  --vtk merged_dfn_raw_time.vtk

echo "[step8-balance] complete"

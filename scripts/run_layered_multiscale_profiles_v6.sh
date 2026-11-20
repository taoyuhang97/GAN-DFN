#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SAND_DFN_ROOT="优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_medium_count36000_v1"
SAND_OUTPUT="优化阶段二/正式主线/output/step9_section_presentation/mine_v1/layered_v8_sand_medium36000_10km_faults_all_scales_large_outline"
TAIGU_DFN="太古界/supp_layer/output/taigu_supp_merged_density_balance_v1/merged_dfn_patches.csv"
TAIGU_OUTPUT="太古界/supp_layer/output/taigu_supp_step9_layered_v6_5km_faults_all_scales_large_outline"

echo "[layered-v6] sandstone"
"${PYTHON_BIN}" 优化阶段二/正式主线/step9_section_visualize/build_cheye1_scale_wiggle_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json \
  --dfn-patch-csv "${SAND_DFN_ROOT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${SAND_DFN_ROOT}/well_corrected_dfn_raw_time.vtk" \
  --output-dir "${SAND_OUTPUT}" \
  --layer-output-dir "${SAND_OUTPUT}" \
  --section-window-m 10000 \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding \
  --max-traces 200
"${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
  --sand-output "${SAND_OUTPUT}"

echo "[layered-v6] Taigu"
"${PYTHON_BIN}" 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_scale_wiggle.json \
  --dfn-patch-csv "${TAIGU_DFN}" \
  --output-dir "${TAIGU_OUTPUT}" \
  --layer-output-dir "${TAIGU_OUTPUT}" \
  --section-window-m 5000 \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding
"${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
  --taigu-output "${TAIGU_OUTPUT}"

echo "[layered-v6] complete"
echo "sandstone: ${SAND_OUTPUT}"
echo "Taigu:     ${TAIGU_OUTPUT}"

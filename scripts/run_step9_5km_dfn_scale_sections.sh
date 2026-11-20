#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SECTION_WINDOW_M="${SECTION_WINDOW_M:-5000}"
SECTION_WINDOW_KM="$(awk -v meters="${SECTION_WINDOW_M}" 'BEGIN {printf "%.3g", meters / 1000.0}')"

SAND_DFN_ROOT="优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_v1"
SAND_OUTPUT="优化阶段二/正式主线/output/step9_section_presentation/mine_v1/scale_wiggle_t4_t7_seg1_density_balance_${SECTION_WINDOW_KM}km_dfn_pad20"
TAIGU_DFN="太古界/supp_layer/output/taigu_supp_merged_density_balance_v1/merged_dfn_patches.csv"
TAIGU_OUTPUT="太古界/supp_layer/output/taigu_supp_step9_scale_wiggle_density_balance_${SECTION_WINDOW_KM}km_dfn_pad20"

echo "[${SECTION_WINDOW_KM}km-dfn-sections] sandstone"
"${PYTHON_BIN}" 优化阶段二/正式主线/step9_section_visualize/build_cheye1_scale_wiggle_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json \
  --dfn-patch-csv "${SAND_DFN_ROOT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${SAND_DFN_ROOT}/well_corrected_dfn_raw_time.vtk" \
  --output-dir "${SAND_OUTPUT}" \
  --section-window-m "${SECTION_WINDOW_M}" \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding \
  --max-traces 200

echo "[${SECTION_WINDOW_KM}km-dfn-sections] Taigu expanded DFN"
"${PYTHON_BIN}" 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_scale_wiggle.json \
  --dfn-patch-csv "${TAIGU_DFN}" \
  --output-dir "${TAIGU_OUTPUT}" \
  --section-window-m "${SECTION_WINDOW_M}" \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding

echo "[${SECTION_WINDOW_KM}km-dfn-sections] standalone legends"
"${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
  --sand-output "${SAND_OUTPUT}" \
  --taigu-output "${TAIGU_OUTPUT}"

echo "[${SECTION_WINDOW_KM}km-dfn-sections] complete"
echo "sandstone: ${SAND_OUTPUT}"
echo "Taigu:     ${TAIGU_OUTPUT}"

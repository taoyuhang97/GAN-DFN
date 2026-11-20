#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SAND_DFN_ROOT="优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_medium_count36000_v1"
SAND_OUTPUT="优化阶段二/正式主线/output/step9_section_presentation/mine_v1/split_imaging_dfn_200m_staged_v4"
TAIGU_DFN="太古界/supp_layer/output/taigu_supp_merged_density_balance_v1/merged_dfn_patches.csv"
TAIGU_OUTPUT="太古界/supp_layer/output/taigu_supp_step9_imaging_dfn_200m_staged_v4"

echo "[imaging-200m-v4] sandstone"
"${PYTHON_BIN}" 优化阶段二/正式主线/step9_section_visualize/build_cheye1_imaging_dfn_split_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json \
  --dfn-patch-csv "${SAND_DFN_ROOT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${SAND_DFN_ROOT}/well_corrected_dfn_raw_time.vtk" \
  --local-axis-radius-m 200 \
  --dfn-geometry-source csv \
  --output-dir "${SAND_OUTPUT}"

echo "[imaging-200m-v4] Taigu"
"${PYTHON_BIN}" 太古界/supp_layer/plot_supp_step9_wiggle_split.py \
  --config 太古界/supp_layer/configs/taigu_supp_step9_wiggle_split.json \
  --dfn-csv "${TAIGU_DFN}" \
  --output-dir "${TAIGU_OUTPUT}"

echo "[imaging-200m-v4] complete"
echo "sandstone: ${SAND_OUTPUT}"
echo "Taigu:     ${TAIGU_OUTPUT}"

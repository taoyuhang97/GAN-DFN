#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SAND_DFN_ROOT="优化阶段二/正式主线/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_v1"
SAND_OUTPUT="优化阶段二/正式主线/output/step9_section_presentation/mine_v1/scale_wiggle_t4_t7_seg1_density_balance_10km_dfn_pad20"

"${PYTHON_BIN}" 优化阶段二/正式主线/step9_section_visualize/build_cheye1_scale_wiggle_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json \
  --dfn-patch-csv "${SAND_DFN_ROOT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${SAND_DFN_ROOT}/well_corrected_dfn_raw_time.vtk" \
  --output-dir "${SAND_OUTPUT}" \
  --section-window-m 10000 \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding \
  --max-traces 200

"${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
  --sand-output "${SAND_OUTPUT}"

echo "[10km-sand-dfn-sections] complete"
echo "sandstone: ${SAND_OUTPUT}"

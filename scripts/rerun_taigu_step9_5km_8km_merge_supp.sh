#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TAIGU_DFN="太古界/supp_layer/output/taigu_supp_merged_density_balance_v1/merged_dfn_patches.csv"
CONFIG="太古界/supp_layer/configs/taigu_supp_step9_scale_wiggle.json"

for SECTION_WINDOW_M in 5000 8000; do
  SECTION_WINDOW_KM="$(awk -v meters="${SECTION_WINDOW_M}" 'BEGIN {printf "%g", meters / 1000.0}')"
  OUTPUT_DIR="太古界/supp_layer/output/taigu_supp_step9_scale_wiggle_density_balance_${SECTION_WINDOW_KM}km_dfn_pad20"
  "${PYTHON_BIN}" 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
    --config "${CONFIG}" \
    --dfn-patch-csv "${TAIGU_DFN}" \
    --output-dir "${OUTPUT_DIR}" \
    --section-window-m "${SECTION_WINDOW_M}" \
    --dfn-time-padding-ms 20 \
    --section-time-mode dfn_extent_plus_padding
  "${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
    --taigu-output "${OUTPUT_DIR}"
done

echo "[taigu-merge-supp] complete"

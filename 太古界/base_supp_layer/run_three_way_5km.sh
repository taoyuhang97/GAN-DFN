#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SECTION_DIR="太古界/base_supp_layer/output/taigu_three_way_merge_v1/5km_sections"

"$PYTHON" 太古界/base_supp_layer/merge_three_way_layers.py
"$PYTHON" 太古界/supp_layer/plot_supp_step9_scale_wiggle.py \
  --config 太古界/base_supp_layer/configs/taigu_three_way_5km_sections.json
"$PYTHON" scripts/build_step9_5km_legends.py \
  --taigu-output "$SECTION_DIR"

echo "[three-way-5km] outputs: $SECTION_DIR"

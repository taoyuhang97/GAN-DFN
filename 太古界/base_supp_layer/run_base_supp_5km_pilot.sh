#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="太古界/base_supp_layer/output/taigu_base_supp_5km_v1"
PYTHON="${PYTHON:-python3}"

"$PYTHON" 太古界/base_supp_layer/build_base_supp_inputs.py
"$PYTHON" 太古界/base_supp_layer/predict_base_supp_density.py
"$PYTHON" 太古界/base_supp_layer/build_base_supp_inputs.py \
  --density-sgy "$OUT/step6a_density/supp_density_block.sgy"

"$PYTHON" 太古界/step6a_density_volume/build_taigu_small_background.py \
  --config "$OUT/configs/step6a_small.json"
"$PYTHON" 太古界/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config "$OUT/configs/step6b_medium.json"
"$PYTHON" 太古界/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config "$OUT/configs/step6c_large.json"
"$PYTHON" 太古界/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config "$OUT/configs/step6d_bundle.json"

"$PYTHON" 太古界/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config "$OUT/configs/step7a.json" --replace-output
"$PYTHON" 太古界/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config "$OUT/configs/step7b.json"
"$PYTHON" 太古界/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config "$OUT/configs/step7c.json"
"$PYTHON" 太古界/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config "$OUT/configs/step7d.json"

"$PYTHON" 太古界/base_supp_layer/relabel_merge_base_supp.py

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RUN_NAME="demo_bx6_13_by28_35_$(date +%Y%m%d_%H%M%S)"
FIXED_SPLIT_CSV="${SCRIPT_DIR}/fixed_unit_splits/demo_bx6_13_by28_35_fixed_split.csv"
LAYER_TRAINING_CONFIG_PY="${SCRIPT_DIR}/layerwise_training_plan_demo_v1.py"

has_run_name="0"
for arg in "$@"; do
  if [[ "$arg" == "--run-name" ]]; then
    has_run_name="1"
    break
  fi
done

forward_args=(
  --train-unit-count 20
  --val-unit-count 9
  --test-unit-count 5
  --fixed-unit-split-csv "$FIXED_SPLIT_CSV"
  --layer-training-config-py "$LAYER_TRAINING_CONFIG_PY"
  --center-positive-weight 24.0
  --center-negative-weight 0.25
  --center-focal-gamma 2.0
  --count-positive-weight 12.0
  --count-negative-weight 0.25
  --center-loss-weight 2.5
  --count-loss-weight 1.0
  --calibration-center-thresholds 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90
  --calibration-count-thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50
  --calibration-window-weight 0.35
  --layer-density-source robust
  --layer-density-calibration-min-scale 0.25
  --layer-density-calibration-max-scale 4.0
  --eval-count-activation-threshold 0.5
  --eval-min-count-if-active 1
  --count-activation-threshold 0.5
  --min-count-if-active 1
  --skip-smoke-infer
  --full-block-x-start 6
  --full-block-x-end 13
  --full-block-y-start 28
  --full-block-y-end 35
  --artifact-profile compact
  --postprocess-phase 2
  --postprocess-boundary-connect
  --postprocess-bc-seam-fill
)

if [[ "$has_run_name" != "1" ]]; then
  forward_args=(--run-name "$DEFAULT_RUN_NAME" "${forward_args[@]}")
fi

exec "${SCRIPT_DIR}/start_layerwise_full_pipeline_nohup.sh" "${forward_args[@]}" "$@"

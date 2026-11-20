#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/tyh/projects/petroleum/code/GAN-DFN
STEP="$ROOT/优化阶段二/正式主线/step5d_reference_and_conditioning_samples"
PY=/home/tyh/anaconda3/envs/gan-dfn/bin/python
SCRIPT="$STEP/build_fracture_reference_samples.py"
LOG_DIR="$STEP/logs"

mkdir -p "$LOG_DIR"

run_one() {
  local name="$1"
  local config="$2"
  echo "===== ${name} start $(date '+%F %T') ====="
  "$PY" "$SCRIPT" --config "$config" --overwrite 2>&1 | tee "$LOG_DIR/${name}.log"
  local status=${PIPESTATUS[0]}
  echo "===== ${name} end status=${status} $(date '+%F %T') ====="
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
}

run_one "sand_high_fidelity_200" "$STEP/configs/formal_sand_high_fidelity_v1.json" || exit 1
run_one "taigu_high_fidelity_40" "$STEP/configs/formal_taigu_high_fidelity_v1.json" || exit 1
run_one "sand_direct_conditioning_80" "$STEP/configs/formal_sand_direct_conditioning_v1.json" || exit 1
run_one "taigu_direct_conditioning_40" "$STEP/configs/formal_taigu_direct_conditioning_v1.json" || exit 1

echo "===== all finished $(date '+%F %T') ====="

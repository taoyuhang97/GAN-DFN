#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/tyh/projects/petroleum/code/GAN-DFN
PY=/home/tyh/anaconda3/envs/gan-dfn/bin/python
SCRIPT="$ROOT/优化阶段二/正式主线/step5c_seismic_fracture_grid_samples/build_seismic_fracture_grid_samples.py"
SAND_CONFIG="$ROOT/优化阶段二/正式主线/step5c_seismic_fracture_grid_samples/configs/formal_sand_16x26_v1.json"
TAIGU_CONFIG="$ROOT/优化阶段二/正式主线/step5c_seismic_fracture_grid_samples/configs/formal_taigu_8x26_v1.json"
SAND_OUTPUT="$ROOT/优化阶段二/正式主线/output/seismic_fracture_grid_samples_sand_v1"
TAIGU_OUTPUT="$ROOT/太古界/output/taigu_seismic_fracture_grid_samples_v1"
SAND_LOG_DIR="$SAND_OUTPUT/logs"
TAIGU_LOG_DIR="$TAIGU_OUTPUT/logs"

mkdir -p "$SAND_LOG_DIR" "$TAIGU_LOG_DIR"

echo "===== Sand 16x26 start $(date '+%F %T') ====="
"$PY" "$SCRIPT" --config "$SAND_CONFIG" --overwrite 2>&1 | tee "$SAND_LOG_DIR/sand_16x26.log" || exit 1

echo "===== Taigu 8x26 start $(date '+%F %T') ====="
"$PY" "$SCRIPT" --config "$TAIGU_CONFIG" --overwrite 2>&1 | tee "$TAIGU_LOG_DIR/taigu_8x26.log" || exit 1

echo "===== Finished $(date '+%F %T') ====="

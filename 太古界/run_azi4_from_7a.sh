#!/usr/bin/env bash
# azi4 链条（Step7A → 7B → 7C → 7D → 8 → 9 + QC）
#   方案：太古界/太古界DFN产状赋值逻辑修正方案_20260923.md
#   D1 全部成像点进库；D2 最近几口井高斯加权 + 空间相关抽井；D3 局部几何 β 门控；
#   D4 中尺度（7B/7C）不动（沿用 azi1 配置，只把输出改到 azi4）。
#
#   用法:
#     tmux new-session -d -s taigu_azi4 "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#       bash 太古界/run_azi4_from_7a.sh 2>&1 | tee 太古界/output_azi1_logs/chain_azi4.log"
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_azi1_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

if [ "${SKIP_7A:-0}" != "1" ]; then
  step "Step7A 小尺度 DFN（azi4：按井经验分布重采样）"
  $PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
    --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_azi4.json \
    --replace-output 2>&1 | tee "$LOG/step7a_azi4.log" || fail "step7a"
fi

step "Step7B 中尺度 DFN（azi4；按 D4 保持蚂蚁体来源，配置同 azi1）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi4.json \
  2>&1 | tee "$LOG/step7b_azi4.log" || fail "step7b"

step "Step7C 大尺度 DFN（azi4；按 D4 不变）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi4.json \
  2>&1 | tee "$LOG/step7c_azi4.log" || fail "step7c"

step "Step7D 多尺度融合（azi4）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi4.json \
  2>&1 | tee "$LOG/step7d_azi4.log" || fail "step7d"

step "Step8 井控校正（azi4）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi4.json \
  2>&1 | tee "$LOG/step8_azi4.log" || fail "step8"

step "Step9 剖面出图（azi4）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_azi4.json \
  --replace-output 2>&1 | tee "$LOG/step9_azi4.log" || fail "step9"

step "S10 产状口径 QC（azi4）"
$PY $ROOT/tools/check_orientation_convention.py \
  --step7a-csv $ROOT/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_azi4/fracture_patches.csv \
  --step7b-csv $ROOT/step7b_multiscale_initial_dfn/output/taigu_medium_v7_anttrack_led_azi4/medium_dfn_patches.csv \
  --step7c-csv $ROOT/step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_azi4/large_fault_dfn_patches.csv \
  --step8-csv  $ROOT/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_azi4/well_corrected_dfn_fracture_patches.csv \
  --output $ROOT/output_azi1_logs/orientation_convention_qc_azi4.json \
  2>&1 | tee "$LOG/g3_orientation_qc_azi4.log" || fail "orientation qc"

echo; echo "[$(date '+%F %T')] ALL DONE (Step7A→Step9 + QC, azi4)"

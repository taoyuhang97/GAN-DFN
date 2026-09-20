#!/usr/bin/env bash
# v6 下游链条（Step7A → 7B → 7C → 7D → 8 → 9），承接 v5 的 Step6 系列输出
#   用法: tmux new-session -d -s taigu_v6 "cd <repo> && bash 太古界/run_v6_from_7.sh 2>&1 | tee 太古界/output_v5_logs/chain_v6.log"
#   说明：Step9 本版含"图例分组 + 405 三段区间标尺（成像段/常规段/目标地层内）"。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_v5_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step7A 小尺度 DFN（v6：输入=6D v5 背景+断层损伤；Step2/3/4 v5）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_v6.json \
  --replace-output 2>&1 | tee "$LOG/step7a_v6.log" || fail "step7a"

step "Step7B 中尺度 DFN（v6：输入=6B v5）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_v6.json \
  2>&1 | tee "$LOG/step7b_v6.log" || fail "step7b"

step "Step7C 大尺度 DFN（v6：输入=6C v5）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_v6.json \
  2>&1 | tee "$LOG/step7c_v6.log" || fail "step7c"

step "Step7D 多尺度融合（v6）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_v6.json \
  2>&1 | tee "$LOG/step7d_v6.log" || fail "step7d"

step "Step8 井控校正（v6）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_v6.json \
  2>&1 | tee "$LOG/step8_v6.log" || fail "step8"

step "Step9 剖面展示（v6：图例分组 + 405 三段区间标尺）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_v6.json \
  --replace-output \
  2>&1 | tee "$LOG/step9_v6.log" || fail "step9"

echo "[$(date '+%F %T')] ALL DONE (Step7A→Step9, v6)"

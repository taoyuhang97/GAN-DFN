#!/usr/bin/env bash
# azi1 续跑（Step7B → 7C → 7D → 8 → 9）
#   用途：`run_azi1_from_6b.sh` 在 Step7B 因脚本 bug 中断后，从 7B 续跑。
#   Step6B / Step6C 的 azi1 产物已生成，不重复执行。
#
#   用法:
#     tmux new-session -d -s taigu_azi1b "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#       bash 太古界/run_azi1_from_7b.sh 2>&1 | tee -a 太古界/output_azi1_logs/chain_azi1.log"
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_azi1_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step7B 中尺度 DFN（azi1，续跑）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi1.json \
  2>&1 | tee "$LOG/step7b_azi1.log" || fail "step7b"

step "Step7C 大尺度 DFN（azi1，续跑）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi1.json \
  2>&1 | tee "$LOG/step7c_azi1.log" || fail "step7c"

step "Step7D 多尺度融合（azi1）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi1.json \
  2>&1 | tee "$LOG/step7d_azi1.log" || fail "step7d"

step "Step8 井控校正（azi1，全量重建顶点）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi1.json \
  2>&1 | tee "$LOG/step8_azi1.log" || fail "step8"

step "Step9 剖面出图（azi1）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_azi1.json \
  --replace-output 2>&1 | tee "$LOG/step9_azi1.log" || fail "step9"

step "S10 产状口径 QC 闸门"
$PY $ROOT/tools/check_orientation_convention.py 2>&1 | tee "$LOG/g3_orientation_qc.log" || fail "orientation qc"

echo; echo "[$(date '+%F %T')] ALL DONE (Step7B→Step9 + QC, azi1)"

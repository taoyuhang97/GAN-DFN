#!/usr/bin/env bash
# v5：Step7A 改成"背景小尺度 + 受限断层派生小尺度"分域采样（对齐砂砾岩口径）
#   Step6D 重跑只为刷新 summary 里的 output_paths（数据不变，声明背景体/断层派生体），
#   然后 7A → 7D → 8 → 9 全链重跑。
#   用法: tmux new-session -d -s taigu_v5 "cd <repo> && bash 太古界/run_v5_from_7a.sh 2>&1 | tee 太古界/output_v5_logs/chain_v5.log"
set -uo pipefail
# 约定：每步 stdout/stderr 双写（tmux 面板 + 日志文件），见 太古界/AGENTS.md

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_v5_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step6D 多尺度打包（刷新 summary：声明 background/damage 密度体）"
$PY $ROOT/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config $ROOT/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_v4.json \
  2>&1 | tee "$LOG/step6d_refresh.log" || fail "step6d"

step "Step7A 小尺度 DFN（v5：背景域 + 受限断层派生域）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_attribute_v3_regen_v5.json \
  --replace-output 2>&1 | tee "$LOG/step7a_v5.log" || fail "step7a"

step "Step7D 多尺度融合（v5）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_v5.json \
  2>&1 | tee "$LOG/step7d_v5.log" || fail "step7d"

step "Step8 井控校正（v5）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_v5.json \
  2>&1 | tee "$LOG/step8_v5.log" || fail "step8"

step "Step9 剖面展示（v5）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_v5.json \
  --replace-output \
  2>&1 | tee "$LOG/step9_v5.log" || fail "step9"

echo "[$(date '+%F %T')] ALL DONE (v5)"

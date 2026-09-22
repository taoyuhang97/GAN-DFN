#!/usr/bin/env bash
# azi1 链条（Step6B → 6C → 7A → 7B → 7C → 7D → 8 → 9）
#   口径：把全链产状统一成"真倾向方位 0–360 + 倾角"（见
#   太古界/太古界产状口径统一施工方案_20260922.md）。
#
#   用法:
#     tmux new-session -d -s taigu_azi1 "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#       bash 太古界/run_azi1_from_6b.sh 2>&1 | tee 太古界/output_azi1_logs/chain_azi1.log"
#
#   为什么从这里开始：
#     * Step1–5、Step6A、Step6D 不含方位字段（Step6D 只打包密度体），无需重跑；
#     * Step6B/6C 的分量 PCA 方位是 7B/7C 的输入，必须重跑；
#     * 7A 的输入密度取 Step6D md1（密度与方位无关）。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_azi1_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "S1 产状口径公共模块单测（闸门 G1）"
$PY $ROOT/common/orientation_frame/test_convention.py 2>&1 | tee "$LOG/g1_convention_test.log" || fail "G1 convention test"

step "Step6B 中尺度裂缝带（azi1）"
$PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_azi1.json \
  2>&1 | tee "$LOG/step6b_azi1.log" || fail "step6b"

step "Step6C 大尺度断层先验（azi1）"
$PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_azi1.json \
  2>&1 | tee "$LOG/step6c_azi1.log" || fail "step6c"

step "Step7A 小尺度 DFN（azi1，全圆倾向方位家族）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_azi1.json \
  --replace-output 2>&1 | tee "$LOG/step7a_azi1.log" || fail "step7a"

step "Step7B 中尺度 DFN（azi1）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi1.json \
  2>&1 | tee "$LOG/step7b_azi1.log" || fail "step7b"

step "Step7C 大尺度 DFN（azi1）"
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

echo; echo "[$(date '+%F %T')] ALL DONE (Step6B→Step9, azi1)"

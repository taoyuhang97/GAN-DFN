#!/usr/bin/env bash
# v4 局部重跑：Step6A 特征只保留 3 个属性分数（位置/几何特征剔除）后的下游重跑
#   用法: tmux new-session -d -s taigu_v4_6a "cd <repo> && bash 太古界/run_v4_from_6a.sh 2>&1 | tee 太古界/output_v4_logs/chain_from_6a.log"
# 说明：Step6A 模型与密度体都会变 → 6B/6C/7A/7B/7C/7D/8/9 必须一起重跑；Step3/4/5 不动。
# 注意：Step7A 的小尺度口径已被 v5 取代（分域采样，见问题记录 0.25）。
#       重跑当前正式口径请用 太古界/run_v5_from_7a.sh；本脚本保留用于复现 v4 历史结果。
set -uo pipefail
# 约定：每步 stdout/stderr 双写（tmux 面板 + 日志文件），见 太古界/AGENTS.md

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_v4_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step6A 训练（v4：特征=3 属性分数，含样本构成硬校验）"
OMP_NUM_THREADS=16 $PY $ROOT/step6a_density_volume/train_taigu_two_stage_density.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_v4.json \
  --replace-output 2>&1 | tee "$LOG/step6a_train_v4.log" || fail "step6a train"

step "Step6A 预测（v4：demo 区 5km，含层内纵向分布 QC）"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_v4.json \
  --replace-output 2>&1 | tee "$LOG/step6a_predict_v4.log" || fail "step6a predict"

step "Step6A 小尺度背景分数（v4：体数据再矫正，密度 0.4 / 曲率 0.6）"
$PY $ROOT/step6a_density_volume/build_taigu_small_background.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_small_background_v4.json \
  --replace-output 2>&1 | tee "$LOG/step6a_small_background_v4.log" || fail "step6a small background"

step "Step6B 中尺度裂缝带（v4）"
$PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_v4.json \
  2>&1 | tee "$LOG/step6b_v4.log" || fail "step6b"

step "Step6C 大尺度断层先验（v4）"
$PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_v4.json \
  2>&1 | tee "$LOG/step6c_v4.log" || fail "step6c"

step "Step6D 多尺度打包（v4：6A 小尺度背景 + 6B 中尺度 + 6C 大尺度）"
$PY $ROOT/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config $ROOT/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_v4.json \
  2>&1 | tee "$LOG/step6d_v4.log" || fail "step6d"

step "Step7B 中尺度 DFN（v4）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_v4.json \
  2>&1 | tee "$LOG/step7b_v4.log" || fail "step7b"

step "Step7C 大尺度 DFN（v4）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_v4.json \
  2>&1 | tee "$LOG/step7c_v4.log" || fail "step7c"

step "Step7A 小尺度 DFN（v4：输入源 = Step6D final_small_density）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_attribute_v3_regen_v4.json \
  --replace-output 2>&1 | tee "$LOG/step7a_v4.log" || fail "step7a"

step "Step7D 多尺度融合（v4）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_v4.json \
  2>&1 | tee "$LOG/step7d_v4.log" || fail "step7d"

step "Step8 井控校正（v4）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_v4.json \
  2>&1 | tee "$LOG/step8_v4.log" || fail "step8"

step "Step9 剖面展示（v4，405 居中 demo 区域）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_v4.json \
  --replace-output \
  2>&1 | tee "$LOG/step9_v4.log" || fail "step9"

echo "[$(date '+%F %T')] ALL DONE (from 6A)"

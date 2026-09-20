#!/usr/bin/env bash
# v5 链条（Step5B → Step6A 训练/预测/小尺度背景 → 6B → 6C → 6D）
#   用法: tmux new-session -d -s taigu_v5_6 "cd <repo> && bash 太古界/run_v5_from_step5b.sh 2>&1 | tee 太古界/output_v5_logs/chain_v5_6.log"
#   说明：Step5 v5 已产出统一样本；本链条把取样窗口口径（成像段管标签、层位段管预测）
#         传导到三维密度体。日志双写（tmux 面板 + 文件），见 太古界/AGENTS.md。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_v5_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step5B 统一样本交付（v5）"
$PY $ROOT/step5_virtual_wells/build_taigu_step5b_unified_samples.py \
  --input-csv $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v5/taigu_step5_unified_samples.csv \
  --output-dir $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v5/step5b \
  --replace-output 2>&1 | tee "$LOG/step5b_v5.log" || fail "step5b"

step "Step6A 训练（v5，样本来自 Step5B v5）"
OMP_NUM_THREADS=16 $PY $ROOT/step6a_density_volume/train_taigu_two_stage_density.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_v5.json \
  --replace-output 2>&1 | tee "$LOG/step6a_train_v5.log" || fail "step6a train"

step "Step6A 预测（v5，demo 区 5km）"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_v5.json \
  --replace-output 2>&1 | tee "$LOG/step6a_predict_v5.log" || fail "step6a predict"

step "Step6A 小尺度背景分数（v5，密度 0.4 / 曲率 0.6）"
$PY $ROOT/step6a_density_volume/build_taigu_small_background.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_small_background_v5.json \
  --replace-output 2>&1 | tee "$LOG/step6a_small_background_v5.log" || fail "step6a small background"

step "Step6B 中尺度裂缝带（v5）"
$PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_v5.json \
  2>&1 | tee "$LOG/step6b_v5.log" || fail "step6b"

step "Step6C 大尺度断层先验（v5）"
$PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_v5.json \
  2>&1 | tee "$LOG/step6c_v5.log" || fail "step6c"

step "Step6D 多尺度打包（v5：6A 小尺度背景 + 6B + 6C）"
$PY $ROOT/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config $ROOT/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_v5.json \
  2>&1 | tee "$LOG/step6d_v5.log" || fail "step6d"

echo "[$(date '+%F %T')] ALL DONE (Step5B→Step6D, v5)"

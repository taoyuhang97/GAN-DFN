#!/usr/bin/env bash
# v4 全链：Step3 标签口径修正（窗口/填充/标定/门控/点吸附放宽）+ Step4 标定·区间·细化改造
#        + demo 区域移到 405 居中（taigu_attribute_demo_grid_v2_405center）
#   用法: tmux new-session -d -s taigu_v4 'bash 太古界/run_v4_chain.sh'
# 说明：Step3 密度标签与 Step4 输出都变了 → 5/6A/6B/6C/7A/7B/7C/7D/8/9 必须整链重跑。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_v4_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step4 分段预测 + 同井合并（v4：门控 v4c + 总量守恒标定 + 区间/细化改造）"
$PY $ROOT/step4_fracture_prediction/train_and_predict_taigu_gr_rd_rs.py \
  --config $ROOT/step4_fracture_prediction/configs/taigu_step4_gr_rd_rs_v4.json \
  --replace-existing-output > "$LOG/step4_v4.log" 2>&1 || fail "step4"
tail -3 "$LOG/step4_v4.log"

step "Step4 裂缝点坐标自检"
$PY $ROOT/tools/check_step4_point_geometry.py \
  --points $ROOT/step4_fracture_prediction/output/taigu_step4_gr_rd_rs_v4/predictions/all_wells_merged_fracture_points.csv \
  > "$LOG/step4_point_geometry.log" 2>&1 || fail "step4 point geometry"
cat "$LOG/step4_point_geometry.log"

step "Step5 虚拟井 + 统一训练样本（v4）"
$PY $ROOT/step5_virtual_wells/build_taigu_virtual_wells.py \
  --config $ROOT/step5_virtual_wells/configs/taigu_step5_attribute_v3_v4.json \
  --replace-output > "$LOG/step5_v4.log" 2>&1 || fail "step5"
tail -3 "$LOG/step5_v4.log"

step "Step5B 统一训练样本交付（v4）"
$PY $ROOT/step5_virtual_wells/build_taigu_step5b_unified_samples.py \
  --input-csv $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v4/taigu_step5_unified_samples.csv \
  --output-dir $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_v4/step5b \
  --replace-output > "$LOG/step5b_v4.log" 2>&1 || fail "step5b"
tail -3 "$LOG/step5b_v4.log"

step "Step6A 训练（v4）"
OMP_NUM_THREADS=16 $PY $ROOT/step6a_density_volume/train_taigu_two_stage_density.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_v4.json \
  --replace-output > "$LOG/step6a_train_v4.log" 2>&1 || fail "step6a train"
tail -3 "$LOG/step6a_train_v4.log"

step "Step6A 预测（v4，demo 区域已移到 405 居中）"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_v4.json \
  --replace-output > "$LOG/step6a_predict_v4.log" 2>&1 || fail "step6a predict"
tail -3 "$LOG/step6a_predict_v4.log"

step "Step6B 中尺度裂缝带（v4）"
$PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_v4.json \
  > "$LOG/step6b_v4.log" 2>&1 || fail "step6b"
tail -3 "$LOG/step6b_v4.log"

step "Step7B 中尺度 DFN（v4）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_v4.json \
  > "$LOG/step7b_v4.log" 2>&1 || fail "step7b"
tail -3 "$LOG/step7b_v4.log"

step "Step6C 大尺度断层先验（v4）"
$PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_v4.json \
  > "$LOG/step6c_v4.log" 2>&1 || fail "step6c"
tail -3 "$LOG/step6c_v4.log"

step "Step7C 大尺度 DFN（v4）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_v4.json \
  > "$LOG/step7c_v4.log" 2>&1 || fail "step7c"
tail -3 "$LOG/step7c_v4.log"

step "Step7A 小尺度 DFN（v4）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_attribute_v3_regen_v4.json \
  --replace-output > "$LOG/step7a_v4.log" 2>&1 || fail "step7a"
tail -3 "$LOG/step7a_v4.log"

step "Step7D 多尺度融合（v4）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_v4.json \
  > "$LOG/step7d_v4.log" 2>&1 || fail "step7d"
tail -3 "$LOG/step7d_v4.log"

step "Step8 井控校正（v4）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_v4.json \
  > "$LOG/step8_v4.log" 2>&1 || fail "step8"
tail -4 "$LOG/step8_v4.log"

step "Step9 剖面展示（v4，405 居中 demo 区域）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_v4.json \
  > "$LOG/step9_v4.log" 2>&1 || fail "step9"
tail -3 "$LOG/step9_v4.log"

echo "[$(date '+%F %T')] ALL DONE"

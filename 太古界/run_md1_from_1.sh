#!/usr/bin/env bash
# md1 全链：MD 深度口径修正（Step1 合同 → Step2 取样窗口 → Step3 标签回接）后的整链重跑
#   用法: tmux new-session -d -s taigu_md1 "cd <repo> && bash 太古界/run_md1_from_1.sh 2>&1 | tee 太古界/output_md1_logs/chain_md1.log"
#   说明：Step1/2/3 口径变了（甲方成像解释深度 = 测深 MD）→ Step4…Step9 必须整链重跑；
#         方案见 太古界/太古界MD深度口径修正施工方案_20260922.md。
#   约定：每步 stdout/stderr 双写（tmux 面板 + 日志文件），见 太古界/AGENTS.md §1。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_md1_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step1 合同（md1：成像解释区间为 MD）"
$PY $ROOT/step1_strata_contracts/build_taigu_strata_contracts.py \
  --config $ROOT/step1_strata_contracts/configs/taigu_step1_contracts_md1.json \
  --replace-output 2>&1 | tee "$LOG/step1_md1.log" || fail "step1"

step "Step2 常规/成像井段（md1：窗口按 MD）"
$PY $ROOT/step2_well_log_segments/build_taigu_regular_samples.py \
  --config $ROOT/step2_well_log_segments/configs/taigu_step2_contracts_md1.json \
  --replace-output 2>&1 | tee "$LOG/step2_md1.log" || fail "step2"

step "Step3 成像标签（md1：密度/产状按 MD 回接）"
$PY $ROOT/step3_imaging_groups/build_taigu_imaging_groups.py \
  --config $ROOT/step3_imaging_groups/configs/taigu_step3_imaging_groups_md1.json \
  --replace-output 2>&1 | tee "$LOG/step3_md1.log" || fail "step3"

step "Step3 交付校验"
$PY $ROOT/step3_imaging_groups/validate_taigu_rebuild_delivery.py \
  --config $ROOT/step3_imaging_groups/configs/taigu_step3_imaging_groups_md1.json \
  2>&1 | tee "$LOG/step3_validate_md1.log" || fail "step3 validate"

step "闸门 G1：Step2 与 v5 逐行比对（预期只有 405 变化）"
$PY $ROOT/tools/compare_md1_vs_v5_step2.py \
  --md1 $ROOT/step2_well_log_segments/output/taigu_step2_regular_md1 \
  --v5  $ROOT/step2_well_log_segments/output/taigu_step2_regular_v5 \
  --out $LOG/gate_g1_step2.json > "$LOG/gate_g1_step2.log" || fail "gate G1 step2"
$PY $ROOT/tools/compare_md1_vs_v5_step3.py \
  --md1 $ROOT/step3_imaging_groups/output/taigu_step3_imaging_md1 \
  --v5  $ROOT/step3_imaging_groups/output/taigu_step3_imaging_v5 \
  --out $LOG/gate_g1_step3.json > "$LOG/gate_g1_step3.log" || fail "gate G1 step3"
echo "[$(date '+%F %T')] G1 PASS"

step "Step4 分段预测 + 同井合并（md1）"
$PY $ROOT/step4_fracture_prediction/train_and_predict_taigu_gr_rd_rs.py \
  --config $ROOT/step4_fracture_prediction/configs/taigu_step4_gr_rd_rs_md1.json \
  --replace-existing-output 2>&1 | tee "$LOG/step4_md1.log" || fail "step4"

step "Step4 裂缝点坐标自检"
$PY $ROOT/tools/check_step4_point_geometry.py \
  --points $ROOT/step4_fracture_prediction/output/taigu_step4_gr_rd_rs_md1/predictions/all_wells_merged_fracture_points.csv \
  2>&1 | tee "$LOG/step4_point_geometry_md1.log" || fail "step4 point geometry"

step "Step5 虚拟井 + 统一训练样本（md1）"
$PY $ROOT/step5_virtual_wells/build_taigu_virtual_wells.py \
  --config $ROOT/step5_virtual_wells/configs/taigu_step5_attribute_v3_md1.json \
  --replace-output 2>&1 | tee "$LOG/step5_md1.log" || fail "step5"

step "Step5B 统一训练样本交付（md1）"
$PY $ROOT/step5_virtual_wells/build_taigu_step5b_unified_samples.py \
  --input-csv $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_md1/taigu_step5_unified_samples.csv \
  --output-dir $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_md1/step5b \
  --replace-output 2>&1 | tee "$LOG/step5b_md1.log" || fail "step5b"

step "Step6A 训练（md1）"
OMP_NUM_THREADS=16 $PY $ROOT/step6a_density_volume/train_taigu_two_stage_density.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_md1.json \
  --replace-output 2>&1 | tee "$LOG/step6a_train_md1.log" || fail "step6a train"

step "Step6A 预测（md1，全区体）"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_md1.json \
  --replace-output 2>&1 | tee "$LOG/step6a_predict_md1.log" || fail "step6a predict"

step "Step6A 小尺度背景分数（md1）"
$PY $ROOT/step6a_density_volume/build_taigu_small_background.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_small_background_md1.json \
  --replace-output 2>&1 | tee "$LOG/step6a_small_background_md1.log" || fail "step6a small background"

step "Step6B 中尺度裂缝带（md1）"
$PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
  --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_md1.json \
  2>&1 | tee "$LOG/step6b_md1.log" || fail "step6b"

step "Step6C 大尺度断层先验（md1）"
$PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
  --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_md1.json \
  2>&1 | tee "$LOG/step6c_md1.log" || fail "step6c"

step "Step6D 多尺度打包（md1）"
$PY $ROOT/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
  --config $ROOT/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_md1.json \
  2>&1 | tee "$LOG/step6d_md1.log" || fail "step6d"

step "Step7B 中尺度 DFN（md1）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_md1.json \
  2>&1 | tee "$LOG/step7b_md1.log" || fail "step7b"

step "Step7C 大尺度 DFN（md1）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_md1.json \
  2>&1 | tee "$LOG/step7c_md1.log" || fail "step7c"

step "Step7A 小尺度 DFN（md1）"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_md1.json \
  --replace-output 2>&1 | tee "$LOG/step7a_md1.log" || fail "step7a"

step "Step7D 多尺度融合（md1）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_md1.json \
  2>&1 | tee "$LOG/step7d_md1.log" || fail "step7d"

step "Step8 井控校正（md1）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_md1.json \
  2>&1 | tee "$LOG/step8_md1.log" || fail "step8"

step "Step9 剖面展示（md1）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_md1.json \
  --replace-output 2>&1 | tee "$LOG/step9_md1.log" || fail "step9"

echo "[$(date '+%F %T')] ALL DONE (Step1→Step9, md1)"

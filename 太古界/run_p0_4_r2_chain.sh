#!/usr/bin/env bash
# P0-4（测井段回接健壮性 + 审计）r2 全链验证
#   用法: tmux new-session -d -s taigu_p04_r2 'bash 太古界/run_p0_4_r2_chain.sh'
# 约定：所有 r2 输出走独立版本目录，正式链（非 r2）不被覆盖。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_p04_r2_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step5 虚拟井 + 统一训练样本 (r2)"
$PY $ROOT/step5_virtual_wells/build_taigu_virtual_wells.py \
  --config $ROOT/step5_virtual_wells/configs/taigu_step5_attribute_v3_r2.json \
  --replace-output > "$LOG/step5_r2.log" 2>&1 || fail "step5"
tail -3 "$LOG/step5_r2.log"

step "Step5B 统一训练样本交付 (r2)"
$PY $ROOT/step5_virtual_wells/build_taigu_step5b_unified_samples.py \
  --input-csv $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_r2/taigu_step5_unified_samples.csv \
  --output-dir $ROOT/step5_virtual_wells/output/taigu_step5_attribute_v3_common_contract_r2/step5b \
  --replace-output \
  > "$LOG/step5b_r2.log" 2>&1 || fail "step5b"
tail -3 "$LOG/step5b_r2.log"

step "Step6A 训练 (r2)"
OMP_NUM_THREADS=16 $PY $ROOT/step6a_density_volume/train_taigu_two_stage_density.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_train_attribute_v3_r2.json \
  --replace-output > "$LOG/step6a_train_r2.log" 2>&1 || fail "step6a train"
tail -3 "$LOG/step6a_train_r2.log"

step "Step6A 预测 (r2)"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_r2.json \
  --replace-output > "$LOG/step6a_predict_r2.log" 2>&1 || fail "step6a predict"
tail -3 "$LOG/step6a_predict_r2.log"

step "Step6A 体等价性判定（决定 6B/6C/7B/7C 是否跳过）"
$PY $ROOT/tools/compare_step6a_volume.py \
  --old $ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_full/volume/predicted_fracture_density.sgy \
  --new $ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_full_r2/volume/predicted_fracture_density.sgy \
  --report $LOG/step6a_volume_compare.json > "$LOG/step6a_compare.log" 2>&1 || fail "compare 6a"
cat "$LOG/step6a_volume_compare.json"

if grep -q '"identical": true' "$LOG/step6a_volume_compare.json"; then
  echo "[$(date '+%F %T')] Step6A 体逐样点一致 -> 跳过 Step6B/6C/7B/7C（沿用现有正式输出）"
else
  step "Step6A 体有差异 -> 重跑 Step6B/6C/7B/7C (r2)"
  $PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
    --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_r2.json \
    > "$LOG/step6b_r2.log" 2>&1 || fail "step6b"
  $PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
    --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_r2.json \
    > "$LOG/step7b_r2.log" 2>&1 || fail "step7b"
  $PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
    --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_r2.json \
    > "$LOG/step6c_r2.log" 2>&1 || fail "step6c"
  $PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
    --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_r2.json \
    > "$LOG/step7c_r2.log" 2>&1 || fail "step7c"
fi

step "Step7A 小尺度 DFN (r2)"
$PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
  --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_attribute_v3_regen_r2.json \
  --replace-output > "$LOG/step7a_r2.log" 2>&1 || fail "step7a"
tail -3 "$LOG/step7a_r2.log"

step "Step7D 多尺度融合 (r2)"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_r2.json \
  > "$LOG/step7d_r2.log" 2>&1 || fail "step7d"
tail -3 "$LOG/step7d_r2.log"

step "Step8 井控校正 (r2)"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_r2.json \
  > "$LOG/step8_r2.log" 2>&1 || fail "step8"
tail -4 "$LOG/step8_r2.log"

step "Step9 剖面展示 (r2)"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_r2.json \
  > "$LOG/step9_r2.log" 2>&1 || fail "step9"
tail -3 "$LOG/step9_r2.log"

step "全链验收 + 新旧对比报告"
$PY $ROOT/tools/compare_p0_4_runs.py --report $LOG/p0_4_r2_report.json \
  > "$LOG/p0_4_report.log" 2>&1 || fail "compare"

echo "[$(date '+%F %T')] ALL DONE"

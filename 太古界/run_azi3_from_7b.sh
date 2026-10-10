#!/usr/bin/env bash
# azi3 链条（Step7B → 7C → 7D → 8 → 9 + QC）
#   目的：中尺度"全局兜底产状"改用多井统计倾角先验（见 tools/make_azi3_configs.py 文件头）。
#   上游：7A 复用 azi2 产物；6B/6C 仍用 azi1 产物。
#
#   用法:
#     tmux new-session -d -s taigu_azi3 "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#       bash 太古界/run_azi3_from_7b.sh 2>&1 | tee 太古界/output_azi1_logs/chain_azi3.log"
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_azi1_logs
mkdir -p "$LOG"
PY=python3

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

step "Step7B 中尺度 DFN（azi3，兜底倾角改用多井先验）"
$PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
  --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_azi3.json \
  2>&1 | tee "$LOG/step7b_azi3.log" || fail "step7b"

step "Step7C 大尺度 DFN（azi3）"
$PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
  --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_azi3.json \
  2>&1 | tee "$LOG/step7c_azi3.log" || fail "step7c"

step "Step7D 多尺度融合（azi3；小尺度复用 7A azi2）"
$PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
  --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_azi3.json \
  2>&1 | tee "$LOG/step7d_azi3.log" || fail "step7d"

step "Step8 井控校正（azi3）"
$PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
  --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_azi3.json \
  2>&1 | tee "$LOG/step8_azi3.log" || fail "step8"

step "Step9 剖面出图（azi3）"
$PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
  --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_azi3.json \
  --replace-output 2>&1 | tee "$LOG/step9_azi3.log" || fail "step9"

step "S10 产状口径 QC（azi3）"
$PY $ROOT/tools/check_orientation_convention.py \
  --step7a-csv $ROOT/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_azi2/fracture_patches.csv \
  --step7b-csv $ROOT/step7b_multiscale_initial_dfn/output/taigu_medium_v7_anttrack_led_azi3/medium_dfn_patches.csv \
  --step7c-csv $ROOT/step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_azi3/large_fault_dfn_patches.csv \
  --step8-csv  $ROOT/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_azi3/well_corrected_dfn_fracture_patches.csv \
  --output $ROOT/output_azi1_logs/orientation_convention_qc_azi3.json \
  2>&1 | tee "$LOG/g3_orientation_qc_azi3.log" || fail "orientation qc"

echo; echo "[$(date '+%F %T')] ALL DONE (Step7B→Step9 + QC, azi3)"

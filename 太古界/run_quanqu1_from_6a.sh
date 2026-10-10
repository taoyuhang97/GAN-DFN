#!/usr/bin/env bash
# =============================================================================
# quanqu1：太古界**矿区/全区**结果（Step6A 预测 → Step9）
#
#   范围：1650×1307 = 2,156,550 道，X 658887–679500（20.6 km）/ Y 4231662–4247987（16.3 km）
#         采样格网 = 太古界/common/attribute_demo_grid/output/taigu_attribute_grid_quanqu1/
#   口径：Step6A/6B/6C/6D 沿用 md1（深度 = MD）；7A→9 沿用 azi4（倾向方位为唯一真值 +
#         按井经验分布重采样 + 井周投影线按真实视倾角画）。
#
#   **为什么从 Step6A 预测开始、不重训**：
#     Step5 的训练样本本就基于全区 trace header（2,156,550 道）生成，
#     因此 taigu_step6a_attribute_v3_md1/models/*.joblib 可直接复用 → 不重训；
#     但预测范围由 demo_grid_csv 决定，旧配置只覆盖 5×5 km，必须从 6A 预测起重跑。
#
#   用法（tmux + 面板/日志双写，见 太古界/AGENTS.md §1）：
#     tmux new-session -d -s taigu_quanqu1 "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#       bash 太古界/run_quanqu1_from_6a.sh 2>&1 | tee 太古界/output_quanqu1_logs/chain_quanqu1.log"
#
#   资源与耗时（按 demo 5 km 实测 ×13.4 估算）：
#     F1/F2/F3 之后：6A 预测预计 10–20 min（原来单进程需约 70 h）；
#     6D ≈ 1.9 h；其它合计 ≈ 1–2 h
#     磁盘：方案 A+B 之后预计新增约 62 GB（6D 只写 2 个体 + summary，约 24 GB）
#           2026-09-23 已清理 34 个历史产物释放 19.6 GB；脚本仍会做前置磁盘检查
#     内存/CPU：503 GB / 128 核，充足
#
#   断点续跑：QUANQU1_START=<步骤号>
#     1=6A预测 2=6A小背景 3=6B 4=6C 5=6D 6=7A 7=7B 8=7C 9=7D 10=Step8 11=Step9
# =============================================================================
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_quanqu1_logs
mkdir -p "$LOG"
PY=python3
START="${QUANQU1_START:-1}"
MIN_FREE_GB="${QUANQU1_MIN_FREE_GB:-110}"
WORKERS="${QUANQU1_WORKERS:-8}"

step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }
run_step() { [ "$START" -le "$1" ]; }
fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }

# ------------------------------------------------------------------ 前置检查
step "前置检查（磁盘 / 模型 / 全区格网）"
FREE_GB=$(df -BG --output=avail /home/tyh/projects/petroleum/code/GAN-DFN | tail -1 | tr -dc '0-9')
echo "  可用磁盘 ${FREE_GB} GB（阈值 ${MIN_FREE_GB} GB）；断点起始步骤 START=${START}"
[ "$FREE_GB" -ge "$MIN_FREE_GB" ] || fail "磁盘不足：需 ≥${MIN_FREE_GB} GB，当前 ${FREE_GB} GB。请先清理历史实验产物。"
MODEL="$ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_md1/models/step6_two_stage_density_models.joblib"
GRID="$ROOT/common/attribute_demo_grid/output/taigu_attribute_grid_quanqu1/attribute_demo_grid.csv"
[ -f "$MODEL" ] || fail "训练模型不存在: $MODEL"
[ -f "$GRID" ] || fail "全区格网不存在: $GRID"
NG=$($PY -c "import pandas as pd,sys; print(len(pd.read_csv('$GRID', encoding='utf-8-sig')))")
echo "  全区格网道数: $NG"
[ "$NG" = "2156550" ] || fail "格网道数应为 2156550，实际 $NG"

# ------------------------------------------------------------------ Step6A 预测
if run_step 1; then
  step "Step6A 预测（quanqu1，全区 2,156,550 道；按块并行 workers=${WORKERS}）"
  # F1（O(N^2)→O(N log N) 采样分组）+ F2（只采层窗内采样，去掉 ~10.8x 冗余）
  # + F3（按块并行 + 拼装）：冒烟验证已确认与旧实现**逐位一致**，
  # 且 demo 实测 44 min → 169 s（15.6x）。矿区预计 10–20 min。
  $PY $ROOT/tools/run_step6a_parallel.py \
    --config $ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_quanqu1.json \
    --output-dir $ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_full_quanqu1/volume \
    --parts-root $ROOT/step6a_density_volume/output/_quanqu1_predict_parts \
    --workers "$WORKERS" \
    2>&1 | tee "$LOG/step6a_predict_quanqu1.log" || fail "step6a predict (parallel)"
  V="$ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_full_quanqu1/volume/predicted_fracture_density.sgy"
  NT=$($PY -c "import segyio; h=segyio.open('$V','r',ignore_geometry=True); print(h.tracecount); h.close()")
  echo "  预测体道数: $NT"
  [ "$NT" = "2156550" ] || fail "预测体道数应为 2156550，实际 $NT"
fi

# ------------------------------------------------------------------ 6A 小尺度背景
if run_step 2; then
  step "Step6A 小尺度背景分数（quanqu1）"
  $PY $ROOT/step6a_density_volume/build_taigu_small_background.py \
    --config $ROOT/step6a_density_volume/configs/taigu_step6a_small_background_quanqu1.json \
    --replace-output 2>&1 | tee "$LOG/step6a_small_background_quanqu1.log" || fail "step6a small background"
fi

# ------------------------------------------------------------------ 6B / 6C / 6D
if run_step 3; then
  step "Step6B 中尺度裂缝带（quanqu1）"
  $PY $ROOT/step6b_medium_corridor/code/build_step6b_medium_corridor_prior.py \
    --config $ROOT/step6b_medium_corridor/configs/taigu_step6b_medium_v7_anttrack_led_quanqu1.json \
    2>&1 | tee "$LOG/step6b_quanqu1.log" || fail "step6b"
fi

if run_step 4; then
  step "Step6C 大尺度断层先验（quanqu1）"
  $PY $ROOT/step6c_large_fault/code/build_step6c_large_fault_prior.py \
    --config $ROOT/step6c_large_fault/configs/taigu_step6c_large_v3_attribute_v3_quanqu1.json \
    2>&1 | tee "$LOG/step6c_quanqu1.log" || fail "step6c"
fi

if run_step 5; then
  step "Step6D 多尺度打包（quanqu1；预计约 1.9 h，只写 2 个体 ≈24 GB）"
  $PY $ROOT/step6d_multiscale_bundle/code/build_step6d_multiscale_bundle.py \
    --config $ROOT/step6d_multiscale_bundle/configs/taigu_step6d_multiscale_quanqu1.json \
    2>&1 | tee "$LOG/step6d_quanqu1.log" || fail "step6d"
fi

# ------------------------------------------------------------------ 7A → 7D
if run_step 6; then
  step "Step7A 小尺度 DFN（quanqu1；按井经验分布重采样）"
  $PY $ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py \
    --config $ROOT/step7a_small_scale_dfn/configs/taigu_step7a_quanqu1.json \
    --replace-output 2>&1 | tee "$LOG/step7a_quanqu1.log" || fail "step7a"
fi

if run_step 7; then
  step "Step7B 中尺度 DFN（quanqu1）"
  $PY $ROOT/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py \
    --config $ROOT/step7b_multiscale_initial_dfn/configs/taigu_step7b_medium_v7_anttrack_led_quanqu1.json \
    2>&1 | tee "$LOG/step7b_quanqu1.log" || fail "step7b"
fi

if run_step 8; then
  step "Step7C 大尺度 DFN（quanqu1）"
  $PY $ROOT/step7c_large_fault_dfn/build_large_fault_dfn.py \
    --config $ROOT/step7c_large_fault_dfn/configs/taigu_step7c_large_v3_attribute_v3_quanqu1.json \
    2>&1 | tee "$LOG/step7c_quanqu1.log" || fail "step7c"
fi

if run_step 9; then
  step "Step7D 多尺度融合（quanqu1）"
  $PY $ROOT/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py \
    --config $ROOT/step7d_multiscale_fused_dfn/configs/taigu_step7d_fused_v1_quanqu1.json \
    2>&1 | tee "$LOG/step7d_quanqu1.log" || fail "step7d"
fi

# ------------------------------------------------------------------ Step8 / Step9
if run_step 10; then
  step "Step8 井控校正（quanqu1）"
  $PY $ROOT/step8_well_correction/correct_taigu_multiscale_dfn_with_well_controls.py \
    --config $ROOT/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1_quanqu1.json \
    2>&1 | tee "$LOG/step8_quanqu1.log" || fail "step8"
fi

if run_step 11; then
  step "Step9 剖面展示（quanqu1）"
  $PY $ROOT/step9_sections/build_taigu_multibackground_sections.py \
    --config $ROOT/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1.json \
    --replace-output 2>&1 | tee "$LOG/step9_quanqu1.log" || fail "step9"
fi

# ------------------------------------------------------------------ 收尾闸门
step "收尾闸门：口径 + 产状赋值（quanqu1）"
$PY $ROOT/tools/check_orientation_convention.py \
  --step7a-csv $ROOT/step7a_small_scale_dfn/output/taigu_step7a_attribute_v3_regen_full_quanqu1/fracture_patches.csv \
  --step7b-csv $ROOT/step7b_multiscale_initial_dfn/output/taigu_medium_v7_anttrack_led_quanqu1/medium_dfn_patches.csv \
  --step7c-csv $ROOT/step7c_large_fault_dfn/output/taigu_step7c_large_v3_attribute_v3_quanqu1/large_fault_dfn_patches.csv \
  --step8-csv  $ROOT/step8_well_correction/output/taigu_step8_attribute_multiscale_v1_quanqu1/well_corrected_dfn_fracture_patches.csv \
  --output $LOG/orientation_convention_qc_quanqu1.json \
  2>&1 | tee "$LOG/gate_orientation_convention_quanqu1.log" || fail "orientation convention qc"

$PY $ROOT/tools/check_orientation_assignment_qc.py \
  --tag quanqu1 \
  --output $LOG/orientation_assignment_qc_quanqu1.json \
  2>&1 | tee "$LOG/gate_orientation_assignment_quanqu1.log" || fail "orientation assignment qc"

echo; echo "[$(date '+%F %T')] ALL DONE (Step6A 预测 → Step9 + 闸门, quanqu1)"
echo "断点续跑：QUANQU1_START=<n>；n=1 6A预测 2 6A小背景 3 6B 4 6C 5 6D 6 7A 7 7B 8 7C 9 7D 10 Step8 11 Step9"

#!/usr/bin/env bash
# =============================================================================
# 2026-09-29 Step9 展示口径 v2（对齐砂砾岩）：正式重跑
#
#   1) 矿区/全区（quanqu1）→ **probeB**：变面积按米抽道 25 m（825/1650 条，50%）
#   2) 5 km（azi4）        → **全量**：变面积画剖面上全部道（约 400 条，12.5 m）
#
#   两者共同的展示口径：
#     * 地震变密度 = 黑白（|振幅| 上 gray_r，0=白/强振幅=黑）；红白蓝存 red_white_blue_backup/
#     * 画布 = 5280 x 1968 px @240 dpi（像素口径，数据区宽 5280 px）
#     * 变面积 = 线宽 0.35 / 正瓣填充 alpha 0.55
#     * 全 NaN 道（剖面伸出 OBN 覆盖）不画
#
#   正式重跑**不复用采样缓存**（要 real sample QC）；约 30 min + 15 min。
#
#   用法：
#     tmux new-session -d -s taigu_step9_v2 \
#       "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#        bash 太古界/run_step9_display_v2.sh 2>&1 | tee 太古界/output_quanqu1_logs/step9_display_v2.log"
# =============================================================================
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_quanqu1_logs
PY="${STEP9_PY:-/home/tyh/anaconda3/envs/gan-dfn/bin/python}"
SCRIPT="$ROOT/step9_sections/build_taigu_multibackground_sections.py"
mkdir -p "$LOG"

run_one() {
  local label="$1" cfg="$2" out="$3"
  echo; echo "[$(date '+%F %T')] ===== ${label} ====="
  "$PY" "$SCRIPT" --config "$cfg" --replace-output 2>&1 | tee "$LOG/$out" \
    || { echo "[$(date '+%F %T')] FAILED: ${label}"; exit 1; }
}

run_one "Step9 quanqu1 矿区（probeB：按米抽道 25 m）" \
  "$ROOT/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1.json" \
  "step9_quanqu1_v2display.log"

run_one "Step9 azi4 5 km（全量：全部道）" \
  "$ROOT/step9_sections/configs/taigu_step9_multibackground_v2_azi4.json" \
  "step9_azi4_v2display.log"

echo; echo "[$(date '+%F %T')] DONE：quanqu1 + azi4 展示口径 v2 重跑完成"

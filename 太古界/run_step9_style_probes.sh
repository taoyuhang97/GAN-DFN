#!/usr/bin/env bash
# 2026-09-29：Step9 展示口径对照小样（配色/抽道/图幅）快速迭代。
#
# 复用正式 quanqu1 的 section_samples NPZ 缓存（--reuse-samples），只重画图，
# 三组配置各出 2–4 张 overview 地震剖面，供选参数用。
#
# 用法：
#   tmux new-session -d -s taigu_step9_probe \
#     "cd /home/tyh/projects/petroleum/code/GAN-DFN && \
#      bash 太古界/run_step9_style_probes.sh 2>&1 | tee 太古界/output_quanqu1_logs/step9_style_probe.log"
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
LOG=$ROOT/output_quanqu1_logs
PY="${STEP9_PY:-/home/tyh/anaconda3/envs/gan-dfn/bin/python}"
mkdir -p "$LOG"

for probe in probeA probeB probeC; do
  cfg="$ROOT/step9_sections/configs/taigu_step9_multibackground_v2_quanqu1_${probe}.json"
  echo; echo "[$(date '+%F %T')] ===== ${probe} ====="
  "$PY" "$ROOT/step9_sections/build_taigu_multibackground_sections.py" \
    --config "$cfg" --reuse-samples --replace-output \
    2>&1 | tee "$LOG/step9_${probe}.log" || { echo "[$(date '+%F %T')] FAILED: $probe"; exit 1; }
done

echo; echo "[$(date '+%F %T')] 三组小样完成：$ROOT/step9_sections/output/_step9_style_probe/"

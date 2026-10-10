#!/usr/bin/env bash
# F1/F2/F3 冒烟验证：用与 md1 **完全相同**的 demo 配置跑新代码，
# 与 md1 旧产物逐位比较（数值中性验证），并单独验证 part 模式 + 拼装。
#
#   用法:
#     tmux new-session -d -s taigu_smoke "cd <repo> && bash 太古界/tools/smoke_step6a_f123.sh 2>&1 | tee 太古界/output_quanqu1_logs/smoke_f123.log"
#
#   产物写到 /tmp/quanqu1_smoke/，验证完成后按需求方要求删除（脚本末尾不自动删，
#   由调用者确认后删除；派生产物不留在仓库里）。
set -uo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT=太古界
PY=python3
SMOKE=/tmp/quanqu1_smoke
REF=$ROOT/step6a_density_volume/output/taigu_step6a_attribute_v3_full_md1/volume
CFG=$ROOT/step6a_density_volume/configs/taigu_step6a_predict_attribute_v3_md1.json

fail() { echo "[$(date '+%F %T')] FAILED: $1"; exit 1; }
step() { echo; echo "[$(date '+%F %T')] ===== $1 ====="; }

mkdir -p "$SMOKE"

step "① 整卷（新代码：F1+F2）"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config "$CFG" --output-dir "$SMOKE/full" --replace-output \
  2>&1 | tail -5 || fail "full run"

step "② 与 md1 旧产物逐位比较（等价性）"
$PY $ROOT/tools/compare_step6a_runs.py --reference "$REF" --candidate "$SMOKE/full" \
  || fail "bitwise compare vs md1"

step "③ part 模式：1-2 与 3-21 两个 part"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config "$CFG" --output-dir "$SMOKE/part_a" --block-start 1 --block-end 2 --replace-output \
  2>&1 | tail -3 || fail "part_a"
$PY $ROOT/step6a_density_volume/predict_taigu_density_volume.py \
  --config "$CFG" --output-dir "$SMOKE/part_b" --block-start 3 --block-end 21 --replace-output \
  2>&1 | tail -3 || fail "part_b"

step "④ 拼装并与整卷比较"
$PY $ROOT/tools/assemble_step6a_parts.py --parts-root "$SMOKE" --output-dir "$SMOKE/assembled" \
  2>&1 | tail -3 || fail "assemble"
$PY $ROOT/tools/compare_step6a_runs.py --reference "$SMOKE/full" --candidate "$SMOKE/assembled" \
  || fail "bitwise compare assembled vs full"

echo; echo "[$(date '+%F %T')] SMOKE ALL PASS（F1/F2 数值中性 + F3 拼装与整卷一致）"
echo "冒烟产物在 $SMOKE（确认后删除）"

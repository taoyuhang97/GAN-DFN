#!/usr/bin/env bash
# 太古界 Step5-Step9 v2 全链验证（tmux 后台执行）
# 用法: tmux new-session -d -s taigu_v2 'bash 太古界/run_taigu_step5_to_step9_v2.sh'
set -euo pipefail

cd /home/tyh/projects/petroleum/code/GAN-DFN
ROOT="太古界"
LOGDIR="$ROOT/output_v2_logs"
mkdir -p "$LOGDIR"

echo "[flow] $(date '+%F %T') Step5 v2 start"
python3 "$ROOT/step5_virtual_wells/build_taigu_virtual_wells.py" \
  --config "$ROOT/step5_virtual_wells/configs/taigu_step5_v1.json" \
  --replace-output > "$LOGDIR/step5.log" 2>&1
echo "[flow] $(date '+%F %T') Step5 v2 done"

echo "[flow] $(date '+%F %T') Step6A train v2 start"
OMP_NUM_THREADS=16 python3 "$ROOT/step6a_density_volume/train_taigu_two_stage_density.py" \
  --config "$ROOT/step6a_density_volume/configs/taigu_step6a_train_v1.json" \
  --replace-output > "$LOGDIR/step6a_train.log" 2>&1
echo "[flow] $(date '+%F %T') Step6A train v2 done"

echo "[flow] $(date '+%F %T') Step6A predict v2 start"
python3 "$ROOT/step6a_density_volume/predict_taigu_density_volume.py" \
  --config "$ROOT/step6a_density_volume/configs/taigu_step6a_predict_v1.json" \
  --replace-output > "$LOGDIR/step6a_predict.log" 2>&1
echo "[flow] $(date '+%F %T') Step6A predict v2 done"

echo "[flow] $(date '+%F %T') Step7A v2 start"
python3 "$ROOT/step7a_small_scale_dfn/build_taigu_small_scale_dfn.py" \
  --config "$ROOT/step7a_small_scale_dfn/configs/taigu_step7a_v1.json" \
  --replace-output > "$LOGDIR/step7a.log" 2>&1
echo "[flow] $(date '+%F %T') Step7A v2 done"

echo "[flow] $(date '+%F %T') Step8 v2 start"
python3 "$ROOT/step8_well_correction/correct_taigu_dfn_with_well_controls.py" \
  --config "$ROOT/step8_well_correction/configs/taigu_step8_v1.json" \
  --replace-output > "$LOGDIR/step8.log" 2>&1
echo "[flow] $(date '+%F %T') Step8 v2 done"

echo "[flow] $(date '+%F %T') Step9 v2 start"
python3 "$ROOT/step9_sections/build_taigu_well_sections.py" \
  --config "$ROOT/step9_sections/configs/taigu_step9_v1.json" \
  --replace-output > "$LOGDIR/step9.log" 2>&1
echo "[flow] $(date '+%F %T') Step9 v2 done"

echo "[flow] $(date '+%F %T') verify start"
python3 - <<'PYEOF'
import json, os
root = "/home/tyh/projects/petroleum/code/GAN-DFN"
checks = [
    ("Step5",           "太古界/step5_virtual_wells/output/taigu_step5_v1/taigu_step5_acceptance_summary.json"),
    ("Step6A train",    "太古界/step6a_density_volume/output/taigu_step6a_v1/models/step6_acceptance_summary.json"),
    ("Step6A volume",   "太古界/step6a_density_volume/output/taigu_step6a_v1/volume/prediction_summary.json"),
    ("Step7A",          "太古界/step7a_small_scale_dfn/output/taigu_step7a_v1/step7a_summary.json"),
    ("Step8",           "太古界/step8_well_correction/output/taigu_step8_v1/step8_summary.json"),
    ("Step9",           "太古界/step9_sections/output/taigu_step9_v1/step9_summary.json"),
]
ok = True
for name, rel in checks:
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        print(f"{name:16s} MISSING {rel}")
        ok = False
        continue
    status = json.load(open(path)).get("status")
    print(f"{name:16s} status={status}")
    ok = ok and status == "pass"
print("ALL PASS" if ok else "SOME FAILED")
raise SystemExit(0 if ok else 1)
PYEOF
echo "[flow] $(date '+%F %T') verify done (exit=$?)"

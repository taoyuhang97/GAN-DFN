#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/configs/formal_mine_curvature_led_2ms_v1.json"
TRAINER="${SCRIPT_DIR}/train_step6_two_stage_density_models.py"
OUTPUT_DIR="${SCRIPT_DIR}/output/formal_mine_curvature_led_2ms_v1/models"
LOG_ROOT="${PROJECT_ROOT}/优化阶段二/正式主线/logs/step6_mine_curvature_led_2ms_v1"
RUN_STAMP="${STEP6_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_ROOT}/step6_model_training_${RUN_STAMP}.log"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[step6-runner] started_at=$(date --iso-8601=seconds)"
echo "[step6-runner] config=${CONFIG}"
echo "[step6-runner] output_dir=${OUTPUT_DIR}"
echo "[step6-runner] log=${LOG_PATH}"

python -m py_compile "${TRAINER}"
python - <<'PY' "${CONFIG}" "${OUTPUT_DIR}"
import json
import shutil
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
config = json.loads(config_path.read_text(encoding="utf-8"))
input_csv = Path(config["unified_samples_csv"])
if not input_csv.exists():
    raise SystemExit(f"missing Step5B input: {input_csv}")
expected = {
    output_dir / "step6_two_stage_density_models.joblib",
    output_dir / "step6_two_stage_training_summary.json",
    output_dir / "step6_two_stage_grouped_validation.csv",
}
existing = sorted(str(path) for path in expected if path.exists())
if existing:
    raise SystemExit("refusing to overwrite existing formal Step6 model outputs: " + ", ".join(existing))
free_gib = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else config_path.parent).free / 1024**3
print(f"[step6-runner] step5b_bytes={input_csv.stat().st_size} free_gib={free_gib:.2f}")
if free_gib < 10.0:
    raise SystemExit(f"insufficient free disk for model training: {free_gib:.2f} GiB")
PY

python "${TRAINER}" --config "${CONFIG}"

python - <<'PY' "${OUTPUT_DIR}"
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = output_dir / "step6_two_stage_training_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "pass":
    raise SystemExit(f"Step6 training status is not pass: {summary.get('status')}")
checks = summary.get("checks", {})
failed = sorted(key for key, value in checks.items() if not value)
if failed:
    raise SystemExit("Step6 training failed checks: " + ", ".join(failed))
print("[step6-runner] validation_passed=true")
for layer, payload in summary["layer_summaries"].items():
    print(
        f"[step6-runner] layer={layer} rows={payload['rows']} positives={payload['positive_rows']} "
        f"mean_auc={payload['mean_fold_roc_auc']:.6f} mean_ap={payload['mean_fold_average_precision']:.6f}"
    )
PY

echo "[step6-runner] finished_at=$(date --iso-8601=seconds)"
echo "[step6-runner] status=pass"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${SCRIPT_DIR}/configs/formal_demo_10km_curvature_led_2ms_v1.json"
PREDICTOR="${SCRIPT_DIR}/predict_step6_two_stage_density_volume.py"
DIAGNOSTICS="${SCRIPT_DIR}/build_step6_density_diagnostics.py"
OUTPUT_DIR="${SCRIPT_DIR}/output/formal_demo_10km_curvature_led_2ms_v1/base_density"
LOG_DIR="${PROJECT_ROOT}/优化阶段二/正式主线/logs/step6_demo_10km_curvature_led_2ms_v1"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/step6_demo_10km_density_${RUN_STAMP}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[step6-run] started_at=$(date --iso-8601=seconds)"
echo "[step6-run] project_root=${PROJECT_ROOT}"
echo "[step6-run] config=${CONFIG}"
echo "[step6-run] output_dir=${OUTPUT_DIR}"
echo "[step6-run] log_file=${LOG_FILE}"

"${PYTHON_BIN}" -m py_compile "${PREDICTOR}" "${DIAGNOSTICS}"
"${PYTHON_BIN}" -m json.tool "${CONFIG}" >/dev/null

"${PYTHON_BIN}" - "${CONFIG}" "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
required = [
    Path(config["model_joblib"]),
    Path(config["trace_header_csv"]),
    Path(config["layer_dir"]),
    Path(config["source_sgy_for_headers"]),
    *(Path(value) for value in config["volume_paths"].values()),
    Path(config["diagnostics"]["well_main_csv"]),
    *(Path(value) for value in config["diagnostics"]["step3_group_csvs"]),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("missing required Step6 inputs:\n" + "\n".join(missing))

formal_outputs = [
    output_dir / "predicted_fracture_density.sgy",
    output_dir / "predicted_fracture_density.sgy.partial",
    output_dir / "trace_mapping.npz",
    output_dir / "prediction_block_qc.csv",
    output_dir / "prediction_summary.json",
]
existing = [str(path) for path in formal_outputs if path.exists()]
if existing:
    raise FileExistsError("formal output already exists; refusing overwrite:\n" + "\n".join(existing))

free_bytes = shutil.disk_usage(output_dir.parent).free
minimum_bytes = 8 * 1024**3
if free_bytes < minimum_bytes:
    raise RuntimeError(f"insufficient free space: {free_bytes / 1024**3:.2f} GiB < 8 GiB")
print(f"[step6-run] preflight=pass free_space_gib={free_bytes / 1024**3:.2f}")
PY

"${PYTHON_BIN}" "${PREDICTOR}" --config "${CONFIG}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/prediction_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "status": "pass",
    "trace_count": 641601,
    "x_line_count": 801,
    "y_line_count": 801,
    "sample_interval_ms": 2.0,
}
actual = {
    "status": summary.get("status"),
    "trace_count": summary["grid"]["trace_count"],
    "x_line_count": summary["grid"]["x_line_count"],
    "y_line_count": summary["grid"]["y_line_count"],
    "sample_interval_ms": summary["sample_axis"]["sample_interval_ms"],
}
if actual != expected or not all(summary.get("checks", {}).values()):
    raise RuntimeError(f"prediction acceptance failed: actual={actual} expected={expected}")
print(f"[step6-run] prediction_acceptance=pass sample_count={summary['sample_axis']['sample_count']}")
PY

"${PYTHON_BIN}" "${DIAGNOSTICS}" --config "${CONFIG}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}/diagnostics/diagnostic_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
if summary.get("status") != "pass" or len(summary.get("images", [])) != 6:
    raise RuntimeError(f"diagnostic acceptance failed: {summary}")
if not all(summary.get("checks", {}).values()):
    raise RuntimeError(f"diagnostic checks failed: {summary['checks']}")
print(f"[step6-run] diagnostic_acceptance=pass valid_well_samples={summary['well_prediction_valid_count']}")
PY

echo "[step6-run] completed_at=$(date --iso-8601=seconds)"
echo "[step6-run] status=pass"

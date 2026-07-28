#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MAIN_DIR}/../.." && pwd)"
STEP5B_DIR="${MAIN_DIR}/step5b_unified_samples_t4_t7"
PYTHON_BIN="${PYTHON_BIN:-python}"

STEP5A_CONFIG="${SCRIPT_DIR}/configs/formal_single_source_virtual_wells.json"
STEP5B_CONFIG="${STEP5B_DIR}/configs/formal_unified_t4_t7_density_samples.json"
STEP5A_OUTPUT="${SCRIPT_DIR}/output/formal_curvature_led_v2"
STEP5B_OUTPUT="${STEP5B_DIR}/output/formal_curvature_led_v2"
LOG_DIR="${MAIN_DIR}/logs/step5_curvature_led_v2"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/step5_curvature_led_v2_${RUN_ID}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[$(date --iso-8601=seconds)] Step5 curvature-led v2 pipeline started"
echo "repo_root=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "step5a_output=${STEP5A_OUTPUT}"
echo "step5b_output=${STEP5B_OUTPUT}"
echo "log_path=${LOG_PATH}"

available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
required_kb=$((10 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: available disk is below 10 GiB: ${available_kb} KiB"
  exit 1
fi
echo "available_disk_kb=${available_kb}"

"${PYTHON_BIN}" -m py_compile \
  "${SCRIPT_DIR}/build_single_source_virtual_wells.py" \
  "${STEP5B_DIR}/build_unified_t4_t7_density_samples.py"
"${PYTHON_BIN}" -m json.tool "${STEP5A_CONFIG}" >/dev/null
"${PYTHON_BIN}" -m json.tool "${STEP5B_CONFIG}" >/dev/null
echo "[$(date --iso-8601=seconds)] Preflight checks passed"

echo "[$(date --iso-8601=seconds)] Step5A full build started"
/usr/bin/time -f 'STEP5A_ELAPSED_SEC=%e STEP5A_CPU_SEC=%U STEP5A_MAX_RSS_KB=%M' \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_single_source_virtual_wells.py" \
  --config "${STEP5A_CONFIG}" \
  --progress-every 25
echo "[$(date --iso-8601=seconds)] Step5A build completed; validation started"

"${PYTHON_BIN}" - "${STEP5A_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
index = pd.read_csv(root / "virtual_well_index.csv")
samples = pd.read_csv(root / "virtual_well_training_samples.csv")
audit = json.loads((root / "virtual_well_build_audit.json").read_text(encoding="utf-8"))
expected_columns = [
    "SourceSampleID", "SourceWellName", "VirtualWellName", "X", "Y", "TIME",
    "StrataName", "PresenceLabel", "DensityLabel", "PointConfidence", "SampleWeight",
    "SeisAmp", "Coherence", "AntTrack", "CurvatureMax",
]
checks = {
    "compact_15_column_contract": samples.columns.tolist() == expected_columns,
    "25_virtual_tracks_per_source": index.groupby("SourceWellName").size().eq(25).all(),
    "one_center_virtual_track_per_source": index.groupby("SourceWellName")["IsCenterVirtualTrace"].sum().eq(1).all(),
    "training_rows_match_audit": len(samples) == int(audit["training_sample_count"]),
    "presence_complete": samples["PresenceLabel"].notna().all(),
    "positive_density_complete": samples.loc[samples["PresenceLabel"].eq(1), "DensityLabel"].notna().all(),
    "negative_density_null": samples.loc[samples["PresenceLabel"].eq(0), "DensityLabel"].isna().all(),
    "curvature_complete": pd.to_numeric(samples["CurvatureMax"], errors="coerce").notna().all(),
    "curvature_pos_absent": "CurvaturePos" not in samples.columns,
    "confidence_bounded": pd.to_numeric(samples["PointConfidence"], errors="coerce").between(0, 1).all(),
    "child_weight_bounded": samples.groupby(["SourceWellName", "SourceSampleID"])["SampleWeight"].sum().le(1.0 + 1.0e-9).all(),
    "38_source_wells": samples["SourceWellName"].nunique() == 38,
}
failed = [name for name, passed in checks.items() if not bool(passed)]
print(json.dumps({
    "stage": "Step5A",
    "status": "pass" if not failed else "fail",
    "checks": {name: bool(value) for name, value in checks.items()},
    "rows": int(len(samples)),
    "source_wells": int(samples["SourceWellName"].nunique()),
    "virtual_tracks": int(index["VirtualWellName"].nunique()),
    "candidate_rows": int(audit["candidate_sample_count"]),
    "unknown_rejected_rows": int(audit["unknown_rejected_count"]),
}, ensure_ascii=False, indent=2))
if failed:
    raise SystemExit(f"Step5A validation failed: {failed}")
PY
echo "[$(date --iso-8601=seconds)] Step5A validation passed"

echo "[$(date --iso-8601=seconds)] Step5B full build started"
/usr/bin/time -f 'STEP5B_ELAPSED_SEC=%e STEP5B_CPU_SEC=%U STEP5B_MAX_RSS_KB=%M' \
  "${PYTHON_BIN}" "${STEP5B_DIR}/build_unified_t4_t7_density_samples.py" \
  --config "${STEP5B_CONFIG}"
echo "[$(date --iso-8601=seconds)] Step5B build completed; validation started"

"${PYTHON_BIN}" - "${STEP5B_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
summary = json.loads((root / "unified_t4_t7_density_samples_summary.json").read_text(encoding="utf-8"))
expected_columns = [
    "SourceKind", "SourceWellName", "TrackWellName", "X", "Y", "TIME", "LayerGroup",
    "PresenceLabel", "DensityLabel", "HasFracture", "PointConfidence", "SampleWeight",
    "SeisAmp", "Coherence", "AntTrack", "CurvatureMax",
]
csv_path = root / "unified_t4_t7_density_samples.csv"
header = pd.read_csv(csv_path, nrows=0).columns.tolist()
counts = {"rows": 0, "real_rows": 0, "virtual_rows": 0, "real_wells": set()}
checks = {
    "summary_pass": summary.get("status") == "pass",
    "compact_16_column_contract": header == expected_columns,
    "curvature_pos_absent": "CurvaturePos" not in header,
    "presence_complete": True,
    "positive_density_complete": True,
    "negative_density_null": True,
    "curvature_complete": True,
    "weights_nonnegative": True,
}
for chunk in pd.read_csv(csv_path, chunksize=250000, low_memory=False):
    presence = pd.to_numeric(chunk["PresenceLabel"], errors="coerce")
    density = pd.to_numeric(chunk["DensityLabel"], errors="coerce")
    weights = pd.to_numeric(chunk["SampleWeight"], errors="coerce")
    curvature = pd.to_numeric(chunk["CurvatureMax"], errors="coerce")
    counts["rows"] += int(len(chunk))
    real = chunk["SourceKind"].eq("real_well")
    virtual = chunk["SourceKind"].eq("virtual_well")
    counts["real_rows"] += int(real.sum())
    counts["virtual_rows"] += int(virtual.sum())
    counts["real_wells"].update(chunk.loc[real, "SourceWellName"].dropna().astype(str).unique())
    checks["presence_complete"] &= bool(presence.notna().all())
    checks["positive_density_complete"] &= bool(density[presence.eq(1)].notna().all())
    checks["negative_density_null"] &= bool(density[presence.eq(0)].isna().all())
    checks["curvature_complete"] &= bool(curvature.notna().all())
    checks["weights_nonnegative"] &= bool(weights.notna().all() and weights.ge(0).all())
checks["38_real_wells"] = len(counts["real_wells"]) == 38
checks["real_and_virtual_present"] = counts["real_rows"] > 0 and counts["virtual_rows"] > 0
checks["rows_match_summary"] = counts["rows"] == int(summary["summary"]["total_rows"])
checks["virtual_weight_bounded"] = bool(summary["checks"].get("virtual_weight_bounded"))
failed = [name for name, passed in checks.items() if not bool(passed)]
print(json.dumps({
    "stage": "Step5B",
    "status": "pass" if not failed else "fail",
    "checks": {name: bool(value) for name, value in checks.items()},
    "rows": counts["rows"],
    "real_rows": counts["real_rows"],
    "virtual_rows": counts["virtual_rows"],
    "real_wells": len(counts["real_wells"]),
    "virtual_to_real_weight_ratio": summary.get("virtual_to_real_weight_ratio"),
}, ensure_ascii=False, indent=2))
if failed:
    raise SystemExit(f"Step5B validation failed: {failed}")
PY

echo "[$(date --iso-8601=seconds)] Step5B validation passed"
echo "[$(date --iso-8601=seconds)] FINAL_STATUS=pass"

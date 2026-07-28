#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${FORMAL_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="formal_demo_10km_multiscale_flow_v1"
RESUME_COMPLETED="${RESUME_COMPLETED:-0}"
MASTER_CONFIG="${SCRIPT_DIR}/configs/${VERSION}.json"
BASE_CONFIG="${SCRIPT_DIR}/configs/formal_demo_10km_curvature_led_2ms_v1.json"
STEP6_CONFIG="${SCRIPT_DIR}/configs/${VERSION}_expanded.json"
STEP6_ROOT="${SCRIPT_DIR}/output/${VERSION}"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_${RUN_STAMP}.log"

STEP7A_CONFIG="${FORMAL_ROOT}/step7a_small_scale_dfn/configs/${VERSION}.json"
STEP7B_CONFIG="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/configs/${VERSION}.json"
STEP7C_CONFIG="${FORMAL_ROOT}/step7c_large_fault_dfn/configs/${VERSION}.json"
STEP7D_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/${VERSION}.json"
STEP8_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/${VERSION}.json"
STEP9_CONFIG="${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"

mkdir -p "${LOG_DIR}" "${STEP6_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[flow] started_at=$(date --iso-8601=seconds)"
echo "[flow] version=${VERSION}"
echo "[flow] log=${LOG_FILE}"

"${PYTHON_BIN}" -m py_compile \
  "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" \
  "${SCRIPT_DIR}/build_step6a_small_background.py" \
  "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
  "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
  "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
  "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" \
  "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_demo_10km_flow_configs.py" --master-config "${MASTER_CONFIG}"

"${PYTHON_BIN}" - "${MASTER_CONFIG}" "${STEP6_ROOT}" "${RESUME_COMPLETED}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
resume_completed = sys.argv[3] == "1"
required = [
    Path(config["input_density_sgy"]),
    Path(config["trace_mapping_npz"]),
    *(Path(value) for value in config["volume_paths"].values()),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("missing flow inputs:\n" + "\n".join(missing))
weights = config["small_evidence"]
if abs(float(weights["density_weight"]) - 0.5) > 1.0e-12 or abs(float(weights["curvature_weight"]) - 0.5) > 1.0e-12:
    raise ValueError(f"unexpected Step6A weights: {weights}")
formal_markers = [
    root / "step6a_small/small_background_qc.json",
    root / "step6b_medium/medium_corridor_qc.json",
    root / "step6c_large/large_fault_qc.json",
    root / "step6d_bundle/multiscale_bundle_summary.json",
]
existing = [str(path) for path in formal_markers if path.exists()]
if existing and not resume_completed:
    raise FileExistsError("refusing to mix with existing completed stage outputs:\n" + "\n".join(existing))
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 30.0:
    raise RuntimeError(f"insufficient free space for compact 10 km flow: {free_gib:.1f} GiB")
print(
    f"[flow] preflight=pass free_space_gib={free_gib:.1f} "
    f"step6a_weights=0.5/0.5 resume_completed={resume_completed}"
)
PY

if [[ "${RESUME_COMPLETED}" == "1" ]] && [[ -f "${STEP6_ROOT}/input_qc/input_qc_summary.json" ]]; then
  echo "[flow] stage=input_qc action=skip_existing"
else
  echo "[flow] stage=input_qc"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/qc_multiscale_rebalance_inputs.py" \
    --base-config "${BASE_CONFIG}" \
    --multiscale-config "${STEP6_CONFIG}" \
    --output-dir "${STEP6_ROOT}/input_qc"
fi

if [[ "${RESUME_COMPLETED}" == "1" ]] && [[ -f "${STEP6_ROOT}/step6a_small/small_background_qc.json" ]]; then
  "${PYTHON_BIN}" - "${STEP6_ROOT}/step6a_small/small_background_qc.json" "${STEP6_ROOT}/step6a_small/small_background_score.sgy" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
score_path = Path(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
weights = payload.get("small_evidence", {})
if payload.get("status") != "pass":
    raise RuntimeError(f"cannot resume from failed Step6A QC: {path}")
if abs(float(weights.get("density_weight", -1.0)) - 0.5) > 1.0e-12:
    raise RuntimeError(f"resume Step6A density weight mismatch: {weights}")
if abs(float(weights.get("curvature_weight", -1.0)) - 0.5) > 1.0e-12:
    raise RuntimeError(f"resume Step6A curvature weight mismatch: {weights}")
if not score_path.exists() or score_path.stat().st_size <= 0:
    raise FileNotFoundError(f"resume Step6A score SGY missing or empty: {score_path}")
PY
  echo "[flow] stage=step6a action=skip_existing status=pass weights=0.5/0.5"
else
  echo "[flow] stage=step6a"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6a_small_background.py" \
    --config "${STEP6_CONFIG}" \
    --output-dir "${STEP6_ROOT}/step6a_small" \
    --candidate-q 0.88 \
    --core-q 0.95
fi

echo "[flow] stage=step6b"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6b_medium_corridor_prior.py" \
  --config "${STEP6_CONFIG}" \
  --output-dir "${STEP6_ROOT}/step6b_medium" \
  --enable-lowcoh-vertical-branch \
  --min-component-voxels 120 \
  --max-component-voxels-before-split 12000 \
  --split-time-samples 8

echo "[flow] stage=step6c"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6c_large_fault_prior.py" \
  --config "${STEP6_CONFIG}" \
  --input-qc-dir "${STEP6_ROOT}/input_qc" \
  --output-dir "${STEP6_ROOT}/step6c_large" \
  --inferred-extraction-mode component_tiles \
  --min-component-voxels 40 \
  --max-component-voxels-before-split 25000 \
  --split-time-samples 20

echo "[flow] stage=step6d"
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_step6d_multiscale_bundle.py" \
  --config "${STEP6_CONFIG}" \
  --rebalance-root "${STEP6_ROOT}" \
  --output-dir "${STEP6_ROOT}/step6d_bundle"

echo "[flow] stage=step7a"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7a_small_scale_dfn/build_small_scale_dfn.py" --config "${STEP7A_CONFIG}"
echo "[flow] stage=step7b"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" --config "${STEP7B_CONFIG}"
echo "[flow] stage=step7c"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" --config "${STEP7C_CONFIG}"
echo "[flow] stage=step7d"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" --config "${STEP7D_CONFIG}"

echo "[flow] stage=step8"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" --config "${STEP8_CONFIG}"

echo "[flow] stage=step9"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" --config "${STEP9_CONFIG}"

"${PYTHON_BIN}" - "${FORMAL_ROOT}" "${STEP6_ROOT}" "${VERSION}" <<'PY'
import json
import sys
from pathlib import Path

formal = Path(sys.argv[1]); step6 = Path(sys.argv[2]); version = sys.argv[3]
summaries = [
    step6 / "input_qc/input_qc_summary.json",
    step6 / "step6a_small/small_background_qc.json",
    step6 / "step6b_medium/medium_corridor_qc.json",
    step6 / "step6c_large/large_fault_qc.json",
    step6 / "step6d_bundle/multiscale_bundle_summary.json",
    formal / f"step7a_small_scale_dfn/output/{version}/small_dfn_summary.json",
    formal / f"step7b_multiscale_initial_dfn/output/{version}/medium_dfn_summary.json",
    formal / f"step7c_large_fault_dfn/output/{version}/large_fault_dfn_summary.json",
    formal / f"step7d_multiscale_fused_dfn/output/{version}/fused_multiscale_summary.json",
    formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_summary.json",
    formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections/section_summary.json",
]
required_outputs = [
    step6 / "step6a_small/small_background_score.sgy",
    step6 / "step6b_medium/medium_corridor_prior.sgy",
    step6 / "step6b_medium/medium_corridor_components.npz",
    step6 / "step6b_medium/medium_corridor_component_summary.csv",
    step6 / "step6c_large/large_fault_prior.sgy",
    step6 / "step6c_large/large_fault_prior_components.npz",
    step6 / "step6c_large/large_fault_component_summary.csv",
    step6 / "step6d_bundle/multiscale_damage_context_10ms.npz",
]
missing_outputs = [str(path) for path in required_outputs if not path.exists() or path.stat().st_size <= 0]
if missing_outputs:
    raise FileNotFoundError("missing compact flow outputs:\n" + "\n".join(missing_outputs))
rows = []
for path in summaries:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status", "pass")
    if status != "pass":
        raise RuntimeError(f"stage summary failed: {path} status={status}")
    rows.append({"path": str(path), "status": status})
for stage_name, path in {
    "step6b": step6 / "step6b_medium/medium_corridor_qc.json",
    "step6c": step6 / "step6c_large/large_fault_qc.json",
    "step6d": step6 / "step6d_bundle/multiscale_bundle_summary.json",
}.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    interval = payload.get("sample_interval_ms")
    if interval is None and stage_name == "step6d":
        interval = payload.get("sample_interval_ms")
    if abs(float(interval) - 10.0) > 1.0e-6:
        raise RuntimeError(f"{stage_name} expected 10 ms output, got {interval}")
image_dir = formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections"
pngs = sorted(image_dir.glob("*.png"))
if len(pngs) != 20 or any(path.stat().st_size < 10000 for path in pngs):
    raise RuntimeError(f"Step9 expected 20 nonempty PNGs, found {len(pngs)}")
acceptance = {"status": "pass", "summaries": rows, "image_count": len(pngs), "images": [path.name for path in pngs]}
(step6 / "flow_acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[flow] acceptance=pass images={len(pngs)}")
PY

echo "[flow] completed_at=$(date --iso-8601=seconds)"
echo "[flow] status=pass"

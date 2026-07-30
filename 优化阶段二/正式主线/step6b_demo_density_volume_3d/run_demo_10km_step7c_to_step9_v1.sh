#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORMAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="${VERSION:-formal_demo_10km_multiscale_flow_v2}"
STEP6_ROOT="${SCRIPT_DIR}/output/${VERSION}"
LOG_DIR="${FORMAL_ROOT}/logs/${VERSION}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${VERSION}_step7c_to_step9_${RUN_STAMP}.log"

STEP7A_ROOT="${FORMAL_ROOT}/step7a_small_scale_dfn/output/${VERSION}"
STEP7B_ROOT="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/output/${VERSION}"
STEP7C_CONFIG="${FORMAL_ROOT}/step7c_large_fault_dfn/configs/${VERSION}.json"
STEP7D_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/${VERSION}.json"
STEP8_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/${VERSION}.json"
STEP9_CONFIG="${FORMAL_ROOT}/step9_section_visualize/configs/${VERSION}.json"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[continuation] started_at=$(date --iso-8601=seconds)"
echo "[continuation] version=${VERSION}"
echo "[continuation] log=${LOG_FILE}"

"${PYTHON_BIN}" -m py_compile \
  "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" \
  "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
  "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py"

"${PYTHON_BIN}" - "${STEP6_ROOT}" "${STEP7A_ROOT}" "${STEP7B_ROOT}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

step6 = Path(sys.argv[1])
step7a = Path(sys.argv[2])
step7b = Path(sys.argv[3])
summaries = {
    "step6c": step6 / "step6c_large/large_fault_qc.json",
    "step7a": step7a / "small_dfn_summary.json",
    "step7b": step7b / "medium_dfn_summary.json",
}
for name, path in summaries.items():
    if not path.exists():
        raise FileNotFoundError(f"missing continuation input summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"continuation input {name} is not passing: {path}")
required = [
    step6 / "step6c_large/large_fault_prior_components.npz",
    step6 / "step6c_large/large_fault_component_summary.csv",
    step6 / "step6c_large/original_fault_units_demo_raw_time.vtk",
    step6 / "step6c_large/original_fault_unit_manifest.csv",
    step7a / "small_dfn_patches.csv",
    step7b / "medium_dfn_patches.csv",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size <= 0]
if missing:
    raise FileNotFoundError("missing continuation inputs:\n" + "\n".join(missing))
free_gib = shutil.disk_usage(step6).free / 1024**3
if free_gib < 20.0:
    raise RuntimeError(f"insufficient project disk space: {free_gib:.1f} GiB")
print(f"[continuation] preflight=pass free_space_gib={free_gib:.1f}")
PY

echo "[continuation] stage=step7c"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7c_large_fault_dfn/build_large_fault_dfn.py" --config "${STEP7C_CONFIG}"

echo "[continuation] stage=step7d"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" --config "${STEP7D_CONFIG}"

echo "[continuation] stage=step8"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" --config "${STEP8_CONFIG}"

echo "[continuation] stage=step9"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_dfn_multibackground_sections.py" --config "${STEP9_CONFIG}"

"${PYTHON_BIN}" - "${FORMAL_ROOT}" "${STEP6_ROOT}" "${VERSION}" <<'PY'
import json
import sys
from pathlib import Path

formal = Path(sys.argv[1])
step6 = Path(sys.argv[2])
version = sys.argv[3]
summaries = [
    formal / f"step7c_large_fault_dfn/output/{version}/large_fault_dfn_summary.json",
    formal / f"step7d_multiscale_fused_dfn/output/{version}/fused_multiscale_summary.json",
    formal / f"step8_dfn_well_correction/output/{version}/well_corrected_dfn_summary.json",
    formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections/section_summary.json",
]
rows = []
for path in summaries:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise RuntimeError(f"continuation stage failed: {path}")
    rows.append({"path": str(path), "status": "pass"})
step7c_dir = formal / f"step7c_large_fault_dfn/output/{version}"
step7c_products = [
    step7c_dir / "original_fault_units_demo_raw_time.vtk",
    step7c_dir / "original_fault_units_demo_raw_time.vtp",
    step7c_dir / "original_fault_unit_manifest.csv",
    step7c_dir / "large_inferred_fault_surfaces_raw_time.vtk",
    step7c_dir / "large_inferred_fault_surfaces_raw_time.vtp",
    step7c_dir / "large_inferred_fault_surface_patches.csv",
    step7c_dir / "large_fault_dfn_raw_time.vtk",
    step7c_dir / "large_fault_dfn_patches.csv",
    step7c_dir / "large_fault_result_raw_time.vtm",
]
missing_step7c = [str(path) for path in step7c_products if not path.exists() or path.stat().st_size <= 0]
if missing_step7c:
    raise FileNotFoundError("missing Step7C products:\n" + "\n".join(missing_step7c))
step7c_summary = json.loads((step7c_dir / "large_fault_dfn_summary.json").read_text(encoding="utf-8"))
if step7c_summary.get("damage_zone_included_in_formal_dfn") is not False:
    raise RuntimeError("Step7C formal DFN unexpectedly includes damage-zone patches")
image_dir = formal / f"step9_section_visualize/output/{version}/cheye1_dfn_multibackground_sections"
pngs = sorted(image_dir.glob("*.png"))
if len(pngs) != 20 or any(path.stat().st_size < 10000 for path in pngs):
    raise RuntimeError(f"expected 20 nonempty PNGs, found {len(pngs)}")
payload = {
    "status": "pass",
    "scope": "step7c_to_step9_continuation",
    "summaries": rows,
    "step7c_products": [str(path) for path in step7c_products],
    "image_count": len(pngs),
    "images": [path.name for path in pngs],
}
output = step6 / "step7c_to_step9_acceptance.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[continuation] acceptance=pass images={len(pngs)} output={output}")
PY

echo "[continuation] completed_at=$(date --iso-8601=seconds)"
echo "[continuation] status=pass"

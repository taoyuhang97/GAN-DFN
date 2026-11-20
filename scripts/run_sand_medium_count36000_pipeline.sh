#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/tyh/projects/petroleum/code/GAN-DFN"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VERSION="medium_count36000_v1"

FORMAL_ROOT="优化阶段二/正式主线"
STEP7B_BASE_CONFIG="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/configs/formal_mine_multiscale_flow_v1.json"
STEP7B_CONFIG="${FORMAL_ROOT}/step7b_multiscale_initial_dfn/configs/formal_mine_multiscale_flow_v1_${VERSION}.json"
STEP7B_OUTPUT="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1_seg1/step7b_${VERSION}"

STEP7D_BASE_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/formal_mine_multiscale_flow_v1_seg1.json"
STEP7D_CONFIG="${FORMAL_ROOT}/step7d_multiscale_fused_dfn/configs/formal_mine_multiscale_flow_v1_seg1_${VERSION}.json"
STEP7D_OUTPUT="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1_seg1/step7d_fused_${VERSION}"

STEP8_BASE_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/formal_mine_multiscale_flow_v1_seg1_density_balance_v1.json"
STEP8_CONFIG="${FORMAL_ROOT}/step8_dfn_well_correction/configs/formal_mine_multiscale_flow_v1_seg1_density_balance_${VERSION}.json"
STEP8_OUTPUT="${FORMAL_ROOT}/output/formal_mine_multiscale_flow_v1_seg1/step8_well_correction_density_balance_${VERSION}"
STEP8_NEG_OUTPUT="${STEP8_OUTPUT}_z_negative"

SECTION_10KM_OUTPUT="${FORMAL_ROOT}/output/step9_section_presentation/mine_v1/scale_wiggle_t4_t7_${VERSION}_10km_dfn_pad20"
LOCAL_200M_OUTPUT="${FORMAL_ROOT}/output/step9_section_presentation/mine_v1/split_imaging_dfn_${VERSION}_200m"

"${PYTHON_BIN}" - "${PWD}" "${VERSION}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
version = sys.argv[2]
formal = root / "优化阶段二/正式主线"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


step7b = load(
    formal
    / "step7b_multiscale_initial_dfn/configs/formal_mine_multiscale_flow_v1.json"
)
step7b["target_patch_count"] = 36000
step7b["output_dir"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7b_{version}"
)
write(
    formal
    / "step7b_multiscale_initial_dfn/configs"
    / f"formal_mine_multiscale_flow_v1_{version}.json",
    step7b,
)

step7d = load(
    formal
    / "step7d_multiscale_fused_dfn/configs/formal_mine_multiscale_flow_v1_seg1.json"
)
step7d["medium_dfn_csv"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7b_{version}/medium_dfn_patches.csv"
)
step7d["output_dir"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7d_fused_{version}"
)
write(
    formal
    / "step7d_multiscale_fused_dfn/configs"
    / f"formal_mine_multiscale_flow_v1_seg1_{version}.json",
    step7d,
)

step8 = load(
    formal
    / "step8_dfn_well_correction/configs/formal_mine_multiscale_flow_v1_seg1_density_balance_v1.json"
)
step8["initial_dfn_csv"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7d_fused_{version}/fused_multiscale_dfn_patches.csv"
)
step8["initial_dfn_vtk"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7d_fused_{version}/fused_multiscale_dfn_raw_time.vtk"
)
step8["initial_dfn_summary_json"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step7d_fused_{version}/fused_multiscale_summary.json"
)
step8["output_dir"] = str(
    formal
    / "output/formal_mine_multiscale_flow_v1_seg1"
    / f"step8_well_correction_density_balance_{version}"
)
write(
    formal
    / "step8_dfn_well_correction/configs"
    / f"formal_mine_multiscale_flow_v1_seg1_density_balance_{version}.json",
    step8,
)
PY

echo "[medium36000] Step7B"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/build_medium_scale_dfn_v4.py" \
  --config "${STEP7B_CONFIG}"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7b_multiscale_initial_dfn/export_existing_dfn_vtk.py" \
  --config "${STEP7B_CONFIG}"

echo "[medium36000] Step7D"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step7d_multiscale_fused_dfn/build_multiscale_fused_dfn.py" \
  --config "${STEP7D_CONFIG}"

echo "[medium36000] Step8"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step8_dfn_well_correction/correct_dfn_with_well_controls.py" \
  --config "${STEP8_CONFIG}"
"${PYTHON_BIN}" scripts/negate_dfn_z.py \
  --input-dir "${STEP8_OUTPUT}" \
  --output-dir "${STEP8_NEG_OUTPUT}" \
  --csv well_corrected_dfn_fracture_patches.csv \
  --vtk well_corrected_dfn_raw_time.vtk

echo "[medium36000] 10 km scale sections"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_scale_wiggle_sections.py" \
  --config "${FORMAL_ROOT}/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json" \
  --dfn-patch-csv "${STEP8_OUTPUT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${STEP8_OUTPUT}/well_corrected_dfn_raw_time.vtk" \
  --output-dir "${SECTION_10KM_OUTPUT}" \
  --section-window-m 10000 \
  --dfn-time-padding-ms 20 \
  --section-time-mode dfn_extent_plus_padding \
  --max-traces 200
"${PYTHON_BIN}" scripts/build_step9_5km_legends.py \
  --sand-output "${SECTION_10KM_OUTPUT}"

echo "[medium36000] local 200 m imaging comparison"
"${PYTHON_BIN}" "${FORMAL_ROOT}/step9_section_visualize/build_cheye1_imaging_dfn_split_sections.py" \
  --config "${FORMAL_ROOT}/step9_section_visualize/configs/formal_mine_multiscale_flow_v1.json" \
  --dfn-patch-csv "${STEP8_OUTPUT}/well_corrected_dfn_fracture_patches.csv" \
  --input-vtk "${STEP8_OUTPUT}/well_corrected_dfn_raw_time.vtk" \
  --local-axis-radius-m 200 \
  --dfn-geometry-source csv \
  --legend-width-in 3.6 \
  --output-dir "${LOCAL_200M_OUTPUT}"

echo "[medium36000] complete"
echo "Step7B:   ${STEP7B_OUTPUT}"
echo "Step7D:   ${STEP7D_OUTPUT}"
echo "Step8:    ${STEP8_OUTPUT}"
echo "10km:     ${SECTION_10KM_OUTPUT}"
echo "local200: ${LOCAL_200M_OUTPUT}"

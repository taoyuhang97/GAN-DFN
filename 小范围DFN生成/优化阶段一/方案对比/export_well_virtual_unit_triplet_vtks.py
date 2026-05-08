from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTANCE_DIR = REPO_ROOT / "研究内容三" / "优化阶段一" / "DFN实例表达互转实验"
if str(INSTANCE_DIR) not in sys.path:
    sys.path.append(str(INSTANCE_DIR))

from instance_roundtrip_common import VtkPatchExportConfig, export_patch_vtk_files  # type: ignore


DEFAULT_SHARED_ROOT = Path("/data/shared/project-oil/wx数据/砂砾岩")
DEFAULT_PRIVATE_ROOT = Path("/home/tyh/data/project-oil/砂砾岩")
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PRIVATE_ROOT / "优化阶段一" / "方案对比" / "BX49_BY5_三联图_VTK_20260507"
)


def scalar_float(value: Any, default: float = 0.0) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    value_float = float(numeric)
    if not np.isfinite(value_float):
        return float(default)
    return value_float


def scalar_int(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return int(default)
    value_float = float(numeric)
    if not np.isfinite(value_float):
        return int(default)
    return int(round(value_float))


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() == "nan" else text


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def build_patch_basis(azimuth_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_rad = math.radians(float(azimuth_deg))
    dip_rad = math.radians(float(dip_deg))
    normal = np.array(
        [
            math.sin(dip_rad) * math.sin(azimuth_rad),
            math.sin(dip_rad) * math.cos(azimuth_rad),
            math.cos(dip_rad),
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / norm

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(normal, reference))) >= 0.98:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    u_dir = np.cross(normal, reference)
    u_norm = float(np.linalg.norm(u_dir))
    if u_norm <= 1e-12:
        u_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u_dir = u_dir / u_norm

    v_dir = np.cross(normal, u_dir)
    v_norm = float(np.linalg.norm(v_dir))
    if v_norm <= 1e-12:
        v_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v_dir = v_dir / v_norm

    return normal, u_dir, v_dir


def build_patch_vertices(
    center_xyz: np.ndarray,
    azimuth_deg: float,
    dip_deg: float,
    patch_length: float,
    patch_height: float,
) -> dict[str, float]:
    normal, u_dir, v_dir = build_patch_basis(azimuth_deg, dip_deg)
    half_length = float(patch_length) * 0.5
    half_height = float(patch_height) * 0.5
    vertices = [
        center_xyz - half_length * u_dir - half_height * v_dir,
        center_xyz + half_length * u_dir - half_height * v_dir,
        center_xyz + half_length * u_dir + half_height * v_dir,
        center_xyz - half_length * u_dir + half_height * v_dir,
    ]
    payload: dict[str, float] = {
        "NormalX": float(normal[0]),
        "NormalY": float(normal[1]),
        "NormalZ": float(normal[2]),
    }
    for idx, vertex in enumerate(vertices, start=1):
        payload[f"V{idx}X"] = float(vertex[0])
        payload[f"V{idx}Y"] = float(vertex[1])
        payload[f"V{idx}Z"] = float(vertex[2])
    return payload


def resolve_point_patch_length(row: pd.Series) -> float:
    seg_length = max(scalar_float(row.get("SegLength"), default=8.0), 1.0)
    pred_point_count = max(scalar_int(row.get("PredPointCount"), default=1), 1)
    point_mass = max(scalar_float(row.get("PointDensityMassAllocated"), default=1.0), 0.25)
    base_length = seg_length * 4.0 / float(pred_point_count)
    scaled_length = base_length * clamp(math.sqrt(point_mass), 0.8, 1.6)
    return clamp(scaled_length, 6.0, 24.0)


def resolve_point_patch_height(row: pd.Series, patch_length: float) -> float:
    dip_deg = clamp(scalar_float(row.get("PointDip"), default=45.0), 0.0, 90.0)
    point_mass_per_length = max(scalar_float(row.get("PointDensityMassPerLengthAllocated"), default=0.5), 0.1)
    base_height = 0.32 * float(patch_length) + 1.2 * point_mass_per_length + 0.8 * (dip_deg / 90.0)
    return clamp(base_height, 3.0, 9.0)


def build_point_patch_table(
    point_df: pd.DataFrame,
    unit_id: str,
    source_kind: str,
    source_name: str,
    block_x: int,
    block_y: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patch_index, (_, point_row) in enumerate(point_df.iterrows(), start=1):
        azimuth_deg = scalar_float(point_row.get("PointAzimuth"), default=45.0)
        dip_deg = scalar_float(point_row.get("PointDip"), default=45.0)
        center_x = scalar_float(point_row.get("X"), default=0.0)
        center_y = scalar_float(point_row.get("Y"), default=0.0)
        center_time = scalar_float(point_row.get("TIME"), default=0.0)
        center_depth = scalar_float(point_row.get("TVD"), default=center_time)
        patch_length = resolve_point_patch_length(point_row)
        patch_height = resolve_point_patch_height(point_row, patch_length)
        seed_id = (
            f"{source_name}::{normalize_text(point_row.get('GeoIntervalKey'))}"
            f"::SEG{scalar_int(point_row.get('Segment_ID'), default=0):03d}"
            f"::PT{scalar_int(point_row.get('Point_ID_In_Segment'), default=0):03d}"
        )
        patch_row: dict[str, Any] = {
            "PatchIndex": patch_index,
            "PatchID": f"{unit_id}_PATCH_{patch_index:04d}",
            "UnitID": unit_id,
            "BlockX": int(block_x),
            "BlockY": int(block_y),
            "GeoIntervalKey": normalize_text(point_row.get("GeoIntervalKey")),
            "StrataName": normalize_text(point_row.get("StrataName")),
            "TopSurfaceCode": normalize_text(point_row.get("TopSurfaceCode")),
            "BaseSurfaceCode": normalize_text(point_row.get("BaseSurfaceCode")),
            "SeedID": seed_id,
            "SourceKind": source_kind,
            "SourceName": source_name,
            "SeedType": "point",
            "ParentSeedID": (
                f"{source_name}::{normalize_text(point_row.get('GeoIntervalKey'))}"
                f"::SEG{scalar_int(point_row.get('Segment_ID'), default=0):03d}"
            ),
            "ParentSourceKind": source_kind,
            "CenterX": center_x,
            "CenterY": center_y,
            "CenterTIME": center_time,
            "CenterDepth": center_depth,
            "Azimuth": azimuth_deg,
            "Dip": dip_deg,
            "DensityWeight": scalar_float(point_row.get("PointDensityMassAllocated"), default=1.0),
            "LengthWeight": scalar_float(point_row.get("PointDensityMassPerLengthAllocated"), default=1.0),
            "Confidence": clamp(scalar_float(point_row.get("PredOrientationConfidence"), default=1.0), 0.0, 1.0),
            "PatchLength": patch_length,
            "PatchHeight": patch_height,
            "PatchArea": float(patch_length * patch_height),
            "SegLength": scalar_float(point_row.get("SegLength"), default=0.0),
            "PredPointCount": scalar_int(point_row.get("PredPointCount"), default=1),
            "SegmentID": scalar_int(point_row.get("Segment_ID"), default=0),
            "PointIDInSegment": scalar_int(point_row.get("Point_ID_In_Segment"), default=0),
        }
        patch_row.update(
            build_patch_vertices(
                center_xyz=np.array([center_x, center_y, center_time], dtype=float),
                azimuth_deg=azimuth_deg,
                dip_deg=dip_deg,
                patch_length=patch_length,
                patch_height=patch_height,
            )
        )
        rows.append(patch_row)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_case_vtk(
    patch_df: pd.DataFrame,
    output_dir: Path,
    base_name: str,
    title_prefix: str,
) -> dict[str, Any]:
    default_float_series = pd.Series(np.full(len(patch_df), -9999.0, dtype=float))
    default_int_series = pd.Series(np.full(len(patch_df), -9999, dtype=int))
    patch_area_series = pd.to_numeric(
        patch_df.get(
            "PatchArea",
            pd.to_numeric(
                patch_df.get("PatchLength", default_float_series),
                errors="coerce",
            ).fillna(0.0)
            * pd.to_numeric(
                patch_df.get("PatchHeight", default_float_series),
                errors="coerce",
            ).fillna(0.0),
        ),
        errors="coerce",
    ).fillna(-9999.0)
    extra_cell_data = {
        "SegLength": pd.to_numeric(
            patch_df.get("SegLength", default_float_series),
            errors="coerce",
        ).fillna(-9999.0).to_numpy(dtype=float),
        "PredPointCount": pd.to_numeric(
            patch_df.get("PredPointCount", default_int_series),
            errors="coerce",
        ).fillna(-9999).to_numpy(dtype=int),
        "SegmentID": pd.to_numeric(
            patch_df.get("SegmentID", default_int_series),
            errors="coerce",
        ).fillna(-9999).to_numpy(dtype=int),
        "PointIDInSegment": pd.to_numeric(
            patch_df.get("PointIDInSegment", default_int_series),
            errors="coerce",
        ).fillna(-9999).to_numpy(dtype=int),
        "PatchArea": patch_area_series.to_numpy(dtype=float),
    }
    vtk_payload = export_patch_vtk_files(
        patch_df=patch_df,
        output_dir=output_dir,
        base_name=base_name,
        title_prefix=title_prefix,
        config=VtkPatchExportConfig(display_z_scale=5.0, invert_time=False),
        extra_cell_data=extra_cell_data,
    )
    write_json(output_dir / f"{base_name}_vtk_mappings.json", vtk_payload.get("mappings", {}))
    return vtk_payload


def build_readme_text(summary_payload: dict[str, Any]) -> str:
    lines = [
        "# BX49_BY5 三联图 VTK 导出",
        "",
        "主展示文件优先看 `*_display.vtk`。",
        "如果需要保持原始时间值，则使用对应的 `*_raw_time.vtk`。",
        "第一张和第二张已经按 `BX49_BY5` 单元 DFN 中对应 `real / virtual` 裂缝片的实际几何进行回填，尺寸与第三张单元 DFN 保持一致。",
        "",
        "## 三个主对象",
        f"- 常规测井裂缝预测：`{summary_payload['real_case']['display_vtk']}`",
        f"- 虚拟测井裂缝预测：`{summary_payload['virtual_case']['display_vtk']}`",
        f"- 单元 DFN：`{summary_payload['unit_case']['display_vtk']}`",
        "",
        "## 颜色区分建议",
        "- 第一张图：直接按 `Azimuth`、`Dip` 或 `DensityWeight` 着色。",
        "- 第二张图：直接按 `Azimuth`、`Dip` 或 `DensityWeight` 着色。",
        "- 第三张图：优先按 `SourceKindCode` 着色，可区分 `real / virtual / seismic_gradient_fill`。",
        "",
        "## 对应对象",
        f"- 常规测井井名：`{summary_payload['real_case']['well_name']}`",
        f"- 虚拟测井名称：`{summary_payload['virtual_case']['well_name']}`",
        f"- 单元编号：`{summary_payload['unit_case']['unit_id']}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="导出常规测井 / 虚拟测井 / 单元 DFN 三联图 VTK")
    parser.add_argument("--unit-id", default="BX49_BY5")
    parser.add_argument("--real-well-name", default="车103")
    parser.add_argument("--virtual-unit-id", default="BX49_BY5")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    unit_id = str(args.unit_id).strip()
    real_well_name = str(args.real_well_name).strip()
    virtual_unit_id = str(args.virtual_unit_id).strip()
    output_dir = Path(str(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_points_csv = (
        DEFAULT_SHARED_ROOT
        / "优化阶段一"
        / "研究内容一"
        / "两阶段裂缝预测流程结果"
        / "现有常规测井裂缝预测"
        / real_well_name
        / "final_fracture_points.csv"
    )
    virtual_points_csv = (
        DEFAULT_SHARED_ROOT
        / "优化阶段一"
        / "研究内容二"
        / "虚拟测井构建"
        / "虚拟裂缝预测"
        / virtual_unit_id
        / "final_fracture_points.csv"
    )
    unit_patch_csv = (
        DEFAULT_SHARED_ROOT
        / "优化阶段一"
        / "研究内容二"
        / "单元DFN构建"
        / "批量生成"
        / unit_id
        / "unit_dfn_patches.csv"
    )

    real_points_df = pd.read_csv(real_points_csv, encoding="utf-8-sig")
    virtual_points_df = pd.read_csv(virtual_points_csv, encoding="utf-8-sig")
    unit_patch_df = pd.read_csv(unit_patch_csv, encoding="utf-8-sig")

    block_x = scalar_int(unit_id.split("_BY")[0].replace("BX", ""), default=0)
    block_y = scalar_int(unit_id.split("_BY")[1], default=0)
    real_slug = real_well_name.replace("/", "_")
    virtual_slug = f"virtual_{virtual_unit_id}".replace("/", "_")
    unit_slug = unit_id.replace("/", "_")

    real_patch_df = unit_patch_df[
        (unit_patch_df.get("SourceKind", pd.Series(dtype=object)).astype(str) == "real")
        & (unit_patch_df.get("SourceName", pd.Series(dtype=object)).astype(str) == real_well_name)
    ].copy()
    virtual_patch_df = unit_patch_df[
        (unit_patch_df.get("SourceKind", pd.Series(dtype=object)).astype(str) == "virtual")
        & (unit_patch_df.get("SourceName", pd.Series(dtype=object)).astype(str) == f"virtual_{virtual_unit_id}")
    ].copy()
    real_patch_df["UnitID"] = f"{unit_id}_REAL"
    virtual_patch_df["UnitID"] = f"{unit_id}_VIRTUAL"
    real_patch_df["BlockX"] = int(block_x)
    real_patch_df["BlockY"] = int(block_y)
    virtual_patch_df["BlockX"] = int(block_x)
    virtual_patch_df["BlockY"] = int(block_y)

    real_patch_csv = output_dir / f"01_real_conventional_{real_slug}_unit_sized_patches.csv"
    virtual_patch_csv = output_dir / f"02_virtual_{virtual_slug}_unit_sized_patches.csv"
    unit_patch_export_csv = output_dir / f"03_unit_{unit_slug}_all_source_patches.csv"
    real_patch_df.to_csv(real_patch_csv, index=False, encoding="utf-8-sig")
    virtual_patch_df.to_csv(virtual_patch_csv, index=False, encoding="utf-8-sig")
    unit_patch_df.to_csv(unit_patch_export_csv, index=False, encoding="utf-8-sig")

    real_vtk = export_case_vtk(
        patch_df=real_patch_df,
        output_dir=output_dir,
        base_name=f"01_real_conventional_{real_slug}",
        title_prefix=f"real_conventional_{real_slug}",
    )
    virtual_vtk = export_case_vtk(
        patch_df=virtual_patch_df,
        output_dir=output_dir,
        base_name=f"02_virtual_{virtual_slug}",
        title_prefix=f"virtual_{virtual_slug}",
    )
    unit_vtk = export_case_vtk(
        patch_df=unit_patch_df,
        output_dir=output_dir,
        base_name=f"03_unit_{unit_slug}_all_sources",
        title_prefix=f"unit_{unit_slug}_all_sources",
    )

    summary_payload = {
        "export_dir": str(output_dir),
        "real_case": {
            "well_name": real_well_name,
            "raw_point_count": int(len(real_points_df)),
            "patch_count": int(len(real_patch_df)),
            "patch_csv": str(real_patch_csv),
            "raw_vtk": str(real_vtk.get("raw_vtk", "")),
            "display_vtk": str(real_vtk.get("display_vtk", "")),
        },
        "virtual_case": {
            "well_name": f"virtual_{virtual_unit_id}",
            "raw_point_count": int(len(virtual_points_df)),
            "patch_count": int(len(virtual_patch_df)),
            "patch_csv": str(virtual_patch_csv),
            "raw_vtk": str(virtual_vtk.get("raw_vtk", "")),
            "display_vtk": str(virtual_vtk.get("display_vtk", "")),
        },
        "unit_case": {
            "unit_id": unit_id,
            "patch_count": int(len(unit_patch_df)),
            "source_kind_counts": {
                str(key): int(value)
                for key, value in unit_patch_df.get("SourceKind", pd.Series(dtype=str)).value_counts().to_dict().items()
            },
            "patch_csv": str(unit_patch_export_csv),
            "raw_vtk": str(unit_vtk.get("raw_vtk", "")),
            "display_vtk": str(unit_vtk.get("display_vtk", "")),
        },
    }
    write_json(output_dir / "export_summary.json", summary_payload)
    (output_dir / "README.md").write_text(build_readme_text(summary_payload), encoding="utf-8")

    print(f"export_dir={output_dir}")
    print(f"real_display_vtk={real_vtk.get('display_vtk', '')}")
    print(f"virtual_display_vtk={virtual_vtk.get('display_vtk', '')}")
    print(f"unit_display_vtk={unit_vtk.get('display_vtk', '')}")


if __name__ == "__main__":
    main()

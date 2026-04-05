from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN


DEFAULT_PACKAGE_RUN_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\虚拟测井构建\单元裂缝归属\smoke_20260328_pkg_min"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\虚拟测井构建\单元DFN构建"
)


DEFAULT_TRACE_HEADER_CSV = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
)
DEFAULT_SEGY_FILE = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"
)

DEFAULT_PACKAGE_RUN_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\单元测井裂缝"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\单元DFN预览"
)
DEFAULT_BLOCK_SIZE_TRACES = 25
DEFAULT_BLOCK_STRIDE_TRACES = 24

# Override legacy defaults with the merged unit-fracture package used in phase 2.
DEFAULT_PACKAGE_RUN_DIR = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\单元测井裂缝"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\单元DFN预览"
)
DEFAULT_TRACE_HEADER_CSV = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv"
)
DEFAULT_SEGY_FILE = Path(
    r"E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\psdm_final_time.sgy"
)

@dataclass(frozen=True)
class PatchConfig:
    point_base_length: float = 18.0
    point_base_height: float = 5.0
    segment_base_length: float = 26.0
    segment_base_height: float = 7.0
    density_gain: float = 0.35
    segment_length_gain: float = 1.1
    max_length_fraction: float = 0.72
    max_height_fraction: float = 0.70
    min_length: float = 8.0
    min_height: float = 2.0
    virtual_point_scale: float = 0.78
    virtual_segment_scale: float = 0.88
    gradient_fill_scale: float = 0.72


@dataclass(frozen=True)
class GradientFillConfig:
    trace_header_csv: Path
    segy_file: Path
    time_padding_ms: float = 20.0
    gradient_threshold: float = 0.65
    layer_threshold_mode: str = "hybrid"
    layer_gradient_quantile: float = 0.985
    layer_threshold_floor: float = 0.60
    enable_multiscale_gradient: bool = True
    multiscale_sigma_levels: tuple[float, ...] = (0.0, 1.0, 2.0)
    multiscale_time_sigma_scale: float = 1.5
    multiscale_combine_mode: str = "max"
    expansion_radius_m: float = 180.0
    vertical_radius_ms: float = 120.0
    dbscan_eps_xy_m: float = 35.0
    dbscan_eps_time_ms: float = 14.0
    min_cluster_samples: int = 2
    max_clusters_per_seed: int = 3
    max_candidate_voxels_per_seed: int = 450
    fill_density_scale: float = 0.82
    fill_length_scale: float = 0.90
    virtual_only_threshold_bonus: float = 0.0
    virtual_only_radius_scale: float = 1.0
    virtual_only_vertical_scale: float = 1.0
    virtual_only_max_clusters_per_seed: int = 2
    enable_seedless_layer_fill: bool = True
    seedless_layer_max_clusters_per_layer: int = 4
    seedless_layer_min_cluster_samples: int = 2
    seedless_layer_min_voxels: int = 12
    seedless_layer_max_candidate_voxels: int = 900
    seed_proximity_exclusion_xy_m: float = 10.0
    seed_proximity_exclusion_time_ms: float = 4.0
    dedup_xy_m: float = 12.5
    dedup_time_ms: float = 4.0
    dedup_azimuth_deg: float = 10.0
    dedup_dip_deg: float = 5.0
    enable_target_fill_ratio: bool = True
    target_fill_to_input_ratio: float = 1.5
    target_fill_layer_weighted: bool = True
    target_fill_max_candidate_voxels_per_layer: int = 4000


@dataclass(frozen=True)
class VtkExportConfig:
    display_z_scale: float = 5.0
    invert_time: bool = True


def read_csv_utf8(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv_utf8(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_boolean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off", ""}
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        normalized = df[col].fillna("").astype(str).str.strip().str.lower()
        df[col] = np.where(normalized.isin(truthy), True, np.where(normalized.isin(falsy), False, False))
    return df


def safe_int(value: object, default: int = 0) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return int(default)
    return int(numeric)


def safe_str(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def parse_float_tuple(text: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = str(text).strip()
    if not value:
        return tuple(float(item) for item in default)
    parts = [part.strip() for part in value.split(",") if part.strip() != ""]
    if not parts:
        return tuple(float(item) for item in default)
    parsed = tuple(float(part) for part in parts)
    if not parsed:
        return tuple(float(item) for item in default)
    return parsed


def make_fill_dedup_key(
    center_x: float,
    center_y: float,
    center_time: float,
    azimuth: float,
    dip: float,
    config: GradientFillConfig,
) -> tuple[int, int, int, int, int]:
    dedup_xy = max(float(config.dedup_xy_m), 1e-6)
    dedup_time = max(float(config.dedup_time_ms), 1e-6)
    dedup_az = max(float(config.dedup_azimuth_deg), 1e-6)
    dedup_dip = max(float(config.dedup_dip_deg), 1e-6)
    return (
        int(round(float(center_x) / dedup_xy)),
        int(round(float(center_y) / dedup_xy)),
        int(round(float(center_time) / dedup_time)),
        int(round(float(azimuth) / dedup_az)),
        int(round(float(dip) / dedup_dip)),
    )


def with_sequential_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    work = df.copy().reset_index(drop=True)
    if index_col in work.columns:
        return work
    work.insert(0, index_col, np.arange(1, len(work) + 1, dtype=int))
    return work


def build_category_code_map(series: pd.Series) -> tuple[np.ndarray, dict[int, str]]:
    values = series.fillna("").astype(str)
    unique_values = sorted([value for value in values.unique().tolist() if value != ""])
    code_map = {value: idx for idx, value in enumerate(unique_values, start=1)}
    codes = values.map(code_map).fillna(0).astype(int).to_numpy(dtype=int)
    reverse_map = {code: value for value, code in code_map.items()}
    reverse_map[0] = ""
    return codes, reverse_map


def transform_display_z(values: np.ndarray, config: VtkExportConfig) -> np.ndarray:
    z_values = np.asarray(values, dtype=float) * float(config.display_z_scale)
    if config.invert_time:
        z_values = -z_values
    return z_values


def normalize_vtk_numeric(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(int)
    numeric = array.astype(float)
    return np.where(np.isfinite(numeric), numeric, -9999.0)


def write_vtk_scalar_block(lines: list[str], name: str, values: np.ndarray) -> None:
    array = normalize_vtk_numeric(values)
    if np.issubdtype(array.dtype, np.integer):
        vtk_type = "int"
        formatter = lambda value: str(int(value))
    else:
        vtk_type = "float"
        formatter = lambda value: f"{float(value):.6f}"
    lines.append(f"SCALARS {name} {vtk_type} 1")
    lines.append("LOOKUP_TABLE default")
    lines.extend(formatter(value) for value in array.tolist())


def write_legacy_vtk_polygons(
    path: Path,
    title: str,
    points: np.ndarray,
    polygons: list[list[int]],
    cell_data: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_polygon_size = sum(len(polygon) + 1 for polygon in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}" for point in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(polygon)} {' '.join(str(int(idx)) for idx in polygon)}" for polygon in polygons)
    if cell_data:
        lines.append(f"CELL_DATA {len(polygons)}")
        for name, values in cell_data.items():
            write_vtk_scalar_block(lines, name, values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_legacy_vtk_vertices(
    path: Path,
    title: str,
    points: np.ndarray,
    point_data: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_size = len(points) * 2
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}" for point in points)
    lines.append(f"VERTICES {len(points)} {vertex_size}")
    lines.extend(f"1 {idx}" for idx in range(len(points)))
    if point_data:
        lines.append(f"POINT_DATA {len(points)}")
        for name, values in point_data.items():
            write_vtk_scalar_block(lines, name, values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_patch_vtk_cell_data(patch_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[int, str]]]:
    source_codes, source_map = build_category_code_map(patch_df["SourceKind"])
    seed_type_codes, seed_type_map = build_category_code_map(patch_df["SeedType"])
    layer_codes, layer_map = build_category_code_map(patch_df["GeoIntervalKey"])
    parent_source_codes: np.ndarray | None = None
    parent_source_map: dict[int, str] = {}
    if "ParentSourceKind" in patch_df.columns:
        parent_source_codes, parent_source_map = build_category_code_map(patch_df["ParentSourceKind"])
    cell_data = {
        "PatchIndex": patch_df["PatchIndex"].to_numpy(dtype=int),
        "BlockX": patch_df["BlockX"].fillna(-9999).to_numpy(dtype=int),
        "BlockY": patch_df["BlockY"].fillna(-9999).to_numpy(dtype=int),
        "SourceKindCode": source_codes,
        "SeedTypeCode": seed_type_codes,
        "LayerCode": layer_codes,
        "CenterTime": patch_df["CenterTIME"].to_numpy(dtype=float),
        "CenterDepth": patch_df["CenterDepth"].to_numpy(dtype=float),
        "Azimuth": patch_df["Azimuth"].to_numpy(dtype=float),
        "Dip": patch_df["Dip"].to_numpy(dtype=float),
        "PatchLength": patch_df["PatchLength"].to_numpy(dtype=float),
        "PatchHeight": patch_df["PatchHeight"].to_numpy(dtype=float),
        "DensityWeight": patch_df["DensityWeight"].to_numpy(dtype=float),
        "LengthWeight": patch_df["LengthWeight"].to_numpy(dtype=float),
        "Confidence": patch_df["Confidence"].to_numpy(dtype=float),
    }
    if parent_source_codes is not None:
        cell_data["ParentSourceKindCode"] = parent_source_codes
    mappings = {
        "SourceKindCode": source_map,
        "SeedTypeCode": seed_type_map,
        "LayerCode": layer_map,
    }
    if parent_source_codes is not None:
        mappings["ParentSourceKindCode"] = parent_source_map
    return cell_data, mappings


def build_seed_vtk_point_data(seed_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, dict[int, str]]]:
    source_codes, source_map = build_category_code_map(seed_df["SourceKind"])
    seed_type_codes, seed_type_map = build_category_code_map(seed_df["SeedType"])
    layer_codes, layer_map = build_category_code_map(seed_df["GeoIntervalKey"])
    parent_source_codes: np.ndarray | None = None
    parent_source_map: dict[int, str] = {}
    if "ParentSourceKind" in seed_df.columns:
        parent_source_codes, parent_source_map = build_category_code_map(seed_df["ParentSourceKind"])
    point_data = {
        "SeedIndex": seed_df["SeedIndex"].to_numpy(dtype=int),
        "BlockX": seed_df["BlockX"].fillna(-9999).to_numpy(dtype=int),
        "BlockY": seed_df["BlockY"].fillna(-9999).to_numpy(dtype=int),
        "SourceKindCode": source_codes,
        "SeedTypeCode": seed_type_codes,
        "LayerCode": layer_codes,
        "CenterTime": seed_df["CenterTime"].to_numpy(dtype=float),
        "CenterDepth": seed_df["CenterDepth"].to_numpy(dtype=float),
        "Azimuth": seed_df["Azimuth"].to_numpy(dtype=float),
        "Dip": seed_df["Dip"].to_numpy(dtype=float),
        "DensityWeight": seed_df["DensityWeight"].to_numpy(dtype=float),
        "LengthWeight": seed_df["LengthWeight"].to_numpy(dtype=float),
        "Confidence": seed_df["Confidence"].to_numpy(dtype=float),
    }
    if parent_source_codes is not None:
        point_data["ParentSourceKindCode"] = parent_source_codes
    mappings = {
        "SourceKindCode": source_map,
        "SeedTypeCode": seed_type_map,
        "LayerCode": layer_map,
    }
    if parent_source_codes is not None:
        mappings["ParentSourceKindCode"] = parent_source_map
    return point_data, mappings


def export_patch_vtk_files(
    patch_df: pd.DataFrame,
    output_dir: Path,
    config: VtkExportConfig,
) -> dict[str, object]:
    if patch_df.empty:
        return {"patch_file_count": 0}
    cell_data, mappings = build_patch_vtk_cell_data(patch_df)
    raw_points: list[list[float]] = []
    display_points: list[list[float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        vertices = np.array([[row[f"V{i}X"], row[f"V{i}Y"], row[f"V{i}Z"]] for i in range(1, 5)], dtype=float)
        start_idx = len(raw_points)
        raw_points.extend(vertices.tolist())
        display_vertices = vertices.copy()
        display_vertices[:, 2] = transform_display_z(display_vertices[:, 2], config)
        display_points.extend(display_vertices.tolist())
        polygons.append([start_idx, start_idx + 1, start_idx + 2, start_idx + 3])

    raw_path = output_dir / "unit_dfn_patches_raw_time.vtk"
    display_path = output_dir / "unit_dfn_patches_display.vtk"
    write_legacy_vtk_polygons(raw_path, "unit_dfn_patches_raw_time", np.asarray(raw_points, dtype=float), polygons, cell_data)
    write_legacy_vtk_polygons(display_path, "unit_dfn_patches_display", np.asarray(display_points, dtype=float), polygons, cell_data)
    return {
        "patch_file_count": 2,
        "patch_raw_vtk": str(raw_path),
        "patch_display_vtk": str(display_path),
        "PatchMappings": mappings,
    }


def export_seed_vtk_files(
    seed_df: pd.DataFrame,
    output_dir: Path,
    config: VtkExportConfig,
) -> dict[str, object]:
    if seed_df.empty:
        return {"seed_file_count": 0}
    point_data, mappings = build_seed_vtk_point_data(seed_df)
    raw_points = seed_df[["CenterX", "CenterY", "CenterTime"]].to_numpy(dtype=float)
    display_points = raw_points.copy()
    display_points[:, 2] = transform_display_z(display_points[:, 2], config)
    raw_path = output_dir / "fracture_seeds_merged_raw_time.vtk"
    display_path = output_dir / "fracture_seeds_merged_display.vtk"
    write_legacy_vtk_vertices(raw_path, "fracture_seeds_merged_raw_time", raw_points, point_data)
    write_legacy_vtk_vertices(display_path, "fracture_seeds_merged_display", display_points, point_data)
    return {
        "seed_file_count": 2,
        "seed_raw_vtk": str(raw_path),
        "seed_display_vtk": str(display_path),
        "SeedMappings": mappings,
    }


def write_vtk_mapping_json(output_dir: Path, export_summary: dict[str, object], config: VtkExportConfig) -> Path:
    mapping_path = output_dir / "vtk_attribute_mappings.json"
    payload = {
        "DisplayCoordinates": {
            "XAxis": "X(m)",
            "YAxis": "Y(m)",
            "ZAxis": ("-TIME(ms)" if config.invert_time else "TIME(ms)") + f" * {float(config.display_z_scale):.3f}",
            "Formula": f"display_z = {'-' if config.invert_time else ''}raw_time * {float(config.display_z_scale):.3f}",
        },
        "PatchMappings": export_summary.get("PatchMappings", {}),
        "SeedMappings": export_summary.get("SeedMappings", {}),
        "LookupHint": {
            "PatchIndexColumn": "unit_dfn_patches.csv -> PatchIndex",
            "SeedIndexColumn": "fracture_seeds_merged.csv -> SeedIndex",
        },
    }
    mapping_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping_path


def write_paraview_quickstart(output_dir: Path, unit_id: str, config: VtkExportConfig) -> Path:
    quickstart_path = output_dir / "paraview_quickstart.md"
    content = "\n".join(
        [
            f"# {unit_id} ParaView Quickstart",
            "",
            "推荐直接打开以下两个文件：",
            "- `unit_dfn_patches_display.vtk`：裂缝片面，适合观察最终 DFN 分布和做面片选择。",
            "- `fracture_seeds_merged_display.vtk`：裂缝中心点，适合先快速看点云分布。",
            "",
            "建议操作顺序：",
            "1. 在 ParaView 中打开 `unit_dfn_patches_display.vtk`。",
            "2. 将 `Representation` 设为 `Surface With Edges`。",
            "3. 打开 `Spreadsheet View`，方便看到被选中的属性。",
            "4. 选择 `Select Cells On` 或 `Select Cells Through`，在三维视窗中框选裂缝片。",
            "5. 在 `Spreadsheet View` 中读取选中的 `PatchIndex`、`Azimuth`、`Dip`、`LayerCode`、`SourceKindCode`。",
            "6. 用 `PatchIndex` 回查 `unit_dfn_patches.csv`，即可拿到对应 `PatchID/SeedID/层位`。",
            "",
            "点云查看：",
            "1. 打开 `fracture_seeds_merged_display.vtk`。",
            "2. 用 `Select Points On` 或 `Select Points Through` 做点选或框选。",
            "3. 用 `SeedIndex` 回查 `fracture_seeds_merged.csv`。",
            "",
            "编码映射：",
            "- `SourceKindCode`、`SeedTypeCode`、`LayerCode` 的含义见 `vtk_attribute_mappings.json`。",
            "",
            "当前显示坐标：",
            f"- display_z = {'-' if config.invert_time else ''}raw_time * {float(config.display_z_scale):.3f}",
            "- 如果你想看原始时间坐标，可改为打开 `*_raw_time.vtk`。",
        ]
    )
    quickstart_path.write_text(content + "\n", encoding="utf-8")
    return quickstart_path


def resolve_windows_short_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt":
        return text
    buffer = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.kernel32.GetShortPathNameW(text, buffer, len(buffer))
    if result > 0:
        return buffer.value
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-unit DFN preview patches from packaged unit seeds.")
    parser.add_argument("--package-run-dir", type=Path, default=DEFAULT_PACKAGE_RUN_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--segy-file", type=Path, default=DEFAULT_SEGY_FILE)
    parser.add_argument("--run-name", type=str, default="unit_dfn_preview")
    parser.add_argument("--unit-id", type=str, default="")
    parser.add_argument("--data-mode", type=str, default="all", choices=["all", "real_virtual", "real_only", "virtual_only"])
    parser.add_argument("--auto-select-richest", action="store_true")
    parser.add_argument("--block-size-traces", type=int, default=DEFAULT_BLOCK_SIZE_TRACES)
    parser.add_argument("--block-stride-traces", type=int, default=DEFAULT_BLOCK_STRIDE_TRACES)
    parser.add_argument("--point-base-length", type=float, default=18.0)
    parser.add_argument("--point-base-height", type=float, default=5.0)
    parser.add_argument("--segment-base-length", type=float, default=26.0)
    parser.add_argument("--segment-base-height", type=float, default=7.0)
    parser.add_argument("--density-gain", type=float, default=0.35)
    parser.add_argument("--segment-length-gain", type=float, default=1.1)
    parser.add_argument("--virtual-point-size-scale", type=float, default=0.78)
    parser.add_argument("--virtual-segment-size-scale", type=float, default=0.88)
    parser.add_argument("--gradient-fill-size-scale", type=float, default=0.72)
    parser.add_argument("--gradient-fill-threshold", type=float, default=0.65)
    parser.add_argument("--gradient-threshold-mode", type=str, default="hybrid", choices=["global", "layer_quantile", "hybrid"])
    parser.add_argument("--gradient-layer-quantile", type=float, default=0.985)
    parser.add_argument("--gradient-layer-threshold-floor", type=float, default=0.60)
    parser.add_argument("--disable-multiscale-gradient", action="store_true")
    parser.add_argument("--gradient-multiscale-sigmas", type=str, default="0,1,2")
    parser.add_argument("--gradient-multiscale-time-scale", type=float, default=1.5)
    parser.add_argument("--gradient-multiscale-combine", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--gradient-fill-radius-m", type=float, default=180.0)
    parser.add_argument("--gradient-fill-vertical-ms", type=float, default=120.0)
    parser.add_argument("--gradient-fill-time-padding-ms", type=float, default=20.0)
    parser.add_argument("--gradient-fill-dbscan-eps-xy", type=float, default=35.0)
    parser.add_argument("--gradient-fill-dbscan-eps-time", type=float, default=14.0)
    parser.add_argument("--gradient-fill-min-samples", type=int, default=2)
    parser.add_argument("--gradient-fill-max-clusters-per-seed", type=int, default=3)
    parser.add_argument("--gradient-fill-max-candidate-voxels-per-seed", type=int, default=450)
    parser.add_argument("--gradient-fill-density-scale", type=float, default=0.82)
    parser.add_argument("--gradient-fill-length-scale", type=float, default=0.90)
    parser.add_argument("--virtual-only-gradient-threshold-bonus", type=float, default=0.0)
    parser.add_argument("--virtual-only-gradient-radius-scale", type=float, default=1.0)
    parser.add_argument("--virtual-only-gradient-vertical-scale", type=float, default=1.0)
    parser.add_argument("--virtual-only-gradient-max-clusters-per-seed", type=int, default=2)
    parser.add_argument("--disable-seedless-layer-fill", action="store_true")
    parser.add_argument("--seedless-layer-max-clusters-per-layer", type=int, default=4)
    parser.add_argument("--seedless-layer-min-samples", type=int, default=2)
    parser.add_argument("--seedless-layer-min-voxels", type=int, default=12)
    parser.add_argument("--seedless-layer-max-candidate-voxels", type=int, default=900)
    parser.add_argument("--seed-proximity-exclusion-xy-m", type=float, default=10.0)
    parser.add_argument("--seed-proximity-exclusion-time-ms", type=float, default=4.0)
    parser.add_argument("--dedup-xy-m", type=float, default=12.5)
    parser.add_argument("--dedup-time-ms", type=float, default=4.0)
    parser.add_argument("--dedup-azimuth-deg", type=float, default=10.0)
    parser.add_argument("--dedup-dip-deg", type=float, default=5.0)
    parser.add_argument("--disable-target-fill-ratio", action="store_true")
    parser.add_argument("--target-fill-to-input-ratio", type=float, default=1.5)
    parser.add_argument("--disable-target-fill-layer-weighted", action="store_true")
    parser.add_argument("--target-fill-max-candidate-voxels-per-layer", type=int, default=4000)
    parser.add_argument("--disable-gradient-fill", action="store_true")
    parser.add_argument("--disable-vtk-export", action="store_true")
    parser.add_argument("--vtk-display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    return parser.parse_args()


def load_unit_catalog(package_run_dir: Path) -> pd.DataFrame:
    path = package_run_dir / "unit_catalog.csv"
    if not path.exists():
        raise FileNotFoundError(f"unit_catalog.csv not found: {path}")
    catalog = read_csv_utf8(path)
    catalog = ensure_numeric(
        catalog,
        [
            "BlockX",
            "BlockY",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "XCenter",
            "YCenter",
            "TopDepth",
            "BaseDepth",
            "TopTime",
            "BaseTime",
            "SeedCount",
            "LayerCount",
            "RealSeedCount",
            "VirtualSeedCount",
            "RealPointSeedCount",
            "RealSegmentSeedCount",
            "VirtualPointSeedCount",
            "VirtualSegmentSeedCount",
            "RealLayerCount",
            "VirtualLayerCount",
            "CoveredIntervalCount",
        ],
    )
    catalog = ensure_boolean(catalog, ["HasRealData", "HasVirtualData"])
    return catalog


def select_target_unit(catalog_df: pd.DataFrame, unit_id: str, auto_select_richest: bool, data_mode: str) -> pd.Series:
    work = catalog_df.copy()
    if unit_id:
        matched = work[work["UnitID"].astype(str) == unit_id].copy()
        if matched.empty:
            raise ValueError(f"Unit not found in unit_catalog.csv: {unit_id}")
        if str(data_mode) != "all" and str(matched.iloc[0].get("DataMode", "")) != str(data_mode):
            raise ValueError(f"Unit {unit_id} does not match requested data mode: {data_mode}")
        return matched.iloc[0]

    candidates = work[work["SeedCount"].fillna(0) > 0].copy()
    if str(data_mode) != "all":
        candidates = candidates[candidates["DataMode"].astype(str) == str(data_mode)].copy()
    if candidates.empty:
        raise ValueError("No unit with seeds found in unit_catalog.csv.")
    if auto_select_richest or not unit_id:
        mode_rank = {"real_virtual": 3, "real_only": 2, "virtual_only": 1}
        candidates["DataModeRank"] = candidates["DataMode"].astype(str).map(mode_rank).fillna(0).astype(int)
        candidates = candidates.sort_values(
            ["DataModeRank", "RealSeedCount", "SeedCount", "LayerCount", "VirtualSeedCount"],
            ascending=[False, False, False, False, False],
        )
        return candidates.iloc[0]
    raise ValueError("Either provide --unit-id or use --auto-select-richest.")


def root_unit_dir(root: Path) -> Path:
    if (root / "units").exists():
        return root / "units"
    return root


def load_unit_package(package_run_dir: Path, unit_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    unit_dir = root_unit_dir(package_run_dir) / unit_id
    if not unit_dir.exists():
        raise FileNotFoundError(f"Unit directory not found: {unit_dir}")
    layers = ensure_numeric(
        read_csv_utf8(unit_dir / "unit_layers.csv"),
        [
            "BlockX",
            "BlockY",
            "TopDepth",
            "BaseDepth",
            "TopTime",
            "BaseTime",
            "RealPointSeedCount",
            "RealSegmentSeedCount",
            "VirtualPointSeedCount",
            "VirtualSegmentSeedCount",
            "RealSeedCount",
            "VirtualSeedCount",
        ],
    )
    seeds = ensure_numeric(
        read_csv_utf8(unit_dir / "fracture_seeds.csv"),
        [
            "BlockX",
            "BlockY",
            "CenterX",
            "CenterY",
            "CenterDepth",
            "CenterTime",
            "DepthStart",
            "DepthEnd",
            "TimeStart",
            "TimeEnd",
            "Azimuth",
            "Dip",
            "DensityWeight",
            "LengthWeight",
            "Confidence",
        ],
    )
    meta = json.loads((unit_dir / "unit_meta.json").read_text(encoding="utf-8"))
    return layers, seeds, meta


def load_trace_header(trace_header_csv: Path) -> pd.DataFrame:
    trace_header = read_csv_utf8(trace_header_csv)
    return ensure_numeric(trace_header, ["TraceIdx", "X", "Y"])


def derive_unit_bounds_from_block(
    unit_row: pd.Series,
    trace_header_df: pd.DataFrame,
    block_size_traces: int,
    block_stride_traces: int,
) -> pd.Series:
    work = unit_row.copy()
    required_cols = ["XMin", "XMax", "YMin", "YMax", "XCenter", "YCenter"]
    has_complete_bounds = all(col in work.index and pd.notna(pd.to_numeric(work.get(col), errors="coerce")) for col in required_cols)
    if has_complete_bounds:
        return work

    block_x = int(pd.to_numeric(work.get("BlockX"), errors="coerce"))
    block_y = int(pd.to_numeric(work.get("BlockY"), errors="coerce"))
    x_unique = np.array(sorted(trace_header_df["X"].dropna().astype(float).unique().tolist()), dtype=float)
    y_unique = np.array(sorted(trace_header_df["Y"].dropna().astype(float).unique().tolist()), dtype=float)
    if len(x_unique) == 0 or len(y_unique) == 0:
        raise ValueError("Trace header does not contain valid X/Y coordinates.")

    x_start = int(block_x * int(block_stride_traces))
    y_start = int(block_y * int(block_stride_traces))
    x_end = min(x_start + int(block_size_traces) - 1, len(x_unique) - 1)
    y_end = min(y_start + int(block_size_traces) - 1, len(y_unique) - 1)
    if x_start < 0 or y_start < 0 or x_start >= len(x_unique) or y_start >= len(y_unique):
        raise ValueError(f"Block indices out of grid range for unit {work.get('UnitID', '')}: BX={block_x}, BY={block_y}")

    x_slice = x_unique[x_start : x_end + 1]
    y_slice = y_unique[y_start : y_end + 1]
    work["XMin"] = float(x_slice.min())
    work["XMax"] = float(x_slice.max())
    work["YMin"] = float(y_slice.min())
    work["YMax"] = float(y_slice.max())
    work["XCenter"] = float((x_slice.min() + x_slice.max()) / 2.0)
    work["YCenter"] = float((y_slice.min() + y_slice.max()) / 2.0)
    work["TraceCountX"] = int(len(x_slice))
    work["TraceCountY"] = int(len(y_slice))
    return work


def extract_unit_trace_header(trace_header_df: pd.DataFrame, unit_row: pd.Series) -> pd.DataFrame:
    mask = (
        (trace_header_df["X"] >= float(unit_row["XMin"]))
        & (trace_header_df["X"] <= float(unit_row["XMax"]))
        & (trace_header_df["Y"] >= float(unit_row["YMin"]))
        & (trace_header_df["Y"] <= float(unit_row["YMax"]))
    )
    unit_traces = trace_header_df.loc[mask, ["TraceIdx", "X", "Y"]].copy()
    if unit_traces.empty:
        raise ValueError(f"No traces found for unit {unit_row['UnitID']}.")
    return unit_traces


def load_unit_seismic_cube(
    unit_row: pd.Series,
    layers_df: pd.DataFrame,
    trace_header_df: pd.DataFrame,
    config: GradientFillConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    unit_trace_df = extract_unit_trace_header(trace_header_df, unit_row)
    x_unique = np.array(sorted(unit_trace_df["X"].dropna().unique().tolist()), dtype=float)
    y_unique = np.array(sorted(unit_trace_df["Y"].dropna().unique().tolist()), dtype=float)
    x_index = {float(value): idx for idx, value in enumerate(x_unique)}
    y_index = {float(value): idx for idx, value in enumerate(y_unique)}

    lower_time = float(min(layers_df["TopTime"].min(), layers_df["BaseTime"].min()) - config.time_padding_ms)
    upper_time = float(max(layers_df["TopTime"].max(), layers_df["BaseTime"].max()) + config.time_padding_ms)
    segy_path = resolve_windows_short_path(config.segy_file)
    with segyio.open(segy_path, "r", ignore_geometry=True) as segy_file:
        segy_file.mmap()
        samples = np.asarray(segy_file.samples, dtype=float)
        start_idx = int(np.searchsorted(samples, lower_time, side="left"))
        end_idx = int(np.searchsorted(samples, upper_time, side="right"))
        start_idx = max(0, min(start_idx, len(samples) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(samples)))
        time_axis = samples[start_idx:end_idx]
        cube = np.zeros((len(x_unique), len(y_unique), len(time_axis)), dtype=np.float32)

        for row in unit_trace_df.itertuples(index=False):
            xi = x_index[float(row.X)]
            yi = y_index[float(row.Y)]
            cube[xi, yi, :] = np.asarray(segy_file.trace[int(row.TraceIdx)][start_idx:end_idx], dtype=np.float32)

    summary = {
        "TraceCount": int(len(unit_trace_df)),
        "GridShape": [int(len(x_unique)), int(len(y_unique)), int(len(time_axis))],
        "XCount": int(len(x_unique)),
        "YCount": int(len(y_unique)),
        "TimeSampleCount": int(len(time_axis)),
        "TimeMin": float(time_axis[0]),
        "TimeMax": float(time_axis[-1]),
        "TimePaddingMs": float(config.time_padding_ms),
    }
    return cube, x_unique, y_unique, time_axis, summary


def compute_normalized_gradient_volume(
    cube: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    gx, gy, gz = np.gradient(cube.astype(np.float32), dx, dy, dt, edge_order=1)
    gradient = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    grad_min = float(np.nanmin(gradient))
    grad_max = float(np.nanmax(gradient))
    if grad_max - grad_min <= 1e-8:
        grad_norm = np.zeros_like(gradient, dtype=np.float32)
    else:
        grad_norm = ((gradient - grad_min) / (grad_max - grad_min)).astype(np.float32)
    return grad_norm, grad_min, grad_max


def detect_high_gradient_voxels(
    seismic_cube: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    time_axis: np.ndarray,
    layers_df: pd.DataFrame,
    config: GradientFillConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dx = float(np.median(np.diff(x_axis))) if len(x_axis) > 1 else 12.5
    dy = float(np.median(np.diff(y_axis))) if len(y_axis) > 1 else 12.5
    dt = float(np.median(np.diff(time_axis))) if len(time_axis) > 1 else 2.0

    scale_stats: list[dict[str, object]] = []
    gradient_norm_volumes: list[np.ndarray] = []
    sigma_levels = tuple(float(value) for value in config.multiscale_sigma_levels)
    if not sigma_levels:
        sigma_levels = (0.0,)

    for sigma_xy in sigma_levels:
        sigma_time = float(max(0.0, sigma_xy * float(config.multiscale_time_sigma_scale)))
        if bool(config.enable_multiscale_gradient) and (sigma_xy > 0 or sigma_time > 0):
            working_cube = gaussian_filter(
                seismic_cube.astype(np.float32),
                sigma=(float(sigma_xy), float(sigma_xy), float(sigma_time)),
                mode="nearest",
            )
        else:
            working_cube = seismic_cube.astype(np.float32, copy=False)
        grad_norm_scale, grad_min_scale, grad_max_scale = compute_normalized_gradient_volume(working_cube, dx, dy, dt)
        gradient_norm_volumes.append(grad_norm_scale)
        scale_stats.append(
            {
                "SigmaXY": float(sigma_xy),
                "SigmaTime": float(sigma_time),
                "GradientMin": float(grad_min_scale),
                "GradientMax": float(grad_max_scale),
            }
        )

    if len(gradient_norm_volumes) == 1:
        grad_norm = gradient_norm_volumes[0]
    elif str(config.multiscale_combine_mode).strip().lower() == "mean":
        grad_norm = np.mean(np.stack(gradient_norm_volumes, axis=0), axis=0).astype(np.float32)
    else:
        grad_norm = np.max(np.stack(gradient_norm_volumes, axis=0), axis=0).astype(np.float32)

    combined_grad_min = float(np.nanmin(grad_norm)) if grad_norm.size else 0.0
    combined_grad_max = float(np.nanmax(grad_norm)) if grad_norm.size else 0.0

    rows: list[dict[str, object]] = []
    layer_stats: list[dict[str, object]] = []
    threshold_mode = str(config.layer_threshold_mode).strip().lower()
    total_selected_voxels = 0

    for _, layer in layers_df.iterrows():
        top_time = pd.to_numeric(layer.get("TopTime"), errors="coerce")
        base_time = pd.to_numeric(layer.get("BaseTime"), errors="coerce")
        if pd.isna(top_time) or pd.isna(base_time):
            continue
        lower = float(min(top_time, base_time))
        upper = float(max(top_time, base_time))
        time_mask = (time_axis >= lower) & (time_axis <= upper)
        time_indices = np.flatnonzero(time_mask)
        if time_indices.size == 0:
            continue

        layer_gradient = grad_norm[:, :, time_mask]
        layer_values = layer_gradient.reshape(-1)
        if layer_values.size == 0:
            continue

        layer_quantile_threshold = float(np.quantile(layer_values, float(config.layer_gradient_quantile)))
        if threshold_mode == "global":
            effective_threshold = float(config.gradient_threshold)
        elif threshold_mode == "layer_quantile":
            effective_threshold = max(float(config.layer_threshold_floor), layer_quantile_threshold)
        else:
            effective_threshold = max(
                float(config.layer_threshold_floor),
                min(float(config.gradient_threshold), layer_quantile_threshold),
            )

        layer_indices = np.argwhere(layer_gradient >= effective_threshold)
        total_selected_voxels += int(len(layer_indices))
        layer_stats.append(
            {
                "GeoIntervalKey": str(layer.get("GeoIntervalKey", "")),
                "TopSurfaceCode": str(layer.get("TopSurfaceCode", "")),
                "BaseSurfaceCode": str(layer.get("BaseSurfaceCode", "")),
                "ThresholdMode": threshold_mode,
                "EffectiveThreshold": float(effective_threshold),
                "LayerQuantileThreshold": float(layer_quantile_threshold),
                "LayerVoxelCount": int(layer_values.size),
                "SelectedVoxelCount": int(len(layer_indices)),
            }
        )

        for ix, iy, it_local in layer_indices:
            time_value = float(time_axis[int(time_indices[int(it_local)])])
            rows.append(
                {
                    "X": float(x_axis[int(ix)]),
                    "Y": float(y_axis[int(iy)]),
                    "TIME": time_value,
                    "GradientValue": float(layer_gradient[int(ix), int(iy), int(it_local)]),
                    "GeoIntervalKey": str(layer.get("GeoIntervalKey", "")),
                    "StrataName": str(layer.get("StrataName", "")),
                    "TopSurfaceCode": str(layer.get("TopSurfaceCode", "")),
                    "BaseSurfaceCode": str(layer.get("BaseSurfaceCode", "")),
                }
            )

    summary = {
        "GradientThreshold": float(config.gradient_threshold),
        "GradientVoxelCount": int(np.prod(seismic_cube.shape)),
        "GradientThresholdMode": threshold_mode,
        "GradientLayerQuantile": float(config.layer_gradient_quantile),
        "GradientLayerThresholdFloor": float(config.layer_threshold_floor),
        "MultiscaleGradientEnabled": bool(config.enable_multiscale_gradient),
        "MultiscaleSigmaLevels": [float(value) for value in sigma_levels],
        "MultiscaleTimeSigmaScale": float(config.multiscale_time_sigma_scale),
        "MultiscaleCombineMode": str(config.multiscale_combine_mode),
        "MultiscaleScaleStats": scale_stats,
        "HighGradientVoxelCountRaw": int(total_selected_voxels),
        "HighGradientVoxelCountInLayers": int(len(rows)),
        "GradientMin": combined_grad_min,
        "GradientMax": combined_grad_max,
        "LayerThresholds": layer_stats,
    }
    return pd.DataFrame(rows), summary


def deduplicate_fill_seeds(fill_df: pd.DataFrame, config: GradientFillConfig) -> pd.DataFrame:
    if fill_df.empty:
        return fill_df
    work = fill_df.copy()
    dedup_xy = max(float(config.dedup_xy_m), 1e-6)
    dedup_time = max(float(config.dedup_time_ms), 1e-6)
    dedup_az = max(float(config.dedup_azimuth_deg), 1e-6)
    dedup_dip = max(float(config.dedup_dip_deg), 1e-6)
    work["DedupX"] = (work["CenterX"] / dedup_xy).round().astype(int)
    work["DedupY"] = (work["CenterY"] / dedup_xy).round().astype(int)
    work["DedupT"] = (work["CenterTime"] / dedup_time).round().astype(int)
    work["DedupAz"] = (work["Azimuth"] / dedup_az).round().astype(int)
    work["DedupDip"] = (work["Dip"] / dedup_dip).round().astype(int)
    work = work.sort_values(["GradientValue", "ClusterPointCount"], ascending=[False, False])
    work = work.drop_duplicates(
        subset=["GeoIntervalKey", "TopSurfaceCode", "BaseSurfaceCode", "DedupX", "DedupY", "DedupT", "DedupAz", "DedupDip"],
        keep="first",
    )
    return work.drop(columns=["DedupX", "DedupY", "DedupT", "DedupAz", "DedupDip"]).reset_index(drop=True)


def circular_mean_degrees(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    radians = np.deg2rad(np.mod(array, 360.0))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) <= 1e-8 and abs(cos_mean) <= 1e-8:
        return float(np.mod(np.nanmedian(array), 360.0))
    return float(np.mod(np.degrees(np.arctan2(sin_mean, cos_mean)), 360.0))


def layer_time_center(layer_row: pd.Series) -> float:
    top_time = pd.to_numeric(layer_row.get("TopTime"), errors="coerce")
    base_time = pd.to_numeric(layer_row.get("BaseTime"), errors="coerce")
    if pd.isna(top_time) or pd.isna(base_time):
        return float("nan")
    return float((float(top_time) + float(base_time)) / 2.0)


def summarize_seed_template(seed_df: pd.DataFrame) -> dict[str, float | str]:
    azimuth = circular_mean_degrees(pd.to_numeric(seed_df.get("Azimuth"), errors="coerce")) if not seed_df.empty else float("nan")
    dip = float(np.nanmedian(pd.to_numeric(seed_df.get("Dip"), errors="coerce"))) if not seed_df.empty else float("nan")
    density = float(np.nanmedian(pd.to_numeric(seed_df.get("DensityWeight"), errors="coerce"))) if not seed_df.empty else float("nan")
    length = float(np.nanmedian(pd.to_numeric(seed_df.get("LengthWeight"), errors="coerce"))) if not seed_df.empty else float("nan")
    confidence = float(np.nanmedian(pd.to_numeric(seed_df.get("Confidence"), errors="coerce"))) if not seed_df.empty else float("nan")
    source_kind = ""
    if not seed_df.empty and "SourceKind" in seed_df.columns:
        source_mode = seed_df["SourceKind"].fillna("").astype(str)
        source_kind = str(source_mode.mode(dropna=False).iloc[0]) if not source_mode.empty else ""
    return {
        "Azimuth": azimuth if np.isfinite(azimuth) else 330.0,
        "Dip": dip if np.isfinite(dip) else 75.0,
        "DensityWeight": density if np.isfinite(density) and density > 0 else 2.0,
        "LengthWeight": length if np.isfinite(length) and length > 0 else 1.0,
        "Confidence": confidence if np.isfinite(confidence) and confidence > 0 else 0.55,
        "SourceKind": source_kind or "layer_direct",
    }


def resolve_seedless_layer_template(target_layer: pd.Series, layers_df: pd.DataFrame, seed_df: pd.DataFrame) -> dict[str, float | str]:
    if seed_df.empty:
        return summarize_seed_template(seed_df)
    target_center = layer_time_center(target_layer)
    best_subset = pd.DataFrame()
    best_key: tuple[float, int] | None = None
    for _, layer in layers_df.iterrows():
        layer_key = layer_lookup_key(layer)
        subset = seed_df[
            (seed_df["GeoIntervalKey"].astype(str) == layer_key[0])
            & (seed_df["TopSurfaceCode"].astype(str) == layer_key[1])
            & (seed_df["BaseSurfaceCode"].astype(str) == layer_key[2])
        ].copy()
        if subset.empty:
            continue
        center = layer_time_center(layer)
        distance = abs(center - target_center) if np.isfinite(center) and np.isfinite(target_center) else float("inf")
        rank_key = (distance, -int(len(subset)))
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_subset = subset
    return summarize_seed_template(best_subset if not best_subset.empty else seed_df)


def filter_voxels_by_seed_exclusion(
    voxel_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    config: GradientFillConfig,
) -> pd.DataFrame:
    if voxel_df.empty or seed_df.empty:
        return voxel_df
    exclusion_xy = max(0.0, float(config.seed_proximity_exclusion_xy_m))
    exclusion_time = max(0.0, float(config.seed_proximity_exclusion_time_ms))
    if exclusion_xy <= 0.0 and exclusion_time <= 0.0:
        return voxel_df

    candidate_x = voxel_df["X"].to_numpy(dtype=float)
    candidate_y = voxel_df["Y"].to_numpy(dtype=float)
    candidate_t = voxel_df["TIME"].to_numpy(dtype=float)
    seed_x = pd.to_numeric(seed_df.get("CenterX"), errors="coerce").to_numpy(dtype=float)
    seed_y = pd.to_numeric(seed_df.get("CenterY"), errors="coerce").to_numpy(dtype=float)
    seed_t = pd.to_numeric(seed_df.get("CenterTime"), errors="coerce").to_numpy(dtype=float)
    valid_seed_mask = np.isfinite(seed_x) & np.isfinite(seed_y) & np.isfinite(seed_t)
    if not np.any(valid_seed_mask):
        return voxel_df
    seed_x = seed_x[valid_seed_mask]
    seed_y = seed_y[valid_seed_mask]
    seed_t = seed_t[valid_seed_mask]

    dx = candidate_x[:, None] - seed_x[None, :]
    dy = candidate_y[:, None] - seed_y[None, :]
    dt = np.abs(candidate_t[:, None] - seed_t[None, :])
    xy = np.sqrt(dx ** 2 + dy ** 2)
    blocked = ((xy < exclusion_xy) & (dt < exclusion_time)).any(axis=1)
    if not np.any(blocked):
        return voxel_df
    return voxel_df.loc[~blocked].copy()


def build_fill_seed_row(
    unit_row: pd.Series,
    layer_row: pd.Series,
    center_x: float,
    center_y: float,
    center_time: float,
    gradient_value: float,
    cluster_point_count: int,
    seed_id: str,
    source_name: str,
    azimuth: float,
    dip: float,
    density_weight: float,
    length_weight: float,
    confidence: float,
    parent_seed_id: str,
    parent_source_kind: str,
) -> dict[str, object]:
    return {
        "UnitID": str(unit_row["UnitID"]),
        "BlockX": int(unit_row["BlockX"]),
        "BlockY": int(unit_row["BlockY"]),
        "GeoIntervalKey": str(layer_row.get("GeoIntervalKey", "")),
        "StrataName": str(layer_row.get("StrataName", "")),
        "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
        "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
        "SeedID": seed_id,
        "SourceKind": "seismic_gradient_fill",
        "SourceName": source_name,
        "SeedType": "gradient_fill",
        "CenterX": float(center_x),
        "CenterY": float(center_y),
        "CenterDepth": float("nan"),
        "CenterTime": float(center_time),
        "DepthStart": float("nan"),
        "DepthEnd": float("nan"),
        "TimeStart": float(center_time),
        "TimeEnd": float(center_time),
        "Azimuth": float(azimuth),
        "Dip": float(dip),
        "DensityWeight": float(density_weight),
        "LengthWeight": float(length_weight),
        "Confidence": float(confidence),
        "GradientValue": float(gradient_value),
        "ClusterPointCount": int(cluster_point_count),
        "ParentSeedID": str(parent_seed_id),
        "ParentSourceKind": str(parent_source_kind),
    }


def cluster_gradient_voxels(
    voxel_df: pd.DataFrame,
    config: GradientFillConfig,
    max_clusters: int,
    max_candidate_voxels: int | None = None,
    min_cluster_samples: int | None = None,
) -> pd.DataFrame:
    if voxel_df.empty or int(max_clusters) <= 0:
        return pd.DataFrame(columns=["GradientValue", "ClusterPointCount", "CenterX", "CenterY", "CenterTime"])

    work = voxel_df.copy()
    candidate_limit = int(max_candidate_voxels or config.max_candidate_voxels_per_seed)
    if candidate_limit > 0:
        work = work.nlargest(candidate_limit, "GradientValue")
    if work.empty:
        return pd.DataFrame(columns=["GradientValue", "ClusterPointCount", "CenterX", "CenterY", "CenterTime"])

    time_scale = float(config.dbscan_eps_xy_m / max(config.dbscan_eps_time_ms, 1e-6))
    scaled = np.column_stack(
        [
            work["X"].to_numpy(dtype=float),
            work["Y"].to_numpy(dtype=float),
            work["TIME"].to_numpy(dtype=float) * time_scale,
        ]
    )
    labels = DBSCAN(
        eps=float(config.dbscan_eps_xy_m),
        min_samples=int(min_cluster_samples or config.min_cluster_samples),
    ).fit_predict(scaled)
    work["ClusterLabel"] = labels

    cluster_rows: list[dict[str, object]] = []
    for label, group in work.groupby("ClusterLabel"):
        if int(label) < 0:
            continue
        cluster_rows.append(
            {
                "GradientValue": float(group["GradientValue"].mean()),
                "ClusterPointCount": int(len(group)),
                "CenterX": float(group["X"].mean()),
                "CenterY": float(group["Y"].mean()),
                "CenterTime": float(group["TIME"].mean()),
            }
        )

    if not cluster_rows:
        fallback_rows: list[dict[str, object]] = []
        chosen_centers: list[tuple[float, float, float]] = []
        for candidate in work.sort_values("GradientValue", ascending=False).itertuples(index=False):
            if len(fallback_rows) >= int(max_clusters):
                break
            if any(
                (math.hypot(float(candidate.X) - cx, float(candidate.Y) - cy) <= float(config.dbscan_eps_xy_m))
                and (abs(float(candidate.TIME) - ct) <= float(config.dbscan_eps_time_ms))
                for cx, cy, ct in chosen_centers
            ):
                continue
            local_group = work[
                (np.sqrt((work["X"] - float(candidate.X)) ** 2 + (work["Y"] - float(candidate.Y)) ** 2) <= float(config.dbscan_eps_xy_m))
                & ((work["TIME"] - float(candidate.TIME)).abs() <= float(config.dbscan_eps_time_ms))
            ].copy()
            fallback_rows.append(
                {
                    "GradientValue": float(local_group["GradientValue"].mean()),
                    "ClusterPointCount": int(len(local_group)),
                    "CenterX": float(local_group["X"].mean()),
                    "CenterY": float(local_group["Y"].mean()),
                    "CenterTime": float(local_group["TIME"].mean()),
                }
            )
            chosen_centers.append((float(candidate.X), float(candidate.Y), float(candidate.TIME)))
        cluster_rows = fallback_rows

    if not cluster_rows:
        return pd.DataFrame(columns=["GradientValue", "ClusterPointCount", "CenterX", "CenterY", "CenterTime"])
    return (
        pd.DataFrame(cluster_rows)
        .sort_values(["GradientValue", "ClusterPointCount"], ascending=[False, False])
        .head(int(max_clusters))
        .reset_index(drop=True)
    )


def build_target_ratio_fill_seeds(
    unit_row: pd.Series,
    layers_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    candidate_voxels: pd.DataFrame,
    current_fill_df: pd.DataFrame,
    config: GradientFillConfig,
) -> pd.DataFrame:
    if seeds_df.empty or candidate_voxels.empty or not bool(config.enable_target_fill_ratio):
        return pd.DataFrame(columns=list(current_fill_df.columns) if not current_fill_df.empty else list(seeds_df.columns))

    target_ratio = float(config.target_fill_to_input_ratio)
    if target_ratio <= 0:
        return pd.DataFrame(columns=list(current_fill_df.columns) if not current_fill_df.empty else list(seeds_df.columns))

    current_count = int(len(current_fill_df))
    target_count = int(math.ceil(len(seeds_df) * target_ratio))
    total_deficit = max(0, target_count - current_count)
    if total_deficit <= 0:
        return pd.DataFrame(columns=current_fill_df.columns if not current_fill_df.empty else list(seeds_df.columns))

    output_columns = list(current_fill_df.columns) if not current_fill_df.empty else list(seeds_df.columns)
    layer_seed_counts = (
        seeds_df.assign(
            _LayerKey=seeds_df.apply(layer_lookup_key, axis=1),
        )["_LayerKey"].value_counts().to_dict()
        if not seeds_df.empty
        else {}
    )
    current_fill_counts = (
        current_fill_df.assign(_LayerKey=current_fill_df.apply(layer_lookup_key, axis=1))["_LayerKey"].value_counts().to_dict()
        if not current_fill_df.empty
        else {}
    )
    layers_by_key = {layer_lookup_key(layer): layer for _, layer in layers_df.iterrows()}

    existing_keys: set[tuple[int, int, int, int, int]] = set()
    if not current_fill_df.empty:
        for row in current_fill_df.itertuples(index=False):
            existing_keys.add(
                make_fill_dedup_key(
                    float(row.CenterX),
                    float(row.CenterY),
                    float(row.CenterTime),
                    float(row.Azimuth),
                    float(row.Dip),
                    config,
                )
            )

    layer_templates: dict[tuple[str, str, str], dict[str, float | str]] = {}
    layer_candidates: dict[tuple[str, str, str], pd.DataFrame] = {}
    for layer_key, layer_row in layers_by_key.items():
        layer_voxels = candidate_voxels[
            (candidate_voxels["GeoIntervalKey"] == layer_key[0])
            & (candidate_voxels["TopSurfaceCode"] == layer_key[1])
            & (candidate_voxels["BaseSurfaceCode"] == layer_key[2])
        ].copy()
        if layer_voxels.empty:
            continue
        layer_seed_df = seeds_df[
            (seeds_df["GeoIntervalKey"].astype(str) == layer_key[0])
            & (seeds_df["TopSurfaceCode"].astype(str) == layer_key[1])
            & (seeds_df["BaseSurfaceCode"].astype(str) == layer_key[2])
        ].copy()
        layer_voxels = filter_voxels_by_seed_exclusion(layer_voxels, layer_seed_df, config)
        if layer_voxels.empty:
            continue
        layer_voxels = layer_voxels.sort_values("GradientValue", ascending=False).head(int(config.target_fill_max_candidate_voxels_per_layer)).copy()
        if layer_voxels.empty:
            continue
        layer_candidates[layer_key] = layer_voxels
        layer_templates[layer_key] = (
            summarize_seed_template(layer_seed_df)
            if not layer_seed_df.empty
            else resolve_seedless_layer_template(layer_row, layers_df, seeds_df)
        )

    if not layer_candidates:
        return pd.DataFrame(columns=output_columns)

    selected_rows: list[dict[str, object]] = []
    pending_layer_keys = [key for key in layers_by_key if key in layer_candidates and layer_seed_counts.get(key, 0) > 0]

    if bool(config.target_fill_layer_weighted):
        for layer_key in pending_layer_keys:
            input_count = int(layer_seed_counts.get(layer_key, 0))
            if input_count <= 0:
                continue
            current_layer_fill = int(current_fill_counts.get(layer_key, 0))
            target_layer_fill = int(math.ceil(input_count * target_ratio))
            layer_deficit = max(0, target_layer_fill - current_layer_fill)
            if layer_deficit <= 0:
                continue

            layer_row = layers_by_key[layer_key]
            template = layer_templates[layer_key]
            selected_for_layer = 0
            for candidate in layer_candidates[layer_key].itertuples(index=False):
                if len(selected_rows) >= total_deficit or selected_for_layer >= layer_deficit:
                    break
                dedup_key = make_fill_dedup_key(
                    float(candidate.X),
                    float(candidate.Y),
                    float(candidate.TIME),
                    float(template["Azimuth"]),
                    float(template["Dip"]),
                    config,
                )
                if dedup_key in existing_keys:
                    continue
                existing_keys.add(dedup_key)
                selected_rows.append(
                    build_fill_seed_row(
                        unit_row=unit_row,
                        layer_row=layer_row,
                        center_x=float(candidate.X),
                        center_y=float(candidate.Y),
                        center_time=float(candidate.TIME),
                        gradient_value=float(candidate.GradientValue),
                        cluster_point_count=1,
                        seed_id=f"{unit_row['UnitID']}::GF_TARGET::{layer_key[0]}::{selected_for_layer + 1:04d}",
                        source_name="seismic_gradient_fill_target",
                        azimuth=float(template["Azimuth"]),
                        dip=float(template["Dip"]),
                        density_weight=float(template["DensityWeight"]) * float(config.fill_density_scale) * (0.75 + 0.75 * float(candidate.GradientValue)),
                        length_weight=float(template["LengthWeight"]) * float(config.fill_length_scale),
                        confidence=min(0.98, float(template["Confidence"]) * 0.72 + 0.12),
                        parent_seed_id="",
                        parent_source_kind=str(template["SourceKind"]),
                    )
                )
                selected_for_layer += 1

    if len(selected_rows) < total_deficit:
        global_candidates = []
        for layer_key, layer_voxels in layer_candidates.items():
            layer_row = layers_by_key[layer_key]
            template = layer_templates[layer_key]
            for candidate in layer_voxels.itertuples(index=False):
                global_candidates.append((float(candidate.GradientValue), layer_key, layer_row, template, candidate))
        global_candidates.sort(key=lambda item: item[0], reverse=True)

        for _, layer_key, layer_row, template, candidate in global_candidates:
            if len(selected_rows) >= total_deficit:
                break
            dedup_key = make_fill_dedup_key(
                float(candidate.X),
                float(candidate.Y),
                float(candidate.TIME),
                float(template["Azimuth"]),
                float(template["Dip"]),
                config,
            )
            if dedup_key in existing_keys:
                continue
            existing_keys.add(dedup_key)
            layer_count = sum(1 for row in selected_rows if str(row.get("GeoIntervalKey", "")) == str(layer_key[0]))
            selected_rows.append(
                build_fill_seed_row(
                    unit_row=unit_row,
                    layer_row=layer_row,
                    center_x=float(candidate.X),
                    center_y=float(candidate.Y),
                    center_time=float(candidate.TIME),
                    gradient_value=float(candidate.GradientValue),
                    cluster_point_count=1,
                    seed_id=f"{unit_row['UnitID']}::GF_TARGET::{layer_key[0]}::{layer_count + 1:04d}",
                    source_name="seismic_gradient_fill_target",
                    azimuth=float(template["Azimuth"]),
                    dip=float(template["Dip"]),
                    density_weight=float(template["DensityWeight"]) * float(config.fill_density_scale) * (0.75 + 0.75 * float(candidate.GradientValue)),
                    length_weight=float(template["LengthWeight"]) * float(config.fill_length_scale),
                    confidence=min(0.98, float(template["Confidence"]) * 0.72 + 0.12),
                    parent_seed_id="",
                    parent_source_kind=str(template["SourceKind"]),
                )
            )

    if not selected_rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(selected_rows)


def build_gradient_fill_seeds(
    unit_row: pd.Series,
    layers_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    gradient_voxel_df: pd.DataFrame,
    config: GradientFillConfig,
) -> pd.DataFrame:
    extra_cols = ["GradientValue", "ClusterPointCount", "ParentSeedID", "ParentSourceKind"]
    output_cols = list(seeds_df.columns) + [col for col in extra_cols if col not in seeds_df.columns]
    if gradient_voxel_df.empty:
        return pd.DataFrame(columns=output_cols)

    data_mode = str(unit_row.get("DataMode", "")).strip().lower()
    runtime_threshold = float(config.gradient_threshold)
    runtime_radius = float(config.expansion_radius_m)
    runtime_vertical = float(config.vertical_radius_ms)
    runtime_max_clusters = int(config.max_clusters_per_seed)

    if data_mode == "virtual_only":
        runtime_threshold = min(0.99, runtime_threshold + float(config.virtual_only_threshold_bonus))
        runtime_radius = float(runtime_radius * config.virtual_only_radius_scale)
        runtime_vertical = float(runtime_vertical * config.virtual_only_vertical_scale)
        runtime_max_clusters = min(runtime_max_clusters, int(config.virtual_only_max_clusters_per_seed))

    if str(config.layer_threshold_mode).strip().lower() == "global":
        candidate_threshold = runtime_threshold
    else:
        candidate_threshold = float(config.layer_threshold_floor)
        if data_mode == "virtual_only":
            candidate_threshold = max(
                candidate_threshold,
                float(config.layer_threshold_floor) + float(config.virtual_only_threshold_bonus),
            )
    candidate_voxels = gradient_voxel_df[gradient_voxel_df["GradientValue"] >= float(candidate_threshold)].copy()
    if candidate_voxels.empty:
        return pd.DataFrame(columns=output_cols)

    real_seed_layer_keys = set(
        layers_df.loc[pd.to_numeric(layers_df.get("RealSeedCount"), errors="coerce").fillna(0) > 0, "GeoIntervalKey"]
        .astype(str)
        .tolist()
    ) if not layers_df.empty else set()

    parent_seed_df = seeds_df.copy()
    if data_mode == "real_virtual":
        layer_keys = parent_seed_df["GeoIntervalKey"].astype(str)
        source_kinds = parent_seed_df["SourceKind"].astype(str).str.lower()
        use_real_mask = layer_keys.isin(real_seed_layer_keys)
        parent_seed_df = parent_seed_df[(~use_real_mask) | source_kinds.eq("real")].copy()
    if data_mode == "real_only":
        parent_seed_df = parent_seed_df[parent_seed_df["SourceKind"].astype(str).str.lower().eq("real")].copy()
    if data_mode == "virtual_only":
        parent_seed_df = parent_seed_df[parent_seed_df["SourceKind"].astype(str).str.lower().eq("virtual")].copy()

    fill_rows: list[dict[str, object]] = []
    parent_layer_keys = {
        layer_lookup_key(seed)
        for _, seed in parent_seed_df.iterrows()
    }

    for _, seed in parent_seed_df.iterrows():
        center_x = pd.to_numeric(seed.get("CenterX"), errors="coerce")
        center_y = pd.to_numeric(seed.get("CenterY"), errors="coerce")
        center_time = pd.to_numeric(seed.get("CenterTime"), errors="coerce")
        azimuth = pd.to_numeric(seed.get("Azimuth"), errors="coerce")
        dip = pd.to_numeric(seed.get("Dip"), errors="coerce")
        if pd.isna(center_x) or pd.isna(center_y) or pd.isna(center_time) or pd.isna(azimuth) or pd.isna(dip):
            continue

        layer_key = layer_lookup_key(seed)
        seed_voxels = candidate_voxels[
            (candidate_voxels["GeoIntervalKey"] == layer_key[0])
            & (candidate_voxels["TopSurfaceCode"] == layer_key[1])
            & (candidate_voxels["BaseSurfaceCode"] == layer_key[2])
        ].copy()
        if seed_voxels.empty:
            continue

        seed_voxels["DX"] = seed_voxels["X"] - float(center_x)
        seed_voxels["DY"] = seed_voxels["Y"] - float(center_y)
        seed_voxels["DT"] = seed_voxels["TIME"] - float(center_time)
        seed_voxels["XYDistance"] = np.sqrt(seed_voxels["DX"] ** 2 + seed_voxels["DY"] ** 2)
        seed_voxels["TimeDistance"] = seed_voxels["DT"].abs()
        seed_voxels = seed_voxels[
            (seed_voxels["XYDistance"] <= runtime_radius)
            & (seed_voxels["TimeDistance"] <= runtime_vertical)
        ].copy()
        exclusion_xy = max(0.0, float(config.seed_proximity_exclusion_xy_m))
        exclusion_time = max(0.0, float(config.seed_proximity_exclusion_time_ms))
        seed_voxels = seed_voxels[
            (seed_voxels["XYDistance"] >= exclusion_xy) | (seed_voxels["TimeDistance"] >= exclusion_time)
        ].copy()
        if seed_voxels.empty:
            continue

        cluster_df = cluster_gradient_voxels(
            seed_voxels,
            config,
            max_clusters=runtime_max_clusters,
        )
        if cluster_df.empty:
            continue

        parent_source_kind = str(seed.get("SourceKind", "")).strip().lower()
        parent_density_scale = 1.0 if parent_source_kind == "real" else 0.78
        parent_length_scale = 1.0 if parent_source_kind == "real" else 0.82
        parent_confidence_scale = 0.85 if parent_source_kind == "real" else 0.72

        for cluster_idx, cluster in enumerate(cluster_df.itertuples(index=False), start=1):
            fill_rows.append(
                {
                    "UnitID": str(unit_row["UnitID"]),
                    "BlockX": int(unit_row["BlockX"]),
                    "BlockY": int(unit_row["BlockY"]),
                    "GeoIntervalKey": str(seed.get("GeoIntervalKey", "")),
                    "StrataName": str(seed.get("StrataName", "")),
                    "TopSurfaceCode": str(seed.get("TopSurfaceCode", "")),
                    "BaseSurfaceCode": str(seed.get("BaseSurfaceCode", "")),
                    "SeedID": f"{unit_row['UnitID']}::GF::{seed['SeedID']}::{cluster_idx:02d}",
                    "SourceKind": "seismic_gradient_fill",
                    "SourceName": "seismic_gradient_fill",
                    "SeedType": "gradient_fill",
                    "CenterX": float(cluster.CenterX),
                    "CenterY": float(cluster.CenterY),
                    "CenterDepth": float("nan"),
                    "CenterTime": float(cluster.CenterTime),
                    "DepthStart": float("nan"),
                    "DepthEnd": float("nan"),
                    "TimeStart": float(cluster.CenterTime),
                    "TimeEnd": float(cluster.CenterTime),
                    "Azimuth": float(azimuth),
                    "Dip": float(dip),
                    "DensityWeight": (
                        float(seed["DensityWeight"]) * float(config.fill_density_scale) * parent_density_scale * (0.6 + 0.6 * float(cluster.GradientValue))
                        if pd.notna(seed.get("DensityWeight"))
                        else float(cluster.GradientValue) * parent_density_scale
                    ),
                    "LengthWeight": (
                        float(seed["LengthWeight"]) * float(config.fill_length_scale) * parent_length_scale
                        if pd.notna(seed.get("LengthWeight"))
                        else 1.0 * parent_length_scale
                    ),
                    "Confidence": float(seed["Confidence"]) * parent_confidence_scale if pd.notna(seed.get("Confidence")) else 0.5 * parent_confidence_scale,
                    "GradientValue": float(cluster.GradientValue),
                    "ClusterPointCount": int(cluster.ClusterPointCount),
                    "ParentSeedID": str(seed["SeedID"]),
                    "ParentSourceKind": str(seed.get("SourceKind", "")),
                }
            )

    if bool(config.enable_seedless_layer_fill):
        template_seed_df = parent_seed_df if not parent_seed_df.empty else seeds_df
        for _, layer in layers_df.iterrows():
            layer_key = layer_lookup_key(layer)
            if layer_key in parent_layer_keys:
                continue
            layer_voxels = candidate_voxels[
                (candidate_voxels["GeoIntervalKey"] == layer_key[0])
                & (candidate_voxels["TopSurfaceCode"] == layer_key[1])
                & (candidate_voxels["BaseSurfaceCode"] == layer_key[2])
            ].copy()
            if len(layer_voxels) < int(config.seedless_layer_min_voxels):
                continue

            cluster_df = cluster_gradient_voxels(
                layer_voxels,
                config,
                max_clusters=int(config.seedless_layer_max_clusters_per_layer),
                max_candidate_voxels=int(config.seedless_layer_max_candidate_voxels),
                min_cluster_samples=int(config.seedless_layer_min_cluster_samples),
            )
            if cluster_df.empty:
                continue

            template = resolve_seedless_layer_template(layer, layers_df, template_seed_df)
            for cluster_idx, cluster in enumerate(cluster_df.itertuples(index=False), start=1):
                fill_rows.append(
                    {
                        "UnitID": str(unit_row["UnitID"]),
                        "BlockX": int(unit_row["BlockX"]),
                        "BlockY": int(unit_row["BlockY"]),
                        "GeoIntervalKey": str(layer.get("GeoIntervalKey", "")),
                        "StrataName": str(layer.get("StrataName", "")),
                        "TopSurfaceCode": str(layer.get("TopSurfaceCode", "")),
                        "BaseSurfaceCode": str(layer.get("BaseSurfaceCode", "")),
                        "SeedID": f"{unit_row['UnitID']}::GF_LAYER::{layer.get('GeoIntervalKey', '')}::{cluster_idx:02d}",
                        "SourceKind": "seismic_gradient_fill",
                        "SourceName": "seismic_gradient_fill_layer",
                        "SeedType": "gradient_fill",
                        "CenterX": float(cluster.CenterX),
                        "CenterY": float(cluster.CenterY),
                        "CenterDepth": float("nan"),
                        "CenterTime": float(cluster.CenterTime),
                        "DepthStart": float("nan"),
                        "DepthEnd": float("nan"),
                        "TimeStart": float(cluster.CenterTime),
                        "TimeEnd": float(cluster.CenterTime),
                        "Azimuth": float(template["Azimuth"]),
                        "Dip": float(template["Dip"]),
                        "DensityWeight": float(template["DensityWeight"]) * float(config.fill_density_scale) * (0.7 + 0.6 * float(cluster.GradientValue)),
                        "LengthWeight": float(template["LengthWeight"]) * float(config.fill_length_scale) * 0.88,
                        "Confidence": float(template["Confidence"]) * 0.68,
                        "GradientValue": float(cluster.GradientValue),
                        "ClusterPointCount": int(cluster.ClusterPointCount),
                        "ParentSeedID": "",
                        "ParentSourceKind": "layer_direct",
                    }
                )

    base_fill_df = deduplicate_fill_seeds(pd.DataFrame(fill_rows), config) if fill_rows else pd.DataFrame(columns=output_cols)
    target_fill_df = build_target_ratio_fill_seeds(
        unit_row=unit_row,
        layers_df=layers_df,
        seeds_df=seeds_df,
        candidate_voxels=candidate_voxels,
        current_fill_df=base_fill_df,
        config=config,
    )
    if not target_fill_df.empty:
        fill_df = deduplicate_fill_seeds(pd.concat([base_fill_df, target_fill_df], ignore_index=True, sort=False), config)
    else:
        fill_df = base_fill_df
    if fill_df.empty:
        return pd.DataFrame(columns=output_cols)
    return fill_df[output_cols]


def plane_basis_from_orientation(azimuth_deg: float, dip_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth = math.radians(float(azimuth_deg))
    dip = math.radians(float(dip_deg))
    normal = np.array(
        [
            math.sin(dip) * math.sin(azimuth),
            math.sin(dip) * math.cos(azimuth),
            math.cos(dip),
        ],
        dtype=float,
    )
    norm = np.linalg.norm(normal)
    if norm <= 1e-8:
        raise ValueError("Invalid fracture orientation.")
    normal = normal / norm
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.allclose(abs(float(np.dot(normal, up))), 1.0, atol=1e-6):
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    u_vec = np.cross(normal, up)
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    return normal, u_vec, v_vec


def build_patch_vertices(center: np.ndarray, u_vec: np.ndarray, v_vec: np.ndarray, length: float, height: float) -> np.ndarray:
    half_u = (length / 2.0) * u_vec
    half_v = (height / 2.0) * v_vec
    return np.array(
        [
            center - half_u - half_v,
            center + half_u - half_v,
            center + half_u + half_v,
            center - half_u + half_v,
        ],
        dtype=float,
    )


def layer_lookup_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row.get("GeoIntervalKey", "")),
        str(row.get("TopSurfaceCode", "")),
        str(row.get("BaseSurfaceCode", "")),
    )


def build_layer_lookup(layers_df: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    lookup: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in layers_df.iterrows():
        lookup[layer_lookup_key(row)] = row
    return lookup


def compute_density_scale(value: float, gain: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(np.clip(1.0 + math.log1p(value) * gain, 1.0, 3.2))


def compute_segment_scale(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(np.clip(1.0 + math.sqrt(value) * 0.22, 1.0, 3.0))


def determine_seed_visual_scale(seed_row: pd.Series, config: PatchConfig) -> float:
    source_kind = str(seed_row.get("SourceKind", "")).strip().lower()
    seed_type = str(seed_row.get("SeedType", "")).strip().lower()
    if source_kind == "seismic_gradient_fill" or seed_type == "gradient_fill":
        return float(config.gradient_fill_scale)
    if source_kind == "virtual":
        if seed_type == "segment":
            return float(config.virtual_segment_scale)
        return float(config.virtual_point_scale)
    return 1.0


def determine_patch_size(
    seed_row: pd.Series,
    unit_row: pd.Series,
    layer_row: pd.Series | None,
    config: PatchConfig,
) -> tuple[float, float]:
    unit_xy_span = max(
        1.0,
        min(float(unit_row["XMax"] - unit_row["XMin"]), float(unit_row["YMax"] - unit_row["YMin"])),
    )
    layer_time_span = float("nan")
    if layer_row is not None and pd.notna(layer_row.get("TopTime")) and pd.notna(layer_row.get("BaseTime")):
        layer_time_span = max(float(layer_row["BaseTime"]) - float(layer_row["TopTime"]), 0.0)

    density_scale = compute_density_scale(float(seed_row.get("DensityWeight", np.nan)), float(config.density_gain))
    segment_scale = compute_segment_scale(float(seed_row.get("LengthWeight", np.nan)))
    seed_type = str(seed_row.get("SeedType", ""))
    if seed_type in {"segment", "gradient_fill"}:
        length = config.segment_base_length * density_scale * (1.0 + config.segment_length_gain * (segment_scale - 1.0))
        height = config.segment_base_height * density_scale
    else:
        length = config.point_base_length * density_scale
        height = config.point_base_height * density_scale

    visual_scale = determine_seed_visual_scale(seed_row, config)
    length *= float(visual_scale)
    height *= float(max(0.6, visual_scale))

    length = float(np.clip(length, config.min_length, unit_xy_span * config.max_length_fraction))
    if np.isfinite(layer_time_span) and layer_time_span > 0:
        max_height = max(config.min_height, layer_time_span * config.max_height_fraction)
        height = float(np.clip(height, config.min_height, max_height))
    else:
        height = float(max(config.min_height, height))
    return length, height


def clamp_vertices_to_layer(vertices: np.ndarray, layer_row: pd.Series | None) -> np.ndarray:
    if layer_row is None:
        return vertices
    top_time = pd.to_numeric(layer_row.get("TopTime"), errors="coerce")
    base_time = pd.to_numeric(layer_row.get("BaseTime"), errors="coerce")
    if pd.isna(top_time) or pd.isna(base_time):
        return vertices
    lower = float(min(top_time, base_time))
    upper = float(max(top_time, base_time))
    clamped = vertices.copy()
    clamped[:, 2] = np.clip(clamped[:, 2], lower, upper)
    return clamped


def build_unit_dfn_patches(
    unit_row: pd.Series,
    layers_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    config: PatchConfig,
) -> pd.DataFrame:
    if seeds_df.empty:
        return pd.DataFrame()
    layer_map = build_layer_lookup(layers_df)
    patch_rows: list[dict[str, object]] = []
    patch_counter = 1

    for _, seed in seeds_df.iterrows():
        azimuth = pd.to_numeric(seed.get("Azimuth"), errors="coerce")
        dip = pd.to_numeric(seed.get("Dip"), errors="coerce")
        center_x = pd.to_numeric(seed.get("CenterX"), errors="coerce")
        center_y = pd.to_numeric(seed.get("CenterY"), errors="coerce")
        center_time = pd.to_numeric(seed.get("CenterTime"), errors="coerce")
        if pd.isna(azimuth) or pd.isna(dip) or pd.isna(center_x) or pd.isna(center_y) or pd.isna(center_time):
            continue

        layer_row = layer_map.get(layer_lookup_key(seed))
        length, height = determine_patch_size(seed, unit_row, layer_row, config)
        normal, u_vec, v_vec = plane_basis_from_orientation(float(azimuth), float(dip))
        center = np.array([float(center_x), float(center_y), float(center_time)], dtype=float)
        vertices = build_patch_vertices(center, u_vec, v_vec, length, height)
        vertices = clamp_vertices_to_layer(vertices, layer_row)

        row = {
            "PatchIndex": int(patch_counter),
            "PatchID": f"{unit_row['UnitID']}_PATCH_{patch_counter:04d}",
            "UnitID": str(unit_row["UnitID"]),
            "BlockX": int(unit_row["BlockX"]),
            "BlockY": int(unit_row["BlockY"]),
            "GeoIntervalKey": str(seed.get("GeoIntervalKey", "")),
            "StrataName": str(seed.get("StrataName", "")),
            "TopSurfaceCode": str(seed.get("TopSurfaceCode", "")),
            "BaseSurfaceCode": str(seed.get("BaseSurfaceCode", "")),
            "SeedID": str(seed.get("SeedID", "")),
            "SourceKind": str(seed.get("SourceKind", "")),
            "SourceName": str(seed.get("SourceName", "")),
            "SeedType": str(seed.get("SeedType", "")),
            "ParentSeedID": safe_str(seed.get("ParentSeedID", "")),
            "ParentSourceKind": safe_str(seed.get("ParentSourceKind", "")),
            "CenterX": float(center[0]),
            "CenterY": float(center[1]),
            "CenterTIME": float(center[2]),
            "CenterDepth": float(seed["CenterDepth"]) if pd.notna(seed.get("CenterDepth")) else float("nan"),
            "Azimuth": float(azimuth),
            "Dip": float(dip),
            "DensityWeight": float(seed["DensityWeight"]) if pd.notna(seed.get("DensityWeight")) else float("nan"),
            "LengthWeight": float(seed["LengthWeight"]) if pd.notna(seed.get("LengthWeight")) else float("nan"),
            "Confidence": float(seed["Confidence"]) if pd.notna(seed.get("Confidence")) else float("nan"),
            "PatchLength": float(length),
            "PatchHeight": float(height),
            "NormalX": float(normal[0]),
            "NormalY": float(normal[1]),
            "NormalZ": float(normal[2]),
        }
        for idx, vertex in enumerate(vertices, start=1):
            row[f"V{idx}X"] = float(vertex[0])
            row[f"V{idx}Y"] = float(vertex[1])
            row[f"V{idx}Z"] = float(vertex[2])
        patch_rows.append(row)
        patch_counter += 1

    return pd.DataFrame(patch_rows)


def build_patch_summary(unit_row: pd.Series, layers_df: pd.DataFrame, seeds_df: pd.DataFrame, patch_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "UnitID": str(unit_row["UnitID"]),
        "BlockX": safe_int(unit_row.get("BlockX")),
        "BlockY": safe_int(unit_row.get("BlockY")),
        "ReliabilityClass": str(unit_row.get("ReliabilityClass", "")),
        "DataMode": str(unit_row.get("DataMode", "")),
        "HasRealData": bool(unit_row.get("HasRealData", False)),
        "HasVirtualData": bool(unit_row.get("HasVirtualData", False)),
        "LayerCount": safe_int(unit_row.get("LayerCount")),
        "SeedCount": int(len(seeds_df)),
        "RealSeedCount": safe_int(unit_row.get("RealSeedCount")),
        "VirtualSeedCount": safe_int(unit_row.get("VirtualSeedCount")),
        "PatchCount": int(len(patch_df)),
        "SeedTypeCounts": seeds_df["SeedType"].value_counts(dropna=False).to_dict() if not seeds_df.empty else {},
        "SourceKindCounts": seeds_df["SourceKind"].value_counts(dropna=False).to_dict() if not seeds_df.empty else {},
        "SourceNameCounts": seeds_df["SourceName"].value_counts(dropna=False).to_dict() if not seeds_df.empty else {},
        "LayerSeedCounts": (
            seeds_df.groupby("GeoIntervalKey")["SeedID"].count().sort_values(ascending=False).to_dict()
            if not seeds_df.empty
            else {}
        ),
    }
    if not seeds_df.empty and "ParentSourceKind" in seeds_df.columns:
        gradient_parent_df = seeds_df[seeds_df["SourceKind"].astype(str).eq("seismic_gradient_fill")].copy()
        summary["GradientFillParentSourceCounts"] = (
            gradient_parent_df["ParentSourceKind"].fillna("").astype(str).value_counts(dropna=False).to_dict()
            if not gradient_parent_df.empty
            else {}
        )
    if not patch_df.empty:
        summary["PatchLengthMean"] = float(patch_df["PatchLength"].mean())
        summary["PatchHeightMean"] = float(patch_df["PatchHeight"].mean())
        summary["AzimuthMean"] = float(patch_df["Azimuth"].mean())
        summary["DipMean"] = float(patch_df["Dip"].mean())
        summary["TimeMin"] = float(patch_df[[f"V{i}Z" for i in range(1, 5)]].min().min())
        summary["TimeMax"] = float(patch_df[[f"V{i}Z" for i in range(1, 5)]].max().max())
    if not layers_df.empty:
        summary["LayerIntervals"] = (
            layers_df[["GeoIntervalKey", "TopTime", "BaseTime", "RealSeedCount", "VirtualSeedCount", "PreferredSource"]]
            .fillna("")
            .to_dict(orient="records")
        )
    return summary


def add_layer_rectangles(ax: plt.Axes, unit_row: pd.Series, layers_df: pd.DataFrame) -> None:
    x_min = float(unit_row["XMin"])
    x_max = float(unit_row["XMax"])
    y_min = float(unit_row["YMin"])
    y_max = float(unit_row["YMax"])
    rectangle_xy = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],
        ],
        dtype=float,
    )
    for _, layer in layers_df.iterrows():
        for time_col in ["TopTime", "BaseTime"]:
            z = pd.to_numeric(layer.get(time_col), errors="coerce")
            if pd.isna(z):
                continue
            ax.plot(rectangle_xy[:, 0], rectangle_xy[:, 1], np.full(len(rectangle_xy), float(z)), color="#999999", linewidth=0.6, alpha=0.6)


def save_preview_figure(unit_row: pd.Series, layers_df: pd.DataFrame, patch_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(14, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    color_map = {
        "real_point": "#c0392b",
        "real_segment": "#2980b9",
        "virtual_point": "#d68910",
        "virtual_segment": "#117a65",
        "seismic_gradient_fill": "#16a085",
    }
    add_layer_rectangles(ax3d, unit_row, layers_df)

    legend_added: set[str] = set()
    for _, row in patch_df.iterrows():
        vertices = np.array(
            [[row[f"V{i}X"], row[f"V{i}Y"], row[f"V{i}Z"]] for i in range(1, 5)],
            dtype=float,
        )
        seed_type = str(row["SeedType"])
        source_kind = str(row["SourceKind"]).lower()
        if source_kind == "seismic_gradient_fill":
            style_key = "seismic_gradient_fill"
        elif source_kind == "virtual" and seed_type == "segment":
            style_key = "virtual_segment"
        elif source_kind == "virtual":
            style_key = "virtual_point"
        elif seed_type == "segment":
            style_key = "real_segment"
        else:
            style_key = "real_point"
        color = color_map.get(style_key, "#7f8c8d")
        poly = Poly3DCollection([vertices], alpha=0.45, facecolor=color, edgecolor="#222222", linewidths=0.4)
        ax3d.add_collection3d(poly)
        ax3d.scatter(row["CenterX"], row["CenterY"], row["CenterTIME"], color=color, s=10)

        label = None if style_key in legend_added else style_key
        ax_top.plot(
            np.append(vertices[:, 0], vertices[0, 0]),
            np.append(vertices[:, 1], vertices[0, 1]),
            color=color,
            linewidth=0.9,
            alpha=0.7,
            label=label,
        )
        legend_added.add(style_key)

    x_min = float(unit_row["XMin"])
    x_max = float(unit_row["XMax"])
    y_min = float(unit_row["YMin"])
    y_max = float(unit_row["YMax"])
    z_min = float(unit_row["TopTime"]) if pd.notna(unit_row.get("TopTime")) else float(patch_df[[f"V{i}Z" for i in range(1, 5)]].min().min())
    z_max = float(unit_row["BaseTime"]) if pd.notna(unit_row.get("BaseTime")) else float(patch_df[[f"V{i}Z" for i in range(1, 5)]].max().max())

    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_max, z_min)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("TIME")
    ax3d.set_title(f"{unit_row['UnitID']} DFN 3D Preview")
    ax3d.view_init(elev=24, azim=-58)

    ax_top.set_xlim(x_min, x_max)
    ax_top.set_ylim(y_min, y_max)
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_title(f"{unit_row['UnitID']} Top View")
    if legend_added:
        ax_top.legend(loc="upper right", frameon=False)

    fig.suptitle(
        f"Unit {unit_row['UnitID']} | Mode={unit_row.get('DataMode', '')} | Real={safe_int(unit_row.get('RealSeedCount'))} | Virtual={safe_int(unit_row.get('VirtualSeedCount'))} | Patches={len(patch_df)}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    package_run_dir = args.package_run_dir.resolve()
    output_dir = (args.output_root / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_df = load_unit_catalog(package_run_dir)
    unit_row = select_target_unit(catalog_df, args.unit_id.strip(), args.auto_select_richest, args.data_mode)
    unit_id = str(unit_row["UnitID"])
    layers_df, seeds_df, meta = load_unit_package(package_run_dir, unit_id)
    trace_header_df = load_trace_header(args.trace_header_csv.resolve())
    unit_row = derive_unit_bounds_from_block(
        unit_row,
        trace_header_df,
        block_size_traces=args.block_size_traces,
        block_stride_traces=args.block_stride_traces,
    )

    config = PatchConfig(
        point_base_length=args.point_base_length,
        point_base_height=args.point_base_height,
        segment_base_length=args.segment_base_length,
        segment_base_height=args.segment_base_height,
        density_gain=args.density_gain,
        segment_length_gain=args.segment_length_gain,
        virtual_point_scale=args.virtual_point_size_scale,
        virtual_segment_scale=args.virtual_segment_size_scale,
        gradient_fill_scale=args.gradient_fill_size_scale,
    )
    gradient_summary: dict[str, object] = {"Enabled": not args.disable_gradient_fill}
    gradient_fill_df = pd.DataFrame(columns=list(seeds_df.columns) + ["GradientValue", "ClusterPointCount", "ParentSeedID", "ParentSourceKind"])
    merged_seeds_df = seeds_df.copy()
    if not args.disable_gradient_fill and not layers_df.empty:
        gradient_config = GradientFillConfig(
            trace_header_csv=args.trace_header_csv.resolve(),
            segy_file=args.segy_file.resolve(),
            time_padding_ms=args.gradient_fill_time_padding_ms,
            gradient_threshold=args.gradient_fill_threshold,
            layer_threshold_mode=args.gradient_threshold_mode,
            layer_gradient_quantile=args.gradient_layer_quantile,
            layer_threshold_floor=args.gradient_layer_threshold_floor,
            enable_multiscale_gradient=not args.disable_multiscale_gradient,
            multiscale_sigma_levels=parse_float_tuple(args.gradient_multiscale_sigmas, (0.0, 1.0, 2.0)),
            multiscale_time_sigma_scale=args.gradient_multiscale_time_scale,
            multiscale_combine_mode=args.gradient_multiscale_combine,
            expansion_radius_m=args.gradient_fill_radius_m,
            vertical_radius_ms=args.gradient_fill_vertical_ms,
            dbscan_eps_xy_m=args.gradient_fill_dbscan_eps_xy,
            dbscan_eps_time_ms=args.gradient_fill_dbscan_eps_time,
            min_cluster_samples=args.gradient_fill_min_samples,
            max_clusters_per_seed=args.gradient_fill_max_clusters_per_seed,
            max_candidate_voxels_per_seed=args.gradient_fill_max_candidate_voxels_per_seed,
            fill_density_scale=args.gradient_fill_density_scale,
            fill_length_scale=args.gradient_fill_length_scale,
            virtual_only_threshold_bonus=args.virtual_only_gradient_threshold_bonus,
            virtual_only_radius_scale=args.virtual_only_gradient_radius_scale,
            virtual_only_vertical_scale=args.virtual_only_gradient_vertical_scale,
            virtual_only_max_clusters_per_seed=args.virtual_only_gradient_max_clusters_per_seed,
            enable_seedless_layer_fill=not args.disable_seedless_layer_fill,
            seedless_layer_max_clusters_per_layer=args.seedless_layer_max_clusters_per_layer,
            seedless_layer_min_cluster_samples=args.seedless_layer_min_samples,
            seedless_layer_min_voxels=args.seedless_layer_min_voxels,
            seedless_layer_max_candidate_voxels=args.seedless_layer_max_candidate_voxels,
            seed_proximity_exclusion_xy_m=args.seed_proximity_exclusion_xy_m,
            seed_proximity_exclusion_time_ms=args.seed_proximity_exclusion_time_ms,
            dedup_xy_m=args.dedup_xy_m,
            dedup_time_ms=args.dedup_time_ms,
            dedup_azimuth_deg=args.dedup_azimuth_deg,
            dedup_dip_deg=args.dedup_dip_deg,
            enable_target_fill_ratio=not args.disable_target_fill_ratio,
            target_fill_to_input_ratio=args.target_fill_to_input_ratio,
            target_fill_layer_weighted=not args.disable_target_fill_layer_weighted,
            target_fill_max_candidate_voxels_per_layer=args.target_fill_max_candidate_voxels_per_layer,
        )
        seismic_cube, x_axis, y_axis, time_axis, seismic_summary = load_unit_seismic_cube(
            unit_row,
            layers_df,
            trace_header_df,
            gradient_config,
        )
        gradient_voxel_df, voxel_summary = detect_high_gradient_voxels(
            seismic_cube,
            x_axis,
            y_axis,
            time_axis,
            layers_df,
            gradient_config,
        )
        gradient_fill_df = build_gradient_fill_seeds(
            unit_row,
            layers_df,
            seeds_df,
            gradient_voxel_df,
            gradient_config,
        )
        if not gradient_fill_df.empty:
            merged_seeds_df = pd.concat([seeds_df, gradient_fill_df], ignore_index=True, sort=False)
        gradient_summary.update(seismic_summary)
        gradient_summary.update(voxel_summary)
        gradient_summary["GradientFillSeedCount"] = int(len(gradient_fill_df))
        gradient_summary["GradientFillFromRealCount"] = int(
            gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("real").sum()
        ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
        gradient_summary["GradientFillFromVirtualCount"] = int(
            gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("virtual").sum()
        ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
        gradient_summary["GradientFillDirectCount"] = int(
            gradient_fill_df["ParentSourceKind"].fillna("").astype(str).str.lower().eq("layer_direct").sum()
        ) if not gradient_fill_df.empty and "ParentSourceKind" in gradient_fill_df.columns else 0
        gradient_summary["GradientFillTargetBoostCount"] = int(
            gradient_fill_df["SourceName"].fillna("").astype(str).eq("seismic_gradient_fill_target").sum()
        ) if not gradient_fill_df.empty and "SourceName" in gradient_fill_df.columns else 0
        gradient_summary["GradientFillToInputRatio"] = (
            float(len(gradient_fill_df)) / float(len(seeds_df))
        ) if len(seeds_df) > 0 else 0.0
        gradient_summary["GradientFillLayerCounts"] = (
            gradient_fill_df.groupby("GeoIntervalKey")["SeedID"].count().to_dict() if not gradient_fill_df.empty else {}
        )
    patch_df = build_unit_dfn_patches(unit_row, layers_df, merged_seeds_df, config)
    patch_df = with_sequential_index(patch_df, "PatchIndex")
    merged_seeds_output_df = with_sequential_index(merged_seeds_df, "SeedIndex")
    seeds_input_output_df = with_sequential_index(seeds_df, "SeedIndex")
    gradient_fill_output_df = with_sequential_index(gradient_fill_df, "SeedIndex")
    summary = build_patch_summary(unit_row, layers_df, merged_seeds_df, patch_df)
    summary["InputSeedCount"] = int(len(seeds_df))
    summary["GradientFillSeedCount"] = int(len(gradient_fill_df))
    summary["MergedSeedCount"] = int(len(merged_seeds_df))
    summary["GradientFillFromRealCount"] = safe_int(gradient_summary.get("GradientFillFromRealCount"))
    summary["GradientFillFromVirtualCount"] = safe_int(gradient_summary.get("GradientFillFromVirtualCount"))
    summary["GradientFillDirectCount"] = safe_int(gradient_summary.get("GradientFillDirectCount"))
    summary["GradientFillTargetBoostCount"] = safe_int(gradient_summary.get("GradientFillTargetBoostCount"))
    summary["GradientFillToInputRatio"] = float(gradient_summary.get("GradientFillToInputRatio", 0.0))
    summary["GradientSummary"] = gradient_summary

    unit_output_dir = output_dir / "units" / unit_id
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_utf8(patch_df, unit_output_dir / "unit_dfn_patches.csv")
    write_csv_utf8(seeds_input_output_df, unit_output_dir / "fracture_seeds_input.csv")
    write_csv_utf8(gradient_fill_output_df, unit_output_dir / "gradient_fill_seeds.csv")
    write_csv_utf8(merged_seeds_output_df, unit_output_dir / "fracture_seeds_merged.csv")
    write_csv_utf8(layers_df, unit_output_dir / "unit_layers_input.csv")
    if not patch_df.empty:
        save_preview_figure(unit_row, layers_df, patch_df, unit_output_dir / "unit_dfn_preview.png")
    vtk_export_summary: dict[str, object] = {"Enabled": not args.disable_vtk_export}
    if not args.disable_vtk_export:
        vtk_config = VtkExportConfig(
            display_z_scale=args.vtk_display_z_scale,
            invert_time=not args.vtk_no_invert_time,
        )
        vtk_export_summary.update(export_patch_vtk_files(patch_df, unit_output_dir, vtk_config))
        vtk_export_summary.update(export_seed_vtk_files(merged_seeds_output_df, unit_output_dir, vtk_config))
        mapping_path = write_vtk_mapping_json(unit_output_dir, vtk_export_summary, vtk_config)
        quickstart_path = write_paraview_quickstart(unit_output_dir, unit_id, vtk_config)
        vtk_export_summary["mapping_json"] = str(mapping_path)
        vtk_export_summary["quickstart_md"] = str(quickstart_path)
    summary["VtkExport"] = vtk_export_summary
    (unit_output_dir / "unit_dfn_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (unit_output_dir / "unit_gradient_summary.json").write_text(
        json.dumps(gradient_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "package_run_dir": str(package_run_dir),
        "output_dir": str(output_dir),
        "unit_id": unit_id,
        "data_mode": str(unit_row.get("DataMode", "")),
        "real_seed_count": safe_int(unit_row.get("RealSeedCount")),
        "virtual_seed_count": safe_int(unit_row.get("VirtualSeedCount")),
        "input_seed_count": int(len(seeds_df)),
        "gradient_fill_seed_count": int(len(gradient_fill_df)),
        "gradient_fill_from_real_count": safe_int(gradient_summary.get("GradientFillFromRealCount")),
        "gradient_fill_from_virtual_count": safe_int(gradient_summary.get("GradientFillFromVirtualCount")),
        "gradient_fill_direct_count": safe_int(gradient_summary.get("GradientFillDirectCount")),
        "gradient_fill_target_boost_count": safe_int(gradient_summary.get("GradientFillTargetBoostCount")),
        "gradient_fill_to_input_ratio": float(gradient_summary.get("GradientFillToInputRatio", 0.0)),
        "merged_seed_count": int(len(merged_seeds_df)),
        "patch_count": int(len(patch_df)),
        "layer_count": int(len(layers_df)),
        "vtk_export": vtk_export_summary,
        "meta": meta,
    }
    write_csv_utf8(pd.DataFrame([unit_row]), output_dir / "selected_unit_catalog.csv")
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

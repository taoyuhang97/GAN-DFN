# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
import torch
from tqdm.auto import tqdm

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - fallback path
    cKDTree = None

THIS_DIR = Path(__file__).resolve().parent
PACK_DIR = THIS_DIR.parent / "GAN训练准备" / "训练样本打包"
if str(PACK_DIR) not in sys.path:
    sys.path.append(str(PACK_DIR))

from baseline_common import (
    DEFAULT_DOCX_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SLOTS_PER_VOXEL,
    DEFAULT_UNIT_DFN_ROOT,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_Z_STEP_MS,
    VtkPatchExportConfig,
    append_lines_to_docx,
    build_grid_spec,
    dedupe_patch_df,
    decode_instance_label_to_patches,
    export_patch_vtk_files,
    load_layer_table,
    load_unit_summary,
    prediction_to_label_payload,
    write_csv_utf8,
    write_json,
)
from baseline_model import SparseInstanceBaselineUNet
from build_sparse_instance_gan_dataset import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_SGY_FILE,
    DEFAULT_TRACE_HEADER_CSV,
    build_input_features,
    build_unit_trace_block,
    compute_unit_bounds_from_trace_arrays,
    extract_fixed_window_volume,
    extract_unit_trace_cube,
    generate_layer_windows,
    load_trace_header,
    traces_to_center_volume,
)


UNIT_ID_PATTERN = re.compile(r"^BX(?P<block_x>\d+)_BY(?P<block_y>\d+)$", flags=re.IGNORECASE)
DEFAULT_SURFACE_DIR = Path(r"/data/shared/project-oil/wx数据/砂砾岩/层位")
AUTO_LAYER_SURFACE_CODES = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
AUTO_LAYER_SURFACE_HINTS = {
    "T1": ["馆陶底"],
    "T2": ["沙一下特殊岩性顶"],
    "T3": ["沙二底"],
    "T4": ["沙三上底"],
    "T5": ["20240715", "DM_Sm", "沙三下顶面"],
    "T6": ["20240715", "AtoInt", "DM_Sm", "沙三下底面"],
    "T7": ["gljmAto", "地震", "DM_Sm", "沙四上底面"],
}
AUTO_LAYER_BLOCK_SIZE = 25
AUTO_LAYER_CENTER_OFFSET = AUTO_LAYER_BLOCK_SIZE // 2


@dataclass
class SurfaceNearestLookup:
    surface_code: str
    surface_name: str
    filepath: Path
    points_xy: np.ndarray
    values_z: np.ndarray
    tree: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run production DFN inference for new units using a trained supervised baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unit-dfn-root", type=Path, default=DEFAULT_UNIT_DFN_ROOT)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--trace-header-csv", type=Path, default=DEFAULT_TRACE_HEADER_CSV)
    parser.add_argument("--sgy-file", type=Path, default=DEFAULT_SGY_FILE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "生产推理")
    parser.add_argument("--run-name", type=str, default=f"baseline_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--docx-path", type=Path, default=DEFAULT_DOCX_PATH)
    parser.add_argument("--unit-id", nargs="+")
    parser.add_argument("--unit-id-csv", type=Path)
    parser.add_argument("--block-x-start", type=int)
    parser.add_argument("--block-x-end", type=int)
    parser.add_argument("--block-y-start", type=int)
    parser.add_argument("--block-y-end", type=int)
    parser.add_argument("--limit-units", type=int)
    parser.add_argument("--limit-windows-per-unit", type=int)
    parser.add_argument("--no-layer-constraint", action="store_true")
    parser.add_argument("--disable-seismic-bound-extension", action="store_true")
    parser.add_argument("--time-min", type=float)
    parser.add_argument("--time-max", type=float)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--z-step-ms", type=float, default=DEFAULT_Z_STEP_MS)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--input-channels", nargs="+", default=list(DEFAULT_INPUT_CHANNELS))
    parser.add_argument("--center-threshold", type=float, default=0.7)
    parser.add_argument("--patch-scale-factor", type=float, default=1.0)
    parser.add_argument("--decode-mode", type=str, default="strict", choices=["strict", "relaxed"])
    parser.add_argument("--relaxed-min-count", type=int, default=1)
    parser.add_argument("--max-slots-per-voxel", type=int, default=DEFAULT_SLOTS_PER_VOXEL)
    parser.add_argument("--max-total-patches-per-window", type=int, default=256)
    parser.add_argument("--dedupe-xy-tol-m", type=float, default=6.25)
    parser.add_argument("--dedupe-time-tol-ms", type=float, default=0.4)
    parser.add_argument("--dedupe-azimuth-tol-deg", type=float, default=20.0)
    parser.add_argument("--dedupe-dip-tol-deg", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    parser.add_argument("--save-window-csv", action="store_true")
    parser.add_argument("--save-window-packages", action="store_true")
    parser.add_argument("--partial-save-every-windows", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[SparseInstanceBaselineUNet, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = SparseInstanceBaselineUNet(
        in_channels=int(model_config["in_channels"]),
        slots_per_voxel=int(model_config["slots_per_voxel"]),
        base_channels=int(model_config["base_channels"]),
        dx=float(model_config.get("dx", 12.5)),
        dy=float(model_config.get("dy", 12.5)),
        dz=float(model_config.get("dz", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def parse_unit_id(unit_id: str) -> tuple[int, int]:
    matched = UNIT_ID_PATTERN.match(str(unit_id).strip())
    if not matched:
        raise ValueError(f"invalid UnitID: {unit_id}")
    return int(matched.group("block_x")), int(matched.group("block_y"))


def load_unit_ids_from_csv(unit_id_csv: Path) -> list[str]:
    df = pd.read_csv(unit_id_csv, encoding="utf-8-sig")
    if "UnitID" in df.columns:
        return df["UnitID"].dropna().astype(str).tolist()
    if {"BlockX", "BlockY"}.issubset(df.columns):
        return [f"BX{int(block_x)}_BY{int(block_y)}" for block_x, block_y in df[["BlockX", "BlockY"]].dropna().to_numpy()]
    if df.shape[1] >= 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()
    return []


def collect_selected_unit_ids(args: argparse.Namespace) -> list[str]:
    unit_ids: set[str] = set()
    if args.unit_id:
        unit_ids.update(str(unit_id).strip() for unit_id in args.unit_id if str(unit_id).strip())
    if args.unit_id_csv:
        unit_ids.update(load_unit_ids_from_csv(Path(args.unit_id_csv)))
    if None not in (args.block_x_start, args.block_x_end, args.block_y_start, args.block_y_end):
        for block_x in range(int(args.block_x_start), int(args.block_x_end) + 1):
            for block_y in range(int(args.block_y_start), int(args.block_y_end) + 1):
                unit_ids.add(f"BX{block_x}_BY{block_y}")
    if not unit_ids:
        raise ValueError("please specify target units by --unit-id, --unit-id-csv, or block range")
    ordered = sorted(unit_ids, key=lambda unit_id: parse_unit_id(unit_id))
    if args.limit_units:
        ordered = ordered[: int(args.limit_units)]
    return ordered


def sort_layers(layers_df: pd.DataFrame) -> pd.DataFrame:
    if layers_df.empty:
        return layers_df.copy()
    work = layers_df.copy()
    top = pd.to_numeric(work.get("TopTime"), errors="coerce")
    base = pd.to_numeric(work.get("BaseTime"), errors="coerce")
    work["SortTop"] = np.nanmin(np.vstack([top.to_numpy(dtype=float), base.to_numpy(dtype=float)]), axis=0)
    work = work.sort_values(["SortTop", "GeoIntervalKey"], na_position="last").reset_index(drop=True)
    return work.drop(columns=["SortTop"], errors="ignore")


def compute_unit_time_bounds(layers_df: pd.DataFrame, unit_summary: dict[str, Any]) -> tuple[float, float]:
    values: list[float] = []
    if not layers_df.empty:
        for col in ("TopTime", "BaseTime"):
            if col in layers_df.columns:
                numeric = pd.to_numeric(layers_df[col], errors="coerce").dropna()
                values.extend(float(v) for v in numeric.tolist())
    for key in ("TimeMin", "TimeMax"):
        value = pd.to_numeric(unit_summary.get(key), errors="coerce")
        if pd.notna(value):
            values.append(float(value))
    if not values:
        raise ValueError("failed to determine unit time bounds from layers and unit summary")
    return float(min(values)), float(max(values))


def build_no_constraint_layers_df(block_x: int, block_y: int, time_min: float, time_max: float) -> pd.DataFrame:
    top_time = float(min(time_min, time_max))
    base_time = float(max(time_min, time_max))
    return pd.DataFrame(
        [
            {
                "UnitID": f"BX{int(block_x)}_BY{int(block_y)}",
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "GeoIntervalKey": "FULL_INTERVAL_001",
                "StrataName": "NO_LAYER_CONSTRAINT",
                "TopSurfaceCode": "",
                "BaseSurfaceCode": "",
                "TopTime": float(top_time),
                "BaseTime": float(base_time),
                "TopDepth": np.nan,
                "BaseDepth": np.nan,
            }
        ]
    )


def resolve_no_constraint_time_bounds(
    args: argparse.Namespace,
    seismic_time_min: float,
    seismic_time_max: float,
) -> tuple[float, float]:
    if args.time_min is not None and args.time_max is not None:
        return float(min(args.time_min, args.time_max)), float(max(args.time_min, args.time_max))
    return float(min(seismic_time_min, seismic_time_max)), float(max(seismic_time_min, seismic_time_max))


def normalize_surface_text(text: str) -> str:
    value = str(text).strip()
    return (
        value.replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("：", ":")
        .replace("　", " ")
    )


def snap_to_interval(value: float | None, interval: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(round(value / float(interval)) * float(interval), 6)


def extract_surface_code_from_name(name: str) -> str:
    matched = re.match(r"^(T[1-7])(?:\b|[_\(（].*)?$", normalize_surface_text(name), flags=re.IGNORECASE)
    return matched.group(1).upper() if matched else ""


def score_surface_file(surface_code: str, filepath: Path) -> tuple[int, int, str]:
    normalized = normalize_surface_text(filepath.stem)
    hints = AUTO_LAYER_SURFACE_HINTS.get(surface_code, [])
    score = 0
    if normalized.startswith(surface_code):
        score += 20
    for hint in hints:
        if hint and hint in normalized:
            score += 10
    score += len(hints)
    return (score, len(normalized), normalized)


def select_preferred_surface_files(surface_dir: Path) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {code: [] for code in AUTO_LAYER_SURFACE_CODES}
    for filepath in sorted(Path(surface_dir).glob("*.dat")):
        if filepath.name.lower().startswith("faultstick"):
            continue
        surface_code = extract_surface_code_from_name(filepath.stem)
        if surface_code in grouped:
            grouped[surface_code].append(filepath)
    selected: dict[str, Path] = {}
    for surface_code, files in grouped.items():
        if not files:
            continue
        selected[surface_code] = max(files, key=lambda path: score_surface_file(surface_code, path))
    return selected


def read_dat_surface_points(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float]] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no valid XYZ rows found in surface file: {filepath}")
    data = np.asarray(rows, dtype=float)
    return data[:, :2].copy(), data[:, 2].copy()


def load_surface_nearest_lookups(surface_dir: Path) -> dict[str, SurfaceNearestLookup]:
    lookups: dict[str, SurfaceNearestLookup] = {}
    selected_files = select_preferred_surface_files(surface_dir)
    for surface_code, filepath in selected_files.items():
        points_xy, values_z = read_dat_surface_points(filepath)
        tree = cKDTree(points_xy) if cKDTree is not None else None
        lookups[surface_code] = SurfaceNearestLookup(
            surface_code=surface_code,
            surface_name=filepath.stem,
            filepath=filepath,
            points_xy=points_xy,
            values_z=values_z,
            tree=tree,
        )
    if not lookups:
        raise FileNotFoundError(f"no T1-T7 surface files found under: {surface_dir}")
    return lookups


def query_surface_nearest(
    center_x: float,
    center_y: float,
    lookup: SurfaceNearestLookup,
    z_step_ms: float,
) -> dict[str, Any]:
    query_xy = np.asarray([[float(center_x), float(center_y)]], dtype=float)
    if lookup.tree is not None:
        distance_arr, index_arr = lookup.tree.query(query_xy, k=1)
        nearest_idx = int(index_arr.reshape(-1)[0])
        nearest_distance = float(distance_arr.reshape(-1)[0])
    else:
        deltas = lookup.points_xy - query_xy[0]
        dist2 = np.sum(deltas * deltas, axis=1)
        nearest_idx = int(np.argmin(dist2))
        nearest_distance = float(np.sqrt(dist2[nearest_idx]))
    nearest_x = float(lookup.points_xy[nearest_idx, 0])
    nearest_y = float(lookup.points_xy[nearest_idx, 1])
    nearest_time = float(lookup.values_z[nearest_idx])
    snapped_time = snap_to_interval(nearest_time, z_step_ms)
    return {
        "SurfaceCode": lookup.surface_code,
        "SurfaceName": lookup.surface_name,
        "SurfaceFile": str(lookup.filepath),
        "QueryX": float(center_x),
        "QueryY": float(center_y),
        "NearestX": nearest_x,
        "NearestY": nearest_y,
        "DistanceXY": nearest_distance,
        "RawTime": nearest_time,
        "SnappedTime": float(snapped_time) if snapped_time is not None else np.nan,
        "PointCount": int(len(lookup.values_z)),
    }


def resolve_unit_center_trace_xy(
    unique_x: np.ndarray,
    unique_y: np.ndarray,
    block_x: int,
    block_y: int,
    block_size: int = AUTO_LAYER_BLOCK_SIZE,
) -> tuple[float, float]:
    start_x = int(block_x) * (int(block_size) - 1)
    start_y = int(block_y) * (int(block_size) - 1)
    center_x_idx = start_x + AUTO_LAYER_CENTER_OFFSET
    center_y_idx = start_y + AUTO_LAYER_CENTER_OFFSET
    if center_x_idx >= len(unique_x) or center_y_idx >= len(unique_y):
        raise ValueError(f"center trace index exceeds trace header extent for BlockX={block_x}, BlockY={block_y}")
    return float(unique_x[center_x_idx]), float(unique_y[center_y_idx])


def build_auto_layers_from_surface_rows(
    unit_id: str,
    block_x: int,
    block_y: int,
    resolved_surface_df: pd.DataFrame,
    z_step_ms: float,
) -> pd.DataFrame:
    if resolved_surface_df.empty:
        return pd.DataFrame()
    work = resolved_surface_df.copy()
    work["SnappedTime"] = pd.to_numeric(work["SnappedTime"], errors="coerce")
    work = work.dropna(subset=["SnappedTime"]).sort_values(["SnappedTime", "SurfaceCode"]).reset_index(drop=True)
    if len(work) < 2:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    interval_idx = 1
    for idx in range(len(work) - 1):
        top_row = work.iloc[idx]
        base_row = work.iloc[idx + 1]
        top_time = float(top_row["SnappedTime"])
        base_time = float(base_row["SnappedTime"])
        if not np.isfinite(top_time) or not np.isfinite(base_time):
            continue
        if (base_time - top_time) < float(z_step_ms):
            continue
        rows.append(
            {
                "UnitID": str(unit_id),
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "GeoIntervalKey": f"{interval_idx:03d}_interval_{interval_idx:03d}",
                "StrataName": f"{top_row['SurfaceCode']}->{base_row['SurfaceCode']}",
                "TopSurfaceCode": str(top_row["SurfaceCode"]),
                "BaseSurfaceCode": str(base_row["SurfaceCode"]),
                "TopTime": float(top_time),
                "BaseTime": float(base_time),
                "TopDepth": float(top_time),
                "BaseDepth": float(base_time),
            }
        )
        interval_idx += 1
    return pd.DataFrame(rows)


def extend_layers_to_seismic_bounds(
    layers_df: pd.DataFrame,
    unit_id: str,
    block_x: int,
    block_y: int,
    seismic_time_min: float,
    seismic_time_max: float,
    z_step_ms: float,
) -> pd.DataFrame:
    if layers_df.empty:
        return layers_df.copy()
    work = sort_layers(layers_df).copy()
    work["TopTime"] = pd.to_numeric(work.get("TopTime"), errors="coerce")
    work["BaseTime"] = pd.to_numeric(work.get("BaseTime"), errors="coerce")
    work = work.dropna(subset=["TopTime", "BaseTime"]).reset_index(drop=True)
    if work.empty:
        return work

    seismic_top = snap_to_interval(float(min(seismic_time_min, seismic_time_max)), z_step_ms)
    seismic_base = snap_to_interval(float(max(seismic_time_min, seismic_time_max)), z_step_ms)
    if seismic_top is None or seismic_base is None:
        return work

    first_row = work.iloc[0]
    last_row = work.iloc[-1]
    first_top = float(min(first_row["TopTime"], first_row["BaseTime"]))
    last_base = float(max(last_row["TopTime"], last_row["BaseTime"]))
    rows_to_add: list[dict[str, Any]] = []

    first_surface_code = str(first_row.get("TopSurfaceCode", "")).strip() or "T1"
    last_surface_code = str(last_row.get("BaseSurfaceCode", "")).strip() or "T7"

    if (first_top - seismic_top) >= float(z_step_ms):
        rows_to_add.append(
            {
                "UnitID": str(unit_id),
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "GeoIntervalKey": "000_interval_pre_top",
                "StrataName": f"SEIS_TOP->{first_surface_code}",
                "TopSurfaceCode": "SEIS_TOP",
                "BaseSurfaceCode": first_surface_code,
                "TopTime": float(seismic_top),
                "BaseTime": float(first_top),
                "TopDepth": float(seismic_top),
                "BaseDepth": float(first_top),
            }
        )

    if (seismic_base - last_base) >= float(z_step_ms):
        rows_to_add.append(
            {
                "UnitID": str(unit_id),
                "BlockX": int(block_x),
                "BlockY": int(block_y),
                "GeoIntervalKey": "999_interval_post_base",
                "StrataName": f"{last_surface_code}->SEIS_BASE",
                "TopSurfaceCode": last_surface_code,
                "BaseSurfaceCode": "SEIS_BASE",
                "TopTime": float(last_base),
                "BaseTime": float(seismic_base),
                "TopDepth": float(last_base),
                "BaseDepth": float(seismic_base),
            }
        )

    if not rows_to_add:
        return work

    extended = pd.concat([work, pd.DataFrame(rows_to_add)], ignore_index=True, sort=False)
    extended = sort_layers(extended).reset_index(drop=True)
    extended["GeoIntervalKey"] = [f"{idx:03d}_interval_{idx:03d}" for idx in range(1, len(extended) + 1)]
    return extended


def scale_patch_geometry_about_center(patch_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    factor = float(scale_factor)
    if patch_df.empty or np.isclose(factor, 1.0):
        return patch_df.copy()
    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError(f"patch_scale_factor must be a finite positive number, got {scale_factor}")

    work = patch_df.copy()
    for size_col in ("PatchLength", "PatchHeight"):
        if size_col in work.columns:
            numeric = pd.to_numeric(work[size_col], errors="coerce")
            mask = numeric.notna()
            if mask.any():
                work.loc[mask, size_col] = numeric.loc[mask] * factor

    center_columns = {"X": "CenterX", "Y": "CenterY", "Z": "CenterTIME"}
    for vertex_idx in range(1, 5):
        for axis, center_col in center_columns.items():
            vertex_col = f"V{vertex_idx}{axis}"
            if vertex_col not in work.columns or center_col not in work.columns:
                continue
            vertex = pd.to_numeric(work[vertex_col], errors="coerce")
            center = pd.to_numeric(work[center_col], errors="coerce")
            mask = vertex.notna() & center.notna()
            if mask.any():
                work.loc[mask, vertex_col] = center.loc[mask] + factor * (vertex.loc[mask] - center.loc[mask])
    return work


def scale_decoded_patch_sizes(decoded_df: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    factor = float(scale_factor)
    if decoded_df.empty or np.isclose(factor, 1.0):
        return decoded_df.copy()
    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError(f"patch_scale_factor must be a finite positive number, got {scale_factor}")

    work = decoded_df.copy()
    for size_col in ("PatchLength", "PatchHeight"):
        if size_col in work.columns:
            numeric = pd.to_numeric(work[size_col], errors="coerce")
            mask = numeric.notna()
            if mask.any():
                work.loc[mask, size_col] = numeric.loc[mask] * factor
    return work


def run_single_window(
    model: SparseInstanceBaselineUNet,
    input_features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(np.asarray(input_features, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        center_probs = torch.sigmoid(outputs["center_logits"]).squeeze(0).detach().cpu().numpy()
        count_pred = outputs["count_pred"].squeeze(0).detach().cpu().numpy()
        geom_pred = outputs["geom_pred"].squeeze(0).detach().cpu().numpy()
    return center_probs, count_pred, geom_pred


def save_window_package(output_path: Path, input_features: np.ndarray, valid_z_mask: np.ndarray, target_z_centers: np.ndarray) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        input_features=np.asarray(input_features, dtype=np.float16),
        valid_z_mask=np.asarray(valid_z_mask, dtype=np.uint8),
        target_z_centers=np.asarray(target_z_centers, dtype=np.float32),
    )
    return output_path


def build_output_unit_summary(
    unit_id: str,
    block_x: int,
    block_y: int,
    layers_df: pd.DataFrame,
    predicted_all: pd.DataFrame,
    dedup_pred: pd.DataFrame,
    predicted_vtk: dict[str, Any],
    unit_source_dir: Path,
) -> dict[str, Any]:
    return {
        "UnitID": str(unit_id),
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "LayerCount": int(len(layers_df)),
        "PredictedWindowPatchCount": int(len(predicted_all)),
        "PredictedDedupPatchCount": int(len(dedup_pred)),
        "PredictedRawVTK": predicted_vtk.get("raw_vtk", ""),
        "PredictedDisplayVTK": predicted_vtk.get("display_vtk", ""),
        "SourceUnitDir": str(unit_source_dir),
    }


def emit_progress_message(message: str, show_progress: bool, progress_bar: tqdm | None = None) -> None:
    if show_progress and progress_bar is not None:
        progress_bar.write(message)
    else:
        print(message)


def build_unit_progress_payload(
    unit_id: str,
    block_x: int,
    block_y: int,
    total_window_count: int,
    completed_window_count: int,
    inferred_window_count: int,
    predicted_all: pd.DataFrame,
    dedup_pred: pd.DataFrame,
    is_final: bool,
) -> dict[str, Any]:
    return {
        "UnitID": str(unit_id),
        "BlockX": int(block_x),
        "BlockY": int(block_y),
        "TotalWindowCount": int(total_window_count),
        "CompletedWindowCount": int(completed_window_count),
        "InferredWindowCount": int(inferred_window_count),
        "PredictedWindowPatchCount": int(len(predicted_all)),
        "PredictedDedupPatchCount": int(len(dedup_pred)),
        "IsFinal": bool(is_final),
        "UpdatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_partial_unit_outputs(
    unit_output_dir: Path,
    unit_id: str,
    block_x: int,
    block_y: int,
    layers_df: pd.DataFrame,
    unit_summary: dict[str, Any],
    per_unit_frames: list[pd.DataFrame],
    total_window_count: int,
    completed_window_count: int,
    inferred_window_count: int,
    dedupe_xy_tol_m: float,
    dedupe_time_tol_ms: float,
    dedupe_azimuth_tol_deg: float,
    dedupe_dip_tol_deg: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unit_output_dir.mkdir(parents=True, exist_ok=True)
    predicted_all = pd.concat(per_unit_frames, ignore_index=True, sort=False) if per_unit_frames else pd.DataFrame()
    dedup_pred = dedupe_patch_df(
        predicted_all,
        xy_tol_m=float(dedupe_xy_tol_m),
        time_tol_ms=float(dedupe_time_tol_ms),
        azimuth_tol_deg=float(dedupe_azimuth_tol_deg),
        dip_tol_deg=float(dedupe_dip_tol_deg),
    )
    write_csv_utf8(layers_df, unit_output_dir / "unit_layers_input.csv")
    write_json(unit_output_dir / "source_unit_summary.json", unit_summary)
    write_csv_utf8(predicted_all, unit_output_dir / "predicted_window_concat_patches.partial.csv")
    write_csv_utf8(dedup_pred, unit_output_dir / "predicted_unit_patches.partial.csv")
    write_json(
        unit_output_dir / "unit_prediction_progress.json",
        build_unit_progress_payload(
            unit_id=unit_id,
            block_x=block_x,
            block_y=block_y,
            total_window_count=total_window_count,
            completed_window_count=completed_window_count,
            inferred_window_count=inferred_window_count,
            predicted_all=predicted_all,
            dedup_pred=dedup_pred,
            is_final=False,
        ),
    )
    return predicted_all, dedup_pred


def build_unit_window_jobs(
    layers_df: pd.DataFrame,
    z_min: float,
    z_max: float,
    window_size: int,
    z_step_ms: float,
    overlap_ratio: float,
    limit_windows_per_unit: int | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for _, layer_row in layers_df.iterrows():
        layer_key = str(layer_row.get("GeoIntervalKey", ""))
        layer_top = pd.to_numeric(layer_row.get("TopTime"), errors="coerce")
        layer_base = pd.to_numeric(layer_row.get("BaseTime"), errors="coerce")
        if pd.isna(layer_top) or pd.isna(layer_base):
            continue
        layer_windows = generate_layer_windows(
            layer_top=float(layer_top),
            layer_base=float(layer_base),
            unit_z_min=float(z_min),
            unit_z_max=float(z_max),
            window_size=int(window_size),
            z_step_ms=float(z_step_ms),
            overlap_ratio=float(overlap_ratio),
        )
        for window_idx, (window_top, window_base) in enumerate(layer_windows, start=1):
            jobs.append(
                {
                    "layer_row": layer_row,
                    "layer_key": layer_key,
                    "layer_top": float(layer_top),
                    "layer_base": float(layer_base),
                    "window_idx": int(window_idx),
                    "window_top": float(window_top),
                    "window_base": float(window_base),
                }
            )
            if limit_windows_per_unit and len(jobs) >= int(limit_windows_per_unit):
                return jobs
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    show_progress = not bool(args.no_progress)
    input_channels = list(dict.fromkeys(args.input_channels))
    if not input_channels:
        raise ValueError("input_channels must not be empty")
    if (args.time_min is None) ^ (args.time_max is None):
        raise ValueError("time_min and time_max must be provided together")
    if (not np.isfinite(float(args.patch_scale_factor))) or float(args.patch_scale_factor) <= 0.0:
        raise ValueError(f"patch_scale_factor must be a finite positive number, got {args.patch_scale_factor}")
    enable_seismic_bound_extension = not bool(args.disable_seismic_bound_extension)

    model, checkpoint = load_checkpoint_model(Path(args.checkpoint), device)
    checkpoint_config = checkpoint.get("model_config") or {}
    checkpoint_in_channels = int(checkpoint_config.get("in_channels", len(input_channels)))
    checkpoint_slots = int(checkpoint_config.get("slots_per_voxel", int(args.max_slots_per_voxel)))
    if checkpoint_in_channels != len(input_channels):
        raise ValueError(
            f"checkpoint expects in_channels={checkpoint_in_channels}, but input_channels={len(input_channels)}: {input_channels}"
        )
    if int(args.max_slots_per_voxel) != checkpoint_slots:
        raise ValueError(
            f"checkpoint expects slots_per_voxel={checkpoint_slots}, but max_slots_per_voxel={int(args.max_slots_per_voxel)}"
        )

    target_unit_ids = collect_selected_unit_ids(args)
    run_dir = Path(args.output_root) / args.run_name
    units_dir = run_dir / "units"
    aggregated_dir = run_dir / "aggregated"
    packages_dir = run_dir / "window_packages"
    units_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    if args.save_window_packages:
        packages_dir.mkdir(parents=True, exist_ok=True)

    trace_df, unique_x, unique_y = load_trace_header(Path(args.trace_header_csv))
    surface_lookups: dict[str, SurfaceNearestLookup] = {}
    if not bool(args.no_layer_constraint):
        try:
            surface_lookups = load_surface_nearest_lookups(Path(args.surface_dir))
        except Exception:
            surface_lookups = {}
    selected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    vtk_config = VtkPatchExportConfig(
        display_z_scale=float(args.display_z_scale),
        invert_time=not bool(args.vtk_no_invert_time),
    )

    with segyio.open(str(args.sgy_file), "r", ignore_geometry=True) as sgy:
        seismic_sample_times = np.asarray(sgy.samples, dtype=float)
        seismic_time_min = float(seismic_sample_times.min())
        seismic_time_max = float(seismic_sample_times.max())
        unit_bar = tqdm(
            total=len(target_unit_ids),
            desc="Units",
            dynamic_ncols=True,
            disable=not bool(show_progress),
        )
        for unit_id in target_unit_ids:
            block_x, block_y = parse_unit_id(unit_id)
            source_unit_dir = Path(args.unit_dfn_root) / unit_id
            source_unit_dir_value = str(source_unit_dir) if source_unit_dir.exists() else ""
            resolved_surface_df = pd.DataFrame()
            layer_resolution_summary: dict[str, Any] = {}
            layer_source_mode = ""
            if bool(args.no_layer_constraint):
                z_min, z_max = resolve_no_constraint_time_bounds(
                    args=args,
                    seismic_time_min=seismic_time_min,
                    seismic_time_max=seismic_time_max,
                )
                layers_df = build_no_constraint_layers_df(block_x=block_x, block_y=block_y, time_min=z_min, time_max=z_max)
                unit_summary = {
                    "UnitID": unit_id,
                    "BlockX": int(block_x),
                    "BlockY": int(block_y),
                    "TimeMin": float(z_min),
                    "TimeMax": float(z_max),
                    "NoLayerConstraint": True,
                    "LayerMode": "full_interval",
                    "SeismicBoundExtensionEnabled": bool(enable_seismic_bound_extension),
                    "SourceUnitDirExists": bool(source_unit_dir.exists()),
                }
                layer_source_mode = "full_interval"
            else:
                layers_df = sort_layers(load_layer_table(source_unit_dir)) if source_unit_dir.exists() else pd.DataFrame()
                if not layers_df.empty:
                    if enable_seismic_bound_extension:
                        layers_df = extend_layers_to_seismic_bounds(
                            layers_df=layers_df,
                            unit_id=unit_id,
                            block_x=block_x,
                            block_y=block_y,
                            seismic_time_min=seismic_time_min,
                            seismic_time_max=seismic_time_max,
                            z_step_ms=float(args.z_step_ms),
                        )
                    unit_summary = load_unit_summary(source_unit_dir)
                    block_x = int(unit_summary.get("BlockX", block_x))
                    block_y = int(unit_summary.get("BlockY", block_y))
                    z_min, z_max = compute_unit_time_bounds(layers_df, unit_summary)
                    layer_source_mode = "existing_layer_table"
                    unit_summary["TimeMin"] = float(z_min)
                    unit_summary["TimeMax"] = float(z_max)
                    unit_summary["LayerMode"] = layer_source_mode
                    unit_summary["SeismicBoundExtensionEnabled"] = bool(enable_seismic_bound_extension)
                    if enable_seismic_bound_extension:
                        unit_summary["ExtendedSeismicTimeMin"] = float(seismic_time_min)
                        unit_summary["ExtendedSeismicTimeMax"] = float(seismic_time_max)
                else:
                    if not surface_lookups:
                        skipped_rows.append({"UnitID": unit_id, "BlockX": block_x, "BlockY": block_y, "Reason": "missing_layer_table_and_surface_catalog"})
                        unit_bar.update(1)
                        continue
                    center_x, center_y = resolve_unit_center_trace_xy(unique_x, unique_y, block_x, block_y)
                    resolved_surface_df = pd.DataFrame(
                        [
                            query_surface_nearest(
                                center_x=center_x,
                                center_y=center_y,
                                lookup=surface_lookups[surface_code],
                                z_step_ms=float(args.z_step_ms),
                            )
                            for surface_code in AUTO_LAYER_SURFACE_CODES
                            if surface_code in surface_lookups
                        ]
                    )
                    layers_df = build_auto_layers_from_surface_rows(
                        unit_id=unit_id,
                        block_x=block_x,
                        block_y=block_y,
                        resolved_surface_df=resolved_surface_df,
                        z_step_ms=float(args.z_step_ms),
                    )
                    if enable_seismic_bound_extension:
                        layers_df = extend_layers_to_seismic_bounds(
                            layers_df=layers_df,
                            unit_id=unit_id,
                            block_x=block_x,
                            block_y=block_y,
                            seismic_time_min=seismic_time_min,
                            seismic_time_max=seismic_time_max,
                            z_step_ms=float(args.z_step_ms),
                        )
                    if layers_df.empty:
                        skipped_rows.append({"UnitID": unit_id, "BlockX": block_x, "BlockY": block_y, "Reason": "auto_surface_layer_resolution_failed"})
                        unit_bar.update(1)
                        continue
                    z_min, z_max = compute_unit_time_bounds(layers_df, {})
                    layer_source_mode = "nearest_surface_auto"
                    valid_distance = pd.to_numeric(resolved_surface_df.get("DistanceXY"), errors="coerce")
                    layer_resolution_summary = {
                        "UnitID": unit_id,
                        "BlockX": int(block_x),
                        "BlockY": int(block_y),
                        "LayerSourceMode": layer_source_mode,
                        "CenterTraceX": float(center_x),
                        "CenterTraceY": float(center_y),
                        "SurfaceDir": str(args.surface_dir),
                        "ValidSurfaceCount": int(len(resolved_surface_df)),
                        "IntervalCount": int(len(layers_df)),
                        "SeismicBoundExtensionEnabled": bool(enable_seismic_bound_extension),
                        "MeanNearestDistance": float(valid_distance.mean()) if len(valid_distance) else None,
                        "MaxNearestDistance": float(valid_distance.max()) if len(valid_distance) else None,
                        "MinNearestDistance": float(valid_distance.min()) if len(valid_distance) else None,
                    }
                    if enable_seismic_bound_extension:
                        layer_resolution_summary["ExtendedSeismicTimeMin"] = float(seismic_time_min)
                        layer_resolution_summary["ExtendedSeismicTimeMax"] = float(seismic_time_max)
                    unit_summary = {
                        "UnitID": unit_id,
                        "BlockX": int(block_x),
                        "BlockY": int(block_y),
                        "TimeMin": float(z_min),
                        "TimeMax": float(z_max),
                        "NoLayerConstraint": False,
                        "LayerMode": layer_source_mode,
                        "SourceUnitDirExists": bool(source_unit_dir.exists()),
                        **layer_resolution_summary,
                    }

            layers_df = sort_layers(layers_df)
            selected_rows.append(
                {
                    "UnitID": unit_id,
                    "BlockX": block_x,
                    "BlockY": block_y,
                    "SourceUnitDir": source_unit_dir_value,
                    "LayerSourceMode": layer_source_mode,
                }
            )
            unit_output_dir = units_dir / unit_id
            unit_output_dir.mkdir(parents=True, exist_ok=True)
            write_csv_utf8(layers_df, unit_output_dir / "unit_layers_input.csv")
            write_json(unit_output_dir / "source_unit_summary.json", unit_summary)
            if not resolved_surface_df.empty:
                write_csv_utf8(resolved_surface_df, unit_output_dir / "resolved_layer_surfaces.csv")
                write_json(unit_output_dir / "layer_resolution_summary.json", layer_resolution_summary)

            x_min, x_max, y_min, y_max = compute_unit_bounds_from_trace_arrays(unique_x, unique_y, block_x, block_y)
            grid = build_grid_spec(
                unit_id=unit_id,
                block_x=block_x,
                block_y=block_y,
                x_bounds=(x_min, x_max),
                y_bounds=(y_min, y_max),
                z_bounds=(z_min, z_max),
                xy_resolution=24,
                z_step_ms=float(args.z_step_ms),
            )

            block_trace_df = build_unit_trace_block(trace_df, unique_x, unique_y, block_x, block_y)
            trace_cube = extract_unit_trace_cube(sgy, block_trace_df, grid.z_centers.astype(float))
            unit_center_volume = traces_to_center_volume(trace_cube).astype(np.float32)

            per_unit_frames: list[pd.DataFrame] = []
            processed_window_count = 0
            completed_window_count = 0
            window_jobs = build_unit_window_jobs(
                layers_df=layers_df,
                z_min=float(z_min),
                z_max=float(z_max),
                window_size=int(args.window_size),
                z_step_ms=float(args.z_step_ms),
                overlap_ratio=float(args.overlap_ratio),
                limit_windows_per_unit=int(args.limit_windows_per_unit) if args.limit_windows_per_unit else None,
            )
            total_window_count = int(len(window_jobs))
            write_json(
                unit_output_dir / "unit_prediction_progress.json",
                build_unit_progress_payload(
                    unit_id=unit_id,
                    block_x=block_x,
                    block_y=block_y,
                    total_window_count=total_window_count,
                    completed_window_count=0,
                    inferred_window_count=0,
                    predicted_all=pd.DataFrame(),
                    dedup_pred=pd.DataFrame(),
                    is_final=False,
                ),
            )
            emit_progress_message(
                (
                    f"[UnitStart] {unit_id} "
                    f"index={unit_bar.n + 1}/{len(target_unit_ids)} "
                    f"windows={total_window_count} "
                    f"grid={grid.nx}x{grid.ny}x{grid.nz} "
                    f"time_range={z_min:.3f}-{z_max:.3f} "
                    f"layer_mode={layer_source_mode}"
                ),
                show_progress=show_progress,
                progress_bar=unit_bar,
            )
            window_bar = tqdm(
                total=len(window_jobs),
                desc=f"{unit_id}",
                leave=False,
                dynamic_ncols=True,
                disable=not bool(show_progress),
            )
            for job in window_jobs:
                layer_row = job["layer_row"]
                layer_key = str(job["layer_key"])
                layer_top = float(job["layer_top"])
                layer_base = float(job["layer_base"])
                window_idx = int(job["window_idx"])
                window_top = float(job["window_top"])
                window_base = float(job["window_base"])
                window_volume, target_z_centers, valid_z_mask = extract_fixed_window_volume(
                        unit_center_volume=unit_center_volume,
                        unit_z_centers=grid.z_centers.astype(np.float32),
                        window_top=float(window_top),
                        window_size=int(args.window_size),
                        z_step_ms=float(args.z_step_ms),
                    )
                if int(valid_z_mask.sum()) <= 0:
                    completed_window_count += 1
                    if int(args.partial_save_every_windows) > 0 and completed_window_count % int(args.partial_save_every_windows) == 0:
                        partial_all, partial_dedup = write_partial_unit_outputs(
                            unit_output_dir=unit_output_dir,
                            unit_id=unit_id,
                            block_x=block_x,
                            block_y=block_y,
                            layers_df=layers_df,
                            unit_summary=unit_summary,
                            per_unit_frames=per_unit_frames,
                            total_window_count=total_window_count,
                            completed_window_count=completed_window_count,
                            inferred_window_count=processed_window_count,
                            dedupe_xy_tol_m=float(args.dedupe_xy_tol_m),
                            dedupe_time_tol_ms=float(args.dedupe_time_tol_ms),
                            dedupe_azimuth_tol_deg=float(args.dedupe_azimuth_tol_deg),
                            dedupe_dip_tol_deg=float(args.dedupe_dip_tol_deg),
                        )
                        emit_progress_message(
                            (
                                f"[UnitProgress] {unit_id} "
                                f"completed={completed_window_count}/{total_window_count} "
                                f"inferred={processed_window_count} "
                                f"raw_patches={len(partial_all)} "
                                f"dedup_patches={len(partial_dedup)}"
                            ),
                            show_progress=show_progress,
                            progress_bar=window_bar,
                        )
                    window_bar.update(1)
                    continue
                input_features, input_stats = build_input_features(
                        unit_center_volume=window_volume,
                        dx=grid.dx,
                        dy=grid.dy,
                        dz=grid.dz,
                        target_z_centers=target_z_centers,
                        valid_z_mask=valid_z_mask,
                        layer_top=float(layer_top),
                        layer_base=float(layer_base),
                        input_channels=input_channels,
                    )
                package_path = ""
                sample_id = f"{unit_id}__{layer_key}__W{window_idx:03d}"
                if args.save_window_packages:
                    pkg_path = packages_dir / unit_id / layer_key / f"W{window_idx:03d}.npz"
                    save_window_package(pkg_path, input_features=input_features, valid_z_mask=valid_z_mask, target_z_centers=target_z_centers)
                    package_path = str(pkg_path)

                center_probs, count_pred, geom_pred = run_single_window(model, input_features, device)
                label_payload, decode_filter_stats = prediction_to_label_payload(
                        center_probs=center_probs,
                        count_pred=count_pred,
                        geom_pred=geom_pred,
                        valid_z_mask=valid_z_mask,
                        threshold=float(args.center_threshold),
                        max_slots_per_voxel=int(args.max_slots_per_voxel),
                        max_total_patches=int(args.max_total_patches_per_window) if args.max_total_patches_per_window else None,
                        decode_mode=str(args.decode_mode),
                        relaxed_min_count=int(args.relaxed_min_count),
                    )
                window_grid = build_grid_spec(
                        unit_id=unit_id,
                        block_x=block_x,
                        block_y=block_y,
                        x_bounds=(x_min, x_max),
                        y_bounds=(y_min, y_max),
                        z_bounds=(float(window_top), float(window_base)),
                        xy_resolution=24,
                        z_step_ms=float(args.z_step_ms),
                    )
                window_patch_df, decoded_df, decode_summary = decode_instance_label_to_patches(
                        label_payload=label_payload,
                        grid=window_grid,
                        layers_df=layers_df,
                        threshold=float(args.center_threshold),
                    )
                window_patch_df = scale_patch_geometry_about_center(
                    patch_df=window_patch_df,
                    scale_factor=float(args.patch_scale_factor),
                )
                decoded_df = scale_decoded_patch_sizes(
                    decoded_df=decoded_df,
                    scale_factor=float(args.patch_scale_factor),
                )
                if not window_patch_df.empty:
                    window_patch_df["PredWindowIndex"] = int(window_idx)
                    window_patch_df["PredWindowTopTime"] = float(window_top)
                    window_patch_df["PredWindowBaseTime"] = float(window_base)
                    window_patch_df["PredSampleID"] = sample_id
                    window_patch_df["PatchScaleFactor"] = float(args.patch_scale_factor)
                if not decoded_df.empty:
                    decoded_df["PatchScaleFactor"] = float(args.patch_scale_factor)
                per_unit_frames.append(window_patch_df)
                processed_window_count += 1
                completed_window_count += 1

                window_rows.append(
                    {
                        "SampleID": sample_id,
                        "UnitID": unit_id,
                        "BlockX": block_x,
                        "BlockY": block_y,
                        "GeoIntervalKey": layer_key,
                        "StrataName": str(layer_row.get("StrataName", "")),
                        "TopSurfaceCode": str(layer_row.get("TopSurfaceCode", "")),
                        "BaseSurfaceCode": str(layer_row.get("BaseSurfaceCode", "")),
                        "WindowIndex": int(window_idx),
                        "WindowTopTime": float(window_top),
                        "WindowBaseTime": float(window_base),
                        "WindowValidZCount": int(valid_z_mask.sum()),
                        "PredictedPatchCount": int(len(window_patch_df)),
                        "DecodedActiveSlotCount": int(decode_summary["active_slot_count"]),
                        "RawCandidateCount": int(decode_filter_stats["raw_candidate_count"]),
                        "KeptCandidateCount": int(decode_filter_stats["kept_candidate_count"]),
                        "MeanPredCount": float(decode_filter_stats["mean_pred_count"]),
                        "MaxPredCount": int(decode_filter_stats["max_pred_count"]),
                        "ForcedVoxelCount": int(decode_filter_stats["forced_voxel_count"]),
                        "PatchScaleFactor": float(args.patch_scale_factor),
                        "DecodeMode": str(decode_filter_stats["decode_mode"]),
                        "AmpAbsQ995": float(input_stats["amp_abs_q995"]),
                        "GradMagAbsQ995": float(input_stats["grad_mag_abs_q995"]),
                        "InputChannels": ",".join(input_channels),
                        "PackagePath": package_path,
                    }
                )
                if args.save_window_csv:
                    window_dir = units_dir / unit_id / "windows" / layer_key
                    window_dir.mkdir(parents=True, exist_ok=True)
                    write_csv_utf8(window_patch_df, window_dir / f"W{window_idx:03d}_predicted_patches.csv")
                    write_csv_utf8(decoded_df, window_dir / f"W{window_idx:03d}_decoded_instances.csv")
                window_bar.update(1)
                window_bar.set_postfix(
                    raw=len(window_patch_df),
                    kept=int(decode_filter_stats["kept_candidate_count"]),
                    forced=int(decode_filter_stats["forced_voxel_count"]),
                )
                if int(args.partial_save_every_windows) > 0 and completed_window_count % int(args.partial_save_every_windows) == 0:
                    partial_all, partial_dedup = write_partial_unit_outputs(
                        unit_output_dir=unit_output_dir,
                        unit_id=unit_id,
                        block_x=block_x,
                        block_y=block_y,
                        layers_df=layers_df,
                        unit_summary=unit_summary,
                        per_unit_frames=per_unit_frames,
                        total_window_count=total_window_count,
                        completed_window_count=completed_window_count,
                        inferred_window_count=processed_window_count,
                        dedupe_xy_tol_m=float(args.dedupe_xy_tol_m),
                        dedupe_time_tol_ms=float(args.dedupe_time_tol_ms),
                        dedupe_azimuth_tol_deg=float(args.dedupe_azimuth_tol_deg),
                        dedupe_dip_tol_deg=float(args.dedupe_dip_tol_deg),
                    )
                    emit_progress_message(
                        (
                            f"[UnitProgress] {unit_id} "
                            f"completed={completed_window_count}/{total_window_count} "
                            f"inferred={processed_window_count} "
                            f"raw_patches={len(partial_all)} "
                            f"dedup_patches={len(partial_dedup)}"
                        ),
                        show_progress=show_progress,
                        progress_bar=window_bar,
                    )
            window_bar.close()

            predicted_all = pd.concat(per_unit_frames, ignore_index=True, sort=False) if per_unit_frames else pd.DataFrame()
            dedup_pred = dedupe_patch_df(
                predicted_all,
                xy_tol_m=float(args.dedupe_xy_tol_m),
                time_tol_ms=float(args.dedupe_time_tol_ms),
                azimuth_tol_deg=float(args.dedupe_azimuth_tol_deg),
                dip_tol_deg=float(args.dedupe_dip_tol_deg),
            )
            write_csv_utf8(layers_df, unit_output_dir / "unit_layers_input.csv")
            write_csv_utf8(predicted_all, unit_output_dir / "predicted_window_concat_patches.csv")
            write_csv_utf8(dedup_pred, unit_output_dir / "predicted_unit_patches.csv")
            predicted_vtk = export_patch_vtk_files(
                patch_df=dedup_pred,
                output_dir=unit_output_dir,
                base_name="predicted_patches",
                title_prefix="predicted_patches",
                config=vtk_config,
            )
            write_json(unit_output_dir / "source_unit_summary.json", unit_summary)
            if predicted_vtk.get("mappings"):
                write_json(unit_output_dir / "vtk_attribute_mappings.json", predicted_vtk["mappings"])
            write_json(
                unit_output_dir / "unit_prediction_progress.json",
                build_unit_progress_payload(
                    unit_id=unit_id,
                    block_x=block_x,
                    block_y=block_y,
                    total_window_count=total_window_count,
                    completed_window_count=completed_window_count,
                    inferred_window_count=processed_window_count,
                    predicted_all=predicted_all,
                    dedup_pred=dedup_pred,
                    is_final=True,
                ),
            )

            unit_summary_row = build_output_unit_summary(
                unit_id=unit_id,
                block_x=block_x,
                block_y=block_y,
                layers_df=layers_df,
                predicted_all=predicted_all,
                dedup_pred=dedup_pred,
                predicted_vtk=predicted_vtk,
                unit_source_dir=source_unit_dir,
            )
            unit_summary_row["NoLayerConstraint"] = bool(args.no_layer_constraint)
            unit_summary_row["LayerSourceMode"] = layer_source_mode
            unit_rows.append(unit_summary_row)
            write_json(unit_output_dir / "unit_prediction_summary.json", unit_summary_row)
            unit_message = (
                f"[Unit] {unit_id} "
                f"windows={processed_window_count} "
                f"raw_patches={len(predicted_all)} "
                f"dedup_patches={len(dedup_pred)}"
            )
            if show_progress:
                unit_bar.write(unit_message)
            else:
                print(unit_message)
            unit_bar.update(1)
            unit_bar.set_postfix(
                unit=unit_id,
                windows=processed_window_count,
                dedup=len(dedup_pred),
            )
        unit_bar.close()

    selected_units_df = pd.DataFrame(selected_rows)
    skipped_units_df = pd.DataFrame(skipped_rows)
    window_df = pd.DataFrame(window_rows)
    unit_df = pd.DataFrame(unit_rows)
    write_csv_utf8(selected_units_df, aggregated_dir / "selected_units.csv")
    write_csv_utf8(skipped_units_df, aggregated_dir / "skipped_units.csv")
    write_csv_utf8(window_df, aggregated_dir / "window_prediction_summary.csv")
    write_csv_utf8(unit_df, aggregated_dir / "unit_prediction.csv")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint),
        "unit_dfn_root": str(args.unit_dfn_root),
        "surface_dir": str(args.surface_dir),
        "trace_header_csv": str(args.trace_header_csv),
        "sgy_file": str(args.sgy_file),
        "device": str(device),
        "selected_unit_count": int(len(selected_units_df)),
        "skipped_unit_count": int(len(skipped_units_df)),
        "processed_unit_count": int(len(unit_df)),
        "window_count": int(len(window_df)),
        "mean_raw_candidate_count": float(window_df["RawCandidateCount"].mean()) if not window_df.empty else 0.0,
        "mean_kept_candidate_count": float(window_df["KeptCandidateCount"].mean()) if not window_df.empty else 0.0,
        "mean_forced_voxel_count": float(window_df["ForcedVoxelCount"].mean()) if not window_df.empty else 0.0,
        "mean_predicted_window_patch_count": float(window_df["PredictedPatchCount"].mean()) if not window_df.empty else 0.0,
        "mean_predicted_dedup_patch_count": float(unit_df["PredictedDedupPatchCount"].mean()) if not unit_df.empty else 0.0,
        "input_channels": list(input_channels),
        "center_threshold": float(args.center_threshold),
        "patch_scale_factor": float(args.patch_scale_factor),
        "decode_mode": str(args.decode_mode),
        "relaxed_min_count": int(args.relaxed_min_count),
        "no_layer_constraint": bool(args.no_layer_constraint),
        "seismic_bound_extension_enabled": bool(enable_seismic_bound_extension),
        "time_min": float(args.time_min) if args.time_min is not None else None,
        "time_max": float(args.time_max) if args.time_max is not None else None,
        "max_total_patches_per_window": int(args.max_total_patches_per_window),
        "window_size": int(args.window_size),
        "z_step_ms": float(args.z_step_ms),
        "overlap_ratio": float(args.overlap_ratio),
        "partial_save_every_windows": int(args.partial_save_every_windows),
        "save_window_csv": bool(args.save_window_csv),
        "save_window_packages": bool(args.save_window_packages),
    }
    summary_path = aggregated_dir / "production_inference_summary.json"
    summary["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    append_lines_to_docx(
        docx_path=args.docx_path,
        title="G-DFN监督基线 - 新单元生产推理",
        lines=[
            f"checkpoint: {args.checkpoint}",
            f"run_dir: {run_dir}",
            f"device: {device}",
            f"surface_dir: {args.surface_dir}",
            f"selected_unit_count: {summary['selected_unit_count']}",
            f"skipped_unit_count: {summary['skipped_unit_count']}",
            f"processed_unit_count: {summary['processed_unit_count']}",
            f"window_count: {summary['window_count']}",
            f"patch_scale_factor: {summary['patch_scale_factor']}",
            f"mean_raw_candidate_count: {summary['mean_raw_candidate_count']}",
            f"mean_kept_candidate_count: {summary['mean_kept_candidate_count']}",
            f"mean_forced_voxel_count: {summary['mean_forced_voxel_count']}",
            f"decode_mode: {summary['decode_mode']}",
            f"relaxed_min_count: {summary['relaxed_min_count']}",
            f"no_layer_constraint: {summary['no_layer_constraint']}",
            f"seismic_bound_extension_enabled: {summary['seismic_bound_extension_enabled']}",
            f"time_min: {summary['time_min']}",
            f"time_max: {summary['time_max']}",
            f"mean_predicted_window_patch_count: {summary['mean_predicted_window_patch_count']}",
            f"mean_predicted_dedup_patch_count: {summary['mean_predicted_dedup_patch_count']}",
            f"summary_json: {summary_path}",
        ],
    )

    print(f"run_dir: {run_dir}")
    print(f"processed_unit_count: {summary['processed_unit_count']}")
    print(f"window_count: {summary['window_count']}")
    print(f"summary_json: {summary_path}")


if __name__ == "__main__":
    main()

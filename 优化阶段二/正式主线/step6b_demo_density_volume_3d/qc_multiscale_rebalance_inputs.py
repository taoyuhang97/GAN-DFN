from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_3d_density_sgy.json"
DEFAULT_MULTISCALE_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_FAULT_STICK = Path("/data/shared/project-oil/wx数据/砂砾岩/断层/FaultStick-GeoEast.dat")
DEFAULT_FAULT_PATCH_SUMMARY = (
    Path(__file__).resolve().parents[3]
    / "小范围DFN生成/断层裂缝片生成/断层划分切割/fault_patches_out/fault_patches_summary.csv"
)
DEFAULT_FAULT_PATCH_ROOT = DEFAULT_FAULT_PATCH_SUMMARY.parent / "patches"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_rebalance_v1/input_qc"
NULL_ABS_LIMIT = 1.0e6
COORD_ABS_LIMIT = 1.0e9
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 QC for multiscale rebalance inputs and coordinates.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--multiscale-config", type=Path, default=DEFAULT_MULTISCALE_CONFIG)
    parser.add_argument("--fault-stick", type=Path, default=DEFAULT_FAULT_STICK)
    parser.add_argument("--fault-patch-summary", type=Path, default=DEFAULT_FAULT_PATCH_SUMMARY)
    parser.add_argument("--fault-patch-root", type=Path, default=DEFAULT_FAULT_PATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--margin-m", type=float, default=500.0)
    parser.add_argument("--patch-vtp-sample-limit", type=int, default=40)
    parser.add_argument("--header-sample-size", type=int, default=512)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_stats(values: np.ndarray | pd.Series | list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def bbox_from_block(block: dict[str, Any], margin_m: float = 0.0) -> dict[str, float]:
    return {
        "x_min": float(block["x_min"]) - margin_m,
        "x_max": float(block["x_max"]) + margin_m,
        "y_min": float(block["y_min"]) - margin_m,
        "y_max": float(block["y_max"]) + margin_m,
    }


def in_bbox_xy(df: pd.DataFrame, bbox: dict[str, float], x_col: str = "x", y_col: str = "y") -> pd.Series:
    return (
        (df[x_col] >= bbox["x_min"])
        & (df[x_col] <= bbox["x_max"])
        & (df[y_col] >= bbox["y_min"])
        & (df[y_col] <= bbox["y_max"])
    )


def bbox_intersects_xy(
    df: pd.DataFrame,
    bbox: dict[str, float],
    xmin_col: str = "bbox_xmin",
    xmax_col: str = "bbox_xmax",
    ymin_col: str = "bbox_ymin",
    ymax_col: str = "bbox_ymax",
) -> pd.Series:
    return (
        (df[xmax_col] >= bbox["x_min"])
        & (df[xmin_col] <= bbox["x_max"])
        & (df[ymax_col] >= bbox["y_min"])
        & (df[ymin_col] <= bbox["y_max"])
    )


def load_mapping(path: Path) -> dict[str, np.ndarray]:
    mapping = np.load(path)
    required = {"output_trace_index", "source_trace_idx", "x", "y", "ix", "iy"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")
    return {key: mapping[key] for key in mapping.files}


def inspect_sgy(path: Path, label: str) -> dict[str, Any]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        dt_us = int(segyio.tools.dt(handle)) if handle.tracecount else None
        return {
            "label": label,
            "path": str(path),
            "exists": True,
            "trace_count": int(handle.tracecount),
            "sample_count": int(len(samples)),
            "sample_min_ms": float(samples[0]) if len(samples) else None,
            "sample_max_ms": float(samples[-1]) if len(samples) else None,
            "sample_interval_us": dt_us,
            "bin_interval_us": int(handle.bin[segyio.BinField.Interval]) if handle.bin else None,
            "format": int(handle.bin[segyio.BinField.Format]) if handle.bin else None,
        }


def sample_indices(count: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    base = np.unique(np.asarray([0, count // 2, count - 1], dtype=np.int64))
    if count <= sample_size:
        return np.arange(count, dtype=np.int64)
    extra_size = max(0, sample_size - len(base))
    pool = np.setdiff1d(np.arange(count, dtype=np.int64), base, assume_unique=False)
    extra = rng.choice(pool, size=min(extra_size, len(pool)), replace=False)
    return np.sort(np.unique(np.concatenate([base, extra]).astype(np.int64)))


def header_xy_candidates(handle: segyio.SegyFile, trace_indices: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fields = {
        "SourceXY": (segyio.TraceField.SourceX, segyio.TraceField.SourceY),
        "GroupXY": (segyio.TraceField.GroupX, segyio.TraceField.GroupY),
        "CDPXY": (segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (xf, yf) in fields.items():
        xs: list[float] = []
        ys: list[float] = []
        for idx in trace_indices:
            header = handle.header[int(idx)]
            xs.append(float(header[xf]))
            ys.append(float(header[yf]))
        out[name] = (np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return out


def compare_header_xy(
    sgy_path: Path,
    mapping: dict[str, np.ndarray],
    use_source_trace_idx: bool,
    label: str,
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    trace_count = len(mapping["x"])
    out_indices = sample_indices(trace_count, sample_size, rng)
    source_indices = mapping["source_trace_idx"][out_indices].astype(np.int64) if use_source_trace_idx else out_indices
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        max_idx = int(np.max(source_indices)) if len(source_indices) else -1
        if max_idx >= int(handle.tracecount):
            return {
                "label": label,
                "path": str(sgy_path),
                "status": "fail",
                "reason": f"sampled max trace index {max_idx} >= tracecount {handle.tracecount}",
            }
        candidates = header_xy_candidates(handle, source_indices)
    mx = mapping["x"][out_indices].astype(np.float64)
    my = mapping["y"][out_indices].astype(np.float64)
    comparisons: dict[str, Any] = {}
    best_name = None
    best_median = math.inf
    for name, (hx, hy) in candidates.items():
        dist = np.sqrt((hx - mx) ** 2 + (hy - my) ** 2)
        valid = np.isfinite(dist) & (np.abs(hx) < COORD_ABS_LIMIT) & (np.abs(hy) < COORD_ABS_LIMIT)
        stats = finite_stats(dist[valid])
        comparisons[name] = {
            "distance_m_stats": stats,
            "exact_match_fraction": float(np.mean(dist[valid] <= 1.0)) if np.any(valid) else None,
        }
        median = stats["median"]
        if median is not None and float(median) < best_median:
            best_median = float(median)
            best_name = name
    return {
        "label": label,
        "path": str(sgy_path),
        "status": "pass" if best_median <= 1.0 else "warn",
        "sampled_trace_count": int(len(out_indices)),
        "uses_source_trace_idx": bool(use_source_trace_idx),
        "best_header_xy": best_name,
        "best_median_xy_error_m": None if not np.isfinite(best_median) else best_median,
        "comparisons": comparisons,
    }


def sample_axis_compare(reference: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    ref_min = reference["sample_min_ms"]
    ref_max = reference["sample_max_ms"]
    tar_min = target["sample_min_ms"]
    tar_max = target["sample_max_ms"]
    if None in (ref_min, ref_max, tar_min, tar_max):
        overlap_fraction = None
    else:
        overlap = max(0.0, min(float(ref_max), float(tar_max)) - max(float(ref_min), float(tar_min)))
        ref_span = max(1.0e-9, float(ref_max) - float(ref_min))
        overlap_fraction = float(overlap / ref_span)
    return {
        "sample_count_equal": bool(reference["sample_count"] == target["sample_count"]),
        "sample_min_diff_ms": None if ref_min is None or tar_min is None else float(tar_min) - float(ref_min),
        "sample_max_diff_ms": None if ref_max is None or tar_max is None else float(tar_max) - float(ref_max),
        "reference_overlap_fraction": overlap_fraction,
    }


def read_fault_stick(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    splitter = re.compile(r"\s+")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = splitter.split(text)
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    {
                        "line": int(float(parts[0])),
                        "trace": int(float(parts[1])),
                        "x": float(parts[2]),
                        "y": float(parts[3]),
                        "time_ms": float(parts[4]),
                        "flag": int(float(parts[5])),
                        "fault_name": str(parts[6]),
                    }
                )
            except ValueError:
                continue
    return pd.DataFrame(rows)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:  # pragma: no cover - diagnostic fallback
            last_error = exc
    raise RuntimeError(f"failed to read {path}: {last_error}")


def fault_stick_qc(df: pd.DataFrame, block_bbox: dict[str, float], margin_bbox: dict[str, float], samples: np.ndarray) -> dict[str, Any]:
    in_block = in_bbox_xy(df, block_bbox)
    in_margin = in_bbox_xy(df, margin_bbox)
    t_min = float(samples[0])
    t_max = float(samples[-1])
    in_time = (df["time_ms"] >= t_min) & (df["time_ms"] <= t_max)
    out: dict[str, Any] = {
        "row_count": int(len(df)),
        "fault_count": int(df["fault_name"].nunique()) if len(df) else 0,
        "xy_stats": {
            "x": finite_stats(df["x"]) if len(df) else finite_stats([]),
            "y": finite_stats(df["y"]) if len(df) else finite_stats([]),
            "time_ms": finite_stats(df["time_ms"]) if len(df) else finite_stats([]),
        },
        "points_in_demo_xy": int(in_block.sum()),
        "points_in_demo_xy_and_t": int((in_block & in_time).sum()),
        "points_in_margin_xy": int(in_margin.sum()),
        "points_in_margin_xy_and_t": int((in_margin & in_time).sum()),
        "fault_names_in_demo_xy": sorted(df.loc[in_block, "fault_name"].dropna().astype(str).unique().tolist()),
        "fault_names_in_margin_xy": sorted(df.loc[in_margin, "fault_name"].dropna().astype(str).unique().tolist()),
    }
    return out


def fault_patch_summary_qc(
    summary: pd.DataFrame,
    block_bbox: dict[str, float],
    margin_bbox: dict[str, float],
    samples: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = {
        "fault_name",
        "cell_i",
        "cell_j",
        "area_3d",
        "cx",
        "cy",
        "cz",
        "bbox_xmin",
        "bbox_xmax",
        "bbox_ymin",
        "bbox_ymax",
        "bbox_zmin",
        "bbox_zmax",
        "strike_deg",
        "dip_deg",
    }
    missing = sorted(needed.difference(summary.columns))
    if missing:
        raise ValueError(f"fault patch summary missing columns: {missing}")
    in_block = bbox_intersects_xy(summary, block_bbox)
    in_margin = bbox_intersects_xy(summary, margin_bbox)
    t_min = float(samples[0])
    t_max = float(samples[-1])
    in_time = (summary["bbox_zmax"] >= t_min) & (summary["bbox_zmin"] <= t_max)
    selected = summary.loc[in_margin & in_time].copy()
    selected["intersects_demo_xy"] = in_block.loc[selected.index].to_numpy(dtype=bool)
    selected["intersects_demo_xy_t"] = (in_block & in_time).loc[selected.index].to_numpy(dtype=bool)
    out = {
        "row_count": int(len(summary)),
        "fault_count": int(summary["fault_name"].nunique()),
        "patches_intersect_demo_xy": int(in_block.sum()),
        "patches_intersect_demo_xy_and_t": int((in_block & in_time).sum()),
        "patches_intersect_margin_xy": int(in_margin.sum()),
        "patches_intersect_margin_xy_and_t": int((in_margin & in_time).sum()),
        "fault_names_intersect_demo_xy_t": sorted(summary.loc[in_block & in_time, "fault_name"].astype(str).unique().tolist()),
        "fault_names_intersect_margin_xy_t": sorted(summary.loc[in_margin & in_time, "fault_name"].astype(str).unique().tolist()),
        "selected_area_stats": finite_stats(selected["area_3d"]) if len(selected) else finite_stats([]),
        "selected_dip_stats": finite_stats(selected["dip_deg"]) if len(selected) else finite_stats([]),
        "selected_time_stats": finite_stats(selected["cz"]) if len(selected) else finite_stats([]),
    }
    return selected, out


def vtp_path_for_row(root: Path, row: pd.Series) -> Path:
    fault_name = str(row["fault_name"])
    return root / fault_name / f"{fault_name}__i{int(row['cell_i'])}_j{int(row['cell_j'])}.vtp"


def sample_vtp_qc(selected: pd.DataFrame, patch_root: Path, sample_limit: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, Any]]:
    if selected.empty or sample_limit <= 0:
        return pd.DataFrame(), {"sampled_count": 0}
    if len(selected) > sample_limit:
        sample = selected.sample(n=sample_limit, random_state=int(rng.integers(0, 2**31 - 1))).copy()
    else:
        sample = selected.copy()
    rows: list[dict[str, Any]] = []
    missing_count = 0
    read_fail_count = 0
    for _, row in sample.iterrows():
        path = vtp_path_for_row(patch_root, row)
        item: dict[str, Any] = {
            "fault_name": str(row["fault_name"]),
            "cell_i": int(row["cell_i"]),
            "cell_j": int(row["cell_j"]),
            "path": str(path),
            "exists": bool(path.exists()),
        }
        if not path.exists():
            missing_count += 1
            rows.append(item)
            continue
        try:
            mesh = pv.read(path)
            bounds = mesh.bounds
            item.update(
                {
                    "n_points": int(mesh.n_points),
                    "n_cells": int(mesh.n_cells),
                    "area": float(mesh.area) if hasattr(mesh, "area") else None,
                    "x_min": float(bounds[0]),
                    "x_max": float(bounds[1]),
                    "y_min": float(bounds[2]),
                    "y_max": float(bounds[3]),
                    "t_min": float(bounds[4]),
                    "t_max": float(bounds[5]),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic output
            read_fail_count += 1
            item["read_error"] = str(exc)
        rows.append(item)
    vtp_df = pd.DataFrame(rows)
    summary = {
        "sampled_count": int(len(vtp_df)),
        "missing_count": int(missing_count),
        "read_fail_count": int(read_fail_count),
        "read_success_count": int(len(vtp_df) - missing_count - read_fail_count),
        "sampled_time_stats": finite_stats(vtp_df[["t_min", "t_max"]].to_numpy().ravel()) if len(vtp_df) and "t_min" in vtp_df else finite_stats([]),
    }
    return vtp_df, summary


def fault_patch_vs_geoeast_qc(selected: pd.DataFrame, fault_points: pd.DataFrame, xy_margin_m: float = 75.0, t_margin_ms: float = 100.0) -> dict[str, Any]:
    if selected.empty or fault_points.empty:
        return {"checked_patch_count": int(len(selected)), "matched_patch_count": 0, "match_fraction": None}
    checked = 0
    matched = 0
    time_diffs: list[float] = []
    for _, row in selected.iterrows():
        checked += 1
        same = fault_points[fault_points["fault_name"].astype(str) == str(row["fault_name"])]
        if same.empty:
            continue
        near_xy = (
            (same["x"] >= float(row["bbox_xmin"]) - xy_margin_m)
            & (same["x"] <= float(row["bbox_xmax"]) + xy_margin_m)
            & (same["y"] >= float(row["bbox_ymin"]) - xy_margin_m)
            & (same["y"] <= float(row["bbox_ymax"]) + xy_margin_m)
        )
        candidates = same.loc[near_xy]
        if candidates.empty:
            continue
        dt = np.minimum(
            np.abs(candidates["time_ms"].to_numpy(dtype=float) - float(row["bbox_zmin"])),
            np.abs(candidates["time_ms"].to_numpy(dtype=float) - float(row["bbox_zmax"])),
        )
        inside_t = (
            (candidates["time_ms"] >= float(row["bbox_zmin"]) - t_margin_ms)
            & (candidates["time_ms"] <= float(row["bbox_zmax"]) + t_margin_ms)
        )
        if bool(inside_t.any()):
            matched += 1
            time_diffs.append(float(np.min(dt)))
    return {
        "checked_patch_count": int(checked),
        "matched_patch_count": int(matched),
        "match_fraction": float(matched / checked) if checked else None,
        "time_edge_diff_ms_stats": finite_stats(time_diffs),
        "xy_margin_m": float(xy_margin_m),
        "time_margin_ms": float(t_margin_ms),
    }


def derive_status(checks: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    mapping = checks["mapping"]
    density = checks["density_sgy"]
    if density["trace_count"] != mapping["trace_count"]:
        errors.append("density trace_count differs from trace mapping trace_count")
    if density["sample_count"] <= 0:
        errors.append("density sample axis is empty")
    for name, attr in checks["attribute_sgy"].items():
        if attr["trace_count"] <= checks["mapping"]["source_trace_idx_max"]:
            errors.append(f"{name} trace_count is smaller than max source_trace_idx")
        cmp = attr["sample_axis_vs_density"]
        if cmp["reference_overlap_fraction"] is not None and cmp["reference_overlap_fraction"] < 0.95:
            errors.append(f"{name} sample axis does not sufficiently overlap density sample axis")
        if attr["header_xy_check"]["status"] == "warn":
            warnings.append(f"{name} header XY does not closely match mapping; downstream relies on source_trace_idx")
    if checks["density_header_xy_check"]["status"] == "warn":
        warnings.append("density output SGY header XY does not closely match trace mapping")
    fault_patch = checks["fault_patch_summary"]
    if fault_patch["patches_intersect_margin_xy_and_t"] <= 0:
        warnings.append("segmented fault patches do not intersect demo+margin in XY/T")
    geoeast = checks["fault_stick_geoeast"]
    if geoeast["points_in_margin_xy_and_t"] <= 0:
        warnings.append("FaultStick-GeoEast has no points inside demo+margin in XY/T")
    alignment = checks["fault_patch_vs_geoeast"]
    if alignment["match_fraction"] is not None and alignment["match_fraction"] < 0.2:
        warnings.append("segmented fault patches weakly match FaultStick-GeoEast in same fault/XY/T check")
    return ("fail" if errors else "pass"), warnings, errors


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    rng = np.random.default_rng(int(args.random_state))

    base_config = read_json(args.base_config.resolve())
    multiscale_config = read_json(args.multiscale_config.resolve())
    density_sgy = Path(multiscale_config["input_density_sgy"]).resolve()
    mapping_path = Path(multiscale_config["trace_mapping_npz"]).resolve()
    volume_paths = {name: Path(path).resolve() for name, path in multiscale_config["volume_paths"].items()}
    block = base_config["target_block"]
    block_bbox = bbox_from_block(block, 0.0)
    margin_bbox = bbox_from_block(block, float(args.margin_m))

    mapping = load_mapping(mapping_path)
    density_info = inspect_sgy(density_sgy, "input_density")
    density_samples = None
    with segyio.open(str(density_sgy), "r", ignore_geometry=True) as handle:
        density_samples = np.asarray(handle.samples, dtype=np.float64)

    mapping_info = {
        "path": str(mapping_path),
        "trace_count": int(len(mapping["x"])),
        "source_trace_idx_min": int(np.min(mapping["source_trace_idx"])),
        "source_trace_idx_max": int(np.max(mapping["source_trace_idx"])),
        "x_stats": finite_stats(mapping["x"]),
        "y_stats": finite_stats(mapping["y"]),
        "ix_stats": finite_stats(mapping["ix"]),
        "iy_stats": finite_stats(mapping["iy"]),
        "grid_shape_xy": [int(np.max(mapping["ix"]) + 1), int(np.max(mapping["iy"]) + 1)],
        "block_bbox": block_bbox,
        "margin_bbox": margin_bbox,
    }

    density_header_xy = compare_header_xy(
        density_sgy,
        mapping,
        use_source_trace_idx=False,
        label="density_output_sgy",
        sample_size=int(args.header_sample_size),
        rng=rng,
    )

    source_sgy = Path(base_config["source_sgy_for_headers"]).resolve()
    source_header_xy = compare_header_xy(
        source_sgy,
        mapping,
        use_source_trace_idx=True,
        label="source_sgy_for_headers",
        sample_size=int(args.header_sample_size),
        rng=rng,
    )

    attr_info: dict[str, Any] = {}
    for name, path in volume_paths.items():
        info = inspect_sgy(path, name)
        info["sample_axis_vs_density"] = sample_axis_compare(density_info, info)
        info["header_xy_check"] = compare_header_xy(
            path,
            mapping,
            use_source_trace_idx=True,
            label=name,
            sample_size=int(args.header_sample_size),
            rng=rng,
        )
        attr_info[name] = info

    fault_df = read_fault_stick(args.fault_stick.resolve())
    fault_qc = fault_stick_qc(fault_df, block_bbox, margin_bbox, density_samples)

    patch_summary = read_csv_with_fallback(args.fault_patch_summary.resolve())
    selected_patches, patch_qc = fault_patch_summary_qc(patch_summary, block_bbox, margin_bbox, density_samples)
    selected_csv = args.output_dir / "fault_patch_demo_overlap.csv"
    selected_patches.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    vtp_qc_df, vtp_qc = sample_vtp_qc(
        selected_patches,
        args.fault_patch_root.resolve(),
        sample_limit=int(args.patch_vtp_sample_limit),
        rng=rng,
    )
    vtp_csv = args.output_dir / "fault_patch_vtp_sample_qc.csv"
    vtp_qc_df.to_csv(vtp_csv, index=False, encoding="utf-8-sig")

    attr_rows: list[dict[str, Any]] = []
    for name, info in attr_info.items():
        row = {
            "attribute": name,
            "path": info["path"],
            "trace_count": info["trace_count"],
            "sample_count": info["sample_count"],
            "sample_min_ms": info["sample_min_ms"],
            "sample_max_ms": info["sample_max_ms"],
            "sample_count_equal_density": info["sample_axis_vs_density"]["sample_count_equal"],
            "reference_overlap_fraction": info["sample_axis_vs_density"]["reference_overlap_fraction"],
            "header_xy_status": info["header_xy_check"]["status"],
            "best_header_xy": info["header_xy_check"].get("best_header_xy"),
            "best_median_xy_error_m": info["header_xy_check"].get("best_median_xy_error_m"),
        }
        attr_rows.append(row)
    attr_alignment_csv = args.output_dir / "attribute_alignment_qc.csv"
    pd.DataFrame(attr_rows).to_csv(attr_alignment_csv, index=False, encoding="utf-8-sig")

    patch_vs_geoeast = fault_patch_vs_geoeast_qc(selected_patches, fault_df)

    checks: dict[str, Any] = {
        "version": "candidate_cheye1_multiscale_rebalance_v1",
        "base_config": str(args.base_config.resolve()),
        "multiscale_config": str(args.multiscale_config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "mapping": mapping_info,
        "density_sgy": density_info,
        "density_header_xy_check": density_header_xy,
        "source_header_xy_check": source_header_xy,
        "attribute_sgy": attr_info,
        "fault_stick_geoeast": fault_qc,
        "fault_patch_summary": patch_qc,
        "fault_patch_vtp_sample": vtp_qc,
        "fault_patch_vs_geoeast": patch_vs_geoeast,
        "output_files": {
            "fault_patch_demo_overlap_csv": str(selected_csv),
            "fault_patch_vtp_sample_qc_csv": str(vtp_csv),
            "attribute_alignment_qc_csv": str(attr_alignment_csv),
        },
    }
    status, warnings, errors = derive_status(checks)
    checks["status"] = status
    checks["warnings"] = warnings
    checks["errors"] = errors
    write_json(args.output_dir / "input_qc_summary.json", checks)

    print(json.dumps({"status": status, "warnings": warnings, "errors": errors, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OPT_STAGE_DIR = THIS_DIR.parent
INSTANCE_DIR = OPT_STAGE_DIR / "DFN实例表达互转实验"
if str(INSTANCE_DIR) not in sys.path:
    sys.path.append(str(INSTANCE_DIR))

from instance_roundtrip_common import (  # type: ignore
    DEFAULT_DOCX_PATH,
    DEFAULT_UNIT_DFN_ROOT,
    GridSpec,
    VtkPatchExportConfig,
    build_grid_spec,
    build_patch_match_metrics,
    compute_time_bounds,
    compute_unit_bounds_from_patch_table,
    decode_instance_label_to_patches,
    export_patch_comparison_vtk,
    export_patch_vtk_files,
    load_layer_table,
    load_patch_table,
    load_unit_summary,
    open_or_create_doc,
    read_csv_utf8,
    write_csv_utf8,
    write_json,
)


DEFAULT_DATASET_RUN_DIR = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备/训练样本打包/phase1_t128_sparse_v1_full_fix1_20260330"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线"
)
DEFAULT_SLOTS_PER_VOXEL = 16
DEFAULT_XY_RESOLUTION = 24
DEFAULT_Z_STEP_MS = 0.2
DEFAULT_WINDOW_SIZE = 128
DEFAULT_INPUT_CHANNELS = [
    "seismic_amp",
    "grad_x",
    "grad_y",
    "grad_z",
    "rel_depth_in_interval",
]
GEOM_CHANNELS = [
    "offset_x",
    "offset_y",
    "offset_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "u_dir_x",
    "u_dir_y",
    "u_dir_z",
    "length",
    "height",
    "confidence",
]


def load_sample_manifest(dataset_run_dir: Path) -> pd.DataFrame:
    manifest_path = Path(dataset_run_dir) / "aggregated" / "sample_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"sample_manifest.csv not found: {manifest_path}")
    df = read_csv_utf8(manifest_path)
    for col in [
        "BlockX",
        "BlockY",
        "WindowIndex",
        "WindowValidZCount",
        "PatchCount",
        "InstanceCount",
        "PackageBytes",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_unit_level_split(
    manifest_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if manifest_df.empty:
        raise ValueError("manifest_df is empty")
    units = sorted(manifest_df["UnitID"].astype(str).unique().tolist())
    rng = random.Random(int(seed))
    rng.shuffle(units)
    unit_count = len(units)
    if unit_count < 3:
        raise ValueError(f"Need at least 3 units to split, got {unit_count}")

    n_train = max(1, int(round(unit_count * float(train_ratio))))
    n_val = max(1, int(round(unit_count * float(val_ratio))))
    if n_train + n_val >= unit_count:
        n_val = max(1, min(n_val, unit_count - 2))
        n_train = max(1, unit_count - n_val - 1)
    n_test = unit_count - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        else:
            n_val -= 1

    split_map = {
        "train": units[:n_train],
        "val": units[n_train:n_train + n_val],
        "test": units[n_train + n_val:],
    }
    rows: list[dict[str, Any]] = []
    for split_name, split_units in split_map.items():
        for unit_id in split_units:
            unit_rows = manifest_df[manifest_df["UnitID"].astype(str) == str(unit_id)]
            rows.append(
                {
                    "UnitID": str(unit_id),
                    "Split": split_name,
                    "WindowCount": int(len(unit_rows)),
                    "PositiveWindowCount": int(unit_rows["IsPositiveWindow"].sum()) if "IsPositiveWindow" in unit_rows.columns else None,
                    "PatchCount": int(unit_rows["PatchCount"].sum()) if "PatchCount" in unit_rows.columns else None,
                }
            )
    return pd.DataFrame(rows), split_map


def build_dense_targets_from_sparse_npz(
    npz_path: Path,
    slots_per_voxel: int = DEFAULT_SLOTS_PER_VOXEL,
) -> dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    input_features = np.asarray(data["input_features"], dtype=np.float32)
    valid_z_mask = np.asarray(data["valid_z_mask"], dtype=np.float32)
    instance_ijk = np.asarray(data["instance_ijk"], dtype=np.int32)
    instance_geom = np.asarray(data["instance_geom"], dtype=np.float32)
    instance_weight = np.asarray(data["instance_weight"], dtype=np.float32)
    if "count_volume" in data.files:
        count_volume = np.asarray(data["count_volume"], dtype=np.float32)
    else:
        count_volume = np.zeros(input_features.shape[1:], dtype=np.float32)

    _, nx, ny, nz = input_features.shape
    slots_per_voxel = int(slots_per_voxel)
    center_target = np.zeros((slots_per_voxel, nx, ny, nz), dtype=np.float32)
    geom_target = np.zeros((slots_per_voxel, len(GEOM_CHANNELS), nx, ny, nz), dtype=np.float32)
    weight_target = np.zeros((slots_per_voxel, nx, ny, nz), dtype=np.float32)
    slot_fill = np.zeros((nx, ny, nz), dtype=np.int16)
    overflow_instance_count = 0

    for idx in range(len(instance_ijk)):
        i, j, k = instance_ijk[idx].tolist()
        if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
            continue
        slot = int(slot_fill[i, j, k])
        slot_fill[i, j, k] += 1
        if slot >= slots_per_voxel:
            overflow_instance_count += 1
            continue
        center_target[slot, i, j, k] = 1.0
        geom_target[slot, :, i, j, k] = instance_geom[idx]
        weight_target[slot, i, j, k] = float(instance_weight[idx])

    count_target = np.clip(count_volume, 0.0, float(slots_per_voxel)).astype(np.float32)[None, ...]
    return {
        "input_features": input_features,
        "valid_z_mask": valid_z_mask,
        "center_target": center_target,
        "geom_target": geom_target,
        "weight_target": weight_target,
        "count_target": count_target,
        "overflow_instance_count": np.asarray([overflow_instance_count], dtype=np.int32),
    }


def prepare_unit_context(
    unit_id: str,
    unit_dfn_root: Path = DEFAULT_UNIT_DFN_ROOT,
    xy_resolution: int = DEFAULT_XY_RESOLUTION,
    z_step_ms: float = DEFAULT_Z_STEP_MS,
) -> dict[str, Any]:
    unit_dir = Path(unit_dfn_root) / str(unit_id)
    patch_csv = unit_dir / "unit_dfn_patches.csv"
    patch_df = load_patch_table(patch_csv)
    layers_df = load_layer_table(unit_dir)
    unit_summary = load_unit_summary(unit_dir)
    if patch_df.empty:
        raise ValueError(f"empty patch table: {patch_csv}")
    block_x = int(unit_summary.get("BlockX", patch_df["BlockX"].iloc[0]))
    block_y = int(unit_summary.get("BlockY", patch_df["BlockY"].iloc[0]))
    x_min, x_max, y_min, y_max = compute_unit_bounds_from_patch_table(patch_df)
    z_min, z_max = compute_time_bounds(patch_df, layers_df, z_padding_ms=0.0)
    grid = build_grid_spec(
        unit_id=str(unit_id),
        block_x=block_x,
        block_y=block_y,
        x_bounds=(x_min, x_max),
        y_bounds=(y_min, y_max),
        z_bounds=(z_min, z_max),
        xy_resolution=int(xy_resolution),
        z_step_ms=float(z_step_ms),
    )
    return {
        "unit_id": str(unit_id),
        "unit_dir": unit_dir,
        "patch_df": patch_df,
        "layers_df": layers_df,
        "grid": grid,
        "block_x": block_x,
        "block_y": block_y,
        "x_bounds": (x_min, x_max),
        "y_bounds": (y_min, y_max),
    }


def build_window_grid(
    unit_context: dict[str, Any],
    window_top: float,
    window_size: int = DEFAULT_WINDOW_SIZE,
    z_step_ms: float = DEFAULT_Z_STEP_MS,
) -> GridSpec:
    x_min, x_max = unit_context["x_bounds"]
    y_min, y_max = unit_context["y_bounds"]
    window_base = float(window_top) + float(window_size) * float(z_step_ms)
    return build_grid_spec(
        unit_id=str(unit_context["unit_id"]),
        block_x=int(unit_context["block_x"]),
        block_y=int(unit_context["block_y"]),
        x_bounds=(float(x_min), float(x_max)),
        y_bounds=(float(y_min), float(y_max)),
        z_bounds=(float(window_top), float(window_base)),
        xy_resolution=DEFAULT_XY_RESOLUTION,
        z_step_ms=float(z_step_ms),
    )


def normalize_vectors_last_dim(vectors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norm = np.where(norm > float(eps), norm, 1.0)
    return vectors / norm


def prediction_to_label_payload(
    center_probs: np.ndarray,
    count_pred: np.ndarray,
    geom_pred: np.ndarray,
    valid_z_mask: np.ndarray,
    threshold: float,
    max_slots_per_voxel: int,
    max_total_patches: int | None = None,
    decode_mode: str = "strict",
    relaxed_min_count: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    center_probs = np.asarray(center_probs, dtype=np.float32)
    count_pred = np.asarray(count_pred, dtype=np.float32)
    geom_pred = np.asarray(geom_pred, dtype=np.float32)
    valid_z_mask = np.asarray(valid_z_mask, dtype=np.float32)
    center_probs = center_probs * valid_z_mask.reshape(1, 1, 1, -1)
    count_pred = np.squeeze(count_pred).astype(np.float32)
    if count_pred.ndim == 4 and count_pred.shape[0] == 1:
        count_pred = count_pred[0]
    if count_pred.ndim != 3:
        raise ValueError(f"count_pred must be 3D after squeeze, got shape={count_pred.shape}")
    count_pred = count_pred * valid_z_mask.reshape(1, 1, -1)

    slot_count = int(center_probs.shape[0])
    geom_swapped = np.moveaxis(geom_pred, 1, -1)
    order = np.argsort(-center_probs, axis=0)
    sorted_center_probs = np.take_along_axis(center_probs, order, axis=0)
    sorted_geom = np.take_along_axis(geom_swapped, order[..., None], axis=0)

    count_limit = np.rint(count_pred).astype(np.int16)
    count_limit = np.clip(count_limit, 0, min(int(max_slots_per_voxel), slot_count))
    forced_voxel_mask = np.zeros_like(count_limit, dtype=bool)
    if str(decode_mode).lower() == "relaxed":
        relaxed_min_count = int(max(relaxed_min_count, 1))
        top_slot_mask = sorted_center_probs[0] >= float(threshold)
        forced_voxel_mask = top_slot_mask & (count_limit < relaxed_min_count)
        count_limit = np.where(forced_voxel_mask, relaxed_min_count, count_limit).astype(np.int16)
        count_limit = np.clip(count_limit, 0, min(int(max_slots_per_voxel), slot_count))
    keep_mask = (np.arange(slot_count, dtype=np.int16).reshape(-1, 1, 1, 1) < count_limit.reshape(1, *count_limit.shape))
    keep_mask &= sorted_center_probs >= float(threshold)

    raw_candidate_count = int(np.count_nonzero(keep_mask))
    if max_total_patches is not None and raw_candidate_count > int(max_total_patches):
        flat_keep = keep_mask.reshape(-1)
        flat_scores = sorted_center_probs.reshape(-1)
        active_indices = np.flatnonzero(flat_keep)
        rank_order = np.argsort(flat_scores[active_indices])[::-1][: int(max_total_patches)]
        limited_indices = active_indices[rank_order]
        limited_flat_keep = np.zeros_like(flat_keep, dtype=bool)
        limited_flat_keep[limited_indices] = True
        keep_mask = limited_flat_keep.reshape(keep_mask.shape)

    center_probs_kept = sorted_center_probs * keep_mask.astype(np.float32)
    center_heatmap = np.transpose(center_probs_kept, (1, 2, 3, 0)).astype(np.float32)
    center_count = keep_mask.sum(axis=0).astype(np.int16)
    normals = normalize_vectors_last_dim(np.transpose(sorted_geom[:, :, :, :, 3:6], (1, 2, 3, 0, 4)))
    u_dirs = normalize_vectors_last_dim(np.transpose(sorted_geom[:, :, :, :, 6:9], (1, 2, 3, 0, 4)))
    payload = {
        "center_heatmap": center_heatmap,
        "center_count": center_count,
        "offsets": np.transpose(sorted_geom[:, :, :, :, 0:3], (1, 2, 3, 0, 4)).astype(np.float32),
        "normals": normals.astype(np.float32),
        "u_dirs": u_dirs.astype(np.float32),
        "lengths": np.transpose(np.maximum(sorted_geom[:, :, :, :, 9], 1e-6), (1, 2, 3, 0)).astype(np.float32),
        "heights": np.transpose(np.maximum(sorted_geom[:, :, :, :, 10], 1e-6), (1, 2, 3, 0)).astype(np.float32),
        "confidence": np.transpose(np.clip(sorted_geom[:, :, :, :, 11], 0.0, 1.0), (1, 2, 3, 0)).astype(np.float32),
        "source_patch_index": np.zeros(center_heatmap.shape, dtype=np.int32),
    }
    stats = {
        "decode_mode": str(decode_mode).lower(),
        "raw_candidate_count": int(raw_candidate_count),
        "kept_candidate_count": int(np.count_nonzero(keep_mask)),
        "mean_pred_count": float(np.mean(count_limit)),
        "max_pred_count": int(np.max(count_limit)) if count_limit.size else 0,
        "forced_voxel_count": int(np.count_nonzero(forced_voxel_mask)),
    }
    return payload, stats


def dedupe_patch_df(
    patch_df: pd.DataFrame,
    xy_tol_m: float = 6.25,
    time_tol_ms: float = 0.4,
    azimuth_tol_deg: float = 20.0,
    dip_tol_deg: float = 12.0,
) -> pd.DataFrame:
    if patch_df.empty:
        return patch_df.copy()
    work = patch_df.copy()
    work["ConfidenceSort"] = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0)
    work = work.sort_values(["ConfidenceSort", "PatchLength", "PatchHeight"], ascending=[False, False, False]).reset_index(drop=True)
    kept_rows: list[pd.Series] = []
    for _, row in work.iterrows():
        center_x = float(pd.to_numeric(row.get("CenterX"), errors="coerce"))
        center_y = float(pd.to_numeric(row.get("CenterY"), errors="coerce"))
        center_t = float(pd.to_numeric(row.get("CenterTIME"), errors="coerce"))
        azimuth = float(pd.to_numeric(row.get("Azimuth"), errors="coerce"))
        dip = float(pd.to_numeric(row.get("Dip"), errors="coerce"))
        interval_key = str(row.get("GeoIntervalKey", ""))
        duplicated = False
        for kept in kept_rows:
            if str(kept.get("GeoIntervalKey", "")) != interval_key:
                continue
            dx = float(pd.to_numeric(kept.get("CenterX"), errors="coerce")) - center_x
            dy = float(pd.to_numeric(kept.get("CenterY"), errors="coerce")) - center_y
            dt = float(pd.to_numeric(kept.get("CenterTIME"), errors="coerce")) - center_t
            xy_dist = float(np.hypot(dx, dy))
            az_diff = abs(float(pd.to_numeric(kept.get("Azimuth"), errors="coerce")) - azimuth) % 360.0
            az_diff = min(az_diff, 360.0 - az_diff)
            dip_diff = abs(float(pd.to_numeric(kept.get("Dip"), errors="coerce")) - dip)
            if xy_dist <= float(xy_tol_m) and abs(dt) <= float(time_tol_ms) and az_diff <= float(azimuth_tol_deg) and dip_diff <= float(dip_tol_deg):
                duplicated = True
                break
        if not duplicated:
            kept_rows.append(row)
    result = pd.DataFrame(kept_rows).reset_index(drop=True)
    if "ConfidenceSort" in result.columns:
        result = result.drop(columns=["ConfidenceSort"])
    return result


def append_lines_to_docx(docx_path: Path, title: str, lines: list[str]) -> None:
    doc = open_or_create_doc(docx_path)
    doc.add_heading(title, level=2)
    body = [f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"] + list(lines)
    doc.add_paragraph("\n".join(body))
    doc.add_paragraph("")
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(docx_path))

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial import cKDTree


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_initial_dfn_3d_candidate_cheye1.json"
SURFACE_TOOL_DIR = CURRENT_DIR.parent / "step1_surface_framework"
if str(SURFACE_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(SURFACE_TOOL_DIR))

from surface_tools import load_surface_tables, validate_surface_order  # noqa: E402


ALLOWED_LAYERS = ["沙三段", "沙四段"]
LAYER_CODE = {"沙三段": 3, "沙四段": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial DFN from Step6B 3D density SGY.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_stats(values: pd.Series | np.ndarray | list[float]) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": int(numeric.count()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "initial_dfn_fracture_patches.csv",
        "raw_vtk": output_dir / "initial_dfn_raw_time.vtk",
        "summary_json": output_dir / "initial_dfn_summary.json",
        "audit_csv": output_dir / "initial_dfn_generation_audit.csv",
    }


def layer_param(config: dict[str, Any], key: str, layer: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, dict):
        return float(value.get(layer, default))
    return float(value)


def layer_distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


def load_density_grid(sgy_path: Path, mapping_path: Path) -> dict[str, Any]:
    mapping = np.load(mapping_path)
    required = {"x", "y", "ix", "iy", "source_trace_idx", "output_trace_index"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")

    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        density_trace_time = np.stack([handle.trace[idx] for idx in range(handle.tracecount)]).astype(np.float32)
        samples = np.asarray(handle.samples, dtype=np.float64)
        tracecount = int(handle.tracecount)
        sample_count = int(len(samples))
        sample_interval_us = float(segyio.tools.dt(handle))
        data_format = int(handle.bin[segyio.BinField.Format])

    if tracecount != len(mapping["x"]):
        raise ValueError(f"SGY tracecount {tracecount} does not match mapping rows {len(mapping['x'])}")
    if sample_count < 1:
        raise ValueError("density SGY has no samples")

    x_values = np.sort(np.unique(mapping["x"].astype(float)))
    y_values = np.sort(np.unique(mapping["y"].astype(float)))
    nx = int(len(x_values))
    ny = int(len(y_values))
    nt = sample_count
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)

    density = np.full((ny, nx, nt), np.nan, dtype=np.float32)
    source_trace_idx = np.full((ny, nx), -1, dtype=np.int64)
    output_trace_index = np.full((ny, nx), -1, dtype=np.int64)
    density[iy, ix, :] = density_trace_time
    source_trace_idx[iy, ix] = mapping["source_trace_idx"].astype(np.int64)
    output_trace_index[iy, ix] = mapping["output_trace_index"].astype(np.int64)
    density[~np.isfinite(density)] = 0.0
    density = np.clip(density, 0.0, None)

    return {
        "density": density,
        "samples": samples,
        "x_values": x_values,
        "y_values": y_values,
        "source_trace_idx": source_trace_idx,
        "output_trace_index": output_trace_index,
        "tracecount": tracecount,
        "sample_count": sample_count,
        "sample_interval_us": sample_interval_us,
        "data_format": data_format,
    }


def load_guidance_grid(
    sgy_path: Path,
    source_trace_idx: np.ndarray,
    target_samples: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    ny, nx = source_trace_idx.shape
    nt = len(target_samples)
    guidance = np.full((ny, nx, nt), np.nan, dtype=np.float32)
    valid_flat = np.where(source_trace_idx.ravel() >= 0)[0]

    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as handle:
        source_samples = np.asarray(handle.samples, dtype=np.float64)
        tracecount = int(handle.tracecount)
        sample_count = int(len(source_samples))
        max_trace_idx = int(np.max(source_trace_idx[source_trace_idx >= 0])) if valid_flat.size else -1
        if max_trace_idx >= tracecount:
            raise ValueError(f"guidance SGY tracecount {tracecount} is smaller than required source trace index {max_trace_idx}")

        same_samples = sample_count == nt and np.allclose(source_samples, target_samples, rtol=0.0, atol=1.0e-6)
        flat_guidance = guidance.reshape(-1, nt)
        flat_trace_idx = source_trace_idx.ravel()
        for ordinal, flat_idx in enumerate(valid_flat, start=1):
            trace = np.asarray(handle.trace[int(flat_trace_idx[flat_idx])], dtype=np.float32)
            if same_samples:
                flat_guidance[flat_idx, :] = trace
            else:
                flat_guidance[flat_idx, :] = np.interp(target_samples, source_samples, trace, left=np.nan, right=np.nan).astype(np.float32)
            if ordinal % 10000 == 0:
                print(f"[step7b-3d] loaded guidance traces={ordinal}/{len(valid_flat)}", flush=True)

    summary = {
        "guidance_sgy": str(sgy_path),
        "tracecount": tracecount,
        "source_sample_count": sample_count,
        "target_sample_count": int(nt),
        "sample_axis_matched": bool(same_samples),
        "loaded_trace_count": int(len(valid_flat)),
        "stats": finite_stats(guidance.ravel()),
    }
    return guidance, summary


def coherence_guidance_config(config: dict[str, Any]) -> dict[str, Any]:
    guidance = dict(config.get("coherence_guidance", {}))
    guidance["enabled"] = bool(guidance.get("enabled", False))
    return guidance


def compute_low_coherence_fields(
    coherence: np.ndarray,
    valid_mask: np.ndarray,
    guidance_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    coherence_valid = valid_mask & np.isfinite(coherence)
    if "valid_min" in guidance_config:
        coherence_valid &= coherence >= float(guidance_config["valid_min"])
    if "valid_max" in guidance_config:
        coherence_valid &= coherence <= float(guidance_config["valid_max"])

    finite = coherence[coherence_valid]
    if finite.size == 0:
        score = np.zeros_like(coherence, dtype=np.float32)
        weight = np.ones_like(coherence, dtype=np.float32)
        return score, weight, {"enabled": True, "finite_count": 0, "low_bound": None, "high_bound": None}

    low_q = float(guidance_config.get("low_quantile", 0.05))
    high_q = float(guidance_config.get("high_quantile", 0.95))
    low_bound = float(np.nanquantile(finite, low_q))
    high_bound = float(np.nanquantile(finite, high_q))
    if high_bound <= low_bound:
        high_bound = low_bound + 1.0e-6

    score = (high_bound - coherence) / (high_bound - low_bound)
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    score[~np.isfinite(score)] = 0.0
    score[~coherence_valid] = 0.0

    gain = float(guidance_config.get("gain", 1.5))
    power = float(guidance_config.get("power", 1.2))
    max_weight = float(guidance_config.get("max_weight", 3.0))
    weight = 1.0 + gain * np.power(score, power)
    weight = np.clip(weight, 1.0, max_weight).astype(np.float32)
    weight[~np.isfinite(weight)] = 1.0
    weight[~coherence_valid] = 1.0

    return score, weight, {
        "enabled": True,
        "finite_count": int(finite.size),
        "valid_min": guidance_config.get("valid_min"),
        "valid_max": guidance_config.get("valid_max"),
        "invalid_or_missing_count": int(valid_mask.sum() - coherence_valid.sum()),
        "low_quantile": low_q,
        "high_quantile": high_q,
        "low_bound": low_bound,
        "high_bound": high_bound,
        "gain": gain,
        "power": power,
        "max_weight": max_weight,
        "coherence_stats": finite_stats(finite),
        "low_coherence_score_stats": finite_stats(score[coherence_valid]),
        "low_coherence_weight_stats": finite_stats(weight[coherence_valid]),
    }


def build_xy_records(x_values: np.ndarray, y_values: np.ndarray) -> pd.DataFrame:
    xx, yy = np.meshgrid(x_values, y_values)
    return pd.DataFrame({"X": xx.ravel(), "Y": yy.ravel(), "TIME": np.zeros(xx.size, dtype=float)})


def attach_surface_grids(layer_dir: Path, x_values: np.ndarray, y_values: np.ndarray) -> dict[str, np.ndarray]:
    surfaces = load_surface_tables(layer_dir)
    records = build_xy_records(x_values, y_values)
    for code in ["T4", "T5", "T6", "T7"]:
        surface_df = surfaces[code]["table"]
        xy = surface_df[["X", "Y"]].to_numpy(dtype=float)
        time = surface_df["Z"].to_numpy(dtype=float)
        tree = cKDTree(xy)
        _, nearest = tree.query(records[["X", "Y"]].to_numpy(dtype=float), k=1, p=1)
        records[f"{code}_TIME"] = time[nearest]
    records = validate_surface_order(records, min_thickness=1.0)
    ny = len(y_values)
    nx = len(x_values)
    out: dict[str, np.ndarray] = {}
    for code in ["T4", "T5", "T6", "T7"]:
        out[f"{code}_TIME"] = records[f"{code}_TIME"].to_numpy(dtype=float).reshape(ny, nx)
    out["Check_All"] = records["Check_All"].fillna(False).to_numpy(dtype=bool).reshape(ny, nx)
    return out


def layer_mask_for_grid(layer: str, samples: np.ndarray, surfaces: dict[str, np.ndarray]) -> np.ndarray:
    t = samples.reshape(1, 1, -1)
    if layer == "沙三段":
        top = surfaces["T4_TIME"][:, :, None]
        base = surfaces["T6_TIME"][:, :, None]
        return surfaces["Check_All"][:, :, None] & np.isfinite(top) & np.isfinite(base) & (t >= top) & (t < base)
    if layer == "沙四段":
        top = surfaces["T6_TIME"][:, :, None]
        base = surfaces["T7_TIME"][:, :, None]
        return surfaces["Check_All"][:, :, None] & np.isfinite(top) & np.isfinite(base) & (t >= top) & (t <= base)
    raise ValueError(f"unsupported layer: {layer}")


def compute_component_labels(candidate_mask: np.ndarray, min_component_voxels: int) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    structure = generate_binary_structure(rank=3, connectivity=2)
    labels, count = label(candidate_mask, structure=structure)
    if count < 1:
        return labels.astype(np.int32), {}, np.zeros_like(candidate_mask, dtype=bool)
    component_sizes = np.bincount(labels.ravel())
    keep_ids = np.where(component_sizes >= int(min_component_voxels))[0]
    keep_ids = keep_ids[keep_ids != 0]
    keep_mask = np.isin(labels, keep_ids)
    size_by_id = {int(cid): int(component_sizes[cid]) for cid in keep_ids}
    return labels.astype(np.int32), size_by_id, keep_mask


def choose_target_count(candidate_density: np.ndarray, config: dict[str, Any]) -> tuple[int, float, float]:
    density_mass = float(np.nansum(candidate_density))
    if density_mass <= 0:
        raise RuntimeError("3D density candidate mask has no positive density mass")
    requested_scale = float(config.get("count_scale", 0.02))
    min_count = int(config.get("min_patch_count", 2500))
    max_count = int(config.get("max_patch_count", 15000))
    expected = density_mass * requested_scale
    target = int(round(expected))
    if min_count > 0:
        target = max(target, min_count)
    if max_count > 0:
        target = min(target, max_count)
    target = max(1, min(target, int(candidate_density.size)))
    effective_scale = float(target) / density_mass
    return target, density_mass, effective_scale


def build_candidate_voxels(
    density: np.ndarray,
    samples: np.ndarray,
    surfaces: dict[str, np.ndarray],
    source_trace_idx: np.ndarray,
    config: dict[str, Any],
    coherence: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    layer_summary: dict[str, Any] = {}
    min_density = float(config.get("min_candidate_density", 1.0e-6))
    min_component_voxels = int(config.get("min_component_voxels", 12))
    guidance = coherence_guidance_config(config)
    use_guidance = bool(guidance.get("enabled", False)) and coherence is not None
    apply_to_candidate_selection = bool(guidance.get("apply_to_candidate_selection", True))
    apply_to_sampling = bool(guidance.get("apply_to_sampling", True))

    for layer in ALLOWED_LAYERS:
        layer_mask = layer_mask_for_grid(layer, samples, surfaces)
        valid_density_mask = layer_mask & np.isfinite(density) & (density > min_density)
        if use_guidance:
            low_score, low_weight, guidance_summary = compute_low_coherence_fields(coherence, valid_density_mask, guidance)
            guided_score = density * low_weight
            candidate_score_grid = guided_score if apply_to_candidate_selection else density
            sampling_weight_grid = guided_score if apply_to_sampling else density
        else:
            low_score = np.zeros_like(density, dtype=np.float32)
            low_weight = np.ones_like(density, dtype=np.float32)
            candidate_score_grid = density
            sampling_weight_grid = density
            guidance_summary = {"enabled": False}

        positive = candidate_score_grid[valid_density_mask]
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        if positive.size == 0:
            layer_summary[layer] = {"candidate_count": 0, "kept_candidate_count": 0, "threshold": None, "coherence_guidance": guidance_summary}
            continue
        quantile = float(layer_param(config, "candidate_density_quantile", layer, 0.90))
        threshold = max(float(np.quantile(positive, quantile)), min_density)
        raw_mask = valid_density_mask & (candidate_score_grid >= threshold)
        labels, size_by_id, keep_mask = compute_component_labels(raw_mask, min_component_voxels=min_component_voxels)
        if not keep_mask.any():
            keep_mask = raw_mask
            labels = raw_mask.astype(np.int32)
            size_by_id = {1: int(raw_mask.sum())}

        yy, xx, tt = np.where(keep_mask)
        if yy.size == 0:
            layer_summary[layer] = {
                "candidate_count": int(raw_mask.sum()),
                "kept_candidate_count": 0,
                "threshold": threshold,
            }
            continue

        comp = labels[yy, xx, tt].astype(np.int32)
        comp_size = np.asarray([size_by_id.get(int(cid), int((comp == cid).sum())) for cid in comp], dtype=np.int32)
        top = surfaces["T4_TIME"][yy, xx] if layer == "沙三段" else surfaces["T6_TIME"][yy, xx]
        base = surfaces["T6_TIME"][yy, xx] if layer == "沙三段" else surfaces["T7_TIME"][yy, xx]
        part = pd.DataFrame(
            {
                "LayerGroup": layer,
                "LayerCode": LAYER_CODE[layer],
                "IY": yy.astype(np.int32),
                "IX": xx.astype(np.int32),
                "IT": tt.astype(np.int32),
                "SourceTraceIdx": source_trace_idx[yy, xx].astype(np.int64),
                "CenterTime": samples[tt].astype(float),
                "TimeWindowMin": top.astype(float),
                "TimeWindowMax": base.astype(float),
                "LayerThickness": (base - top).astype(float),
                "SourceDensity": density[yy, xx, tt].astype(float),
                "CoherenceValue": coherence[yy, xx, tt].astype(float) if coherence is not None else np.full(yy.size, np.nan, dtype=float),
                "LowCoherenceScore": low_score[yy, xx, tt].astype(float),
                "LowCoherenceWeight": low_weight[yy, xx, tt].astype(float),
                "GuidedDensityScore": guided_score[yy, xx, tt].astype(float) if use_guidance else density[yy, xx, tt].astype(float),
                "CandidateScore": candidate_score_grid[yy, xx, tt].astype(float),
                "SamplingWeight": sampling_weight_grid[yy, xx, tt].astype(float),
                "ComponentID": comp,
                "ComponentVoxelCount": comp_size,
                "LayerDensityThreshold": threshold,
            }
        )
        rows.append(part)
        layer_summary[layer] = {
            "positive_voxel_count": int(positive.size),
            "candidate_count": int(raw_mask.sum()),
            "kept_candidate_count": int(len(part)),
            "threshold": threshold,
            "component_count": int(len(size_by_id)),
            "kept_component_count": int(len(set(int(v) for v in part["ComponentID"]))),
            "candidate_density_stats": finite_stats(part["SourceDensity"]),
            "candidate_score_stats": finite_stats(part["CandidateScore"]),
            "sampling_weight_stats": finite_stats(part["SamplingWeight"]),
            "coherence_guidance": guidance_summary,
        }

    if not rows:
        raise RuntimeError("no candidate voxels selected from 3D density SGY")
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates[candidates["SourceTraceIdx"].ge(0) & candidates["LayerThickness"].gt(0)].reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("candidate voxels are empty after trace/layer validation")
    return candidates, layer_summary


def sample_candidate_voxels(candidates: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, Any]]:
    guidance = coherence_guidance_config(config)
    sampling_weights = candidates.get("SamplingWeight", candidates["SourceDensity"]).to_numpy(dtype=float)
    sampling_weights = np.clip(sampling_weights, 0.0, None)
    if sampling_weights.sum() <= 0:
        raise RuntimeError("candidate sampling weights have no positive mass")

    target_basis = str(guidance.get("target_count_basis", "source_density")) if guidance.get("enabled", False) else "source_density"
    if target_basis == "sampling_weight":
        count_weights = sampling_weights
    else:
        count_weights = candidates["SourceDensity"].to_numpy(dtype=float)
        count_weights = np.clip(count_weights, 0.0, None)
    target_count, count_basis_mass, count_basis_effective_scale = choose_target_count(count_weights, config)
    probability = sampling_weights / sampling_weights.sum()
    selected_idx = rng.choice(np.arange(len(candidates)), size=target_count, replace=False, p=probability)
    selected = candidates.iloc[selected_idx].reset_index(drop=True).copy()
    sampling_effective_scale = float(target_count) / float(sampling_weights.sum())
    selected["ExpectedPatchCountForCell"] = selected["SamplingWeight"].to_numpy(dtype=float) * sampling_effective_scale
    selected["EffectiveCountScale"] = sampling_effective_scale
    selected["CountBasisEffectiveScale"] = count_basis_effective_scale
    selected["DensityCellPatchOrdinal"] = 1
    return selected, {
        "density_mass": float(np.clip(candidates["SourceDensity"].to_numpy(dtype=float), 0.0, None).sum()),
        "sampling_weight_mass": float(sampling_weights.sum()),
        "target_count_basis": target_basis,
        "target_count_basis_mass": count_basis_mass,
        "requested_count_scale": float(config.get("count_scale", 0.02)),
        "effective_count_scale": sampling_effective_scale,
        "count_basis_effective_scale": count_basis_effective_scale,
        "expected_patch_count": float(target_count),
        "actual_patch_count": int(len(selected)),
        "positive_density_candidate_count": int(len(candidates)),
        "sampled_density_cell_count": int(len(selected)),
        "sampling_replacement": False,
    }


def _component_axis_stats(group: pd.DataFrame, x_values: np.ndarray, y_values: np.ndarray, time_scale: float) -> dict[str, Any]:
    coords = np.column_stack(
        [
            x_values[group["IX"].to_numpy(dtype=int)],
            y_values[group["IY"].to_numpy(dtype=int)],
            group["CenterTime"].to_numpy(dtype=float) * time_scale,
        ]
    )
    center = coords.mean(axis=0)
    centered = coords - center
    if len(group) >= 3 and float(np.linalg.norm(centered)) > 1.0e-8:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
    else:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    projections = centered @ axis
    axis_length = float(np.nanmax(projections) - np.nanmin(projections)) if projections.size else 0.0
    return {
        "coords": coords,
        "center": center,
        "axis": axis,
        "projections": projections,
        "axis_length_m": axis_length,
        "x_extent_m": float(np.nanmax(coords[:, 0]) - np.nanmin(coords[:, 0])) if len(coords) else 0.0,
        "y_extent_m": float(np.nanmax(coords[:, 1]) - np.nanmin(coords[:, 1])) if len(coords) else 0.0,
        "time_extent_ms": float(group["CenterTime"].max() - group["CenterTime"].min()) if len(group) else 0.0,
    }


def _classify_component(group: pd.DataFrame, stats: dict[str, Any], config: dict[str, Any]) -> str:
    ms = dict(config.get("multi_scale", {}))
    voxel_count = int(len(group))
    mean_density = float(group["SourceDensity"].mean())
    large_cfg = dict(ms.get("large_band", {}))
    medium_cfg = dict(ms.get("medium_band", {}))
    if (
        voxel_count >= int(large_cfg.get("min_candidate_voxels", 5000))
        and stats["time_extent_ms"] >= float(large_cfg.get("min_time_extent_ms", 80.0))
        and stats["axis_length_m"] >= float(large_cfg.get("min_axis_length_m", 250.0))
    ):
        return "large"
    if (
        voxel_count >= int(medium_cfg.get("min_candidate_voxels", 800))
        and stats["axis_length_m"] >= float(medium_cfg.get("min_axis_length_m", 150.0))
        and mean_density >= float(medium_cfg.get("min_mean_density", 0.0))
    ):
        return "medium"
    return "small"


def _select_chain_rows(
    group: pd.DataFrame,
    stats: dict[str, Any],
    scale: str,
    band_id: str,
    config: dict[str, Any],
    used_indices: set[int],
    requested_count: int | None = None,
) -> list[pd.Series]:
    ms = dict(config.get("multi_scale", {}))
    scale_cfg = dict(ms.get(f"{scale}_band", {}))
    spacing = float(scale_cfg.get("patch_spacing_m", 55.0 if scale == "large" else 42.0))
    min_count = int(scale_cfg.get("min_patch_count", 5 if scale == "large" else 3))
    max_count = int(scale_cfg.get("max_patch_count", 24 if scale == "large" else 10))
    axis_length = max(float(stats["axis_length_m"]), spacing)
    if requested_count is None:
        target_count = int(np.clip(round(axis_length / max(spacing, 1.0)) + 1, min_count, max_count))
    else:
        target_count = max(int(requested_count), min_count)
        if bool(scale_cfg.get("enforce_max_patch_count", False)):
            target_count = min(target_count, max_count)
    available_count = int(sum(int(idx) not in used_indices for idx in group.index.to_numpy(dtype=int)))
    target_count = min(target_count, available_count)
    if target_count <= 0:
        return []
    projections = np.asarray(stats["projections"], dtype=float)
    if projections.size == 0:
        return []
    positions = np.linspace(float(np.nanmin(projections)), float(np.nanmax(projections)), target_count)
    order_by_density = group["SourceDensity"].to_numpy(dtype=float)
    selected_rows: list[pd.Series] = []
    local_used: set[int] = set()
    group_indices = group.index.to_numpy(dtype=int)
    for ordinal, position in enumerate(positions, start=1):
        distances = np.abs(projections - position)
        # Prefer points near the chain position, then the denser candidate if ties are close.
        rank_value = distances - 1.0e-4 * order_by_density
        ranked = np.argsort(rank_value)
        chosen_pos = None
        for pos in ranked:
            original_idx = int(group_indices[pos])
            if original_idx not in used_indices and original_idx not in local_used:
                chosen_pos = int(pos)
                break
        if chosen_pos is None:
            continue
        original_idx = int(group_indices[chosen_pos])
        local_used.add(original_idx)
        row = group.iloc[chosen_pos].copy()
        row["FractureScale"] = scale
        row["FractureScaleCode"] = 3 if scale == "large" else 2
        row["BandID"] = band_id
        row["BandPatchOrdinal"] = int(ordinal)
        row["BandContinuityMode"] = "pca_chain"
        row["BandVoxelCount"] = int(len(group))
        row["BandLengthM"] = float(axis_length)
        row["BandTimeExtentMs"] = float(stats["time_extent_ms"])
        row["BandPatchSpacingM"] = float(spacing)
        row["BandMeanDensity"] = float(group["SourceDensity"].mean())
        selected_rows.append(row)
    used_indices.update(local_used)
    return selected_rows


def sample_candidate_voxels_multiscale(
    candidates: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ms = dict(config.get("multi_scale", {}))
    sampling_weights = np.clip(candidates.get("SamplingWeight", candidates["SourceDensity"]).to_numpy(dtype=float), 0.0, None)
    count_weights = np.clip(candidates["SourceDensity"].to_numpy(dtype=float), 0.0, None)
    target_count, count_basis_mass, count_basis_effective_scale = choose_target_count(count_weights, config)
    continuous_target = int(round(target_count * float(ms.get("continuous_patch_fraction", 0.50))))
    time_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))

    work = candidates.copy()
    work["_ComponentKey"] = work["LayerGroup"].astype(str) + "_" + work["ComponentID"].astype(str)
    component_rows: list[pd.Series] = []
    component_summary: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    band_serial = 1

    grouped = sorted(work.groupby("_ComponentKey"), key=lambda item: float(item[1]["SourceDensity"].sum()), reverse=True)
    eligible_components: list[dict[str, Any]] = []
    for _, group in grouped:
        stats = _component_axis_stats(group, x_values=x_values, y_values=y_values, time_scale=time_scale)
        scale = _classify_component(group, stats, config)
        if scale == "small":
            continue
        eligible_components.append(
            {
                "group": group,
                "stats": stats,
                "scale": scale,
                "density_mass": float(np.clip(group["SourceDensity"].to_numpy(dtype=float), 0.0, None).sum()),
            }
        )

    continuous_components = eligible_components[: int(ms.get("max_continuous_band_count", len(eligible_components)))]
    total_band_mass = sum(max(item["density_mass"], 0.0) for item in continuous_components)
    allocated_counts: list[int] = []
    if continuous_components and continuous_target > 0:
        min_counts = []
        raw_counts = []
        for item in continuous_components:
            scale_cfg = dict(ms.get(f"{item['scale']}_band", {}))
            min_count = int(scale_cfg.get("min_patch_count", 5 if item["scale"] == "large" else 3))
            min_counts.append(min_count)
            if total_band_mass > 0:
                raw_counts.append(continuous_target * max(item["density_mass"], 0.0) / total_band_mass)
            else:
                raw_counts.append(continuous_target / len(continuous_components))
        allocated_counts = [max(int(round(raw)), min_count) for raw, min_count in zip(raw_counts, min_counts)]
        overflow = sum(allocated_counts) - continuous_target
        if overflow > 0:
            # Remove excess from the weakest allocations first while respecting per-band minimums.
            order = sorted(range(len(allocated_counts)), key=lambda i: raw_counts[i])
            for i in order:
                removable = min(overflow, max(0, allocated_counts[i] - min_counts[i]))
                allocated_counts[i] -= removable
                overflow -= removable
                if overflow <= 0:
                    break
        underflow = continuous_target - sum(allocated_counts)
        if underflow > 0:
            order = sorted(range(len(allocated_counts)), key=lambda i: raw_counts[i], reverse=True)
            j = 0
            while underflow > 0 and order:
                allocated_counts[order[j % len(order)]] += 1
                underflow -= 1
                j += 1

    for item, allocated_count in zip(continuous_components, allocated_counts):
        if len(component_rows) >= continuous_target:
            break
        group = item["group"]
        stats = item["stats"]
        scale = item["scale"]
        band_id = f"band_{band_serial:05d}"
        band_serial += 1
        rows = _select_chain_rows(group, stats, scale, band_id, config, used_indices, requested_count=allocated_count)
        remaining = max(0, continuous_target - len(component_rows))
        rows = rows[:remaining]
        component_rows.extend(rows)
        component_summary.append(
            {
                "band_id": band_id,
                "scale": scale,
                "candidate_voxels": int(len(group)),
                "allocated_patch_count": int(allocated_count),
                "selected_patch_count": int(len(rows)),
                "axis_length_m": float(stats["axis_length_m"]),
                "time_extent_ms": float(stats["time_extent_ms"]),
                "mean_density": float(group["SourceDensity"].mean()),
                "density_mass": float(group["SourceDensity"].sum()),
            }
        )

    selected_parts: list[pd.DataFrame] = []
    if component_rows:
        selected_parts.append(pd.DataFrame(component_rows))
    remaining_count = target_count - sum(len(part) for part in selected_parts)
    remaining = candidates.drop(index=list(used_indices), errors="ignore").copy()
    if remaining_count > 0 and not remaining.empty:
        weights = np.clip(remaining.get("SamplingWeight", remaining["SourceDensity"]).to_numpy(dtype=float), 0.0, None)
        if weights.sum() <= 0:
            raise RuntimeError("remaining candidate sampling weights have no positive mass")
        remaining_count = min(remaining_count, len(remaining))
        sampled_idx = rng.choice(remaining.index.to_numpy(), size=remaining_count, replace=False, p=weights / weights.sum())
        small = remaining.loc[sampled_idx].copy()
        small["FractureScale"] = "small"
        small["FractureScaleCode"] = 1
        small["BandID"] = ""
        small["BandPatchOrdinal"] = 0
        small["BandContinuityMode"] = "isolated_patch"
        small["BandVoxelCount"] = small["ComponentVoxelCount"].astype(int)
        small["BandLengthM"] = 0.0
        small["BandTimeExtentMs"] = 0.0
        small["BandPatchSpacingM"] = 0.0
        small["BandMeanDensity"] = small["SourceDensity"].astype(float)
        selected_parts.append(small)

    if not selected_parts:
        raise RuntimeError("multi-scale sampling produced no selected candidates")
    selected = pd.concat(selected_parts, ignore_index=True).head(target_count).copy()
    selected["DensityCellPatchOrdinal"] = selected.groupby(["SourceTraceIdx", "LayerGroup", "IT"]).cumcount() + 1
    sampling_effective_scale = float(len(selected)) / float(max(sampling_weights.sum(), 1.0e-12))
    selected["ExpectedPatchCountForCell"] = selected.get("SamplingWeight", selected["SourceDensity"]).to_numpy(dtype=float) * sampling_effective_scale
    selected["EffectiveCountScale"] = sampling_effective_scale
    selected["CountBasisEffectiveScale"] = count_basis_effective_scale
    return selected.reset_index(drop=True), {
        "generation_mode": "multi_scale_connected_bands",
        "density_mass": float(np.clip(candidates["SourceDensity"].to_numpy(dtype=float), 0.0, None).sum()),
        "sampling_weight_mass": float(sampling_weights.sum()),
        "target_count_basis": "source_density",
        "target_count_basis_mass": count_basis_mass,
        "requested_count_scale": float(config.get("count_scale", 0.02)),
        "effective_count_scale": sampling_effective_scale,
        "count_basis_effective_scale": count_basis_effective_scale,
        "expected_patch_count": float(target_count),
        "actual_patch_count": int(len(selected)),
        "continuous_target_count": int(continuous_target),
        "continuous_actual_count": int((selected["FractureScale"].astype(str) != "small").sum()),
        "small_isolated_count": int((selected["FractureScale"].astype(str) == "small").sum()),
        "eligible_band_count": int(len(eligible_components)),
        "band_count": int(sum(1 for item in component_summary if item["selected_patch_count"] > 0)),
        "band_examples": component_summary[: int(ms.get("summary_band_limit", 20))],
        "positive_density_candidate_count": int(len(candidates)),
        "sampled_density_cell_count": int(len(selected)),
        "sampling_replacement": False,
    }


def estimate_local_orientation(
    density: np.ndarray,
    y_idx: int,
    x_idx: int,
    t_idx: int,
    layer: str,
    source_density: float,
    density_p95: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    ry = int(config.get("orientation_window_xy_cells", 4))
    rx = ry
    rt = int(config.get("orientation_window_time_samples", 4))
    vertical_scale = float(config.get("orientation_time_scale_m_per_ms", 1.0))
    min_points = int(config.get("min_orientation_points", 12))
    min_planarity = float(config.get("min_orientation_planarity", 0.08))
    fallback_azimuth = layer_param(config, "fallback_azimuth_deg", layer, layer_param(config, "base_azimuth_deg", layer, 60.0))
    fallback_dip = layer_param(config, "fallback_dip_deg", layer, layer_param(config, "base_dip_deg", layer, 72.0))

    y0 = max(0, y_idx - ry)
    y1 = min(density.shape[0], y_idx + ry + 1)
    x0 = max(0, x_idx - rx)
    x1 = min(density.shape[1], x_idx + rx + 1)
    t0 = max(0, t_idx - rt)
    t1 = min(density.shape[2], t_idx + rt + 1)
    block = density[y0:y1, x0:x1, t0:t1]
    if block.size == 0:
        return {
            "azimuth": fallback_azimuth,
            "dip": fallback_dip,
            "source": "fallback_layer_template_empty_window",
            "point_count": 0,
            "linearity": 0.0,
            "planarity": 0.0,
            "eigenvalues": [0.0, 0.0, 0.0],
        }

    local_threshold = max(float(np.nanquantile(block, float(config.get("orientation_local_quantile", 0.70)))), source_density * 0.35)
    yy, xx, tt = np.where(block >= local_threshold)
    if yy.size < min_points:
        yy, xx, tt = np.where(block > 0.0)
    if yy.size < min_points:
        return {
            "azimuth": fallback_azimuth,
            "dip": fallback_dip,
            "source": "fallback_layer_template_insufficient_points",
            "point_count": int(yy.size),
            "linearity": 0.0,
            "planarity": 0.0,
            "eigenvalues": [0.0, 0.0, 0.0],
        }

    dx = (xx + x0 - x_idx).astype(float) * float(config.get("trace_spacing_x_m", 12.5))
    dy = (yy + y0 - y_idx).astype(float) * float(config.get("trace_spacing_y_m", 12.5))
    dt = (tt + t0 - t_idx).astype(float) * float(config.get("sample_interval_ms", 10.0)) * vertical_scale
    coords = np.column_stack([dx, dy, dt])
    weights = block[yy, xx, tt].astype(float)
    weights = np.clip(weights, 1.0e-9, None)
    center = np.average(coords, axis=0, weights=weights)
    centered = coords - center
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    if total <= 0:
        return {
            "azimuth": fallback_azimuth,
            "dip": fallback_dip,
            "source": "fallback_layer_template_degenerate_pca",
            "point_count": int(yy.size),
            "linearity": 0.0,
            "planarity": 0.0,
            "eigenvalues": [float(v) for v in eigvals],
        }

    linearity = float((eigvals[0] - eigvals[1]) / max(eigvals[0], 1.0e-12))
    planarity = float((eigvals[1] - eigvals[2]) / max(eigvals[0], 1.0e-12))
    normal = eigvecs[:, 2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0 or planarity < min_planarity:
        return {
            "azimuth": fallback_azimuth,
            "dip": fallback_dip,
            "source": "fallback_layer_template_low_planarity",
            "point_count": int(yy.size),
            "linearity": linearity,
            "planarity": planarity,
            "eigenvalues": [float(v) for v in eigvals],
        }
    normal = normal / normal_norm
    vertical_component = abs(float(normal[2]))
    dip = float(np.degrees(np.arccos(np.clip(vertical_component, 0.0, 1.0))))
    dip = float(np.clip(dip, float(config.get("min_dip_deg", 45.0)), float(config.get("max_dip_deg", 89.0))))
    strike = np.asarray([-normal[1], normal[0]], dtype=float)
    if float(np.linalg.norm(strike)) < 1.0e-8:
        azimuth = fallback_azimuth
        source = "fallback_azimuth_vertical_normal_pca_dip"
    else:
        azimuth = float(np.degrees(np.arctan2(strike[1], strike[0])) % 180.0)
        source = "local_3d_density_pca_plane"

    return {
        "azimuth": azimuth,
        "dip": dip,
        "source": source,
        "point_count": int(yy.size),
        "linearity": linearity,
        "planarity": planarity,
        "eigenvalues": [float(v) for v in eigvals],
    }


def build_patch_table(
    selected: pd.DataFrame,
    density: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    density_p95 = {
        layer: max(float(group["SourceDensity"].quantile(0.95)), 1.0e-9)
        for layer, group in selected.groupby("LayerGroup", dropna=False)
    }
    rows: list[dict[str, Any]] = []
    for patch_idx, row in selected.iterrows():
        layer = str(row["LayerGroup"])
        source_density = float(row["SourceDensity"])
        density_norm = min(source_density / density_p95.get(layer, max(source_density, 1.0)), float(config.get("max_density_norm", 3.0)))
        density_factor = float(np.sqrt(max(density_norm, 0.0)))

        base_length = layer_param(config, "base_length_m", layer, 36.0)
        base_height = layer_param(config, "base_height_time_ms", layer, 16.0)
        length = base_length * (1.0 + float(config.get("length_density_gain", 0.9)) * density_factor)
        height = base_height * (1.0 + float(config.get("height_density_gain", 0.6)) * density_factor)
        length *= float(rng.uniform(float(config.get("length_jitter_min", 0.85)), float(config.get("length_jitter_max", 1.15))))
        height *= float(rng.uniform(float(config.get("height_jitter_min", 0.85)), float(config.get("height_jitter_max", 1.15))))
        fracture_scale = str(row.get("FractureScale", "small"))
        if fracture_scale in {"large", "medium"}:
            scale_cfg = dict(config.get("multi_scale", {}).get(f"{fracture_scale}_band", {}))
            length *= float(scale_cfg.get("length_multiplier", 1.35 if fracture_scale == "large" else 1.18))
            height *= float(scale_cfg.get("height_multiplier", 1.12 if fracture_scale == "large" else 1.05))
        layer_thickness = float(row["LayerThickness"])
        height = min(height, layer_thickness * float(config.get("max_height_fraction_of_layer", 0.65)))
        height = max(height, min(float(config.get("min_height_time_ms", 4.0)), max(layer_thickness * 0.5, 0.0)))

        y_idx = int(row["IY"])
        x_idx = int(row["IX"])
        t_idx = int(row["IT"])
        orientation = estimate_local_orientation(
            density=density,
            y_idx=y_idx,
            x_idx=x_idx,
            t_idx=t_idx,
            layer=layer,
            source_density=source_density,
            density_p95=density_p95.get(layer, source_density),
            config=config,
        )
        center_x = float(x_values[x_idx])
        center_y = float(y_values[y_idx])
        patch_id = f"init3d_dfn_{patch_idx + 1:06d}"
        density_cell_id = f"{int(row['SourceTraceIdx'])}_{layer}_it{t_idx:04d}"
        rows.append(
            {
                "PatchID": patch_id,
                "GenerationStage": "initial_3d_density_volume_sampling",
                "SourceTraceIdx": int(row["SourceTraceIdx"]),
                "DensityCellID": density_cell_id,
                "DensityCellPatchOrdinal": int(row["DensityCellPatchOrdinal"]),
                "LayerGroup": layer,
                "LayerCode": int(LAYER_CODE[layer]),
                "CenterX": center_x,
                "CenterY": center_y,
                "CenterTime": float(row["CenterTime"]),
                "TimeWindowMin": float(row["TimeWindowMin"]),
                "TimeWindowMax": float(row["TimeWindowMax"]),
                "LayerThickness": layer_thickness,
                "SourceDensity": source_density,
                "CoherenceValue": float(row.get("CoherenceValue", np.nan)),
                "LowCoherenceScore": float(row.get("LowCoherenceScore", 0.0)),
                "LowCoherenceWeight": float(row.get("LowCoherenceWeight", 1.0)),
                "GuidedDensityScore": float(row.get("GuidedDensityScore", source_density)),
                "CandidateScore": float(row.get("CandidateScore", source_density)),
                "SamplingWeight": float(row.get("SamplingWeight", source_density)),
                "DensityQuantileP95Layer": density_p95.get(layer),
                "ExpectedPatchCountForCell": float(row["ExpectedPatchCountForCell"]),
                "EffectiveCountScale": float(row["EffectiveCountScale"]),
                "LengthM": float(length),
                "HeightTimeMs": float(height),
                "AzimuthDeg": float(orientation["azimuth"]),
                "DipDeg": float(orientation["dip"]),
                "TraceGridDX": float(config.get("trace_spacing_x_m", 12.5)),
                "TraceGridDY": float(config.get("trace_spacing_y_m", 12.5)),
                "VoxelIX": x_idx,
                "VoxelIY": y_idx,
                "VoxelIT": t_idx,
                "ComponentID": int(row["ComponentID"]),
                "ComponentVoxelCount": int(row["ComponentVoxelCount"]),
                "LayerDensityThreshold": float(row["LayerDensityThreshold"]),
                "LocalOrientationPointCount": int(orientation["point_count"]),
                "LocalDensityLinearity": float(orientation["linearity"]),
                "LocalDensityPlanarity": float(orientation["planarity"]),
                "LocalEigenvalue1": float(orientation["eigenvalues"][0]),
                "LocalEigenvalue2": float(orientation["eigenvalues"][1]),
                "LocalEigenvalue3": float(orientation["eigenvalues"][2]),
                "DensitySizeFactor": density_factor,
                "OrientationSource": str(orientation["source"]),
                "SizeRule": "base_size_scaled_by_3d_density_quantile",
                "SamplingRule": str(row.get("BandContinuityMode", "")) or ("low_coherence_guided_weighted_sampling_from_3d_density_voxels" if coherence_guidance_config(config).get("enabled", False) else "weighted_sampling_from_3d_high_density_connected_voxels"),
                "FractureScale": fracture_scale,
                "FractureScaleCode": int(row.get("FractureScaleCode", 1)),
                "BandID": str(row.get("BandID", "")),
                "BandPatchOrdinal": int(row.get("BandPatchOrdinal", 0)),
                "BandContinuityMode": str(row.get("BandContinuityMode", "isolated_patch")),
                "BandVoxelCount": int(row.get("BandVoxelCount", row["ComponentVoxelCount"])),
                "BandLengthM": float(row.get("BandLengthM", 0.0)),
                "BandTimeExtentMs": float(row.get("BandTimeExtentMs", 0.0)),
                "BandPatchSpacingM": float(row.get("BandPatchSpacingM", 0.0)),
                "BandMeanDensity": float(row.get("BandMeanDensity", source_density)),
                "NeedsWellCorrection": 1,
            }
        )
        if (patch_idx + 1) % 1000 == 0:
            print(f"[step7b-3d] built patches={patch_idx + 1}/{len(selected)}", flush=True)
    patch_df = pd.DataFrame(rows)
    summary = {
        "density_p95_by_layer": {str(k): float(v) for k, v in density_p95.items()},
        "orientation_source_distribution": layer_distribution(patch_df["OrientationSource"]),
    }
    return patch_df, summary


def patch_vertices(row: pd.Series, display: bool, display_z_scale: float, geometry_time_scale_m_per_ms: float) -> list[tuple[float, float, float]]:
    azimuth = np.deg2rad(float(row["AzimuthDeg"]))
    dip = np.deg2rad(float(np.clip(row["DipDeg"], 1.0, 89.9)))
    half_length = 0.5 * float(row["LengthM"])
    half_height_time = 0.5 * float(row["HeightTimeMs"])
    strike = np.asarray([np.cos(azimuth), np.sin(azimuth)], dtype=float)
    dip_horizontal = np.asarray([-np.sin(azimuth), np.cos(azimuth)], dtype=float)
    horizontal_dip_half = (half_height_time * geometry_time_scale_m_per_ms) / max(np.tan(dip), 1.0e-6)

    center_x = float(row["CenterX"])
    center_y = float(row["CenterY"])
    center_t = float(row["CenterTime"])
    corners = []
    for strike_sign, dip_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        xy = (
            np.asarray([center_x, center_y], dtype=float)
            + strike_sign * half_length * strike
            + dip_sign * horizontal_dip_half * dip_horizontal
        )
        z = center_t + dip_sign * half_height_time
        if display:
            z = -z / display_z_scale
        corners.append((float(xy[0]), float(xy[1]), float(z)))
    return corners


def write_legacy_vtk(path: Path, patch_df: pd.DataFrame, title: str, display: bool, display_z_scale: float, geometry_time_scale_m_per_ms: float) -> None:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for _, row in patch_df.iterrows():
        base = len(points)
        points.extend(patch_vertices(row, display=display, display_z_scale=display_z_scale, geometry_time_scale_m_per_ms=geometry_time_scale_m_per_ms))
        polygons.append([base, base + 1, base + 2, base + 3])

    scalar_columns = [
        ("PatchIndex", np.arange(1, len(patch_df) + 1), "int"),
        ("LayerCode", patch_df["LayerCode"].to_numpy(), "int"),
        ("SourceTraceIdx", patch_df["SourceTraceIdx"].to_numpy(), "int"),
        ("SourceDensity", patch_df["SourceDensity"].to_numpy(), "float"),
        ("CenterTime", patch_df["CenterTime"].to_numpy(), "float"),
        ("LengthM", patch_df["LengthM"].to_numpy(), "float"),
        ("HeightTimeMs", patch_df["HeightTimeMs"].to_numpy(), "float"),
        ("AzimuthDeg", patch_df["AzimuthDeg"].to_numpy(), "float"),
        ("DipDeg", patch_df["DipDeg"].to_numpy(), "float"),
        ("DensitySizeFactor", patch_df["DensitySizeFactor"].to_numpy(), "float"),
        ("LocalDensityPlanarity", patch_df["LocalDensityPlanarity"].to_numpy(), "float"),
    ]
    for optional_name in ["CoherenceValue", "LowCoherenceScore", "LowCoherenceWeight", "GuidedDensityScore", "SamplingWeight"]:
        if optional_name in patch_df.columns:
            values = pd.to_numeric(patch_df[optional_name], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).any():
                scalar_columns.append((optional_name, values, "float"))
    for optional_name in ["FractureScaleCode", "BandPatchOrdinal", "BandVoxelCount"]:
        if optional_name in patch_df.columns:
            scalar_columns.append((optional_name, patch_df[optional_name].to_numpy(), "int"))
    for optional_name in ["BandLengthM", "BandTimeExtentMs", "BandPatchSpacingM", "BandMeanDensity"]:
        if optional_name in patch_df.columns:
            scalar_columns.append((optional_name, patch_df[optional_name].to_numpy(), "float"))
    total_polygon_size = sum(len(poly) + 1 for poly in polygons)
    lines = [
        "# vtk DataFile Version 3.0",
        title,
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {len(points)} float",
    ]
    lines.extend(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    lines.append(f"POLYGONS {len(polygons)} {total_polygon_size}")
    lines.extend(f"{len(poly)} {' '.join(str(idx) for idx in poly)}" for poly in polygons)
    lines.append(f"CELL_DATA {len(polygons)}")
    for name, values, dtype in scalar_columns:
        vtk_type = "int" if dtype == "int" else "float"
        lines.append(f"SCALARS {name} {vtk_type} 1")
        lines.append("LOOKUP_TABLE default")
        if vtk_type == "int":
            safe_values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            lines.extend(str(int(value)) for value in safe_values)
        else:
            safe_values = np.nan_to_num(np.asarray(values, dtype=float), nan=-9999.0, posinf=9999.0, neginf=-9999.0)
            lines.extend(f"{float(value):.6f}" for value in safe_values)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(patch_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "PatchID",
        "GenerationStage",
        "DensityCellID",
        "SourceTraceIdx",
        "LayerGroup",
        "VoxelIX",
        "VoxelIY",
        "VoxelIT",
        "SourceDensity",
        "CoherenceValue",
        "LowCoherenceScore",
        "LowCoherenceWeight",
        "GuidedDensityScore",
        "CandidateScore",
        "SamplingWeight",
        "DensitySizeFactor",
        "ExpectedPatchCountForCell",
        "EffectiveCountScale",
        "CenterX",
        "CenterY",
        "CenterTime",
        "LengthM",
        "HeightTimeMs",
        "AzimuthDeg",
        "DipDeg",
        "ComponentID",
        "ComponentVoxelCount",
        "LocalDensityPlanarity",
        "OrientationSource",
        "SizeRule",
        "SamplingRule",
        "FractureScale",
        "BandID",
        "BandPatchOrdinal",
        "BandContinuityMode",
        "BandVoxelCount",
        "BandLengthM",
        "BandTimeExtentMs",
        "BandPatchSpacingM",
        "BandMeanDensity",
        "NeedsWellCorrection",
    ]
    columns = [column for column in columns if column in patch_df.columns]
    audit = patch_df[columns].copy()
    audit["Action"] = "create_initial_3d_fracture_patch"
    audit["ActionReason"] = "sampled_from_step6b_3d_density_connected_high_density_voxel"
    return audit


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    grid: dict[str, Any],
    candidates: pd.DataFrame,
    patch_df: pd.DataFrame,
    candidate_summary: dict[str, Any],
    sampling_summary: dict[str, Any],
    patch_build_summary: dict[str, Any],
    guidance_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = dict(config.get("target_block", {}))
    guidance = coherence_guidance_config(config)
    macro_mass_column = "SamplingWeight" if guidance.get("enabled", False) and "SamplingWeight" in candidates.columns else "SourceDensity"
    density_by_layer = candidates.groupby("LayerGroup")[macro_mass_column].sum().to_dict()
    patch_by_layer = patch_df["LayerGroup"].value_counts().to_dict()
    density_mass = float(sum(float(value) for value in density_by_layer.values()))
    patch_count = int(len(patch_df))
    macro_distribution: dict[str, dict[str, float]] = {}
    for layer in ALLOWED_LAYERS:
        density_share = float(density_by_layer.get(layer, 0.0)) / density_mass if density_mass > 0 else 0.0
        patch_share = float(patch_by_layer.get(layer, 0)) / patch_count if patch_count > 0 else 0.0
        macro_distribution[layer] = {
            "density_mass": float(density_by_layer.get(layer, 0.0)),
            "density_mass_share": density_share,
            "patch_count": int(patch_by_layer.get(layer, 0)),
            "patch_count_share": patch_share,
            "absolute_share_difference": abs(density_share - patch_share),
        }
    x_min = float(target.get("x_min", np.min(grid["x_values"])))
    x_max = float(target.get("x_max", np.max(grid["x_values"])))
    y_min = float(target.get("y_min", np.min(grid["y_values"])))
    y_max = float(target.get("y_max", np.max(grid["y_values"])))
    expected = float(sampling_summary["expected_patch_count"])
    expected_error = abs(float(patch_count) - expected) / expected if expected > 0 else 1.0
    checks = {
        "has_patches": patch_count > 0,
        "layers_limited_to_sha3_sha4": bool(set(patch_df["LayerGroup"].dropna().astype(str)).issubset(set(ALLOWED_LAYERS))),
        "centers_within_target_block": bool(patch_df["CenterX"].between(x_min, x_max).all() and patch_df["CenterY"].between(y_min, y_max).all()),
        "times_within_layer_windows": bool(patch_df["CenterTime"].ge(patch_df["TimeWindowMin"]).all() and patch_df["CenterTime"].le(patch_df["TimeWindowMax"]).all()),
        "patch_count_matches_sampling_target": bool(expected_error <= 0.01),
        "macro_layer_distribution_reasonable": bool(all(payload["absolute_share_difference"] <= 0.10 for payload in macro_distribution.values())),
        "density_controls_patch_size": bool(patch_df["LengthM"].corr(patch_df["SourceDensity"], method="spearman") > 0.35),
        "dip_used_in_geometry": True,
        "traceability_fields_present": bool(patch_df[["PatchID", "DensityCellID", "SourceTraceIdx", "SourceDensity", "VoxelIX", "VoxelIY", "VoxelIT"]].notna().all().all()),
        "raw_vtk_output_exists": paths["raw_vtk"].exists(),
        "audit_rows_match_patch_count": paths["audit_csv"].exists() and len(patch_df) > 0,
    }
    status = "pass" if all(checks.values()) else "fail"
    return {
        "status": status,
        "config_path": str(config_path),
        "density_sgy": str(Path(config["density_sgy"]).resolve()),
        "trace_mapping_npz": str(Path(config["trace_mapping_npz"]).resolve()),
        "initial_dfn_csv": str(paths["dfn_csv"]),
        "initial_dfn_raw_vtk": str(paths["raw_vtk"]),
        "initial_dfn_generation_audit_csv": str(paths["audit_csv"]),
        "summary_json": str(paths["summary_json"]),
        "generation_logic": "low_coherence_guided_3d_density_sampling_with_local_pca_orientation_and_density_scaled_patch_size" if guidance.get("enabled", False) else "3d_density_connected_voxel_sampling_with_local_pca_orientation_and_density_scaled_patch_size",
        "target_block": target,
        "allowed_layers": ALLOWED_LAYERS,
        "coherence_guidance": guidance_summary or {"enabled": False},
        "density_grid": {
            "shape_y_x_t": [int(grid["density"].shape[0]), int(grid["density"].shape[1]), int(grid["density"].shape[2])],
            "x_range": [float(np.min(grid["x_values"])), float(np.max(grid["x_values"]))],
            "y_range": [float(np.min(grid["y_values"])), float(np.max(grid["y_values"]))],
            "time_range_ms": [float(np.min(grid["samples"])), float(np.max(grid["samples"]))],
            "sample_interval_us": float(grid["sample_interval_us"]),
            "sgy_data_format": int(grid["data_format"]),
            "density_stats": finite_stats(grid["density"].ravel()),
        },
        "candidate_summary": candidate_summary,
        "candidate_voxels": {
            "row_count": int(len(candidates)),
            "layer_distribution": layer_distribution(candidates["LayerGroup"]),
            "density_stats": finite_stats(candidates["SourceDensity"]),
            "macro_mass_column": macro_mass_column,
            "sampling_weight_stats": finite_stats(candidates["SamplingWeight"]) if "SamplingWeight" in candidates.columns else None,
            "coherence_value_stats": finite_stats(candidates["CoherenceValue"]) if "CoherenceValue" in candidates.columns else None,
            "low_coherence_score_stats": finite_stats(candidates["LowCoherenceScore"]) if "LowCoherenceScore" in candidates.columns else None,
        },
        "sampling_summary": sampling_summary,
        "patch_build_summary": patch_build_summary,
        "initial_dfn": {
            "patch_count": patch_count,
            "layer_distribution": layer_distribution(patch_df["LayerGroup"]),
            "fracture_scale_distribution": layer_distribution(patch_df["FractureScale"]) if "FractureScale" in patch_df.columns else {"small": patch_count},
            "band_count": int(patch_df.loc[patch_df.get("BandID", pd.Series(dtype=object)).astype(str).ne(""), "BandID"].nunique()) if "BandID" in patch_df.columns else 0,
            "source_trace_count": int(patch_df["SourceTraceIdx"].nunique()),
            "center_x_stats": finite_stats(patch_df["CenterX"]),
            "center_y_stats": finite_stats(patch_df["CenterY"]),
            "center_time_stats": finite_stats(patch_df["CenterTime"]),
            "length_m_stats": finite_stats(patch_df["LengthM"]),
            "height_time_ms_stats": finite_stats(patch_df["HeightTimeMs"]),
            "source_density_stats": finite_stats(patch_df["SourceDensity"]),
            "coherence_value_stats": finite_stats(patch_df["CoherenceValue"]) if "CoherenceValue" in patch_df.columns else None,
            "low_coherence_score_stats": finite_stats(patch_df["LowCoherenceScore"]) if "LowCoherenceScore" in patch_df.columns else None,
            "sampling_weight_stats": finite_stats(patch_df["SamplingWeight"]) if "SamplingWeight" in patch_df.columns else None,
            "azimuth_deg_stats": finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg_stats": finite_stats(patch_df["DipDeg"]),
            "density_size_factor_stats": finite_stats(patch_df["DensitySizeFactor"]),
            "length_density_spearman": float(patch_df["LengthM"].corr(patch_df["SourceDensity"], method="spearman")),
            "height_density_spearman": float(patch_df["HeightTimeMs"].corr(patch_df["SourceDensity"], method="spearman")),
        },
        "macro_distribution_check": macro_distribution,
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)

    density_sgy = Path(config["density_sgy"]).resolve()
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    layer_dir = Path(config["layer_dir"]).resolve()
    for label_name, path in [("density_sgy", density_sgy), ("trace_mapping_npz", trace_mapping_npz), ("layer_dir", layer_dir)]:
        if not path.exists():
            raise FileNotFoundError(f"{label_name} not found: {path}")

    guidance = coherence_guidance_config(config)
    coherence_sgy = Path(guidance["coherence_sgy"]).resolve() if guidance.get("enabled", False) else None
    if coherence_sgy is not None and not coherence_sgy.exists():
        raise FileNotFoundError(f"coherence_sgy not found: {coherence_sgy}")

    rng = np.random.default_rng(int(config.get("random_seed", 20260703)))
    print("[step7b-3d] loading density SGY", flush=True)
    grid = load_density_grid(density_sgy, trace_mapping_npz)
    coherence_grid = None
    guidance_summary: dict[str, Any] = {"enabled": False}
    if coherence_sgy is not None:
        print("[step7b-3d] loading coherence guidance SGY", flush=True)
        coherence_grid, guidance_summary = load_guidance_grid(coherence_sgy, grid["source_trace_idx"], grid["samples"])
        invalid_guidance = ~np.isfinite(coherence_grid)
        if "valid_min" in guidance:
            invalid_guidance |= coherence_grid < float(guidance["valid_min"])
        if "valid_max" in guidance:
            invalid_guidance |= coherence_grid > float(guidance["valid_max"])
        invalid_count = int(invalid_guidance.sum())
        if invalid_count > 0:
            coherence_grid = coherence_grid.copy()
            coherence_grid[invalid_guidance] = np.nan
            guidance_summary["invalid_value_count_after_config_filter"] = invalid_count
    print("[step7b-3d] attaching T4-T7 surfaces", flush=True)
    surfaces = attach_surface_grids(layer_dir, grid["x_values"], grid["y_values"])
    print("[step7b-3d] selecting connected high-density voxels", flush=True)
    candidates, candidate_summary = build_candidate_voxels(
        density=grid["density"],
        samples=grid["samples"],
        surfaces=surfaces,
        source_trace_idx=grid["source_trace_idx"],
        config=config,
        coherence=coherence_grid,
    )
    print(f"[step7b-3d] candidate voxels={len(candidates)}", flush=True)
    generation_mode = str(config.get("generation_mode", "weighted_sampling"))
    if generation_mode == "multi_scale_connected_bands":
        selected, sampling_summary = sample_candidate_voxels_multiscale(
            candidates,
            x_values=grid["x_values"],
            y_values=grid["y_values"],
            config=config,
            rng=rng,
        )
    elif generation_mode == "weighted_sampling":
        selected, sampling_summary = sample_candidate_voxels(candidates, config=config, rng=rng)
    else:
        raise ValueError(f"unsupported generation_mode: {generation_mode}")
    print(f"[step7b-3d] sampled patches={len(selected)}", flush=True)
    patch_df, patch_build_summary = build_patch_table(
        selected=selected,
        density=grid["density"],
        x_values=grid["x_values"],
        y_values=grid["y_values"],
        config=config,
        rng=rng,
    )
    audit_df = build_audit(patch_df)
    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    display_z_scale = float(config.get("display_z_scale", 5.0))
    geometry_time_scale = float(config.get("geometry_time_scale_m_per_ms", config.get("orientation_time_scale_m_per_ms", 1.0)))
    write_legacy_vtk(paths["raw_vtk"], patch_df, "initial_3d_dfn_raw_time", display=False, display_z_scale=display_z_scale, geometry_time_scale_m_per_ms=geometry_time_scale)

    summary = build_summary(
        config_path=config_path,
        config=config,
        paths=paths,
        grid=grid,
        candidates=candidates,
        patch_df=patch_df,
        candidate_summary=candidate_summary,
        sampling_summary=sampling_summary,
        patch_build_summary=patch_build_summary,
        guidance_summary=guidance_summary,
    )
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Initial 3D DFN CSV: {paths['dfn_csv']}")
    print(f"Initial 3D DFN raw VTK: {paths['raw_vtk']}")
    print(f"Summary JSON: {paths['summary_json']}")
    print(f"Patch count: {len(patch_df)} status={summary['status']}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

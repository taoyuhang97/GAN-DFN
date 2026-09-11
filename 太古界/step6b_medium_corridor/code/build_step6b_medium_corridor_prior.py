from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
from scipy import ndimage
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
TAIGU_ROOT = CURRENT_DIR.parents[1]
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))
FORMAL_MAINLINE_ROOT = TAIGU_ROOT.parent / "优化阶段二" / "正式主线"
if str(FORMAL_MAINLINE_ROOT) not in sys.path:
    sys.path.append(str(FORMAL_MAINLINE_ROOT))
sys.path.append(str(FORMAL_MAINLINE_ROOT / "common"))

from common.multiscale_density.build_multiscale_density_bundle import (
    ensure_dir,
    finite_stats,
    flat_to_grid,
    grid_to_flat,
    high_score,
    load_mapping,
    load_trace_matrix,
    low_score,
    read_sgy_sample_axis,
    regular_sample_axis,
    valid_values,
    write_sgy_like,
)
from common.multiscale_density.build_multiscale_density_bundle import _resample_matrix
from common.attribute_sampling.attribute_contract import score_attribute


FORMAL_ROOT = CURRENT_DIR.parent
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from horizon_trace_table.horizon_contract import (  # noqa: E402
    HorizonTraceContract,
    apply_validity_inplace,
    contract_summary,
    load_contract_for_mapping,
    surface_grids_from_contract,
    validate_window_contract,
)
from step6a_density_volume.predict_taigu_density_volume import fill_missing_horizons_nearest  # noqa: E402

DEFAULT_CONFIG = CURRENT_DIR.parent / "configs/formal_candidate_cheye1_multiscale_density_v1.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR.parent / "output/default/step6b_medium"


def quantile_bounds(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
    """Return robust finite quantile bounds for local score normalization."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.quantile(finite, low_q))
    high = float(np.quantile(finite, high_q))
    if high <= low:
        high = low + 1.0e-6
    return low, high


def load_attribute_matrix(
    path: Path,
    source_trace_idx: np.ndarray,
    absolute_samples: np.ndarray,
    label: str,
    absolute_time_origin_ms: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an attribute on the project absolute-TWT axis."""
    raw_samples, raw_summary = read_sgy_sample_axis(path)
    raw_target = np.asarray(absolute_samples, dtype=np.float64) - float(absolute_time_origin_ms) + float(raw_samples[0])
    matrix, _, load_summary = load_trace_matrix(path, source_trace_idx, raw_target, label)
    load_summary.update(
        {
            "raw_time_min_ms": float(raw_samples[0]),
            "raw_time_max_ms": float(raw_samples[-1]),
            "absolute_time_origin_ms": float(absolute_time_origin_ms),
            "absolute_time_min_ms": float(raw_samples[0] - raw_samples[0] + absolute_time_origin_ms),
            "absolute_time_max_ms": float(raw_samples[-1] - raw_samples[0] + absolute_time_origin_ms),
            "target_absolute_time_min_ms": float(absolute_samples[0]),
            "target_absolute_time_max_ms": float(absolute_samples[-1]),
            "source_trace_count": int(raw_summary["trace_count"]),
        }
    )
    return matrix, load_summary


def load_taigu_horizon_contract(
    config: dict[str, Any], mapping: dict[str, np.ndarray]
) -> HorizonTraceContract:
    """Load the semantic three-surface table and reproduce Step6A horizon fill."""
    csv_value = config.get("horizon_contract_csv")
    if not csv_value:
        return load_contract_for_mapping(config, mapping)
    path = Path(str(csv_value)).resolve()
    source_trace_idx = np.asarray(mapping["source_trace_idx"], dtype=np.int64)
    contract_trace_idx = source_trace_idx
    table = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["TraceIdx", "TopTimeMs", "MidTimeMs", "BaseTimeMs", "SurfaceValid"],
    )
    for column in table.columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if len(table) == 0 or int(contract_trace_idx.min()) < 0 or int(contract_trace_idx.max()) >= len(table):
        raise ValueError("attribute trace mapping exceeds semantic horizon contract")
    selected = table.iloc[contract_trace_idx].copy().reset_index(drop=True)
    if not np.array_equal(selected["TraceIdx"].to_numpy(dtype=np.int64), contract_trace_idx):
        raise ValueError("semantic horizon contract row index does not equal TraceIdx")
    selected["X"] = np.asarray(mapping["x"], dtype=np.float64)
    selected["Y"] = np.asarray(mapping["y"], dtype=np.float64)
    if bool(config.get("horizon_fill_enabled", True)):
        selected = fill_missing_horizons_nearest(
            selected,
            max_distance_m=float(config.get("max_horizon_fill_distance_m", 25.0)),
            min_thickness_ms=float(config.get("min_horizon_thickness_ms", 1.0)),
        )
        valid = selected["FilledSurfaceValid"].to_numpy(dtype=bool, copy=True)
        correction = selected["HorizonFillUsed"].to_numpy(dtype=np.uint8)
    else:
        valid = selected["SurfaceValid"].fillna(0).to_numpy(dtype=bool, copy=True)
        correction = np.zeros(len(selected), dtype=np.uint8)
    top = selected["TopTimeMs"].to_numpy(dtype=np.float32)
    middle = selected["MidTimeMs"].to_numpy(dtype=np.float32)
    bottom = selected["BaseTimeMs"].to_numpy(dtype=np.float32)
    min_thickness = float(config.get("min_horizon_thickness_ms", 1.0))
    valid &= np.isfinite(top) & np.isfinite(middle) & np.isfinite(bottom)
    valid &= (middle - top >= min_thickness) & (bottom - middle >= min_thickness)
    return HorizonTraceContract(
        table_path=path,
        trace_idx=source_trace_idx.copy(),
        t4=top,
        t5=middle.copy(),
        t6=middle,
        t7=bottom,
        surface_order_valid=valid,
        shasan_present=valid.copy(),
        shasi_present=valid.copy(),
        correction_code=correction,
    )


def validate_attribute_mapping_contract(
    mapping: dict[str, np.ndarray],
    attribute_header_csv: Path,
    step6a_mapping_npz: Path,
    density_trace_count: int,
    xy_tolerance_m: float,
) -> dict[str, Any]:
    """Reject OBN indices and any reordering between Step6A and Step6B."""
    lengths = {key: len(np.asarray(mapping[key])) for key in ("source_trace_idx", "x", "y", "ix", "iy", "output_trace_index")}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"attribute mapping arrays have different lengths: {lengths}")
    trace_count = lengths["source_trace_idx"]
    trace_idx = np.asarray(mapping["source_trace_idx"], dtype=np.int64)
    output_idx = np.asarray(mapping["output_trace_index"], dtype=np.int64)
    ix = np.asarray(mapping["ix"], dtype=np.int32)
    iy = np.asarray(mapping["iy"], dtype=np.int32)
    if density_trace_count != trace_count:
        raise ValueError(f"Step6A density trace count differs from attribute grid: {density_trace_count} != {trace_count}")
    if not np.array_equal(output_idx, np.arange(trace_count, dtype=np.int64)):
        raise ValueError("attribute mapping output_trace_index is not sequential")
    if len(np.unique(trace_idx)) != trace_count:
        raise ValueError("attribute mapping contains duplicate source TraceIdx")
    if len(np.unique(np.column_stack([ix, iy]), axis=0)) != trace_count:
        raise ValueError("attribute mapping contains duplicate IX/IY cells")
    expected_count = int((ix.max() + 1) * (iy.max() + 1))
    if trace_count != expected_count:
        raise ValueError(f"attribute mapping is not a complete rectangle: {trace_count} != {expected_count}")

    header = pd.read_csv(attribute_header_csv, usecols=["TraceIdx", "X", "Y"], encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y"):
        header[column] = pd.to_numeric(header[column], errors="coerce")
    header = header.dropna().drop_duplicates("TraceIdx").set_index("TraceIdx")
    selected = header.reindex(trace_idx)
    if selected[["X", "Y"]].isna().any().any():
        raise ValueError("Step6B mapping contains TraceIdx absent from the attribute trace header")
    header_distance = np.hypot(
        selected["X"].to_numpy(float) - np.asarray(mapping["x"], dtype=float),
        selected["Y"].to_numpy(float) - np.asarray(mapping["y"], dtype=float),
    )
    if float(header_distance.max()) > xy_tolerance_m:
        raise ValueError(
            "Step6B TraceIdx does not match attribute-header X/Y: "
            f"max_distance={float(header_distance.max()):.3f}m; an OBN mapping may have been supplied"
        )

    with np.load(step6a_mapping_npz) as step6a_mapping:
        required = {"trace_idx", "x", "y", "output_trace_index"}
        missing = sorted(required.difference(step6a_mapping.files))
        if missing:
            raise ValueError(f"Step6A trace mapping missing fields: {missing}")
        step6a_trace_idx = np.asarray(step6a_mapping["trace_idx"], dtype=np.int64)
        step6a_output_idx = np.asarray(step6a_mapping["output_trace_index"], dtype=np.int64)
        step6a_x = np.asarray(step6a_mapping["x"], dtype=float)
        step6a_y = np.asarray(step6a_mapping["y"], dtype=float)
    if not np.array_equal(step6a_output_idx, output_idx):
        raise ValueError("Step6A output order differs from the Common attribute demo grid")
    if not np.array_equal(step6a_trace_idx, trace_idx):
        raise ValueError("Step6A attribute TraceIdx differs from the Common attribute demo grid")
    step6a_distance = np.hypot(step6a_x - np.asarray(mapping["x"], float), step6a_y - np.asarray(mapping["y"], float))
    if float(step6a_distance.max()) > xy_tolerance_m:
        raise ValueError("Step6A coordinates differ from the Common attribute demo grid")
    return {
        "status": "pass",
        "contract": "attribute_trace_idx_only_no_obn_trace_idx",
        "trace_count": trace_count,
        "x_line_count": int(ix.max() + 1),
        "y_line_count": int(iy.max() + 1),
        "attribute_header_xy_distance_max_m": float(header_distance.max()),
        "attribute_header_xy_distance_median_m": float(np.median(header_distance)),
        "step6a_grid_xy_distance_max_m": float(step6a_distance.max()),
        "step6a_trace_order_identical": True,
    }


def score_by_semantic_layer(
    values: np.ndarray,
    valid: np.ndarray,
    attribute: str,
    contract: dict[str, Any],
    horizons: HorizonTraceContract,
    samples: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the saved Common normalization limits separately by layer."""
    score = np.zeros_like(values, dtype=np.float32)
    axis = np.asarray(samples, dtype=np.float64)[None, :]
    layer_masks = {
        "上部复合层": horizons.surface_order_valid[:, None] & (axis >= horizons.t4[:, None]) & (axis < horizons.t6[:, None]),
        "太古界风化壳": horizons.surface_order_valid[:, None] & (axis >= horizons.t6[:, None]) & (axis <= horizons.t7[:, None]),
    }
    layer_summary: dict[str, Any] = {}
    for layer, layer_mask in layer_masks.items():
        mask = layer_mask & valid
        scored = score_attribute(values, attribute, contract["layers"][layer][attribute])
        score[mask] = scored[mask]
        layer_summary[layer] = {
            "valid_count": int(mask.sum()),
            "clip_low": float(contract["layers"][layer][attribute]["clip_low"]),
            "clip_high": float(contract["layers"][layer][attribute]["clip_high"]),
            "polarity": str(contract["layers"][layer][attribute]["polarity"]),
            "score_stats": finite_stats(score[mask]),
        }
    score[~valid] = 0.0
    return score, {"contract_version": contract.get("version"), "layers": layer_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step6B medium-scale fracture corridor prior.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-x-lines", type=int, default=0, help="Smoke-test cap; 0 uses the full grid.")
    parser.add_argument("--candidate-quantile", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--anttrack-high-quantile", type=float, default=0.90)
    parser.add_argument("--anttrack-weight", type=float, default=0.70)
    parser.add_argument("--lowcoh-weight", type=float, default=0.20)
    parser.add_argument("--curvature-weight", type=float, default=0.10)
    parser.add_argument("--density-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--seed-medium-score-threshold", type=float, default=0.62)
    parser.add_argument("--growth-anttrack-floor", type=float, default=0.28)
    parser.add_argument("--growth-medium-score-threshold", type=float, default=0.52)
    parser.add_argument("--support-score-threshold", type=float, default=0.35)
    parser.add_argument("--support-neighborhood-cells", type=int, default=1)
    # Retained only so older runners remain callable. These switches no longer
    # activate a separate rescue branch.
    parser.add_argument("--enable-supported-rescue-branch", action="store_true")
    parser.add_argument("--rescue-min-anttrack-score", type=float, default=0.42)
    parser.add_argument("--rescue-min-lowcoh-score", type=float, default=0.72)
    parser.add_argument("--rescue-min-curvature-score", type=float, default=0.62)
    parser.add_argument("--rescue-min-local-support-fraction", type=float, default=0.22)
    parser.add_argument("--enable-lowcoh-structure-rescue", action="store_true")
    parser.add_argument("--lowcoh-rescue-min-lowcoh-score", type=float, default=0.78)
    parser.add_argument("--lowcoh-rescue-min-anttrack-score", type=float, default=0.18)
    parser.add_argument("--lowcoh-rescue-min-curvature-score", type=float, default=0.48)
    parser.add_argument("--lowcoh-rescue-local-score-threshold", type=float, default=0.70)
    parser.add_argument("--lowcoh-rescue-min-local-support-fraction", type=float, default=0.20)
    parser.add_argument("--lowcoh-rescue-score-base", type=float, default=0.82)
    parser.add_argument("--lowcoh-rescue-attribute-weight", type=float, default=0.12)
    parser.add_argument("--lowcoh-rescue-continuity-weight", type=float, default=0.06)
    parser.add_argument("--lowcoh-rescue-min-component-score-mean", type=float, default=0.64)
    parser.add_argument("--lowcoh-rescue-bridge-iterations", type=int, default=1)
    parser.add_argument("--lowcoh-rescue-bridge-min-lowcoh-score", type=float, default=0.62)
    parser.add_argument("--lowcoh-rescue-bridge-min-combined-score", type=float, default=0.58)
    parser.add_argument("--min-component-voxels", type=int, default=20)
    parser.add_argument("--fragment-recovery-min-voxels", type=int, default=15)
    parser.add_argument("--fragment-bridge-score", type=float, default=0.45)
    parser.add_argument("--fragment-bridge-iterations", type=int, default=1)
    parser.add_argument("--fragment-recovery-min-anttrack-mean", type=float, default=0.80)
    parser.add_argument("--fragment-bridge-min-anttrack-score", type=float, default=0.35)
    parser.add_argument("--max-component-voxels-before-split", type=int, default=12000)
    parser.add_argument("--split-tile-cells", type=int, default=16)
    parser.add_argument("--split-time-samples", type=int, default=8)
    parser.add_argument("--orientation-time-scale-m-per-ms", type=float, default=2.0)
    parser.add_argument("--min-vertical-extent-ms", type=float, default=30.0)
    parser.add_argument("--min-component-dip-deg", type=float, default=15.0)
    parser.add_argument("--min-component-linearity", type=float, default=1.05)
    parser.add_argument("--min-component-score-mean", type=float, default=0.45)
    parser.add_argument("--max-horizontal-layer-thickness-ms", type=float, default=8.0)
    parser.add_argument("--horizontal-layer-min-extent-m", type=float, default=700.0)
    parser.add_argument("--layer-like-max-dip-deg", type=float, default=18.0)
    parser.add_argument("--layer-like-min-horizontal-extent-m", type=float, default=500.0)
    parser.add_argument("--layer-like-max-time-extent-ms", type=float, default=80.0)
    parser.add_argument("--strat-following-min-horizontal-extent-m", type=float, default=500.0)
    parser.add_argument("--strat-following-max-relative-std", type=float, default=0.035)
    parser.add_argument("--strat-following-max-relative-span", type=float, default=0.14)
    parser.add_argument("--vtk-max-points", type=int, default=250000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.quantile(finite, q))


def grid_axis_values(mapping: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    x_values = np.zeros(nx, dtype=np.float64)
    y_values = np.zeros(ny, dtype=np.float64)
    for idx in range(nx):
        x_values[idx] = float(np.median(mapping["x"][ix == idx]))
    for idx in range(ny):
        y_values[idx] = float(np.median(mapping["y"][iy == idx]))
    return x_values, y_values


def pca_orientation(points: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if points.shape[0] < 3:
        return None, None, None
    centered = points - points.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    main = vh[0]
    normal = vh[-1]
    azimuth = float((np.degrees(np.arctan2(main[0], main[1])) + 360.0) % 180.0)
    dip = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1.0e-9), 0.0, 1.0))))
    linearity = float(s[0] / max(s[1], 1.0e-9)) if len(s) > 1 else None
    return azimuth, dip, linearity


def build_local_support_grids(
    lowcoh_grid: np.ndarray,
    curv_grid: np.ndarray,
    radius: int,
    support_score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.maximum(lowcoh_grid, curv_grid).astype(np.float32)
    support_binary = support >= float(support_score_threshold)
    if radius <= 0:
        return support, support_binary.astype(np.float32)
    size = 2 * int(radius) + 1
    support_max = ndimage.maximum_filter(support, size=(size, size, size), mode="nearest").astype(np.float32)
    support_fraction = ndimage.uniform_filter(
        support_binary.astype(np.float32),
        size=(size, size, size),
        mode="nearest",
    ).astype(np.float32)
    return support_max, support_fraction


def retain_seeded_growth(seed_grid: np.ndarray, growth_grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep 26-connected growth components that contain at least one strong AntTrack seed."""
    growth = np.asarray(growth_grid, dtype=bool)
    seeds = np.asarray(seed_grid, dtype=bool) & growth
    labels, component_count = ndimage.label(growth, structure=np.ones((3, 3, 3), dtype=np.uint8))
    seed_labels = np.unique(labels[seeds])
    seed_labels = seed_labels[seed_labels > 0]
    retained = np.isin(labels, seed_labels)
    return retained, {
        "growth_component_count": int(component_count),
        "seed_voxel_count": int(seeds.sum()),
        "seeded_component_count": int(len(seed_labels)),
        "retained_growth_voxel_count": int(retained.sum()),
        "discarded_unseeded_growth_voxel_count": int(growth.sum() - retained.sum()),
    }


def repair_fragmented_candidates(
    mask: np.ndarray,
    score: np.ndarray,
    anttrack_score: np.ndarray,
    valid: np.ndarray,
    min_component_voxels: int,
    recovery_min_voxels: int,
    bridge_score: float,
    bridge_iterations: int,
    recovery_min_anttrack_mean: float,
    bridge_min_anttrack_score: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Conservatively recover fragments near an already viable component."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    repaired = np.asarray(mask, dtype=bool).copy()
    labels, count = ndimage.label(repaired, structure=structure)
    sizes = np.bincount(labels.ravel())
    large = np.isin(labels, np.where(sizes >= int(min_component_voxels))[0])
    small_labels = np.where((sizes >= int(recovery_min_voxels)) & (sizes < int(min_component_voxels)))[0]
    small_labels = small_labels[small_labels > 0]
    bridge = np.zeros_like(repaired, dtype=bool)
    recovered_components = 0
    recovered_voxels = 0
    rejected_weak_anttrack_components = 0
    iterations = max(int(bridge_iterations), 1)
    near_large = ndimage.binary_dilation(large, structure=structure, iterations=iterations)
    component_slices = ndimage.find_objects(labels, max_label=count)
    for label_id in tqdm(small_labels, desc="Step6B fragment repair", unit="component"):
        component_slice = component_slices[int(label_id) - 1]
        if component_slice is None:
            continue
        padded_slice = tuple(
            slice(max(int(axis.start) - iterations, 0), min(int(axis.stop) + iterations, labels.shape[dim]))
            for dim, axis in enumerate(component_slice)
        )
        small = labels[padded_slice] == int(label_id)
        local_anttrack = anttrack_score[padded_slice]
        if float(np.mean(local_anttrack[small])) < float(recovery_min_anttrack_mean):
            rejected_weak_anttrack_components += 1
            continue
        local = ndimage.binary_dilation(small, structure=structure, iterations=iterations)
        additions = (
            local
            & near_large[padded_slice]
            & ~repaired[padded_slice]
            & valid[padded_slice]
            & (score[padded_slice] >= float(bridge_score))
            & (local_anttrack >= float(bridge_min_anttrack_score))
        )
        if not np.any(additions):
            continue
        bridge[padded_slice] |= additions
        recovered_components += 1
        recovered_voxels += int(additions.sum())
    repaired |= bridge
    return repaired, {
        "initial_component_count": int(count),
        "small_component_count": int(len(small_labels)),
        "recovered_component_count": int(recovered_components),
        "rejected_weak_anttrack_component_count": int(rejected_weak_anttrack_components),
        "bridge_voxel_count": int(recovered_voxels),
        "repaired_candidate_voxel_count": int(repaired.sum()),
        "fragment_recovery_min_voxels": int(recovery_min_voxels),
        "fragment_bridge_score": float(bridge_score),
        "fragment_bridge_iterations": int(bridge_iterations),
        "fragment_recovery_min_anttrack_mean": float(recovery_min_anttrack_mean),
        "fragment_bridge_min_anttrack_score": float(bridge_min_anttrack_score),
    }


def build_medium_components(
    mask: np.ndarray,
    score: np.ndarray,
    ant_score: np.ndarray,
    lowcoh_score: np.ndarray,
    curvature_score: np.ndarray,
    branch_code: np.ndarray,
    surfaces: dict[str, np.ndarray],
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    raw_labels, raw_count = ndimage.label(mask, structure=structure)
    x_values, y_values = grid_axis_values(mapping)
    time_scale = float(args.orientation_time_scale_m_per_ms)
    component_id_grid = np.zeros(mask.shape, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    next_id = 1
    split_raw_count = 0
    rejected_small = 0
    rejected_horizontal = 0
    rejected_thin = 0
    rejected_layer_like = 0
    rejected_low_dip = 0
    rejected_low_linearity = 0
    rejected_low_score = 0
    rejected_strat_following = 0

    sizes = np.bincount(raw_labels.ravel())
    # ``np.where(raw_labels == raw_id)`` rescans the complete 3-D label volume
    # once for every component.  That was tolerable for the old 3 km demo but
    # becomes prohibitively expensive for the 10 km / 2 ms grid.  Compute the
    # component bounding boxes once and search only inside each local box.
    component_slices = ndimage.find_objects(raw_labels, max_label=raw_count)
    for raw_id in tqdm(range(1, raw_count + 1), desc="Step6B component QC", unit="component"):
        if int(sizes[raw_id]) <= 0:
            continue
        component_slice = component_slices[raw_id - 1]
        if component_slice is None:
            continue
        local_labels = raw_labels[component_slice]
        yy, xx, tt = np.where(local_labels == raw_id)
        yy += int(component_slice[0].start)
        xx += int(component_slice[1].start)
        tt += int(component_slice[2].start)
        if yy.size > int(args.max_component_voxels_before_split):
            split_raw_count += 1
            tile = max(int(args.split_tile_cells), 1)
            time_tile = max(int(args.split_time_samples), 1)
            tile_key = (xx // tile) + 10000 * (yy // tile) + 100000000 * (tt // time_tile)
            unique_keys = np.unique(tile_key)
            groups = [np.where(tile_key == key)[0] for key in unique_keys]
        else:
            groups = [np.arange(yy.size)]

        for group in groups:
            gyy = yy[group]
            gxx = xx[group]
            gtt = tt[group]
            voxel_count = int(len(group))
            if voxel_count < int(args.min_component_voxels):
                rejected_small += 1
                continue
            x_extent = float(x_values[gxx].max() - x_values[gxx].min()) if voxel_count else 0.0
            y_extent = float(y_values[gyy].max() - y_values[gyy].min()) if voxel_count else 0.0
            t_extent = float(samples[gtt].max() - samples[gtt].min()) if voxel_count else 0.0
            horizontal_extent = max(x_extent, y_extent)
            horizontal_like = bool(
                t_extent <= float(args.max_horizontal_layer_thickness_ms)
                and horizontal_extent >= float(args.horizontal_layer_min_extent_m)
            )
            if horizontal_like:
                rejected_horizontal += 1
                continue
            if t_extent < float(args.min_vertical_extent_ms):
                rejected_thin += 1
                continue
            points = np.column_stack([x_values[gxx], y_values[gyy], samples[gtt] * time_scale]).astype(np.float64)
            azimuth, dip, linearity = pca_orientation(points)
            score_mean = float(np.mean(score[gyy, gxx, gtt]))
            branch_values = branch_code[gyy, gxx, gtt].astype(np.uint8)
            required_score_mean = float(args.min_component_score_mean)
            if score_mean < required_score_mean:
                rejected_low_score += 1
                continue
            if dip is not None and dip < float(args.min_component_dip_deg):
                rejected_low_dip += 1
                continue
            if linearity is not None and linearity < float(args.min_component_linearity):
                rejected_low_linearity += 1
                continue
            layer_like = bool(
                dip is not None
                and dip <= float(args.layer_like_max_dip_deg)
                and horizontal_extent >= float(args.layer_like_min_horizontal_extent_m)
                and t_extent <= float(args.layer_like_max_time_extent_ms)
            )
            if layer_like:
                rejected_layer_like += 1
                continue
            local_time = samples[gtt]
            is_shasan = (
                (surfaces["ShasanPresent"][gyy, gxx] > 0)
                & (local_time >= surfaces["T4_TIME"][gyy, gxx])
                & (local_time < surfaces["T6_TIME"][gyy, gxx])
            )
            is_shasi = (
                (surfaces["ShasiPresent"][gyy, gxx] > 0)
                & (local_time >= surfaces["T6_TIME"][gyy, gxx])
                & (local_time <= surfaces["T7_TIME"][gyy, gxx])
            )
            top = np.where(is_shasan, surfaces["T4_TIME"][gyy, gxx], surfaces["T6_TIME"][gyy, gxx])
            base = np.where(is_shasan, surfaces["T6_TIME"][gyy, gxx], surfaces["T7_TIME"][gyy, gxx])
            relative = (local_time - top) / np.maximum(base - top, 1.0e-6)
            relative = relative[is_shasan | is_shasi]
            relative_std = float(np.std(relative)) if relative.size else float("nan")
            relative_span = float(np.max(relative) - np.min(relative)) if relative.size else float("nan")
            strat_following = bool(
                horizontal_extent >= float(args.strat_following_min_horizontal_extent_m)
                and relative.size > 0
                and relative_std <= float(args.strat_following_max_relative_std)
                and relative_span <= float(args.strat_following_max_relative_span)
            )
            if strat_following:
                rejected_strat_following += 1
                continue
            seed_fraction = float(np.mean((branch_values & 1) > 0))
            growth_fraction = float(np.mean((branch_values & 2) > 0))
            dominant_layer = "上部复合层" if int(is_shasan.sum()) >= int(is_shasi.sum()) else "太古界风化壳"
            component_id_grid[gyy, gxx, gtt] = next_id
            rows.append(
                {
                    "component_id": next_id,
                    "raw_component_id": int(raw_id),
                    "voxel_count": voxel_count,
                    "score_mean": score_mean,
                    "score_max": float(np.max(score[gyy, gxx, gtt])),
                    "required_score_mean": required_score_mean,
                    "x_min": float(x_values[gxx].min()),
                    "x_max": float(x_values[gxx].max()),
                    "y_min": float(y_values[gyy].min()),
                    "y_max": float(y_values[gyy].max()),
                    "time_min_ms": float(samples[gtt].min()),
                    "time_max_ms": float(samples[gtt].max()),
                    "x_extent_m": x_extent,
                    "y_extent_m": y_extent,
                    "time_extent_ms": t_extent,
                    "pca_azimuth_deg": azimuth,
                    "pca_dip_deg": dip,
                    "pca_linearity": linearity,
                    "layer_like": layer_like,
                    "dominant_layer": dominant_layer,
                    "relative_position_std": relative_std,
                    "relative_position_span": relative_span,
                    "anttrack_score_mean": float(np.mean(ant_score[gyy, gxx, gtt])),
                    "lowcoh_score_mean": float(np.mean(lowcoh_score[gyy, gxx, gtt])),
                    "curvature_score_mean": float(np.mean(curvature_score[gyy, gxx, gtt])),
                    "anttrack_seed_fraction": seed_fraction,
                    "weighted_growth_fraction": growth_fraction,
                    "dominant_candidate_branch": "anttrack_seeded_weighted_growth",
                }
            )
            next_id += 1

    summary = {
        "raw_component_count": int(raw_count),
        "kept_component_count": int(next_id - 1),
        "split_raw_component_count": int(split_raw_count),
        "rejected_small_group_count": int(rejected_small),
        "rejected_horizontal_group_count": int(rejected_horizontal),
        "rejected_thin_group_count": int(rejected_thin),
        "rejected_layer_like_group_count": int(rejected_layer_like),
        "rejected_low_dip_group_count": int(rejected_low_dip),
        "rejected_low_linearity_group_count": int(rejected_low_linearity),
        "rejected_low_score_group_count": int(rejected_low_score),
        "rejected_strat_following_group_count": int(rejected_strat_following),
        "max_component_voxels_before_split": int(args.max_component_voxels_before_split),
        "split_tile_cells": int(args.split_tile_cells),
        "split_time_samples": int(args.split_time_samples),
        "orientation_time_scale_m_per_ms": time_scale,
    }
    return component_id_grid, pd.DataFrame(rows), summary


def write_component_vtk(
    path: Path,
    component_id_grid: np.ndarray,
    score_grid: np.ndarray,
    mapping: dict[str, np.ndarray],
    samples: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    yy, xx, tt = np.where(component_id_grid > 0)
    total = int(len(yy))
    if total == 0:
        pv.PolyData().save(path)
        return {"point_count": 0, "written_point_count": 0, "subsampled": False}
    if total > max_points:
        idx = rng.choice(np.arange(total), size=max_points, replace=False)
        yy, xx, tt = yy[idx], xx[idx], tt[idx]
        subsampled = True
    else:
        subsampled = False
    x_values, y_values = grid_axis_values(mapping)
    points = np.column_stack([x_values[xx], y_values[yy], samples[tt]]).astype(np.float32)
    cloud = pv.PolyData(points)
    cloud["ComponentID"] = component_id_grid[yy, xx, tt].astype(np.int32)
    cloud["MediumPrior"] = score_grid[yy, xx, tt].astype(np.float32)
    cloud.save(path)
    return {"point_count": total, "written_point_count": int(len(points)), "subsampled": subsampled}


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    # The formal JSON is the source of truth for medium-scale thresholds.
    # Keep CLI defaults for backward compatibility, then apply configured
    # values so reruns cannot silently use stale parser defaults.
    medium_cfg = dict(config.get("medium_prior", {}))
    for name in (
        "min_component_voxels",
        "fragment_recovery_min_voxels",
        "fragment_bridge_score",
        "fragment_bridge_iterations",
        "fragment_recovery_min_anttrack_mean",
        "fragment_bridge_min_anttrack_score",
        "seed_medium_score_threshold",
        "growth_anttrack_floor",
        "growth_medium_score_threshold",
        "min_component_score_mean",
        "min_component_dip_deg",
        "min_component_linearity",
    ):
        if name in medium_cfg:
            setattr(args, name, medium_cfg[name])
    for key, attr in (
        ("ant_weight_base", "anttrack_weight"),
        ("lowcoh_support_weight", "lowcoh_weight"),
        ("curvature_support_weight", "curvature_weight"),
    ):
        if key in medium_cfg:
            setattr(args, attr, medium_cfg[key])
    output_dir = (args.output_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))).resolve()
    ensure_dir(output_dir)
    rng = np.random.default_rng(int(args.random_state))

    input_density_sgy = Path(config["input_density_sgy"]).resolve()
    step6a_summary: dict[str, Any] | None = None
    if config.get("step6a_prediction_summary_json"):
        step6a_summary_path = Path(config["step6a_prediction_summary_json"]).resolve()
        step6a_summary = read_json(step6a_summary_path)
        if step6a_summary.get("status") != "pass":
            raise RuntimeError("configured Step6A prediction summary is not pass")
        declared_density = Path(step6a_summary["output_paths"]["density_sgy"]).resolve()
        if declared_density != input_density_sgy:
            raise RuntimeError(f"Step6B density input mismatch: {input_density_sgy} != {declared_density}")
        expected_contract = config.get("expected_step6a_model_contract_version")
        if expected_contract and step6a_summary.get("model_contract_version") != expected_contract:
            raise RuntimeError(
                f"Step6A contract mismatch: {step6a_summary.get('model_contract_version')} != {expected_contract}"
            )
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    volume_paths = {key: Path(value).resolve() for key, value in dict(config["volume_paths"]).items()}
    mapping = load_mapping(trace_mapping_npz)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)
    source_samples, density_load = read_sgy_sample_axis(input_density_sgy)
    attribute_mapping_qc = validate_attribute_mapping_contract(
        mapping,
        Path(config["attribute_trace_header_csv"]).resolve(),
        Path(config["step6a_trace_mapping_npz"]).resolve(),
        int(density_load["trace_count"]),
        float(config.get("attribute_grid_xy_tolerance_m", 2.0)),
    )
    if args.max_x_lines > 0:
        keep = np.asarray(mapping["ix"], dtype=np.int32) < int(args.max_x_lines)
        if not keep.any():
            raise ValueError("--max-x-lines selected no attribute-grid traces")
        full_count = len(keep)
        mapping = {
            key: np.asarray(value)[keep] if np.asarray(value).ndim == 1 and len(np.asarray(value)) == full_count else value
            for key, value in mapping.items()
        }
        source_trace_idx = np.asarray(mapping["source_trace_idx"], dtype=np.int64)
    interval_ms = float(dict(config.get("medium_evidence", {})).get("sample_interval_ms", 10.0))
    samples = regular_sample_axis(source_samples, interval_ms)
    # Step6A 预测 SGY 已经是 demo 属性网格顺序，不是原始属性体 TraceIdx 顺序；
    # 因此直接按输出道序读取，不能再用 source_trace_idx 二次索引。
    density_output_indices = np.asarray(mapping["output_trace_index"], dtype=np.int64)
    background_density, density_samples, background_load = load_trace_matrix(
        input_density_sgy, density_output_indices, samples, "Step6A density"
    )
    if len(density_samples) != len(samples) or not np.allclose(density_samples, samples, atol=1.0e-6):
        background_density = _resample_matrix(background_density, density_samples, samples)
        background_load["target_sample_count"] = int(len(samples))
        background_load["target_sample_interval_ms"] = float(interval_ms)
    density_load["target_sample_count"] = int(len(samples))
    density_load["target_sample_interval_ms"] = interval_ms
    horizon_contract = load_taigu_horizon_contract(config, mapping)
    try:
        horizon_axis_qc = validate_window_contract(config, horizon_contract, samples)
    except (KeyError, ValueError) as exc:
        # 太古界当前只固化了 2 ms 窗口；10 ms 属性处理直接使用逐道合同插值，
        # 缺少专用窗口文件不应把中尺度流程整体判为无效。
        horizon_axis_qc = {"status": "contract_direct_interpolation", "warning": str(exc)}

    print("[step6b-medium] loading seismic attributes on absolute TWT axis", flush=True)
    time_origins = {key: float(value) for key, value in dict(config["attribute_time_origins_ms"]).items()}
    for attribute in ("Coherence", "AntTrack", "CurvatureMax"):
        if attribute not in time_origins:
            raise ValueError(f"attribute_time_origins_ms missing {attribute}")
    coherence, coh_load = load_attribute_matrix(
        volume_paths["Coherence"], source_trace_idx, samples, "Coherence", time_origins["Coherence"]
    )
    anttrack, ant_load = load_attribute_matrix(
        volume_paths["AntTrack"], source_trace_idx, samples, "AntTrack", time_origins["AntTrack"]
    )
    curvmax, curvmax_load = load_attribute_matrix(
        volume_paths["CurvatureMax"], source_trace_idx, samples, "CurvatureMax", time_origins["CurvatureMax"]
    )
    if "CurvaturePos" in volume_paths:
        if "CurvaturePos" not in time_origins:
            raise ValueError("attribute_time_origins_ms missing CurvaturePos")
        curvpos, curvpos_load = load_attribute_matrix(
            volume_paths["CurvaturePos"], source_trace_idx, samples, "CurvaturePos", time_origins["CurvaturePos"]
        )
    else:
        curvpos = None
        curvpos_load = {"status": "not_configured"}

    score_cfg = dict(config.get("score_config", {}))
    normalization_contract_path = Path(config["attribute_normalization_contract_json"]).resolve()
    normalization_contract = read_json(normalization_contract_path)
    expected_normalization_version = config.get("expected_attribute_normalization_contract_version")
    if expected_normalization_version and normalization_contract.get("version") != expected_normalization_version:
        raise RuntimeError(
            f"attribute normalization contract mismatch: {normalization_contract.get('version')} "
            f"!= {expected_normalization_version}"
        )
    ant_cfg = dict(score_cfg.get("anttrack_score", {"low_quantile": 0.20, "high_quantile": 0.96}))
    coh_cfg = dict(score_cfg.get("coherence_score", {"valid_min": 0.0, "low_quantile": 0.05, "high_quantile": 0.95}))
    curvmax_cfg = dict(score_cfg.get("curvaturemax_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    curvpos_cfg = dict(score_cfg.get("curvaturepos_score", {"low_quantile": 0.50, "high_quantile": 0.98}))
    ant_valid = valid_values(anttrack, ant_cfg)
    coh_valid = valid_values(coherence, coh_cfg)
    curvmax_valid = valid_values(curvmax, curvmax_cfg)
    curvpos_valid = valid_values(curvpos, curvpos_cfg) if curvpos is not None else None
    horizon_mask_qc = apply_validity_inplace(ant_valid, horizon_contract, samples)
    apply_validity_inplace(coh_valid, horizon_contract, samples)
    apply_validity_inplace(curvmax_valid, horizon_contract, samples)
    if curvpos_valid is not None:
        apply_validity_inplace(curvpos_valid, horizon_contract, samples)
    ant_minus_one_mask = np.isfinite(anttrack) & (anttrack <= -0.999)
    ant_minus_one_inside_count = int(np.sum(ant_minus_one_mask & ant_valid))
    # 太古界属性局部缺失较多：候选有效性不再要求三属性同时存在；至少蚂蚁体、
    # 相干体或曲率体之一有效即可，综合评分按有效属性重新归一化。
    valid = ant_valid | coh_valid | curvmax_valid | (curvpos_valid if curvpos_valid is not None else False)

    ant_score, ant_summary = score_by_semantic_layer(
        anttrack, ant_valid, "AntTrack", normalization_contract, horizon_contract, samples
    )
    lowcoh_score, lowcoh_summary = score_by_semantic_layer(
        coherence, coh_valid, "Coherence", normalization_contract, horizon_contract, samples
    )
    curvmax_score, curvmax_summary = score_by_semantic_layer(
        curvmax, curvmax_valid, "CurvatureMax", normalization_contract, horizon_contract, samples
    )
    if curvpos is not None and curvpos_valid is not None:
        curvpos_score, curvpos_summary = high_score(curvpos, curvpos_valid, curvpos_cfg)
        curv_score = np.maximum(curvmax_score, curvpos_score).astype(np.float32)
    else:
        curvpos_summary = {"status": "not_configured"}
        curv_score = curvmax_score.astype(np.float32)

    ant_grid, _, _ = flat_to_grid(ant_score, mapping)
    lowcoh_grid, _, _ = flat_to_grid(lowcoh_score, mapping)
    curv_grid, _, _ = flat_to_grid(curv_score, mapping)
    local_support_grid, local_support_fraction_grid = build_local_support_grids(
        lowcoh_grid,
        curv_grid,
        radius=int(args.support_neighborhood_cells),
        support_score_threshold=float(args.support_score_threshold),
    )
    local_support = grid_to_flat(local_support_grid, mapping)
    local_support_fraction = grid_to_flat(local_support_fraction_grid, mapping)
    if abs(float(args.density_weight)) > 1.0e-12:
        raise ValueError("Step6A density must not participate in Step6B medium-scale scoring")
    weights = np.asarray(
        [float(args.anttrack_weight), float(args.lowcoh_weight), float(args.curvature_weight)],
        dtype=np.float64,
    )
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("Step6B evidence weights must be non-negative and have a positive sum")
    weights /= float(weights.sum())
    # 有效属性逐体素重归一化，避免单个缺失属性把整条候选带判为无效。
    score_terms = np.stack([ant_score, lowcoh_score, curv_score], axis=0)
    valid_terms = np.stack([ant_valid, coh_valid, curvmax_valid | (curvpos_valid if curvpos_valid is not None else False)], axis=0)
    weight_grid = weights[:, None, None]
    medium_score = np.sum(score_terms * (weight_grid * valid_terms), axis=0)
    medium_score /= np.maximum(np.sum(weight_grid * valid_terms, axis=0), 1.0e-6)
    medium_score[~valid] = 0.0
    medium_score = np.clip(medium_score, 0.0, 1.0).astype(np.float32)
    ant_threshold = quantile(ant_score[ant_score > 0], float(args.anttrack_high_quantile))
    seed_flat = (
        (ant_score >= ant_threshold)
        & (medium_score >= float(args.seed_medium_score_threshold))
        & valid
    )
    growth_flat = (
        (ant_score >= float(args.growth_anttrack_floor))
        & (medium_score >= float(args.growth_medium_score_threshold))
        & valid
    )
    seed_grid, _, _ = flat_to_grid(seed_flat.astype(np.float32), mapping)
    growth_grid, _, _ = flat_to_grid(growth_flat.astype(np.float32), mapping)
    retained_growth_grid, seeded_growth_summary = retain_seeded_growth(seed_grid > 0.5, growth_grid > 0.5)
    raw_mask_flat = grid_to_flat(retained_growth_grid.astype(np.float32), mapping) > 0.5
    branch_code_flat = np.zeros_like(raw_mask_flat, dtype=np.uint8)
    branch_code_flat[seed_flat & raw_mask_flat] |= 1
    branch_code_flat[raw_mask_flat] |= 2

    medium_grid, _, _ = flat_to_grid(medium_score, mapping)
    surfaces = surface_grids_from_contract(mapping, horizon_contract)
    branch_code_grid, _, _ = flat_to_grid(branch_code_flat.astype(np.float32), mapping)
    support_grid, _, _ = flat_to_grid(local_support.astype(np.float32), mapping)
    mask_grid, _, _ = flat_to_grid(raw_mask_flat.astype(np.float32), mapping)
    mask_grid = mask_grid > 0.5
    valid_grid, _, _ = flat_to_grid(valid.astype(np.float32), mapping)
    mask_grid, fragment_repair_summary = repair_fragmented_candidates(
        mask_grid,
        medium_grid,
        ant_grid,
        valid_grid > 0.5,
        min_component_voxels=int(args.min_component_voxels),
        recovery_min_voxels=int(args.fragment_recovery_min_voxels),
        bridge_score=float(args.fragment_bridge_score),
        bridge_iterations=int(args.fragment_bridge_iterations),
        recovery_min_anttrack_mean=float(args.fragment_recovery_min_anttrack_mean),
        bridge_min_anttrack_score=float(args.fragment_bridge_min_anttrack_score),
    )
    raw_mask_flat = grid_to_flat(mask_grid.astype(np.float32), mapping) > 0.5
    branch_code_flat[raw_mask_flat] |= 2
    branch_code_grid, _, _ = flat_to_grid(branch_code_flat.astype(np.float32), mapping)
    component_id_grid, component_df, component_summary = build_medium_components(
        mask_grid,
        medium_grid,
        ant_grid,
        lowcoh_grid,
        curv_grid,
        branch_code_grid,
        surfaces,
        mapping,
        samples,
        args,
    )
    kept_mask_grid = component_id_grid > 0
    kept_mask_flat = grid_to_flat(kept_mask_grid.astype(np.float32), mapping)
    component_id_flat = grid_to_flat(component_id_grid.astype(np.float32), mapping).astype(np.int32)
    medium_score_filtered = medium_score.copy()
    medium_score_filtered[kept_mask_flat <= 0.0] = 0.0

    x_pair_count = int(np.sum(kept_mask_grid[:, :-1, :] & kept_mask_grid[:, 1:, :]))
    y_pair_count = int(np.sum(kept_mask_grid[:-1, :, :] & kept_mask_grid[1:, :, :]))
    x_possible = max(int(np.sum(kept_mask_grid[:, :-1, :])), 1)
    y_possible = max(int(np.sum(kept_mask_grid[:-1, :, :])), 1)
    x_continuity = float(x_pair_count / x_possible)
    y_continuity = float(y_pair_count / y_possible)
    continuity_ratio = float(max(x_continuity, y_continuity) / max(min(x_continuity, y_continuity), 1.0e-9))

    write_sgy_like(output_dir / "medium_corridor_prior.sgy", input_density_sgy, medium_score_filtered, samples)
    component_df.to_csv(output_dir / "medium_corridor_component_summary.csv", index=False, encoding="utf-8-sig")
    if bool(config.get("write_intermediate_vtk", False)):
        vtk_summary = write_component_vtk(
            output_dir / "medium_corridor_components_raw_time.vtk",
            component_id_grid,
            medium_grid,
            mapping,
            samples,
            max_points=int(args.vtk_max_points),
            rng=rng,
        )
    else:
        vtk_summary = {"status": "disabled_by_config"}
    np.savez_compressed(
        output_dir / "medium_corridor_components.npz",
        medium_prior=medium_score_filtered.astype(np.float32),
        medium_score=medium_score.astype(np.float32),
        anttrack_score=ant_score.astype(np.float32),
        lowcoh_score=lowcoh_score.astype(np.float32),
        curvature_score=curv_score.astype(np.float32),
        medium_mask=kept_mask_flat.astype(np.uint8),
        medium_component_id=component_id_flat.astype(np.int32),
        candidate_branch_code=branch_code_flat.astype(np.uint8),
        samples=samples.astype(np.float32),
        x=mapping["x"].astype(np.float64),
        y=mapping["y"].astype(np.float64),
        ix=mapping["ix"].astype(np.int32),
        iy=mapping["iy"].astype(np.int32),
        source_trace_idx=horizon_contract.trace_idx.astype(np.int32),
        t4_time=horizon_contract.t4.astype(np.float32),
        t6_time=horizon_contract.t6.astype(np.float32),
        t7_time=horizon_contract.t7.astype(np.float32),
        shasan_present=horizon_contract.shasan_present.astype(np.uint8),
        shasi_present=horizon_contract.shasi_present.astype(np.uint8),
    )

    max_component_fraction = (
        float(component_df["voxel_count"].max() / max(int(kept_mask_grid.sum()), 1)) if len(component_df) else 0.0
    )
    summary = {
        "status": "pass",
        "version": config.get("version"),
        "input_density_sgy": str(input_density_sgy),
        "step6a_input_contract": {
            "summary_json": str(Path(config["step6a_prediction_summary_json"]).resolve())
            if config.get("step6a_prediction_summary_json") else None,
            "model_contract_version": step6a_summary.get("model_contract_version") if step6a_summary else None,
            "attribute_normalization_contract_version": step6a_summary.get("attribute_normalization_contract_version")
            if step6a_summary else None,
        },
        "output_dir": str(output_dir),
        "attribute_mapping_contract": attribute_mapping_qc,
        "attribute_normalization_contract": {
            "path": str(normalization_contract_path),
            "version": normalization_contract.get("version"),
            "mode": "saved Common limits applied separately in upper-composite and weathered-crust windows",
            "full_normalized_volumes_saved": False,
        },
        "sample_interval_ms": interval_ms,
        "sample_count": int(len(samples)),
        "execution_grid": {
            "trace_count": int(len(source_trace_idx)),
            "x_line_count": int(np.asarray(mapping["ix"]).max() + 1),
            "y_line_count": int(np.asarray(mapping["iy"]).max() + 1),
            "max_x_lines_smoke_cap": int(args.max_x_lines),
        },
        "horizon_contract": contract_summary(horizon_contract),
        "horizon_axis_qc": horizon_axis_qc,
        "horizon_mask_qc": horizon_mask_qc,
        "output_contract": {
            "medium_prior_sgy": str(output_dir / "medium_corridor_prior.sgy"),
            "component_npz": str(output_dir / "medium_corridor_components.npz"),
            "mask_storage": "medium_mask inside component_npz",
        },
        "load": {
            "density": density_load,
            "coherence": coh_load,
            "anttrack": ant_load,
            "curvaturemax": curvmax_load,
            "curvaturepos": curvpos_load,
        },
        "ant_score_threshold": ant_threshold,
        "seed_medium_score_threshold": float(args.seed_medium_score_threshold),
        "growth_anttrack_floor": float(args.growth_anttrack_floor),
        "growth_medium_score_threshold": float(args.growth_medium_score_threshold),
        "seeded_growth": seeded_growth_summary,
        "fragment_repair": fragment_repair_summary,
        "score_formula": {
            "formula": "normalized weighted sum with AntTrack dominant",
            "expression": "MediumScore = 0.70*AntTrackScore + 0.20*LowCoherenceScore + 0.10*CurvatureScore (renormalized over available attributes)",
            "anttrack_weight": float(weights[0]),
            "lowcoh_weight": float(weights[1]),
            "curvature_weight": float(weights[2]),
            "density_weight": 0.0,
            "step6a_density_role": "grid, sample axis, and SGY output template only; excluded from medium-scale evidence",
            "candidate_logic": "strong AntTrack seeds plus lower-threshold weighted growth; retain only growth components containing seeds",
            "anttrack_polarity": "high normalized AntTrack response is fracture evidence",
            "anttrack_minus_one_semantics": "valid weak/no-fracture response; included in normalization, not missing",
            "anttrack_valid_min_inclusive": float(ant_cfg.get("valid_min", -1.0e30)),
            "anttrack_minus_one_inside_valid_window_count": ant_minus_one_inside_count,
            "support_score_threshold": float(args.support_score_threshold),
            "support_neighborhood_cells": int(args.support_neighborhood_cells),
            "deprecated_rescue_flags_ignored": bool(
                args.enable_supported_rescue_branch or args.enable_lowcoh_structure_rescue
            ),
        },
        "effective_component_filters": {
            "min_component_voxels": int(args.min_component_voxels),
            "min_component_dip_deg": float(args.min_component_dip_deg),
            "min_component_linearity": float(args.min_component_linearity),
            "min_component_score_mean": float(args.min_component_score_mean),
            "fragment_recovery_min_voxels": int(args.fragment_recovery_min_voxels),
            "fragment_recovery_min_anttrack_mean": float(args.fragment_recovery_min_anttrack_mean),
            "fragment_bridge_min_anttrack_score": float(args.fragment_bridge_min_anttrack_score),
        },
        "spatial_continuity_qc": {
            "x_neighbor_continuity": x_continuity,
            "y_neighbor_continuity": y_continuity,
            "directional_ratio": continuity_ratio,
            "maximum_allowed_directional_ratio": float(config.get("max_directional_continuity_ratio", 3.0)),
            "minimum_required_neighbor_continuity": float(config.get("min_neighbor_continuity", 0.10)),
            "not_single_trace_fragmented": min(x_continuity, y_continuity) >= float(config.get("min_neighbor_continuity", 0.10)),
            "not_directionally_overmerged": continuity_ratio <= float(config.get("max_directional_continuity_ratio", 3.0)),
        },
        "raw_candidate_voxel_count": int(raw_mask_flat.sum()),
        "raw_candidate_voxel_fraction": float(raw_mask_flat.mean()),
        "raw_candidate_branch_counts": {
            "anttrack_seed": int(np.sum((branch_code_flat & 1) > 0)),
            "weighted_growth": int(np.sum((branch_code_flat & 2) > 0)),
            "growth_only": int(np.sum(branch_code_flat == 2)),
        },
        "kept_candidate_voxel_count": int(kept_mask_grid.sum()),
        "kept_candidate_voxel_fraction": float(kept_mask_grid.mean()),
        "kept_candidate_branch_counts": {
            "anttrack_seed": int(np.sum((component_id_grid > 0) & ((branch_code_grid.astype(np.uint8) & 1) > 0))),
            "weighted_growth": int(np.sum((component_id_grid > 0) & ((branch_code_grid.astype(np.uint8) & 2) > 0))),
            "growth_only": int(np.sum((component_id_grid > 0) & (branch_code_grid == 2))),
        },
        "max_component_fraction": max_component_fraction,
        "component_summary": component_summary,
        "component_count": int(len(component_df)),
        "component_voxel_stats": finite_stats(component_df["voxel_count"]) if len(component_df) else finite_stats([]),
        "component_dip_stats": finite_stats(component_df["pca_dip_deg"]) if len(component_df) else finite_stats([]),
        "component_time_extent_stats": finite_stats(component_df["time_extent_ms"]) if len(component_df) else finite_stats([]),
        "component_attribute_contributions": {
            "anttrack_score_mean": finite_stats(component_df["anttrack_score_mean"]) if len(component_df) else finite_stats([]),
            "lowcoh_score_mean": finite_stats(component_df["lowcoh_score_mean"]) if len(component_df) else finite_stats([]),
            "curvature_score_mean": finite_stats(component_df["curvature_score_mean"]) if len(component_df) else finite_stats([]),
            "anttrack_seed_fraction": finite_stats(component_df["anttrack_seed_fraction"]) if len(component_df) else finite_stats([]),
            "weighted_growth_fraction": finite_stats(component_df["weighted_growth_fraction"]) if len(component_df) else finite_stats([]),
        },
        "medium_prior_stats": finite_stats(medium_score_filtered[medium_score_filtered > 0]),
        "local_support_stats_in_raw_candidates": finite_stats(local_support[raw_mask_flat]),
        "local_support_fraction_stats_in_raw_candidates": finite_stats(local_support_fraction[raw_mask_flat]),
        "anttrack_score_stats_in_raw_candidates": finite_stats(ant_score[raw_mask_flat]),
        "local_support_stats_in_kept_candidates": finite_stats(support_grid[kept_mask_grid]),
        "local_support_fraction_stats_in_kept_candidates": finite_stats(local_support_fraction_grid[kept_mask_grid]),
        "attribute_score_summaries": {
            "anttrack": ant_summary,
            "low_coherence": lowcoh_summary,
            "curvaturemax": curvmax_summary,
            "curvaturepos": curvpos_summary,
            "step6a_density_grid_contract": background_load,
        },
        "vtk": vtk_summary,
        "reflection": (
            "Step6B uses AntTrack-dominant additive evidence. Strong AntTrack voxels seed lower-threshold weighted growth, "
            "and coherence or curvature cannot veto a strong AntTrack response. Components still reject low-dip and "
            "stratigraphically conformable anomalies."
        ),
    }
    if max_component_fraction > 0.40:
        summary["status"] = "warn"
        summary["warning"] = "largest medium component still exceeds 40% of kept voxels; Step7B must handle local continuity carefully"
    if not summary["spatial_continuity_qc"]["not_single_trace_fragmented"] or not summary["spatial_continuity_qc"]["not_directionally_overmerged"]:
        summary["status"] = "warn"
        summary["warning"] = "medium components failed spatial continuity QC"
    write_json(output_dir / "medium_corridor_qc.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

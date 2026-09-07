from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import segyio
from scipy import ndimage


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_3d_density_sgy.json"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "output/candidate_cheye1_multiscale_v1/qc_current"
NULL_ABS_LIMIT = 1.0e6
ATTRIBUTE_COLUMNS = ["Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC current Step6/Step6B density volumes against seismic attributes and well-point labels."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Existing Step6B density SGY config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="QC output directory.")
    parser.add_argument(
        "--density-sgy",
        action="append",
        default=[],
        help="Optional density volume as label=path. Can be repeated. Defaults to known candidate_cheye1 outputs.",
    )
    parser.add_argument("--chunksize", type=int, default=250000, help="Unified sample CSV chunksize.")
    parser.add_argument(
        "--max-well-correlation-rows",
        type=int,
        default=1000000,
        help="Maximum rows retained for sampled well-point correlation.",
    )
    parser.add_argument("--correlation-sample-size", type=int, default=1000000, help="Voxel sample size for correlations.")
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


def quantiles(values: np.ndarray, qs: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(q * 1000) / 10:g}": None for q in qs}
    return {f"q{int(q * 1000) / 10:g}": float(np.quantile(arr, q)) for q in qs}


def valid_values(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > -NULL_ABS_LIMIT) & (values < NULL_ABS_LIMIT)


def load_trace_mapping(path: Path) -> dict[str, np.ndarray]:
    mapping = np.load(path)
    required = {"output_trace_index", "source_trace_idx", "x", "y", "ix", "iy"}
    missing = sorted(required.difference(mapping.files))
    if missing:
        raise ValueError(f"trace mapping missing keys: {missing}")
    return {key: mapping[key] for key in mapping.files}


def load_sgy_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        samples = np.asarray(handle.samples, dtype=np.float64)
        tracecount = int(handle.tracecount)
        matrix = np.empty((tracecount, len(samples)), dtype=np.float32)
        for trace_idx in range(tracecount):
            matrix[trace_idx, :] = np.asarray(handle.trace[trace_idx], dtype=np.float32)
            if (trace_idx + 1) % 20000 == 0:
                print(f"[step6-qc] loaded {path.name} traces={trace_idx + 1}/{tracecount}", flush=True)
    matrix[np.abs(matrix) >= NULL_ABS_LIMIT] = np.nan
    return matrix, samples, {"path": str(path), "trace_count": tracecount, "sample_count": int(len(samples))}


def load_attribute_matrix(
    path: Path,
    source_trace_idx: np.ndarray,
    target_samples: np.ndarray,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.empty((len(source_trace_idx), len(target_samples)), dtype=np.float32)
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        source_samples = np.asarray(handle.samples, dtype=np.float64)
        same_samples = len(source_samples) == len(target_samples) and np.allclose(
            source_samples, target_samples, rtol=0.0, atol=1.0e-6
        )
        max_trace_idx = int(np.max(source_trace_idx)) if len(source_trace_idx) else -1
        if max_trace_idx >= int(handle.tracecount):
            raise ValueError(f"{label} tracecount {handle.tracecount} < mapping max trace {max_trace_idx}")
        for out_idx, src_idx in enumerate(source_trace_idx):
            trace = np.asarray(handle.trace[int(src_idx)], dtype=np.float32)
            trace[np.abs(trace) >= NULL_ABS_LIMIT] = np.nan
            if same_samples:
                matrix[out_idx, :] = trace
            else:
                matrix[out_idx, :] = np.interp(target_samples, source_samples, trace, left=np.nan, right=np.nan)
            if (out_idx + 1) % 20000 == 0:
                print(f"[step6-qc] loaded {label} traces={out_idx + 1}/{len(source_trace_idx)}", flush=True)
    return matrix, {
        "path": str(path),
        "sample_axis_matched": bool(same_samples),
        "source_sample_count": int(len(source_samples)),
        "target_sample_count": int(len(target_samples)),
        "trace_count_loaded": int(len(source_trace_idx)),
    }


def sampled_correlations(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    idx = np.flatnonzero(valid.ravel())
    if idx.size < 3:
        return {"count": int(idx.size), "pearson": None, "spearman": None, "sampled": False}
    sampled = False
    if sample_size > 0 and idx.size > sample_size:
        idx = rng.choice(idx, size=sample_size, replace=False)
        sampled = True
    xv = x.ravel()[idx].astype(np.float64)
    yv = y.ravel()[idx].astype(np.float64)
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[finite]
    yv = yv[finite]
    if xv.size < 3 or np.nanstd(xv) <= 0.0 or np.nanstd(yv) <= 0.0:
        return {"count": int(xv.size), "pearson": None, "spearman": None, "sampled": sampled}
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    xr = pd.Series(xv).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(yv).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"count": int(xv.size), "pearson": pearson, "spearman": spearman, "sampled": sampled}


def build_attribute_masks(attributes: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    valid_by_attr = {name: valid_values(values) for name, values in attributes.items()}
    common_valid = np.ones_like(next(iter(attributes.values())), dtype=bool)
    for valid in valid_by_attr.values():
        common_valid &= valid

    thresholds: dict[str, Any] = {}
    masks: dict[str, np.ndarray] = {}
    for name, values in attributes.items():
        valid = valid_by_attr[name]
        finite = values[valid]
        thresholds[name] = {
            "stats": finite_stats(finite),
            "quantiles": quantiles(finite, [0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95]),
        }
        if finite.size == 0:
            continue
        q10 = float(np.quantile(finite, 0.10))
        q20 = float(np.quantile(finite, 0.20))
        q80 = float(np.quantile(finite, 0.80))
        q90 = float(np.quantile(finite, 0.90))
        if name == "Coherence":
            masks["coherence_le_q10"] = valid & (values <= q10)
            masks["coherence_le_q20"] = valid & (values <= q20)
        else:
            key = name.lower()
            masks[f"{key}_ge_q80"] = valid & (values >= q80)
            masks[f"{key}_ge_q90"] = valid & (values >= q90)
    return masks, thresholds, common_valid


def top_density_overlap(
    density: np.ndarray,
    masks: dict[str, np.ndarray],
    valid: np.ndarray,
    top_quantiles: list[float],
) -> dict[str, Any]:
    finite_positive = density[valid & np.isfinite(density) & (density > 0.0)]
    if finite_positive.size == 0:
        return {"positive_density_count": 0, "top_quantiles": {}}
    out: dict[str, Any] = {
        "positive_density_count": int(finite_positive.size),
        "density_stats_positive": finite_stats(finite_positive),
        "density_quantiles_positive": quantiles(finite_positive, top_quantiles),
        "top_quantiles": {},
    }
    baseline = {name: float(mask[valid].sum() / valid.sum()) if valid.sum() else None for name, mask in masks.items()}
    for q in top_quantiles:
        threshold = float(np.quantile(finite_positive, q))
        top = valid & np.isfinite(density) & (density > 0.0) & (density >= threshold)
        top_count = int(top.sum())
        item: dict[str, Any] = {"threshold": threshold, "top_count": top_count, "overlap": {}}
        for name, mask in masks.items():
            fraction = float((top & mask).sum() / top_count) if top_count else None
            base = baseline.get(name)
            item["overlap"][name] = {
                "fraction": fraction,
                "baseline_fraction": base,
                "enrichment": float(fraction / base) if fraction is not None and base not in (None, 0.0) else None,
            }
        out["top_quantiles"][f"top_{(1.0 - q) * 100:g}pct"] = item
    return out


def density_attribute_correlations(
    density: np.ndarray,
    attributes: dict[str, np.ndarray],
    valid: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, values in attributes.items():
        pair_valid = valid & np.isfinite(density) & np.isfinite(values)
        if name == "Coherence":
            low_score = np.full_like(values, np.nan, dtype=np.float32)
            finite = values[pair_valid]
            if finite.size:
                low = float(np.quantile(finite, 0.05))
                high = float(np.quantile(finite, 0.95))
                if high <= low:
                    high = low + 1.0e-6
                low_score[pair_valid] = np.clip((high - values[pair_valid]) / (high - low), 0.0, 1.0)
                out["low_coherence_score"] = sampled_correlations(
                    density, low_score, pair_valid, sample_size=sample_size, rng=rng
                )
        out[name] = sampled_correlations(density, values, pair_valid, sample_size=sample_size, rng=rng)
    return out


def lowcoh_component_qc(
    coherence: np.ndarray,
    mapping: dict[str, np.ndarray],
    quantile: float = 0.20,
    summary_limit: int = 30,
) -> dict[str, Any]:
    valid = valid_values(coherence)
    finite = coherence[valid]
    if finite.size == 0:
        return {"status": "no_valid_coherence"}
    threshold = float(np.quantile(finite, quantile))
    ix = mapping["ix"].astype(np.int32)
    iy = mapping["iy"].astype(np.int32)
    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    nt = int(coherence.shape[1])
    grid = np.zeros((ny, nx, nt), dtype=bool)
    grid[iy, ix, :] = valid & (coherence <= threshold)
    labels, component_count = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=np.uint8))
    objects = ndimage.find_objects(labels)
    component_rows: list[dict[str, Any]] = []
    horizontal_like = 0
    steep_like = 0
    voxel_counts = np.bincount(labels.ravel()) if component_count else np.asarray([], dtype=np.int64)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        y_slice, x_slice, t_slice = slc
        y_extent = int(y_slice.stop - y_slice.start)
        x_extent = int(x_slice.stop - x_slice.start)
        t_extent = int(t_slice.stop - t_slice.start)
        xy_extent = int(max(x_extent, y_extent))
        voxel_count = int(voxel_counts[label_id]) if label_id < len(voxel_counts) else 0
        is_horizontal = bool(t_extent <= 3 and xy_extent >= 40)
        is_steep = bool(t_extent >= 8 and xy_extent <= 80)
        horizontal_like += int(is_horizontal)
        steep_like += int(is_steep)
        component_rows.append(
            {
                "label": int(label_id),
                "voxel_count": voxel_count,
                "x_extent_cells": x_extent,
                "y_extent_cells": y_extent,
                "time_extent_samples": t_extent,
                "xy_extent_cells": xy_extent,
                "horizontal_like": is_horizontal,
                "steep_like": is_steep,
            }
        )
    component_rows.sort(key=lambda item: item["voxel_count"], reverse=True)
    largest = component_rows[:summary_limit]
    sizes = np.asarray([row["voxel_count"] for row in component_rows], dtype=np.float64)
    return {
        "status": "pass",
        "coherence_low_quantile": float(quantile),
        "coherence_threshold": threshold,
        "lowcoh_voxel_count": int(grid.sum()),
        "component_count": int(component_count),
        "component_size_stats": finite_stats(sizes),
        "horizontal_like_component_count": int(horizontal_like),
        "steep_like_component_count": int(steep_like),
        "largest_components": largest,
        "classification_note": "horizontal_like uses time_extent<=3 and xy_extent>=40; steep_like uses time_extent>=8 and xy_extent<=80 for QC only.",
    }


def parse_density_args(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--density-sgy must be label=path, got: {item}")
        label, path = item.split("=", 1)
        out[label.strip()] = Path(path).expanduser().resolve()
    return out


def default_density_sgys(config: dict[str, Any]) -> dict[str, Path]:
    base_output_dir = Path(config["output_dir"]).resolve()
    root = base_output_dir.parent
    candidate = str(config.get("target_block", {}).get("name", "candidate_cheye1"))
    candidates = {
        "original_step6b": base_output_dir / f"{candidate}_3d_predicted_density.sgy",
        "seismic_prior_lowcoh_v1": root
        / "candidate_cheye1_seismic_prior_lowcoh_v1"
        / "candidate_cheye1_3d_predicted_density_seismic_prior_lowcoh_v1.sgy",
        "lowcoh_steep": root
        / "candidate_cheye1_lowcoh_steep"
        / "candidate_cheye1_3d_predicted_density_lowcoh_steep.sgy",
        "lowcoh_add": root
        / "candidate_cheye1_lowcoh_add"
        / "candidate_cheye1_3d_predicted_density_lowcoh_add.sgy",
    }
    return {label: path for label, path in candidates.items() if path.exists()}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def corr_pair(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    if x_col not in df.columns or y_col not in df.columns:
        return {"count": 0, "pearson": None, "spearman": None}
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = safe_numeric(sub[x_col])
    sub[y_col] = safe_numeric(sub[y_col])
    sub = sub.dropna()
    if len(sub) < 3 or sub[x_col].std(ddof=0) <= 0.0 or sub[y_col].std(ddof=0) <= 0.0:
        return {"count": int(len(sub)), "pearson": None, "spearman": None}
    return {
        "count": int(len(sub)),
        "pearson": float(sub[x_col].corr(sub[y_col], method="pearson")),
        "spearman": float(sub[x_col].corr(sub[y_col], method="spearman")),
    }


def stream_well_point_qc(
    unified_samples_csv: Path,
    target_block: dict[str, Any],
    output_dir: Path,
    chunksize: int,
    max_rows: int,
    random_state: int,
) -> dict[str, Any]:
    usecols = [
        "SourceKind",
        "SourceWellName",
        "TrackWellName",
        "X",
        "Y",
        "TIME",
        "LayerGroup",
        "DensityLabel",
        "GT_POINT_FLAG",
        "HasFracture",
        "IsRefinedFracturePoint",
        "PointConfidence",
        "Coherence",
        "AntTrack",
        "CurvatureMax",
        "CurvaturePos",
    ]
    existing_usecols: list[str] | None = None
    sampled_parts: list[pd.DataFrame] = []
    counts: dict[str, Any] = {
        "rows_seen": 0,
        "rows_in_target_block": 0,
        "source_kind_counts": {},
        "layer_counts": {},
        "nominal_training_weight_sum_by_source_kind": {},
    }
    rng = np.random.default_rng(random_state)
    reader = pd.read_csv(
        unified_samples_csv,
        encoding="utf-8-sig",
        chunksize=chunksize,
        low_memory=False,
        usecols=lambda col: col in usecols,
    )
    for chunk_idx, chunk in enumerate(reader, start=1):
        if existing_usecols is None:
            existing_usecols = chunk.columns.tolist()
        counts["rows_seen"] += int(len(chunk))
        for column in ["X", "Y"]:
            chunk[column] = safe_numeric(chunk[column])
        mask = (
            chunk["X"].between(float(target_block["x_min"]), float(target_block["x_max"]))
            & chunk["Y"].between(float(target_block["y_min"]), float(target_block["y_max"]))
        )
        work = chunk.loc[mask].copy()
        if work.empty:
            if chunk_idx % 10 == 0:
                print(f"[step6-qc] scanned sample chunks={chunk_idx}", flush=True)
            continue
        counts["rows_in_target_block"] += int(len(work))
        for key, value in work["SourceKind"].astype(str).value_counts(dropna=False).items():
            counts["source_kind_counts"][str(key)] = counts["source_kind_counts"].get(str(key), 0) + int(value)
        for key, value in work["LayerGroup"].astype(str).value_counts(dropna=False).items():
            counts["layer_counts"][str(key)] = counts["layer_counts"].get(str(key), 0) + int(value)
        confidence = safe_numeric(work.get("PointConfidence", pd.Series(1.0, index=work.index))).fillna(1.0)
        source_kind = work["SourceKind"].astype(str)
        nominal_weight = confidence * np.where(source_kind.eq("real_well"), 3.0, 0.6)
        for key, value in pd.Series(nominal_weight).groupby(source_kind).sum().items():
            counts["nominal_training_weight_sum_by_source_kind"][str(key)] = (
                counts["nominal_training_weight_sum_by_source_kind"].get(str(key), 0.0) + float(value)
            )

        keep_cols = [col for col in usecols if col in work.columns]
        work = work[keep_cols].copy()
        current_rows = sum(len(part) for part in sampled_parts)
        if max_rows <= 0 or current_rows < max_rows:
            remaining = max_rows - current_rows if max_rows > 0 else len(work)
            sampled_parts.append(work.head(max(0, remaining)).copy())
            extra = work.iloc[max(0, remaining) :]
        else:
            extra = work
        if max_rows > 0 and not extra.empty and sampled_parts:
            # Reservoir-like replacement keeps memory bounded and avoids only-head sampling.
            sample_df = pd.concat(sampled_parts, ignore_index=True)
            replace_count = min(len(extra), max(1, int(0.02 * max_rows)))
            if replace_count > 0 and len(sample_df) >= replace_count:
                extra_sample = extra.sample(n=replace_count, random_state=int(rng.integers(0, 2**31 - 1)))
                drop_idx = rng.choice(sample_df.index.to_numpy(), size=replace_count, replace=False)
                sample_df = sample_df.drop(index=drop_idx).reset_index(drop=True)
                sampled_parts = [pd.concat([sample_df, extra_sample], ignore_index=True).head(max_rows)]
        if chunk_idx % 10 == 0:
            print(f"[step6-qc] scanned sample chunks={chunk_idx} target_rows={counts['rows_in_target_block']}", flush=True)

    sample_df = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if len(sample_df) > max_rows > 0:
        sample_df = sample_df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    if "GT_POINT_FLAG" not in sample_df.columns and "HasFracture" in sample_df.columns:
        sample_df["GT_POINT_FLAG"] = sample_df["HasFracture"]
    for col in [
        "DensityLabel",
        "GT_POINT_FLAG",
        "HasFracture",
        "IsRefinedFracturePoint",
        "Coherence",
        "AntTrack",
        "CurvatureMax",
        "CurvaturePos",
    ]:
        if col in sample_df.columns:
            sample_df[col] = safe_numeric(sample_df[col])

    target_cols = [col for col in ["DensityLabel", "GT_POINT_FLAG", "HasFracture", "IsRefinedFracturePoint"] if col in sample_df.columns]
    attr_cols = [col for col in ATTRIBUTE_COLUMNS if col in sample_df.columns]
    correlations: dict[str, Any] = {}
    groups = {"all": sample_df}
    if "SourceKind" in sample_df.columns:
        for key, group in sample_df.groupby("SourceKind", dropna=False):
            groups[f"SourceKind={key}"] = group
    if "LayerGroup" in sample_df.columns:
        for key, group in sample_df.groupby("LayerGroup", dropna=False):
            groups[f"LayerGroup={key}"] = group
    for group_name, group in groups.items():
        correlations[group_name] = {}
        for target in target_cols:
            correlations[group_name][target] = {attr: corr_pair(group, target, attr) for attr in attr_cols}

    summary_rows: list[dict[str, Any]] = []
    for group_name, target_payload in correlations.items():
        for target, attr_payload in target_payload.items():
            for attr, corr in attr_payload.items():
                summary_rows.append(
                    {
                        "group": group_name,
                        "target": target,
                        "attribute": attr,
                        "count": corr["count"],
                        "pearson": corr["pearson"],
                        "spearman": corr["spearman"],
                    }
                )
    corr_csv = output_dir / "well_density_attribute_correlation.csv"
    with corr_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "target", "attribute", "count", "pearson", "spearman"])
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "unified_samples_csv": str(unified_samples_csv),
        "columns_seen": existing_usecols or [],
        "counts": counts,
        "sampled_correlation_rows": int(len(sample_df)),
        "max_sampled_correlation_rows": int(max_rows),
        "correlations": correlations,
        "correlation_csv": str(corr_csv),
        "gt_point_flag_logic": "Use GT_POINT_FLAG when present; current unified samples use HasFracture as GT_POINT_FLAG alias.",
        "note": "Correlations are exact for retained sample rows; large target-block sample rows are bounded by max_well_correlation_rows.",
    }


def write_overlap_csv(path: Path, overlap_payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for density_label, payload in overlap_payload.items():
        for top_name, top_payload in payload.get("top_quantiles", {}).items():
            for mask_name, item in top_payload.get("overlap", {}).items():
                rows.append(
                    {
                        "density_volume": density_label,
                        "top_density_bin": top_name,
                        "threshold": top_payload.get("threshold"),
                        "top_count": top_payload.get("top_count"),
                        "attribute_mask": mask_name,
                        "fraction": item.get("fraction"),
                        "baseline_fraction": item.get("baseline_fraction"),
                        "enrichment": item.get("enrichment"),
                    }
                )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "density_volume",
                "top_density_bin",
                "threshold",
                "top_count",
                "attribute_mask",
                "fraction",
                "baseline_fraction",
                "enrichment",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    mapping_path = Path(config["output_dir"]).resolve() / f"{config['target_block']['name']}_3d_trace_mapping.npz"
    if not mapping_path.exists():
        raise FileNotFoundError(f"trace mapping not found: {mapping_path}")
    mapping = load_trace_mapping(mapping_path)
    source_trace_idx = mapping["source_trace_idx"].astype(np.int64)

    density_paths = default_density_sgys(config)
    density_paths.update(parse_density_args(args.density_sgy))
    if not density_paths:
        raise RuntimeError("no density SGY found for QC")
    for label, path in density_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"density SGY not found for {label}: {path}")

    first_label = next(iter(density_paths))
    first_density, samples, first_load = load_sgy_matrix(density_paths[first_label])
    if first_density.shape[0] != len(source_trace_idx):
        raise ValueError(f"{first_label} tracecount {first_density.shape[0]} != mapping rows {len(source_trace_idx)}")

    volume_paths = {key: Path(value).resolve() for key, value in config["volume_paths"].items() if key in ATTRIBUTE_COLUMNS}
    missing_attrs = sorted(set(ATTRIBUTE_COLUMNS).difference(volume_paths))
    if missing_attrs:
        raise ValueError(f"config volume_paths missing attributes: {missing_attrs}")

    print("[step6-qc] loading seismic attributes", flush=True)
    attributes: dict[str, np.ndarray] = {}
    attribute_load: dict[str, Any] = {}
    for attr in ATTRIBUTE_COLUMNS:
        matrix, load_summary = load_attribute_matrix(volume_paths[attr], source_trace_idx, samples, attr)
        attributes[attr] = matrix
        attribute_load[attr] = load_summary

    masks, attribute_thresholds, common_attr_valid = build_attribute_masks(attributes)
    rng = np.random.default_rng(int(args.random_state))
    top_quantiles = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995]
    density_load: dict[str, Any] = {first_label: first_load}
    density_stats: dict[str, Any] = {}
    overlap_payload: dict[str, Any] = {}
    correlation_payload: dict[str, Any] = {}

    for label, path in density_paths.items():
        if label == first_label:
            density = first_density
            load_summary = first_load
        else:
            density, density_samples, load_summary = load_sgy_matrix(path)
            if density.shape != first_density.shape:
                raise ValueError(f"{label} shape {density.shape} != first density shape {first_density.shape}")
            if not np.allclose(density_samples, samples, rtol=0.0, atol=1.0e-6):
                raise ValueError(f"{label} sample axis differs from first density volume")
        density_valid = common_attr_valid & np.isfinite(density)
        density_load[label] = load_summary
        density_stats[label] = {
            "all_valid": finite_stats(density[density_valid]),
            "positive": finite_stats(density[density_valid & (density > 0.0)]),
            "positive_quantiles": quantiles(density[density_valid & (density > 0.0)], top_quantiles),
        }
        overlap_payload[label] = top_density_overlap(
            density=density,
            masks=masks,
            valid=density_valid,
            top_quantiles=top_quantiles,
        )
        correlation_payload[label] = density_attribute_correlations(
            density=density,
            attributes=attributes,
            valid=density_valid,
            sample_size=int(args.correlation_sample_size),
            rng=rng,
        )

    lowcoh_qc = lowcoh_component_qc(attributes["Coherence"], mapping=mapping, quantile=0.20)
    well_qc = stream_well_point_qc(
        unified_samples_csv=Path(config["unified_samples_csv"]).resolve(),
        target_block=dict(config["target_block"]),
        output_dir=output_dir,
        chunksize=int(args.chunksize),
        max_rows=int(args.max_well_correlation_rows),
        random_state=int(args.random_state),
    )

    overlap_json = output_dir / "top_density_attribute_overlap.json"
    well_json = output_dir / "well_density_attribute_correlation.json"
    lowcoh_json = output_dir / "lowcoh_component_qc.json"
    summary_json = output_dir / "step6_multiscale_qc_current.json"
    overlap_csv = output_dir / "density_attribute_overlap_summary.csv"

    write_json(overlap_json, overlap_payload)
    write_json(well_json, well_qc)
    write_json(lowcoh_json, lowcoh_qc)
    write_overlap_csv(overlap_csv, overlap_payload)

    summary = {
        "status": "pass",
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "target_block": config.get("target_block", {}),
        "trace_mapping_npz": str(mapping_path),
        "trace_count": int(first_density.shape[0]),
        "sample_axis": {
            "time_min_ms": float(samples[0]),
            "time_max_ms": float(samples[-1]),
            "sample_count": int(len(samples)),
        },
        "density_volumes": {label: str(path) for label, path in density_paths.items()},
        "density_load": density_load,
        "density_stats": density_stats,
        "attribute_load": attribute_load,
        "attribute_thresholds": attribute_thresholds,
        "density_attribute_correlations": correlation_payload,
        "lowcoh_component_qc_json": str(lowcoh_json),
        "top_density_attribute_overlap_json": str(overlap_json),
        "top_density_attribute_overlap_csv": str(overlap_csv),
        "well_density_attribute_correlation_json": str(well_json),
        "well_density_attribute_correlation_csv": well_qc.get("correlation_csv"),
        "checks": {
            "no_sgy_written": True,
            "no_existing_result_overwritten": True,
            "has_density_volumes": bool(density_paths),
            "has_attribute_matrices": sorted(attributes) == sorted(ATTRIBUTE_COLUMNS),
            "has_well_qc": bool(well_qc),
            "has_lowcoh_component_qc": lowcoh_qc.get("status") == "pass",
        },
    }
    summary["status"] = "pass" if all(bool(value) for value in summary["checks"].values()) else "fail"
    write_json(summary_json, summary)
    print(f"[step6-qc] summary={summary_json}", flush=True)
    print(f"[step6-qc] status={summary['status']}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

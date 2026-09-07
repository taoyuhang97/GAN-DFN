#!/usr/bin/env python3
"""Step5 (太古界 v2): virtual wells + unified samples with amplitude continuity.

Compared with v1 (pure geometric extrapolation), this version follows the
glutenite Step5A idea adapted to the raw-amplitude-only constraint:
  * amplitude context features are sampled at TIME+[-4,-2,0,+2,+4] ms;
  * a continuity score exp(-|src_amp - virtual_amp|/scale) qualifies the
    transfer of Step4 labels to virtual traces (raw amplitude is the only
    seismic evidence available);
  * DensityLabel = source density * continuity (positive rows only);
  * PointConfidence = distance-decay * continuity; SampleWeight = conf/25;
  * low-continuity rows are excluded from training (audit only);
  * strong supervision assembles ALL usable imaging wells (6 wells) with a
    per-row InputSegmentPath join (multi-segment groups are fully joined);
    only co-located amplitude rows (in-survey) enter the training table.

Outputs (config.output_dir):
  taigu_step5_unified_samples.csv        training table
  taigu_step5_strong_supervision.csv     all imaging wells (reference)
  taigu_step5_no_amplitude_audit.csv     excluded samples (audit)
  taigu_step5_virtual_well_index.csv     virtual track census
  taigu_step5_acceptance_summary.json    status=pass
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


CURRENT_DIR = Path(__file__).resolve().parent
COMMON_DIR = CURRENT_DIR.parent / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(COMMON_DIR / "seismic_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "seismic_sampling"))
if str(COMMON_DIR / "attribute_sampling") not in sys.path:
    sys.path.insert(0, str(COMMON_DIR / "attribute_sampling"))

from attribute_sampling import MultiAttributeSampler  # noqa: E402


class _NoAmplitudeSampler:
    """Compatibility sampler for attribute-only mode; never reads OBN data."""
    def __init__(self, grid):
        self.grid = grid
        self.tree = cKDTree(np.column_stack([grid["x"], grid["y"]]))
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def nearest_trace(self, x, y):
        dist, pos = self.tree.query(np.column_stack([x, y]), k=1)
        return self.grid["trace_idx"][np.asarray(pos, dtype=np.int64)], np.asarray(dist, dtype=float)
    def sample_at_trace(self, trace_idx, times): return np.zeros(len(np.asarray(trace_idx)), dtype=float)


SAMPLE_COLUMNS = [
    "SourceKind",
    "SourceSampleID",
    "SourceWellName",
    "TrackWellName",
    "VirtualWellName",
    "X",
    "Y",
    "TIME",
    "LayerGroup",
    "WindowCode",
    "PresenceLabel",
    "DensityLabel",
    "HasFracture",
    "PointConfidence",
    "SampleWeight",
    "SeisAmp",
    "Coherence",
    "AntTrack",
    "CurvatureMax",
    "CoherenceValid",
    "AntTrackValid",
    "CurvatureMaxValid",
    "AttributeValidCount",
    "SeisAmpM4",
    "SeisAmpM2",
    "SeisAmpP2",
    "SeisAmpP4",
    "AmpMad5",
    "AttributeContinuity",
    "ContinuityCoherence",
    "ContinuityAntTrack",
    "ContinuityCurvatureMax",
    "ContinuitySeisAmp",
    "PrimaryAmpSimilarity",
    "LabelStatus",
    "temporary_neighbor_time_depth",
    "SourceWellType",
    "TimeShiftMs",
    "NearestTraceDistM",
    "VirtualTraceIdx",
]

# Compact Step5A hand-off.  Context windows, continuity components and source
# bookkeeping stay in the audit table; they are not repeated per training row.
VIRTUAL_TRAINING_COLUMNS = [
    "SourceSampleID", "SourceWellName", "VirtualWellName", "X", "Y",
    "TIME", "LayerGroup", "WindowCode", "PresenceLabel",
    "DensityLabel", "PointConfidence", "SampleWeight",
    "Coherence", "AntTrack", "CurvatureMax",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 太古界 Step5 virtual wells + unified samples (v2).")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replace-output", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-source-wells", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument("--max-points-per-well", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument("--no-strong-supervision", action="store_true", help="Validation mode: omit imaging supervision rows.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_surface_lookup(cache_npz: Path) -> tuple[dict[str, cKDTree], Any]:
    payload = np.load(cache_npz)
    lookups: dict[str, cKDTree] = {}
    for code in ("top", "mid", "base"):
        lookups[code] = cKDTree(np.column_stack([payload[f"{code}_x"], payload[f"{code}_y"]]))
    return lookups, payload


def query_surface(
    lookups: dict[str, cKDTree], payload: Any, xy: np.ndarray, max_distance: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    times: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    for code in ("top", "mid", "base"):
        distance, index = lookups[code].query(xy, k=1, p=1)
        distance = np.asarray(distance, dtype=np.float64)
        time_values = payload[f"{code}_t"][np.asarray(index, dtype=np.int64)]
        times[code] = np.where(distance <= max_distance, time_values, np.nan)
        distances[code] = distance
    return times, distances


def build_full_grid(trace_df: pd.DataFrame) -> dict[str, Any]:
    trace_df = trace_df.sort_values("TraceIdx").reset_index(drop=True)
    trace_idx = trace_df["TraceIdx"].to_numpy(dtype=np.int64)
    # Attribute cubes are stored as a complete 1650 x 1307 file-order grid.
    # X/Y contain rounded coordinate jitter, so their unique-value Cartesian
    # product is not a valid index and creates artificial holes.
    shape = (1650, 1307)
    if len(trace_df) != shape[0] * shape[1]:
        raise ValueError(f"attribute trace count {len(trace_df)} != expected grid {shape}")
    ix = np.repeat(np.arange(shape[0], dtype=np.int32), shape[1])
    iy = np.tile(np.arange(shape[1], dtype=np.int32), shape[0])
    idx_grid = trace_idx.reshape(shape)
    xy_grid = trace_df[["X", "Y"]].to_numpy(dtype=np.float64).reshape(shape + (2,))
    x_values = np.nanmedian(xy_grid[:, :, 0], axis=1)
    y_values = np.nanmedian(xy_grid[:, :, 1], axis=0)
    return {
        "x_values": x_values, "y_values": y_values,
        "ix": ix,
        "iy": iy,
        "idx_grid": idx_grid,
        "trace_idx": trace_idx,
        "x": trace_df["X"].to_numpy(dtype=np.float64),
        "y": trace_df["Y"].to_numpy(dtype=np.float64),
        "xy_grid": xy_grid,
        "trace_idx_sorted": np.sort(trace_idx),
    }


def grid_position_for_trace(grid: dict[str, Any], trace_idx: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(grid["trace_idx_sorted"], np.asarray(trace_idx, dtype=np.int64))
    return np.clip(positions, 0, len(grid["trace_idx_sorted"]) - 1)


def confidence_for_well(row: pd.Series, config: dict[str, Any]) -> float:
    if str(row["WellType"]).strip() != "imaging":
        return float(config["confidence"]["regular"])
    distance = float(row.get("BorrowDistanceM") or np.nan)
    if not np.isfinite(distance):
        return float(config["confidence"]["min_confidence"])
    scale = float(config["confidence"]["imaging_borrow_distance_scale_m"])
    minimum = float(config["confidence"]["min_confidence"])
    return float(np.clip(1.0 - distance / scale, minimum, 1.0))


def sample_context(
    sampler: ObnAmplitudeSampler,
    trace_idx: np.ndarray,
    times_ms: np.ndarray,
    offsets_ms: np.ndarray,
) -> np.ndarray:
    """Return (n, len(offsets)) amplitude matrix at times + offsets (batched)."""
    times_2d = np.asarray(times_ms, dtype=np.float64)[:, None] + np.asarray(offsets_ms, dtype=np.float64)[None, :]
    flat_times = times_2d.ravel()
    flat_traces = np.repeat(np.asarray(trace_idx, dtype=np.int64), len(offsets_ms))
    values = sampler.sample_at_trace(flat_traces, flat_times)
    return values.reshape(-1, len(offsets_ms))


def write_per_well_outputs(
    output_dir: Path,
    unified: pd.DataFrame,
    virtual_samples: pd.DataFrame,
    index_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    audit: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Write inspectable Step5A packages without duplicating global hand-off tables."""
    wells_root = output_dir / "wells"
    summaries: dict[str, dict[str, Any]] = {}
    well_names = sorted(set(index_df.get("SourceWellName", pd.Series(dtype=str)).dropna()) | set(unified.get("SourceWellName", pd.Series(dtype=str)).dropna()))
    for well in well_names:
        well_dir = wells_root / str(well)
        well_dir.mkdir(parents=True, exist_ok=True)
        real_dir = well_dir / "real_well"
        virtual_root = well_dir / "virtual_wells"
        real_dir.mkdir(parents=True, exist_ok=True)
        virtual_root.mkdir(parents=True, exist_ok=True)
        real = unified[(unified["SourceWellName"] == well) & (unified["SourceKind"] == "weak_real")].copy()
        virtual = virtual_samples[virtual_samples["SourceWellName"] == well].copy()
        candidates = index_df[index_df["SourceWellName"] == well].copy()
        coverage = coverage_df[coverage_df["SourceWellName"] == well].copy()
        rejected = audit[audit["SourceWellName"] == well].copy() if not audit.empty else pd.DataFrame()
        real.to_csv(real_dir / "real_well_samples.csv", index=False, encoding="utf-8-sig")
        coverage.to_csv(real_dir / "coverage_qc.csv", index=False, encoding="utf-8-sig")
        # Keep a well-level candidate index for machine/audit use, while each
        # virtual trajectory gets its own directory for manual inspection.
        candidates.to_csv(well_dir / "all_virtual_candidates.csv", index=False, encoding="utf-8-sig")
        for virtual_name in sorted(candidates["VirtualWellName"].dropna().astype(str).unique()):
            virtual_dir = virtual_root / virtual_name
            virtual_dir.mkdir(parents=True, exist_ok=True)
            track_samples = virtual[virtual["VirtualWellName"].astype(str) == virtual_name].copy()
            track_candidates = candidates[candidates["VirtualWellName"].astype(str) == virtual_name].copy()
            track_rejected = rejected[rejected["TrackWellName"].astype(str) == virtual_name].copy() if not rejected.empty and "TrackWellName" in rejected.columns else pd.DataFrame()
            compact = [column for column in VIRTUAL_TRAINING_COLUMNS if column in track_samples.columns]
            track_samples.to_csv(virtual_dir / "virtual_well_training_samples.csv", index=False, columns=compact or None, encoding="utf-8-sig")
            track_candidates.to_csv(virtual_dir / "virtual_well_candidates.csv", index=False, encoding="utf-8-sig")
            if not track_rejected.empty:
                track_rejected.to_csv(virtual_dir / "rejected_samples_audit.csv", index=False, encoding="utf-8-sig")
            track_summary = {
                "SourceWellName": str(well),
                "VirtualWellName": virtual_name,
                "virtual_training_sample_count": int(len(track_samples)),
                "candidate_count": int(len(track_candidates)),
                "candidate_reason_counts": track_candidates["ExcludeReason"].value_counts().to_dict() if not track_candidates.empty and "ExcludeReason" in track_candidates.columns else {},
                "sample_weight_sum": float(track_samples["SampleWeight"].sum()) if "SampleWeight" in track_samples.columns else 0.0,
                "positive_sample_count": int((track_samples.get("PresenceLabel", pd.Series(dtype=float)) == 1).sum()),
                "density_label_min": float(track_samples["DensityLabel"].min()) if "DensityLabel" in track_samples.columns and track_samples["DensityLabel"].notna().any() else None,
                "density_label_max": float(track_samples["DensityLabel"].max()) if "DensityLabel" in track_samples.columns and track_samples["DensityLabel"].notna().any() else None,
            }
            write_json(virtual_dir / "virtual_well_summary.json", json_ready(track_summary))
        if not rejected.empty:
            rejected.to_csv(well_dir / "rejected_samples_audit.csv", index=False, encoding="utf-8-sig")
        summary = {
            "SourceWellName": str(well),
            "real_sample_count": int(len(real)),
            "virtual_training_sample_count": int(len(virtual)),
            "candidate_count": int(len(candidates)),
            "candidate_reason_counts": candidates["ExcludeReason"].value_counts().to_dict() if not candidates.empty else {},
            "coverage_source_sample_count": int(len(coverage)),
            "in_obn_coverage_source_samples": int(coverage["InOBNCoverage"].sum()) if not coverage.empty else 0,
            "rejected_audit_count": int(len(rejected)),
            "virtual_tracks": sorted(virtual["VirtualWellName"].dropna().unique().tolist()) if not virtual.empty else [],
        }
        write_json(well_dir / "well_summary.json", summary)
        summaries[str(well)] = summary
    return summaries


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    output_dir = (args.output_dir or Path(config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    unified_path = output_dir / "taigu_step5_unified_samples.csv"
    strong_path = output_dir / "taigu_step5_strong_supervision.csv"
    audit_path = output_dir / "taigu_step5_no_amplitude_audit.csv"
    index_path = output_dir / "taigu_step5_virtual_well_index.csv"
    virtual_path = output_dir / "taigu_step5_virtual_training_samples.csv"
    summary_path = output_dir / "taigu_step5_acceptance_summary.json"
    coverage_path = output_dir / "taigu_step5_coverage_qc.csv"
    for path in (unified_path, strong_path, audit_path, index_path, virtual_path, summary_path, coverage_path):
        if path.exists() and not args.replace_output:
            raise FileExistsError(f"refusing to overwrite existing Step5 output: {path}")

    started = time.time()
    step4 = pd.read_csv(config["step4_merged_predictions_csv"], encoding="utf-8-sig")
    step4 = step4[step4["PredictionValid"].astype(int) == 1].copy()
    for column in ("X", "Y", "TIME", "TVD", "PredDensity", "PredConditionalDensity", "PredHasFracture"):
        step4[column] = pd.to_numeric(step4[column], errors="coerce")
    step4 = step4.dropna(subset=["WellName", "TVD", "X", "Y", "TIME", "PredHasFracture"]).copy()

    metadata = pd.read_csv(config["step2_well_metadata_csv"], encoding="utf-8-sig")
    metadata = metadata[metadata["Step2Status"].astype(str) == "eligible"].copy()
    metadata_lookup = metadata.set_index("WellName")

    trace_df = pd.read_csv(config["trace_header_csv"], encoding="utf-8-sig")
    for column in ("TraceIdx", "X", "Y"):
        trace_df[column] = pd.to_numeric(trace_df[column], errors="coerce")
    trace_df = trace_df.dropna(subset=["TraceIdx", "X", "Y"]).drop_duplicates("TraceIdx").copy()
    trace_df["TraceIdx"] = trace_df["TraceIdx"].astype(np.int64)
    grid = build_full_grid(trace_df)

    lookups, cache_payload = load_surface_lookup(config["surface_lookup_cache_npz"])
    max_horizon_distance = float(config["max_horizon_match_distance_m"])

    max_wells = int(args.max_source_wells)
    max_points = int(args.max_points_per_well)
    radius = int(config["neighborhood_radius"])
    offsets = [(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
    max_shift = float(config["max_time_shift_ms"])
    max_neighbor_distance = float(config["max_virtual_neighbor_distance_m"])
    enforce_neighbor_distance = bool(config.get("enforce_neighbor_distance", False))
    max_sample_distance = float(config.get("max_sample_distance_m", 30.0))
    context_offsets = np.asarray(config["context_time_offsets_ms"], dtype=np.float64)
    continuity_cfg = config.get("continuity", {})
    continuity_enabled = bool(continuity_cfg.get("enabled", True))
    scale_multiplier = float(continuity_cfg.get("scale_multiplier", 1.0))
    scale_floor = float(continuity_cfg.get("scale_floor", 1.0e-6))
    min_continuity = float(continuity_cfg.get("min_continuity", 0.3))
    distance_scale_m = float(continuity_cfg.get("distance_scale_m", 500.0))
    weight_divisor = float(continuity_cfg.get("weight_divisor", 25.0))

    well_names = sorted(step4["WellName"].unique())
    if config.get("selected_wells"):
        selected = set(map(str, config["selected_wells"]))
        well_names = [well for well in well_names if well in selected]
    if max_wells > 0:
        well_names = well_names[:max_wells]

    all_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    index_records: list[dict[str, Any]] = []
    # Keep one audit row for every theoretical offset; stages update its status.
    candidate_lookup: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    coverage_records: list[dict[str, Any]] = []
    well_stats: dict[str, dict[str, Any]] = {}

    use_seis_amp = bool(config.get("use_seis_amp", False))
    attr_sampler = MultiAttributeSampler(
        config["attribute_volume_paths"],
        config.get("attribute_invalid_rules", {}),
        config.get("attribute_time_origins_ms", {}),
    )
    sampler_ctx = (_NoAmplitudeSampler(grid) if not use_seis_amp else None)
    if use_seis_amp:
        from amplitude_sampling import ObnAmplitudeSampler
        sampler_ctx = ObnAmplitudeSampler(Path(config["obn_segy"]), Path(config["trace_header_csv"]))
    with sampler_ctx as sampler, attr_sampler:
        # --- real weak rows per well (needed for continuity scale and source amps) ---
        real_center_amps: dict[str, np.ndarray] = {}
        for well in tqdm(well_names, desc="Step5处理源井", unit="well"):
            if well not in metadata_lookup.index:
                continue
            well_meta = metadata_lookup.loc[well]
            if isinstance(well_meta, pd.DataFrame):
                well_meta = well_meta.iloc[0]
            rows = step4[step4["WellName"] == well].copy()
            if max_points > 0:
                rows = rows.head(max_points).copy()
            is_imaging = str(well_meta["WellType"]).strip() == "imaging"
            confidence = confidence_for_well(well_meta, config)
            temp_flag = 1 if is_imaging else 0
            well_type = "imaging" if is_imaging else "regular"
            nearest_trace, nearest_dist = sampler.nearest_trace(
                rows["X"].to_numpy(dtype=np.float64), rows["Y"].to_numpy(dtype=np.float64)
            )
            ctx = sample_context(sampler, nearest_trace, rows["TIME"].to_numpy(dtype=np.float64), context_offsets)
            source_attrs = attr_sampler.sample_at_xy(rows["X"].to_numpy(dtype=float), rows["Y"].to_numpy(dtype=float), rows["TIME"].to_numpy(dtype=float))
            center = ctx[:, list(context_offsets).index(0.0)]
            real_center_amps[well] = center
            mad5 = np.nanmedian(np.abs(ctx - center[:, None]), axis=1)
            trace_pos = grid_position_for_trace(grid, nearest_trace)
            ix_of_trace = grid["ix"][trace_pos]
            iy_of_trace = grid["iy"][trace_pos]
            source_xy = rows[["X", "Y"]].to_numpy(dtype=np.float64)
            times, dists = query_surface(lookups, cache_payload, source_xy, max_horizon_distance)
            source_top = times["top"]
            source_mid = times["mid"]
            source_base = times["base"]
            for pos in range(len(rows)):
                amp = float(center[pos])
                coverage_records.append({
                    "SourceWellName": well, "SourceRowIndex": int(pos),
                    "X": float(rows.iloc[pos]["X"]), "Y": float(rows.iloc[pos]["Y"]),
                    "TIME": float(rows.iloc[pos]["TIME"]),
                    "NearestTraceIdx": int(nearest_trace[pos]),
                    "NearestTraceDistM": float(nearest_dist[pos]),
                    "InOBNCoverage": int(float(nearest_dist[pos]) <= max_sample_distance),
                    "CoherenceValid": int(np.isfinite(source_attrs["Coherence"][pos])),
                    "AntTrackValid": int(np.isfinite(source_attrs["AntTrack"][pos])),
                    "CurvatureMaxValid": int(np.isfinite(source_attrs["CurvatureMax"][pos])),
                    "AttributeValidCount": int(sum(np.isfinite(source_attrs[name][pos]) for name in ("Coherence", "AntTrack", "CurvatureMax"))),
                })
                for di, dj in offsets:
                    reason = "pending" if float(nearest_dist[pos]) <= max_sample_distance else "outside_obn_coverage"
                    track = f"{well}_VW_{di+2}_{dj+2}"
                    rec = {"SourceWellName": well, "TrackWellName": track, "VirtualWellName": track,
                           "SourceSampleID": f"{well}::row{pos}",
                           "SourceRowIndex": int(pos), "TVD": rows.iloc[pos]["TVD"],
                           "OffsetDx": int(di), "OffsetDy": int(dj), "TrainingEligible": 0,
                           "ExcludeReason": reason}
                    candidate_lookup[(well, int(pos), int(di), int(dj))] = rec
                    index_records.append(rec)
                if float(nearest_dist[pos]) > max_sample_distance or not np.isfinite(amp):
                    audit_records.append(
                        {
                            "SourceWellName": well,
                            "TrackWellName": well,
                            "SourceKind": "weak_real",
                            "MD": rows.iloc[pos]["MD"],
                            "TVD": rows.iloc[pos]["TVD"],
                            "X": rows.iloc[pos]["X"],
                            "Y": rows.iloc[pos]["Y"],
                            "TIME": rows.iloc[pos]["TIME"],
                            "LayerGroup": rows.iloc[pos]["StrataName"],
                            "ExcludeReason": "too_far_from_survey"
                            if float(nearest_dist[pos]) > max_sample_distance
                            else "no_amplitude",
                        }
                    )
                    reason = "outside_obn_coverage" if float(nearest_dist[pos]) > max_sample_distance else "no_source_amplitude"
                    for di, dj in offsets:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = reason
                    continue
                presence = int(rows.iloc[pos]["PredHasFracture"])
                attr_valid = {name: bool(np.isfinite(source_attrs[name][pos])) for name in ("Coherence", "AntTrack", "CurvatureMax")}
                all_records.append(
                    {
                        "SourceKind": "weak_real",
                        "SourceSampleID": f"{well}::row{pos}",
                        "SourceWellName": well,
                        "TrackWellName": well,
                        "VirtualWellName": well,
                        "X": rows.iloc[pos]["X"],
                        "Y": rows.iloc[pos]["Y"],
                        "TIME": rows.iloc[pos]["TIME"],
                        "LayerGroup": rows.iloc[pos]["StrataName"],
                        "WindowCode": "main",
                        "PresenceLabel": presence,
                        "DensityLabel": float(rows.iloc[pos]["PredDensity"]) if presence == 1 else np.nan,
                        "HasFracture": presence,
                        "PointConfidence": confidence,
                        "SampleWeight": confidence,
                        "SeisAmp": amp,
                        "Coherence": float(source_attrs["Coherence"][pos]),
                        "AntTrack": float(source_attrs["AntTrack"][pos]),
                        "CurvatureMax": float(source_attrs["CurvatureMax"][pos]),
                        "CoherenceValid": int(attr_valid["Coherence"]),
                        "AntTrackValid": int(attr_valid["AntTrack"]),
                        "CurvatureMaxValid": int(attr_valid["CurvatureMax"]),
                        "AttributeValidCount": int(sum(attr_valid.values())),
                        "SeisAmpM4": float(ctx[pos, 0]),
                        "SeisAmpM2": float(ctx[pos, 1]),
                        "SeisAmpP2": float(ctx[pos, 3]),
                        "SeisAmpP4": float(ctx[pos, 4]),
                        "AmpMad5": float(mad5[pos]) if np.isfinite(mad5[pos]) else np.nan,
                        "AttributeContinuity": 1.0,
                        "PrimaryAmpSimilarity": 1.0,
                        "LabelStatus": "real_well",
                        "temporary_neighbor_time_depth": temp_flag,
                        "SourceWellType": well_type,
                        "TimeShiftMs": 0.0,
                        "NearestTraceDistM": float(nearest_dist[pos]),
                        "VirtualTraceIdx": int(nearest_trace[pos]),
                    }
                )

            # --- virtual rows (5x5 neighbours) with amplitude continuity (batched) ---
            source_amps = center
            finite_amps = source_amps[np.isfinite(source_amps)]
            amp_scale = max(float(np.median(np.abs(finite_amps - np.median(finite_amps))) * 1.4826), scale_floor) * scale_multiplier
            # Use robust per-well scales for each attribute.  A fixed floor is
            # only a safeguard for constant or sparse source values.
            attr_scales = {"SeisAmp": amp_scale}
            for attr_name in ("Coherence", "AntTrack", "CurvatureMax"):
                values = np.asarray(source_attrs[attr_name], dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    robust = float(np.median(np.abs(values - np.median(values))) * 1.4826)
                else:
                    robust = 0.0
                floor = float(config.get("attribute_scale_floors", {}).get(attr_name, 1.0))
                attr_scales[attr_name] = max(robust, floor, scale_floor) * scale_multiplier
            virt_meta: list[dict[str, Any]] = []
            virt_xy_list: list[np.ndarray] = []
            virt_trace_list: list[int] = []
            virt_time_list: list[float] = []
            for pos in range(len(rows)):
                center_ix = int(ix_of_trace[pos])
                center_iy = int(iy_of_trace[pos])
                if float(nearest_dist[pos]) > max_sample_distance:
                    continue  # already audited in real loop
                source_time = float(rows.iloc[pos]["TIME"])
                layer = str(rows.iloc[pos]["StrataName"])
                if layer == "上部复合层":
                    layer_top_s, layer_base_s = source_top[pos], source_mid[pos]
                elif layer == "太古界风化壳":
                    layer_top_s, layer_base_s = source_mid[pos], source_base[pos]
                else:
                    for di, dj in offsets:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "invalid_source_horizon"
                    continue
                if not (np.isfinite(layer_top_s) and np.isfinite(layer_base_s) and layer_base_s > layer_top_s):
                    for di, dj in offsets:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "invalid_source_horizon"
                    continue
                rel = (source_time - layer_top_s) / (layer_base_s - layer_top_s)
                if not (0.0 <= rel <= 1.0):
                    for di, dj in offsets:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "source_time_outside_horizon"
                    continue
                for di, dj in offsets:
                    target_ix = center_ix + di
                    target_iy = center_iy + dj
                    if not (0 <= target_ix < grid["idx_grid"].shape[0] and 0 <= target_iy < grid["idx_grid"].shape[1]):
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "outside_grid"
                        continue
                    target_trace = int(grid["idx_grid"][target_ix, target_iy])
                    if target_trace < 0:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "missing_attribute_trace"
                        continue
                    target_x = float(grid["xy_grid"][target_ix, target_iy, 0])
                    target_y = float(grid["xy_grid"][target_ix, target_iy, 1])
                    dist_m = np.hypot(target_x - rows.iloc[pos]["X"], target_y - rows.iloc[pos]["Y"])
                    if enforce_neighbor_distance and dist_m > max_neighbor_distance:
                        candidate_lookup[(well, int(pos), int(di), int(dj))]["ExcludeReason"] = "outside_neighbor_distance"
                        continue
                    virt_meta.append(
                        {
                            "row_pos": int(pos),
                            "di": di,
                            "dj": dj,
                            "dist_m": float(dist_m),
                            "layer": layer,
                            "rel": rel,
                        }
                    )
                    virt_xy_list.append(np.array([target_x, target_y]))
                    virt_trace_list.append(int(target_trace))
                    virt_time_list.append(source_time)
            if virt_xy_list:
                virt_xy = np.stack(virt_xy_list)
                vtimes, _ = query_surface(lookups, cache_payload, virt_xy, max_horizon_distance)
                vtop = vtimes["top"]
                vmid = vtimes["mid"]
                vbase = vtimes["base"]
                virtual_times = np.full(len(virt_meta), np.nan, dtype=np.float64)
                for k, meta in enumerate(virt_meta):
                    layer_top_v = vtop[k] if meta["layer"] == "上部复合层" else vmid[k]
                    layer_base_v = vmid[k] if meta["layer"] == "上部复合层" else vbase[k]
                    if np.isfinite(layer_top_v) and np.isfinite(layer_base_v) and layer_base_v > layer_top_v:
                        virtual_times[k] = layer_top_v + meta["rel"] * (layer_base_v - layer_top_v)
                vctx = sample_context(
                    sampler, np.asarray(virt_trace_list, dtype=np.int64), virtual_times, context_offsets
                )
                vattrs = attr_sampler.sample_at_xy(virt_xy[:, 0], virt_xy[:, 1], virtual_times)
                for k, meta in enumerate(virt_meta):
                    if meta["layer"] == "上部复合层":
                        layer_top_v, layer_base_v = vtop[k], vmid[k]
                    else:
                        layer_top_v, layer_base_v = vmid[k], vbase[k]
                    if not (np.isfinite(layer_top_v) and np.isfinite(layer_base_v) and layer_base_v > layer_top_v):
                        candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]["ExcludeReason"] = "invalid_virtual_horizon"
                        continue
                    virtual_time = virtual_times[k]
                    if not (layer_top_v - 0.5 <= virtual_time <= layer_base_v + 0.5):
                        candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]["ExcludeReason"] = "invalid_virtual_time"
                        continue
                    source_time = float(rows.iloc[meta["row_pos"]]["TIME"])
                    if abs(virtual_time - source_time) > max_shift:
                        candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]["ExcludeReason"] = "time_shift_exceeded"
                        continue
                    virtual_center = float(vctx[k, list(context_offsets).index(0.0)])
                    source_row = rows.iloc[meta["row_pos"]]
                    # One source well always owns the same 25 spatial virtual
                    # tracks.  The source depth is represented by SourceSampleID.
                    track = f"{well}_VW_{meta['di'] + 2}_{meta['dj'] + 2}"
                    if use_seis_amp and not np.isfinite(virtual_center):
                        candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]["ExcludeReason"] = "no_virtual_amplitude"
                        audit_records.append(
                            {
                                "SourceWellName": well,
                                "TrackWellName": track,
                                "SourceKind": "weak_virtual",
                                "MD": source_row["MD"],
                                "TVD": source_row["TVD"],
                                "X": float(virt_xy[k][0]),
                                "Y": float(virt_xy[k][1]),
                                "TIME": virtual_time,
                                "LayerGroup": meta["layer"],
                                "SourceRowIndex": int(meta["row_pos"]),
                                "OffsetDx": int(meta["di"]),
                                "OffsetDy": int(meta["dj"]),
                                "ExcludeReason": "no_amplitude",
                            }
                        )
                        continue
                    attr_weights = config.get("attribute_weights", {"CurvatureMax": 0.70, "Coherence": 0.15, "AntTrack": 0.15})
                    if not use_seis_amp:
                        attr_weights = {k: v for k, v in attr_weights.items() if k != "SeisAmp"}
                    similarities = []
                    continuity_components = {}
                    weighted = 0.0
                    weight_sum = 0.0
                    source_attr_row = {name: source_attrs[name][meta["row_pos"]] for name in ("Coherence", "AntTrack", "CurvatureMax")}
                    virtual_attr_row = {name: vattrs[name][k] for name in ("Coherence", "AntTrack", "CurvatureMax")}
                    source_attr_row["SeisAmp"] = source_amps[meta["row_pos"]]
                    virtual_attr_row["SeisAmp"] = virtual_center
                    attr_valid_count = 0
                    for name, weight in attr_weights.items():
                        sv, vv = float(source_attr_row.get(name, np.nan)), float(virtual_attr_row.get(name, np.nan))
                        if np.isfinite(sv) and np.isfinite(vv):
                            scale = attr_scales.get(name, float(config.get("attribute_scale_floors", {}).get(name, 1.0)))
                            score = float(np.exp(-abs(sv - vv) / max(scale, 1.0e-12)))
                            continuity_components[name] = score
                            weighted += float(weight) * score
                            weight_sum += float(weight)
                            attr_valid_count += 1
                    continuity = float(weighted / weight_sum) if weight_sum > 0 else np.nan
                    if continuity_enabled:
                        continuity = continuity
                    else:
                        continuity = 1.0
                    curvature_ok = np.isfinite(source_attr_row["CurvatureMax"]) and np.isfinite(virtual_attr_row["CurvatureMax"])
                    if (not curvature_ok or attr_valid_count < int(config.get("min_support_attribute_count", 2))
                        or not np.isfinite(continuity)):
                        audit_records.append(
                            {
                                "SourceWellName": well,
                                "TrackWellName": track,
                                "SourceKind": "weak_virtual",
                                "MD": source_row["MD"],
                                "TVD": source_row["TVD"],
                                "X": float(virt_xy[k][0]),
                                "Y": float(virt_xy[k][1]),
                                "TIME": virtual_time,
                                "LayerGroup": meta["layer"],
                                "SourceRowIndex": int(meta["row_pos"]),
                                "OffsetDx": int(meta["di"]),
                                "OffsetDy": int(meta["dj"]),
                                "ExcludeReason": "missing_primary_curvature" if not curvature_ok else "insufficient_attribute_support",
                                "AttributeValidCount": attr_valid_count,
                                "AttributeContinuity": continuity,
                            }
                        )
                        rec = candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]
                        rec.update({"VirtualTraceIdx": int(virt_trace_list[k]), "TimeShiftMs": virtual_time - source_time,
                                    "AttributeContinuity": continuity, "TrainingEligible": 0,
                                    "ExcludeReason": "missing_primary_curvature" if not curvature_ok else "insufficient_attribute_support"})
                        continue
                    distance_weight = math.exp(-meta["dist_m"] / max(distance_scale_m, 1.0e-6))
                    # Attribute differences control confidence, while signed
                    # local evidence controls whether density is weakened or strengthened.
                    evidence_signs = config.get("continuity", {}).get(
                        "evidence_signs", {"CurvatureMax": 1.0, "Coherence": -1.0, "AntTrack": 1.0}
                    )
                    evidence_sum = 0.0
                    evidence_weight = 0.0
                    for name, weight in attr_weights.items():
                        sv, vv = float(source_attr_row.get(name, np.nan)), float(virtual_attr_row.get(name, np.nan))
                        if np.isfinite(sv) and np.isfinite(vv):
                            scale = max(attr_scales.get(name, 1.0), 1.0e-12)
                            delta = np.clip((vv - sv) / scale, -2.0, 2.0)
                            evidence_sum += float(weight) * float(evidence_signs.get(name, 0.0)) * delta
                            evidence_weight += float(weight)
                    evidence_score = evidence_sum / evidence_weight if evidence_weight else 0.0
                    evidence_clip = float(config.get("continuity", {}).get("evidence_clip", 2.0))
                    beta = float(config.get("continuity", {}).get("evidence_beta", 0.25))
                    intensity_multiplier = float(np.clip(np.exp(beta * np.clip(evidence_score, -evidence_clip, evidence_clip)), 0.25, 4.0))
                    label_confidence = float(np.clip(distance_weight * continuity, 0.0, 1.0))
                    presence = int(source_row["PredHasFracture"])
                    all_records.append(
                        {
                            "SourceKind": "weak_virtual",
                            "SourceSampleID": f"{well}::row{meta['row_pos']}",
                            "SourceWellName": well,
                            "TrackWellName": track,
                            "VirtualWellName": track,
                            "X": float(virt_xy[k][0]),
                            "Y": float(virt_xy[k][1]),
                            "TIME": virtual_time,
                            "LayerGroup": meta["layer"],
                            "WindowCode": "main",
                            "PresenceLabel": presence,
                            "DensityLabel": float(source_row["PredDensity"]) * intensity_multiplier if presence == 1 else np.nan,
                            "HasFracture": presence,
                            "PointConfidence": label_confidence,
                            "SampleWeight": label_confidence / weight_divisor,
                            "SeisAmp": virtual_center,
                            "Coherence": float(vattrs["Coherence"][k]),
                            "AntTrack": float(vattrs["AntTrack"][k]),
                            "CurvatureMax": float(vattrs["CurvatureMax"][k]),
                            "CoherenceValid": int(np.isfinite(vattrs["Coherence"][k]) and np.isfinite(source_attr_row["Coherence"])),
                            "AntTrackValid": int(np.isfinite(vattrs["AntTrack"][k]) and np.isfinite(source_attr_row["AntTrack"])),
                            "CurvatureMaxValid": int(np.isfinite(vattrs["CurvatureMax"][k]) and np.isfinite(source_attr_row["CurvatureMax"])),
                            "AttributeValidCount": int(attr_valid_count),
                            "SeisAmpM4": float(vctx[k, 0]),
                            "SeisAmpM2": float(vctx[k, 1]),
                            "SeisAmpP2": float(vctx[k, 3]),
                            "SeisAmpP4": float(vctx[k, 4]),
                            "AmpMad5": float(np.nanmedian(np.abs(vctx[k] - virtual_center))),
                            "AttributeContinuity": continuity,
                            "ContinuityCoherence": continuity_components.get("Coherence", np.nan),
                            "ContinuityAntTrack": continuity_components.get("AntTrack", np.nan),
                            "ContinuityCurvatureMax": continuity_components.get("CurvatureMax", np.nan),
                            "ContinuitySeisAmp": continuity_components.get("SeisAmp", np.nan),
                            "PrimaryAmpSimilarity": continuity,
                            "LabelStatus": "attribute_continuity_transfer",
                            "temporary_neighbor_time_depth": temp_flag,
                            "SourceWellType": well_type,
                            "TimeShiftMs": virtual_time - source_time,
                            "NearestTraceDistM": float(nearest_dist[meta["row_pos"]]),
                            "VirtualTraceIdx": int(virt_trace_list[k]),
                        }
                    )
                    rec = candidate_lookup[(well, int(meta["row_pos"]), int(meta["di"]), int(meta["dj"]))]
                    rec.update({"VirtualTraceIdx": int(virt_trace_list[k]), "TimeShiftMs": virtual_time - source_time,
                                "AttributeContinuity": continuity, "EvidenceScore": evidence_score,
                                "IntensityMultiplier": intensity_multiplier,
                                "TrainingEligible": 1, "ExcludeReason": "accepted"})
            well_stats[well] = {
                "well_type": well_type,
                "source_rows": int(len(rows)),
                "temporary_neighbor_time_depth": temp_flag,
                "point_confidence": confidence,
            }
            print(f"[step5] well={well} type={well_type} rows={len(rows)}", flush=True)

        # --- strong supervision: all usable imaging wells, per-segment join ---
        strong_records: list[dict[str, Any]] = []
        strong_well_list = [] if args.no_strong_supervision else config["strong_supervision_wells"]
        for well in strong_well_list:
            group_files = sorted(Path(config["step3_groups_root"]).glob(f"{well}_*.csv"))
            if not group_files:
                continue
            for group_file in group_files:
                group = pd.read_csv(group_file, encoding="utf-8-sig")
                for segment_path, sub in group.groupby("InputSegmentPath"):
                    segment = pd.read_csv(segment_path, encoding="utf-8-sig")
                    for column in ("MD", "TVD"):
                        sub[column] = pd.to_numeric(sub[column], errors="coerce")
                    for column in ("MD", "TVD", "X", "Y", "TIME"):
                        segment[column] = pd.to_numeric(segment[column], errors="coerce")
                    merged = pd.merge_asof(
                        sub.sort_values("MD"),
                        segment[["MD", "X", "Y", "TIME"]].sort_values("MD"),
                        on="MD",
                        direction="nearest",
                        tolerance=float(config["md_merge_tolerance"]),
                    )
                    merged = merged.dropna(subset=["X", "Y", "TIME", "Density"]).copy()
                    if merged.empty:
                        continue
                    nearest_trace, nearest_dist = sampler.nearest_trace(
                        merged["X"].to_numpy(dtype=np.float64), merged["Y"].to_numpy(dtype=np.float64)
                    )
                    ctx = sample_context(sampler, nearest_trace, merged["TIME"].to_numpy(dtype=np.float64), context_offsets)
                    imaging_attrs = attr_sampler.sample_at_xy(merged["X"].to_numpy(dtype=float), merged["Y"].to_numpy(dtype=float), merged["TIME"].to_numpy(dtype=float))
                    center = ctx[:, list(context_offsets).index(0.0)]
                    confidence = confidence_for_well(metadata_lookup.loc[well], config)
                    for pos in range(len(merged)):
                        row = merged.iloc[pos]
                        presence = int(row["HasFractureDensity"])
                        amp = float(center[pos])
                        if float(nearest_dist[pos]) > max_sample_distance or not np.isfinite(amp):
                            audit_records.append(
                                {
                                    "SourceWellName": well,
                                    "TrackWellName": well,
                                    "SourceKind": "imaging_supervision",
                                    "MD": row["MD"],
                                    "TVD": row["TVD"],
                                    "X": row["X"],
                                    "Y": row["Y"],
                                    "TIME": row["TIME"],
                                    "LayerGroup": row["StrataName"],
                                    "ExcludeReason": "outside_survey_no_amplitude"
                                    if float(nearest_dist[pos]) > max_sample_distance
                                    else "no_amplitude",
                                }
                            )
                            continue
                        record = {
                            "SourceKind": "imaging_supervision",
                            "SourceWellName": well,
                            "TrackWellName": well,
                            "X": float(row["X"]),
                            "Y": float(row["Y"]),
                            "TIME": float(row["TIME"]),
                            "LayerGroup": str(row["StrataName"]),
                            "WindowCode": "main",
                            "PresenceLabel": presence,
                            "DensityLabel": float(row["Density"]) if presence == 1 else np.nan,
                            "HasFracture": presence,
                            "PointConfidence": confidence,
                            "SampleWeight": confidence,
                            "SeisAmp": amp,
                            "Coherence": float(imaging_attrs["Coherence"][pos]),
                            "AntTrack": float(imaging_attrs["AntTrack"][pos]),
                            "CurvatureMax": float(imaging_attrs["CurvatureMax"][pos]),
                            "CoherenceValid": int(np.isfinite(imaging_attrs["Coherence"][pos])),
                            "AntTrackValid": int(np.isfinite(imaging_attrs["AntTrack"][pos])),
                            "CurvatureMaxValid": int(np.isfinite(imaging_attrs["CurvatureMax"][pos])),
                            "AttributeValidCount": int(sum(np.isfinite(imaging_attrs[name][pos]) for name in ("Coherence", "AntTrack", "CurvatureMax"))),
                            "SeisAmpM4": float(ctx[pos, 0]),
                            "SeisAmpM2": float(ctx[pos, 1]),
                            "SeisAmpP2": float(ctx[pos, 3]),
                            "SeisAmpP4": float(ctx[pos, 4]),
                            "AmpMad5": float(np.nanmedian(np.abs(ctx[pos] - amp))),
                            "AttributeContinuity": 1.0,
                            "ContinuityCoherence": np.nan,
                            "ContinuityAntTrack": np.nan,
                            "ContinuityCurvatureMax": np.nan,
                            "ContinuitySeisAmp": np.nan,
                            "PrimaryAmpSimilarity": 1.0,
                            "LabelStatus": "imaging_supervision",
                            "temporary_neighbor_time_depth": 1,
                            "SourceWellType": "imaging",
                            "TimeShiftMs": 0.0,
                            "NearestTraceDistM": float(nearest_dist[pos]),
                            "VirtualTraceIdx": int(nearest_trace[pos]),
                        }
                        all_records.append(record)
                        strong_records.append(record)
        strong_full = pd.DataFrame(strong_records, columns=SAMPLE_COLUMNS) if strong_records else pd.DataFrame(columns=SAMPLE_COLUMNS)

    unified = pd.DataFrame(all_records, columns=SAMPLE_COLUMNS)
    for column in ("PointConfidence", "SampleWeight", "SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "SeisAmpM4", "SeisAmpM2", "SeisAmpP2", "SeisAmpP4", "AmpMad5", "AttributeContinuity", "PrimaryAmpSimilarity"):
        unified[column] = pd.to_numeric(unified[column], errors="coerce")
    unified["PointConfidence"] = unified["PointConfidence"].clip(0.0, 1.0)
    unified["SampleWeight"] = unified["SampleWeight"].fillna(0.0)
    unified = unified[unified["CurvatureMax"].notna()].reset_index(drop=True)
    unified.to_csv(unified_path, index=False, encoding="utf-8-sig")
    virtual_samples = unified[unified["SourceKind"] == "weak_virtual"].copy()
    if not virtual_samples.empty:
        virtual_samples = virtual_samples[VIRTUAL_TRAINING_COLUMNS]
    virtual_samples.to_csv(virtual_path, index=False, encoding="utf-8-sig")
    if not strong_full.empty:
        strong_full.to_csv(strong_path, index=False, encoding="utf-8-sig")
    audit = pd.DataFrame(audit_records)
    if not audit.empty:
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    index_df = pd.DataFrame(index_records)
    if not index_df.empty:
        # Finalize any candidate that reached all gates without an explicit
        # rejection (normally this is the accepted path above).
        index_df["ExcludeReason"] = index_df["ExcludeReason"].replace({"pending": "not_evaluated"})
        index_df.to_csv(index_path, index=False, encoding="utf-8-sig")
    coverage_df = pd.DataFrame(coverage_records)
    if not coverage_df.empty:
        coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    per_well_summaries = write_per_well_outputs(
        output_dir, unified, virtual_samples, index_df, coverage_df, audit
    )

    positive = unified["PresenceLabel"].astype(int) == 1
    weight_by_kind = unified.groupby("SourceKind")["SampleWeight"].sum()
    virtual_weight = float(weight_by_kind.get("weak_virtual", 0.0))
    real_weight = float(weight_by_kind.get("weak_real", 0.0))
    audit_wells = set(audit["SourceWellName"].astype(str).unique()) if not audit.empty else set()
    expected_outside = {"埕北305", "埕北310", "埕北313", "埕北816", "桩斜169", "桩海102"}
    strong_wells_in_table = set(unified.loc[unified["SourceKind"] == "imaging_supervision", "SourceWellName"])
    checks = {
        "seismic_amplitude_used": use_seis_amp,
        "all_rows_have_finite_primary_curvature": bool(unified["CurvatureMax"].notna().all()),
        "density_label_only_on_positive": bool(unified.loc[~positive, "DensityLabel"].isna().all())
        and bool(unified.loc[positive, "DensityLabel"].notna().all()),
        "confidence_in_range": bool(unified["PointConfidence"].between(0.0, 1.0).all()),
        "weights_positive": bool((unified["SampleWeight"] > 0).all()),
        "virtual_weight_fraction_within_cap": (
            virtual_weight <= float(config["max_virtual_weight_fraction"]) * max(real_weight, 1.0)
        ),
        "temporary_flag_matches_well_type": bool(
            (unified["temporary_neighbor_time_depth"] == (unified["SourceWellType"] == "imaging").astype(int)).all()
        ),
        "outside_wells_only_in_audit": bool(expected_outside.isdisjoint(set(unified["SourceWellName"]))),
        "strong_supervision_present": bool(args.no_strong_supervision or "imaging_supervision" in set(unified["SourceKind"])),
        "strong_only_405_in_training": bool(args.no_strong_supervision or strong_wells_in_table == {"埕北古斜405"}),
        "low_continuity_excluded": bool((unified["LabelStatus"] != "unlabelled_low_continuity").all()),
        "no_duplicate_key": not unified[["SourceKind", "TrackWellName", "X", "Y", "TIME"]].duplicated().any(),
        "candidate_accounting_complete": bool(index_df.empty or (index_df.groupby(["SourceWellName", "SourceRowIndex"]).size() == len(offsets)).all()),
        "candidate_reasons_complete": bool(index_df.empty or index_df["ExcludeReason"].notna().all()),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "status": status,
        "config_path": str(args.config.resolve()),
        "output_dir": str(output_dir),
        "unified_sample_count": int(len(unified)),
        "unified_counts_by_kind": unified["SourceKind"].value_counts().to_dict(),
        "unified_counts_by_well": unified.groupby("SourceWellName")["SampleWeight"].count().to_dict(),
        "sample_weight_by_kind": {str(k): float(v) for k, v in weight_by_kind.items()},
        "positive_rows": int(positive.sum()),
        "positive_fraction": float(positive.mean()) if len(unified) else 0.0,
        "strong_supervision_rows": int(len(strong_full)),
        "strong_supervision_wells": sorted(strong_full["SourceWellName"].unique().tolist()) if not strong_full.empty else [],
        "continuity_stats": {
            "enabled": continuity_enabled,
            "scale": float(amp_scale) if "amp_scale" in dir() else None,
            "min_continuity": min_continuity,
            "virtual_continuity": {
                "min": float(unified.loc[unified["SourceKind"] == "weak_virtual", "AttributeContinuity"].min()),
                "max": float(unified.loc[unified["SourceKind"] == "weak_virtual", "AttributeContinuity"].max()),
                "mean": float(unified.loc[unified["SourceKind"] == "weak_virtual", "AttributeContinuity"].mean()),
            },
        },
        "audit_rows": int(len(audit)),
        "audit_exclude_reasons": audit["ExcludeReason"].value_counts().to_dict() if not audit.empty else {},
        "audit_wells": sorted(audit_wells),
        "well_stats": well_stats,
        "per_well_output_count": int(len(per_well_summaries)),
        "virtual_track_count": int(len(index_df)),
        "candidate_count": int(len(index_df)),
        "candidate_counts_by_reason": index_df["ExcludeReason"].value_counts().to_dict() if not index_df.empty else {},
        "candidate_counts_by_source": index_df.groupby("SourceWellName").size().to_dict() if not index_df.empty else {},
        "coverage_counts": coverage_df["InOBNCoverage"].value_counts().to_dict() if not coverage_df.empty else {},
        "output_paths": {
            "unified_samples_csv": str(unified_path),
            "virtual_training_samples_csv": str(virtual_path),
            "strong_supervision_csv": str(strong_path) if not strong_full.empty else None,
            "no_amplitude_audit_csv": str(audit_path) if not audit.empty else None,
            "virtual_well_index_csv": str(index_path) if not index_df.empty else None,
            "coverage_qc_csv": str(coverage_path) if not coverage_df.empty else None,
            "per_well_packages_dir": str(output_dir / "wells"),
            "acceptance_summary_json": str(summary_path),
        },
        "checks": checks,
        "elapsed_seconds": float(time.time() - started),
    }
    write_json(summary_path, summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

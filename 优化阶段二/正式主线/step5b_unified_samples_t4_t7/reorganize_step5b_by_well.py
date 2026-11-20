from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_reorganize_wells_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Step5A/5B results into per-source-well directories.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true", help="Move an existing wells directory to a timestamped backup before publishing.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def safe_append_csv(frame: pd.DataFrame, path: Path, *, bom_first: bool = True) -> None:
    exists = path.exists()
    encoding = "utf-8-sig" if bom_first and not exists else "utf-8"
    frame.to_csv(path, mode="a", header=not exists, index=False, encoding=encoding, lineterminator="\n")


def make_dirs(root: Path, source_wells: list[str], virtual_by_source: dict[str, list[str]], imaging_wells: set[str]) -> None:
    for source_well in source_wells:
        well_root = root / source_well
        (well_root / "real_well").mkdir(parents=True, exist_ok=True)
        if source_well in imaging_wells:
            (well_root / "imaging").mkdir(parents=True, exist_ok=True)
        for virtual_well in virtual_by_source[source_well]:
            (well_root / "virtual_wells" / virtual_well).mkdir(parents=True, exist_ok=True)


def partition_virtual_by_source(training_csv: Path, source_wells: set[str], stage_dir: Path, chunk_size: int) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for chunk_id, chunk in enumerate(pd.read_csv(training_csv, chunksize=chunk_size, encoding="utf-8-sig"), start=1):
        required = {"SourceWellName", "VirtualWellName", "PresenceLabel", "SampleWeight"}
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise RuntimeError(f"Step5A training table missing columns: {missing}")
        unknown = sorted(set(chunk["SourceWellName"].astype(str)) - source_wells)
        if unknown:
            raise RuntimeError(f"Step5A training table contains unknown source wells: {unknown[:10]}")
        for source_well, group in chunk.groupby("SourceWellName", sort=False, dropna=False):
            source_well = str(source_well)
            out_path = stage_dir / f"{source_well}.csv"
            safe_append_csv(group, out_path)
        total_rows += int(len(chunk))
        if chunk_id == 1 or chunk_id % 5 == 0:
            print(f"[reorg] virtual_partition_chunk={chunk_id} rows={total_rows}", flush=True)
    print(f"[reorg] virtual_partition_complete rows={total_rows}", flush=True)


def split_virtual_by_track(
    stage_dir: Path,
    root: Path,
    source_wells: list[str],
    virtual_by_source: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], int]:
    stats: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for well_index, source_well in enumerate(source_wells, start=1):
        source_path = stage_dir / f"{source_well}.csv"
        if not source_path.exists():
            raise RuntimeError(f"Missing virtual partition for source well: {source_well}")
        frame = pd.read_csv(source_path, encoding="utf-8-sig")
        if frame["SourceWellName"].astype(str).nunique() != 1:
            raise RuntimeError(f"Virtual partition mixes source wells: {source_path}")
        expected_tracks = set(virtual_by_source[source_well])
        actual_tracks = set(frame["VirtualWellName"].astype(str))
        missing_tracks = sorted(expected_tracks - actual_tracks)
        unexpected_tracks = sorted(actual_tracks - expected_tracks)
        if missing_tracks or unexpected_tracks:
            raise RuntimeError(
                f"Virtual track mismatch for {source_well}: missing={missing_tracks[:5]} unexpected={unexpected_tracks[:5]}"
            )
        for virtual_well, group in frame.groupby("VirtualWellName", sort=False, dropna=False):
            virtual_well = str(virtual_well)
            out_path = root / source_well / "virtual_wells" / virtual_well / "virtual_well_training_samples.csv"
            group.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n")
            presence = pd.to_numeric(group["PresenceLabel"], errors="coerce")
            weight = pd.to_numeric(group["SampleWeight"], errors="coerce")
            stats[virtual_well] = {
                "source_well": source_well,
                "virtual_well": virtual_well,
                "rows": int(len(group)),
                "positive_rows": int(presence.fillna(0).gt(0).sum()),
                "weight_sum": float(weight.fillna(0).sum()),
                "curvature_max_non_null": int(pd.to_numeric(group.get("CurvatureMax"), errors="coerce").notna().sum())
                if "CurvatureMax" in group.columns
                else 0,
            }
            total_rows += int(len(group))
        if well_index == 1 or well_index % 5 == 0 or well_index == len(source_wells):
            print(f"[reorg] virtual_split_wells={well_index}/{len(source_wells)} rows={total_rows}", flush=True)
    return stats, total_rows


def split_real_samples(unified_csv: Path, root: Path, source_wells: set[str], chunk_size: int) -> tuple[dict[str, dict[str, Any]], int]:
    stats: dict[str, dict[str, Any]] = {
        well: {"source_well": well, "rows": 0, "positive_rows": 0, "weight_sum": 0.0, "curvature_max_non_null": 0}
        for well in source_wells
    }
    total_rows = 0
    for chunk_id, chunk in enumerate(pd.read_csv(unified_csv, chunksize=chunk_size, encoding="utf-8-sig"), start=1):
        required = {"SourceKind", "SourceWellName", "PresenceLabel", "SampleWeight"}
        missing = sorted(required - set(chunk.columns))
        if missing:
            raise RuntimeError(f"Step5B unified table missing columns: {missing}")
        real = chunk[chunk["SourceKind"].astype(str).eq("real_well")].copy()
        if real.empty:
            continue
        unknown = sorted(set(real["SourceWellName"].astype(str)) - source_wells)
        if unknown:
            raise RuntimeError(f"Step5B real samples contain unknown source wells: {unknown[:10]}")
        for source_well, group in real.groupby("SourceWellName", sort=False, dropna=False):
            source_well = str(source_well)
            out_path = root / source_well / "real_well" / "real_well_samples.csv"
            safe_append_csv(group, out_path)
            presence = pd.to_numeric(group["PresenceLabel"], errors="coerce")
            weight = pd.to_numeric(group["SampleWeight"], errors="coerce")
            item = stats[source_well]
            item["rows"] += int(len(group))
            item["positive_rows"] += int(presence.fillna(0).gt(0).sum())
            item["weight_sum"] += float(weight.fillna(0).sum())
            if "CurvatureMax" in group.columns:
                item["curvature_max_non_null"] += int(pd.to_numeric(group["CurvatureMax"], errors="coerce").notna().sum())
        total_rows += int(len(real))
        if chunk_id == 1 or chunk_id % 5 == 0:
            print(f"[reorg] real_partition_chunk={chunk_id} rows={total_rows}", flush=True)
    print(f"[reorg] real_partition_complete rows={total_rows}", flush=True)
    return stats, total_rows


def copy_imaging_evidence(root: Path, imaging_config: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for item in imaging_config:
        well = str(item["well_name"])
        points = require_file(Path(str(item["fracture_points_csv"])), f"imaging points for {well}")
        group_paths = [require_file(Path(str(path)), f"imaging group for {well}") for path in item.get("group_csvs", [])]
        points_out = root / well / "imaging" / "imaging_fracture_points.csv"
        points_df = pd.read_csv(points, encoding="utf-8-sig")
        if "WellName" not in points_df.columns:
            points_df.insert(0, "WellName", well)
        points_df.to_csv(points_out, index=False, encoding="utf-8-sig", lineterminator="\n")
        frames: list[pd.DataFrame] = []
        for group_path in group_paths:
            frame = pd.read_csv(group_path, encoding="utf-8-sig")
            frame.insert(0, "SourceGroupFile", group_path.name)
            frames.append(frame)
        if frames:
            group_df = pd.concat(frames, ignore_index=True)
        else:
            group_df = pd.DataFrame()
        group_df.to_csv(root / well / "imaging" / "imaging_supervision_samples.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        summary[well] = {"fracture_points": int(len(points_df)), "supervision_samples": int(len(group_df))}
    return summary


def write_virtual_summaries(
    root: Path,
    index_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    virtual_stats: dict[str, dict[str, Any]],
    well_types: dict[str, str],
) -> None:
    merged = index_df.merge(
        summary_df,
        on=["SourceWellName", "VirtualWellName"],
        how="left",
        validate="one_to_one",
    )
    if merged["NumSamples"].isna().any():
        missing = merged.loc[merged["NumSamples"].isna(), "VirtualWellName"].astype(str).tolist()
        raise RuntimeError(f"Virtual well summary is missing tracks: {missing[:10]}")
    for row in merged.to_dict(orient="records"):
        source_well = str(row["SourceWellName"])
        virtual_well = str(row["VirtualWellName"])
        actual = virtual_stats[virtual_well]
        payload = {
            "SourceWellName": source_well,
            "VirtualWellName": virtual_well,
            "WellType": well_types[source_well],
            "GridOffsetX": int(row["GridOffsetX"]),
            "GridOffsetY": int(row["GridOffsetY"]),
            "SourceAnchorX": float(row["SourceAnchorX"]),
            "SourceAnchorY": float(row["SourceAnchorY"]),
            "VirtualAnchorX": float(row["VirtualAnchorX"]),
            "VirtualAnchorY": float(row["VirtualAnchorY"]),
            "OffsetX": float(row["OffsetX"]),
            "OffsetY": float(row["OffsetY"]),
            "DistanceToSource": float(row["DistanceToSource"]),
            "IsCenterVirtualTrace": int(row["IsCenterVirtualTrace"]),
            "NumSamples": int(row["NumSamples"]),
            "NumFractureSamples": int(row["NumFractureSamples"]) if pd.notna(row["NumFractureSamples"]) else 0,
            "MeanPositiveDensity": float(row["MeanPositiveDensity"]) if pd.notna(row["MeanPositiveDensity"]) else None,
            "MaxPositiveDensity": float(row["MaxPositiveDensity"]) if pd.notna(row["MaxPositiveDensity"]) else None,
            "MeanPointConfidence": float(row["MeanPointConfidence"]) if pd.notna(row["MeanPointConfidence"]) else None,
            "UniqueSourceWellCount": int(row["UniqueSourceWellCount"]),
            "SingleSourceCheck": str(row["SingleSourceCheck"]),
            "ActualRows": int(actual["rows"]),
            "ActualPositiveRows": int(actual["positive_rows"]),
            "ActualWeightSum": float(actual["weight_sum"]),
            "CurvatureMaxNonNull": int(actual["curvature_max_non_null"]),
            "TrainingFile": "virtual_well_training_samples.csv",
            "CandidateLevelOutputsAvailable": False,
        }
        write_json(root / source_well / "virtual_wells" / virtual_well / "virtual_well_summary.json", payload)


def write_well_summaries(
    root: Path,
    source_wells: list[str],
    index_df: pd.DataFrame,
    real_stats: dict[str, dict[str, Any]],
    virtual_stats: dict[str, dict[str, Any]],
    well_types: dict[str, str],
    imaging_summary: dict[str, dict[str, int]],
) -> None:
    for source_well in source_wells:
        tracks = index_df.loc[index_df["SourceWellName"].astype(str).eq(source_well), "VirtualWellName"].astype(str).tolist()
        vstats = [virtual_stats[track] for track in tracks]
        well_type = well_types[source_well]
        payload = {
            "SourceWellName": source_well,
            "WellType": well_type,
            "HasImagingSupervision": well_type == "imaging",
            "RealSampleCount": int(real_stats[source_well]["rows"]),
            "RealPositiveCount": int(real_stats[source_well]["positive_rows"]),
            "RealWeightSum": float(real_stats[source_well]["weight_sum"]),
            "RealCurvatureMaxNonNull": int(real_stats[source_well]["curvature_max_non_null"]),
            "VirtualTrackCount": int(len(tracks)),
            "VirtualSampleCount": int(sum(item["rows"] for item in vstats)),
            "VirtualPositiveCount": int(sum(item["positive_rows"] for item in vstats)),
            "VirtualWeightSum": float(sum(item["weight_sum"] for item in vstats)),
            "VirtualCurvatureMaxNonNull": int(sum(item["curvature_max_non_null"] for item in vstats)),
            "VirtualWellNames": tracks,
            "Imaging": imaging_summary.get(source_well),
            "CandidateLevelOutputsAvailable": False,
            "CandidateLevelOutputsNote": "Step5A write_intermediate_tables=false; only accepted training samples are available.",
            "Files": {
                "real_well": "real_well/real_well_samples.csv",
                "coverage_qc": "real_well/coverage_qc.csv",
                "virtual_wells": "virtual_wells/<VirtualWellName>/virtual_well_training_samples.csv",
            },
        }
        write_json(root / source_well / "well_summary.json", payload)
        qc = pd.DataFrame([{
            "SourceWellName": source_well,
            "WellType": well_type,
            "HasImagingSupervision": well_type == "imaging",
            "RealSampleCount": payload["RealSampleCount"],
            "RealPositiveCount": payload["RealPositiveCount"],
            "RealWeightSum": payload["RealWeightSum"],
            "VirtualTrackCount": payload["VirtualTrackCount"],
            "VirtualSampleCount": payload["VirtualSampleCount"],
            "VirtualPositiveCount": payload["VirtualPositiveCount"],
            "VirtualWeightSum": payload["VirtualWeightSum"],
            "CandidateLevelOutputsAvailable": False,
        }])
        qc.to_csv(root / source_well / "real_well" / "coverage_qc.csv", index=False, encoding="utf-8-sig", lineterminator="\n")


def write_root_manifest(
    root: Path,
    source_wells: list[str],
    index_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    virtual_stats: dict[str, dict[str, Any]],
    well_types: dict[str, str],
) -> None:
    merged = index_df.merge(summary_df, on=["SourceWellName", "VirtualWellName"], how="left", validate="one_to_one")
    rows = []
    for row in merged.to_dict(orient="records"):
        source_well = str(row["SourceWellName"])
        virtual_well = str(row["VirtualWellName"])
        actual = virtual_stats[virtual_well]
        rows.append({
            "SourceWellName": source_well,
            "WellType": well_types[source_well],
            "VirtualWellName": virtual_well,
            "VirtualTrainingSamplesFile": f"{source_well}/virtual_wells/{virtual_well}/virtual_well_training_samples.csv",
            "VirtualWellSummaryFile": f"{source_well}/virtual_wells/{virtual_well}/virtual_well_summary.json",
            "Rows": actual["rows"],
            "PositiveRows": actual["positive_rows"],
            "SampleWeightSum": actual["weight_sum"],
            "DistanceToSource": float(row["DistanceToSource"]),
            "IsCenterVirtualTrace": int(row["IsCenterVirtualTrace"]),
        })
    pd.DataFrame(rows).sort_values(["SourceWellName", "VirtualWellName"]).to_csv(
        root / "wells_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n"
    )


def write_readme(root: Path) -> None:
    text = """# 砂砾岩 Step5 逐井重组结果

本目录是现有 Step5A/5B 正式结果的逐井追溯视图，不替代顶层正式训练表。

每口源井包含：

- `real_well/real_well_samples.csv`: 来自 Step5B 的真实井样本。
- `real_well/coverage_qc.csv`: 该井真实/虚拟样本覆盖统计。
- `virtual_wells/<VirtualWellName>/virtual_well_training_samples.csv`: 来自 Step5A 的虚拟井训练样本。
- `virtual_wells/<VirtualWellName>/virtual_well_summary.json`: 虚拟井位置、距离和统计摘要。
- `well_summary.json`: 源井级汇总。
- `imaging/`: 仅成像井（车页1导眼、车151HF）保存成像裂缝点位和监督样本。

顶层 `wells_manifest.csv` 是 950 条虚拟井的索引和校验账目。

注意：当前 Step5A 未启用 `write_intermediate_tables=true`，因此没有保存全部候选样本和拒绝明细；本目录不生成伪造的 `all_virtual_candidates.csv` 或 `virtual_well_candidates.csv`。
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    output_dir = (args.output_dir or Path(str(config["output_dir"]))).resolve()
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force to replace it with a backup.")
        backup = output_dir.with_name(f"{output_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(output_dir), str(backup))
        print(f"[reorg] existing output moved to {backup}", flush=True)

    step5a_index = require_file(Path(str(config["step5a_index_csv"])), "Step5A virtual well index")
    step5a_training = require_file(Path(str(config["step5a_training_csv"])), "Step5A virtual training samples")
    step5a_summary = require_file(Path(str(config["step5a_summary_csv"])), "Step5A virtual well summary")
    step5a_audit = require_file(Path(str(config["step5a_audit_json"])), "Step5A build audit")
    step5b_unified = require_file(Path(str(config["step5b_unified_csv"])), "Step5B unified samples")
    step5b_summary = require_file(Path(str(config["step5b_summary_json"])), "Step5B summary")
    inputs = [step5a_index, step5a_training, step5a_summary, step5a_audit, step5b_unified, step5b_summary]
    fingerprints_before = {str(path): file_fingerprint(path) for path in inputs}

    index_df = pd.read_csv(step5a_index, encoding="utf-8-sig")
    summary_df = pd.read_csv(step5a_summary, encoding="utf-8-sig")
    audit = read_json(step5a_audit)
    unified_summary = read_json(step5b_summary)
    required_index = {"SourceWellName", "VirtualWellName", "VirtualAnchorX", "VirtualAnchorY", "DistanceToSource", "IsCenterVirtualTrace"}
    required_summary = {"SourceWellName", "VirtualWellName", "NumSamples", "NumFractureSamples", "MeanPointConfidence", "SingleSourceCheck"}
    if required_index - set(index_df.columns):
        raise RuntimeError(f"Step5A index missing columns: {sorted(required_index - set(index_df.columns))}")
    if required_summary - set(summary_df.columns):
        raise RuntimeError(f"Step5A summary missing columns: {sorted(required_summary - set(summary_df.columns))}")
    if index_df.duplicated("VirtualWellName").any():
        raise RuntimeError("Duplicate VirtualWellName in Step5A index")
    if summary_df.duplicated("VirtualWellName").any():
        raise RuntimeError("Duplicate VirtualWellName in Step5A summary")

    virtual_by_source: dict[str, list[str]] = {}
    for source_well, group in index_df.groupby("SourceWellName", sort=False):
        virtual_by_source[str(source_well)] = group["VirtualWellName"].astype(str).tolist()
    source_wells = sorted(virtual_by_source)
    track_counts = {well: len(tracks) for well, tracks in virtual_by_source.items()}
    bad_tracks = {well: count for well, count in track_counts.items() if count != 25}
    if bad_tracks:
        raise RuntimeError(f"Each source well must have exactly 25 virtual tracks: {bad_tracks}")

    imaging_config = list(config.get("imaging_wells", []))
    imaging_wells = {str(item["well_name"]) for item in imaging_config}
    well_types = {well: ("imaging" if well in imaging_wells else "regular") for well in source_wells}

    temp_root = Path(tempfile.mkdtemp(prefix=".step5b_wells_tmp_", dir=output_parent))
    stage_dir = temp_root / ".virtual_by_source"
    try:
        make_dirs(temp_root, source_wells, virtual_by_source, imaging_wells)
        partition_virtual_by_source(step5a_training, set(source_wells), stage_dir, int(config.get("chunk_size", 250000)))
        virtual_stats, virtual_rows = split_virtual_by_track(stage_dir, temp_root, source_wells, virtual_by_source)
        shutil.rmtree(stage_dir)
        real_stats, real_rows = split_real_samples(step5b_unified, temp_root, set(source_wells), int(config.get("chunk_size", 250000)))
        imaging_summary = copy_imaging_evidence(temp_root, imaging_config)
        write_virtual_summaries(temp_root, index_df, summary_df, virtual_stats, well_types)
        write_well_summaries(temp_root, source_wells, index_df, real_stats, virtual_stats, well_types, imaging_summary)
        write_root_manifest(temp_root, source_wells, index_df, summary_df, virtual_stats, well_types)
        write_readme(temp_root)

        expected_source_wells = int(audit.get("source_well_count", 0))
        expected_virtual_wells = int(audit.get("candidate_virtual_well_count", 0))
        expected_virtual_rows = int(audit.get("training_sample_count", 0))
        expected_virtual_positive = int(audit.get("accepted_positive_count", 0))
        source_kind = {row["SourceKind"]: row for row in unified_summary.get("summary", {}).get("source_kind", [])}
        expected_real_rows = int(source_kind.get("real_well", {}).get("rows", 0))
        expected_real_positive = int(source_kind.get("real_well", {}).get("presence_positive", 0))
        expected_virtual_rows_unified = int(source_kind.get("virtual_well", {}).get("rows", 0))
        expected_virtual_weight = float(source_kind.get("virtual_well", {}).get("sample_weight", 0.0))
        expected_real_weight = float(source_kind.get("real_well", {}).get("sample_weight", 0.0))
        actual_virtual_weight = sum(float(item["weight_sum"]) for item in virtual_stats.values())
        actual_virtual_positive = sum(int(item["positive_rows"]) for item in virtual_stats.values())
        actual_real_weight = sum(float(item["weight_sum"]) for item in real_stats.values())
        actual_real_positive = sum(int(item["positive_rows"]) for item in real_stats.values())
        checks = {
            "source_well_count_matches": len(source_wells) == expected_source_wells,
            "virtual_well_count_matches": len(virtual_stats) == expected_virtual_wells,
            "virtual_tracks_per_source_25": all(count == 25 for count in track_counts.values()),
            "virtual_rows_match_step5a": virtual_rows == expected_virtual_rows,
            "virtual_rows_match_step5b": virtual_rows == expected_virtual_rows_unified,
            "virtual_positive_match_step5a": actual_virtual_positive == expected_virtual_positive,
            "real_rows_match_step5b": real_rows == expected_real_rows,
            "real_positive_match_step5b": actual_real_positive == expected_real_positive,
            "virtual_weight_match_step5b": abs(actual_virtual_weight - expected_virtual_weight) <= 1.0e-6,
            "real_weight_match_step5b": abs(actual_real_weight - expected_real_weight) <= 1.0e-6,
            "every_source_has_real_file": all((temp_root / well / "real_well" / "real_well_samples.csv").exists() for well in source_wells),
            "every_virtual_has_training_file": all((temp_root / well / "virtual_wells" / track / "virtual_well_training_samples.csv").exists() for well in source_wells for track in virtual_by_source[well]),
            "single_source_mapping": all(
                pd.read_csv(temp_root / well / "virtual_wells" / track / "virtual_well_training_samples.csv", nrows=5, encoding="utf-8-sig")["SourceWellName"].astype(str).eq(well).all()
                for well in source_wells for track in virtual_by_source[well]
            ),
        }
        fingerprints_after = {str(path): file_fingerprint(path) for path in inputs}
        checks["input_files_unchanged"] = fingerprints_before == fingerprints_after
        status = "pass" if all(checks.values()) else "fail"
        root_summary = {
            "status": status,
            "output_dir": str(output_dir),
            "config_path": str(args.config.resolve()),
            "source_well_count": len(source_wells),
            "virtual_well_count": len(virtual_stats),
            "real_row_count": int(real_rows),
            "virtual_row_count": int(virtual_rows),
            "real_positive_count": int(actual_real_positive),
            "virtual_positive_count": int(actual_virtual_positive),
            "real_weight_sum": float(actual_real_weight),
            "virtual_weight_sum": float(actual_virtual_weight),
            "imaging_wells": imaging_summary,
            "candidate_level_outputs_available": False,
            "candidate_level_outputs_note": "Step5A formal output was built with write_intermediate_tables=false; candidate/rejection rows are not available for per-well reconstruction.",
            "input_fingerprints": fingerprints_before,
            "checks": {key: bool(value) for key, value in checks.items()},
        }
        write_json(temp_root / "wells_reorganization_summary.json", root_summary)
        if status != "pass":
            raise RuntimeError(f"Well reorganization validation failed; temporary output kept at {temp_root}: {root_summary['checks']}")
        os.replace(temp_root, output_dir)
        print(f"[reorg] output={output_dir}", flush=True)
        print(f"[reorg] status={status}", flush=True)
        return 0
    except Exception:
        print(f"[reorg] failed; temporary output kept at {temp_root}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


TARGET_COVERAGE = {"FULL_T4_T7", "PARTIAL_T4_T7"}
WELL_ALIAS_MAP = {
    "导眼井": "车页1导眼",
}


@dataclass(frozen=True)
class WellRecord:
    well_name: str
    canonical_well_name: str
    coverage_class: str
    source_kind: str
    has_track: bool
    xy_source: str
    target_time_min: str
    target_time_max: str
    target_depth_min: str
    target_depth_max: str
    sha3_rows: str
    sha4_rows: str
    in_target_rows: str
    partial_keep_rule: str
    audit_reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare T4-T7 real-well batch manifests from the existing coverage audit."
    )
    parser.add_argument("--config", required=True, help="Path to batch prep config JSON.")
    return parser


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def canonicalize_well_name(well_name: str) -> str:
    return WELL_ALIAS_MAP.get(str(well_name).strip(), str(well_name).strip())


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def read_audit_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_usable_wells(rows: list[dict]) -> list[WellRecord]:
    seen: set[str] = set()
    selected: list[WellRecord] = []
    for row in rows:
        if row.get("Status") != "ok":
            continue
        if row.get("CoverageClass") not in TARGET_COVERAGE:
            continue
        canonical_name = canonicalize_well_name(row.get("WellName", ""))
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        coverage_class = row["CoverageClass"]
        partial_keep_rule = (
            "keep_only_points_inside_t4_t7" if coverage_class == "PARTIAL_T4_T7" else "keep_full_t4_t7_interval"
        )
        selected.append(
            WellRecord(
                well_name=row["WellName"],
                canonical_well_name=canonical_name,
                coverage_class=coverage_class,
                source_kind=row.get("SourceKind", ""),
                has_track=parse_bool(row.get("HasTrack", "")),
                xy_source=row.get("XYSource", ""),
                target_time_min=row.get("TargetTimeMin", ""),
                target_time_max=row.get("TargetTimeMax", ""),
                target_depth_min=row.get("TargetDepthMin", ""),
                target_depth_max=row.get("TargetDepthMax", ""),
                sha3_rows=row.get("Sha3Rows", ""),
                sha4_rows=row.get("Sha4Rows", ""),
                in_target_rows=row.get("InTargetRows", ""),
                partial_keep_rule=partial_keep_rule,
                audit_reason=row.get("Reason", ""),
            )
        )
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest_rows(wells: list[WellRecord], output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for well in wells:
        well_dir = output_root / well.canonical_well_name
        rows.append(
            {
                "WellName": well.well_name,
                "CanonicalWellName": well.canonical_well_name,
                "CoverageClass": well.coverage_class,
                "SourceKind": well.source_kind,
                "HasTrack": "true" if well.has_track else "false",
                "XYSource": well.xy_source,
                "TargetTimeMin": well.target_time_min,
                "TargetTimeMax": well.target_time_max,
                "TargetDepthMin": well.target_depth_min,
                "TargetDepthMax": well.target_depth_max,
                "Sha3Rows": well.sha3_rows,
                "Sha4Rows": well.sha4_rows,
                "InTargetRows": well.in_target_rows,
                "PartialKeepRule": well.partial_keep_rule,
                "CoreTableCsv": str(well_dir / "real_well_core_t4_t7.csv"),
                "ContextTableCsv": str(well_dir / "real_well_context_3x3_t4_t7.csv"),
                "HorizonTableCsv": str(well_dir / "real_well_horizon_map_t4_t7.csv"),
                "PerWellSummaryJson": str(well_dir / "real_well_t4_t7_summary.json"),
                "AuditReason": well.audit_reason,
            }
        )
    return rows


def build_batch_config(config: dict, wells: list[WellRecord], output_root: Path) -> dict:
    return {
        "step_name": "t4_t7_real_well_formal_batch",
        "audit_csv": config["audit_csv"],
        "source_data": config["source_data"],
        "well_alias_map": WELL_ALIAS_MAP,
        "selection_rule": {
            "include_coverage_classes": sorted(TARGET_COVERAGE),
            "exclude_statuses": ["missing_las", "missing_timedepth"],
            "partial_rule": "retain only the actual T4-T7 interval points for PARTIAL_T4_T7 wells",
        },
        "output_root": str(output_root),
        "per_well_outputs": {
            "core_table": "real_well_core_t4_t7.csv",
            "context_table": "real_well_context_3x3_t4_t7.csv",
            "horizon_table": "real_well_horizon_map_t4_t7.csv",
            "summary_json": "real_well_t4_t7_summary.json",
        },
        "selected_wells": [well.canonical_well_name for well in wells],
    }


def build_summary_template_rows(wells: list[WellRecord]) -> list[dict]:
    rows: list[dict] = []
    for well in wells:
        rows.append(
            {
                "CanonicalWellName": well.canonical_well_name,
                "CoverageClass": well.coverage_class,
                "SourceKind": well.source_kind,
                "BatchStatus": "pending",
                "CoreRows": "",
                "ContextRows": "",
                "HorizonRows": "",
                "TargetPointRows": "",
                "RuntimeSeconds": "",
                "ValidationNote": "",
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)

    audit_csv = Path(config["audit_csv"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    manifest_dir = Path(config["manifest_dir"]).resolve()
    summary_dir = Path(config["summary_dir"]).resolve()

    rows = read_audit_rows(audit_csv)
    wells = select_usable_wells(rows)

    manifest_rows = build_manifest_rows(wells, output_root)
    manifest_csv = manifest_dir / "t4_t7_real_well_batch_manifest.csv"
    write_csv(manifest_csv, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else [])

    selected_wells_csv = manifest_dir / "t4_t7_real_well_selected_wells.csv"
    write_csv(
        selected_wells_csv,
        [
            {
                "WellName": well.well_name,
                "CanonicalWellName": well.canonical_well_name,
                "CoverageClass": well.coverage_class,
                "SourceKind": well.source_kind,
                "HasTrack": "true" if well.has_track else "false",
                "PartialKeepRule": well.partial_keep_rule,
            }
            for well in wells
        ],
        ["WellName", "CanonicalWellName", "CoverageClass", "SourceKind", "HasTrack", "PartialKeepRule"],
    )

    batch_config_json = manifest_dir / "t4_t7_real_well_batch_config.generated.json"
    write_json(batch_config_json, build_batch_config(config, wells, output_root))

    summary_template_csv = summary_dir / "t4_t7_real_well_batch_summary_template.csv"
    write_csv(
        summary_template_csv,
        build_summary_template_rows(wells),
        [
            "CanonicalWellName",
            "CoverageClass",
            "SourceKind",
            "BatchStatus",
            "CoreRows",
            "ContextRows",
            "HorizonRows",
            "TargetPointRows",
            "RuntimeSeconds",
            "ValidationNote",
        ],
    )

    summary_json = summary_dir / "t4_t7_real_well_batch_prep_summary.json"
    write_json(
        summary_json,
        {
            "audit_csv": str(audit_csv),
            "usable_well_count": len(wells),
            "full_well_count": sum(w.coverage_class == "FULL_T4_T7" for w in wells),
            "partial_well_count": sum(w.coverage_class == "PARTIAL_T4_T7" for w in wells),
            "vertical_well_count": sum(w.source_kind == "vertical" for w in wells),
            "deviated_well_count": sum(w.source_kind == "deviated" for w in wells),
            "track_well_count": sum(w.has_track for w in wells),
            "partial_wells": [w.canonical_well_name for w in wells if w.coverage_class == "PARTIAL_T4_T7"],
            "manifest_csv": str(manifest_csv),
            "selected_wells_csv": str(selected_wells_csv),
            "batch_config_json": str(batch_config_json),
            "summary_template_csv": str(summary_template_csv),
            "output_root": str(output_root),
        },
    )

    for well in wells:
        ensure_dir(output_root / well.canonical_well_name)


if __name__ == "__main__":
    main()

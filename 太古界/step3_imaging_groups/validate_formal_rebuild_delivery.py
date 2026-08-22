from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from build_formal_rebuild_groups import (
    CONTEXT_COLUMNS,
    GR_RESISTIVITY_PAIR_COLUMNS,
    GR_RESISTIVITY_TABLE_COLUMNS,
    GROUP_COLUMNS,
    INNER_SEGMENTS,
    MANUAL_STRATA,
    OUTER_SEGMENTS,
    STEP2_FORMAL_ROOT,
    STEP2_MAIN_COLUMNS,
    STEP2_OUTER_ROOT,
    STEP3_OUTPUT_ROOT,
    gr_resistivity_path,
    group_gr_resistivity_coverage,
)


FORBIDDEN_GROUP_COLUMNS = {"P10", "P21", "P33", "RAW_POINT_COUNT", "GT_LABEL"}
EXPECTED_GROUP_IDS = [f"{well}_{strata}" for well, strata, _, _ in MANUAL_STRATA]


def bool_status(value: bool) -> str:
    return "pass" if bool(value) else "fail"


def validate_group_files(groups_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for group_id in EXPECTED_GROUP_IDS:
        path = groups_dir / f"{group_id}.csv"
        if not path.exists():
            errors.append(f"missing group csv: {path}")
            rows.append({"GroupID": group_id, "Status": "missing"})
            continue

        df = pd.read_csv(path)
        columns = df.columns.tolist()
        forbidden = sorted(FORBIDDEN_GROUP_COLUMNS.intersection(columns))
        schema_ok = columns == GROUP_COLUMNS
        density = pd.to_numeric(df["Density"], errors="coerce") if "Density" in df else pd.Series(dtype=float)
        has_fracture = pd.to_numeric(df["HasFracture"], errors="coerce") if "HasFracture" in df else pd.Series(dtype=float)
        gt_point = pd.to_numeric(df["GT_POINT_FLAG"], errors="coerce") if "GT_POINT_FLAG" in df else pd.Series(dtype=float)
        usable = df["SampleUsableForModel"].astype(bool) if "SampleUsableForModel" in df else pd.Series(dtype=bool)

        has_fracture_ok = has_fracture.equals(density.fillna(0.0).gt(0.0).astype(int))
        gt_point_ok = set(gt_point.dropna().astype(int).unique().tolist()).issubset({0, 1})
        density_ok = len(df) > 0 and density.notna().all()
        usable_ok = len(df) > 0 and usable.all()
        no_forbidden_ok = not forbidden

        row = {
            "GroupID": group_id,
            "Path": str(path),
            "Status": "pass" if all([schema_ok, density_ok, has_fracture_ok, gt_point_ok, usable_ok, no_forbidden_ok]) else "fail",
            "RowCount": int(len(df)),
            "ColumnCount": int(len(columns)),
            "SchemaOK": bool_status(schema_ok),
            "DensityOK": bool_status(density_ok),
            "DensityPositiveRows": int(density.fillna(0.0).gt(0.0).sum()) if len(df) else 0,
            "HasFractureDefinitionOK": bool_status(has_fracture_ok),
            "GTPointFlagOK": bool_status(gt_point_ok),
            "GTPointRows": int(gt_point.fillna(0).astype(int).sum()) if len(df) else 0,
            "SampleUsableOK": bool_status(usable_ok),
            "ForbiddenColumns": ",".join(forbidden),
            "SeisAmpValidCountUnique": ",".join(map(str, sorted(pd.to_numeric(df["SeisAmpValidCount"], errors="coerce").dropna().astype(int).unique().tolist()))),
            "CoherenceNonNullRows": int(df["Coherence"].notna().sum()) if "Coherence" in df else 0,
            "AntTrackNonNullRows": int(df["AntTrack"].notna().sum()) if "AntTrack" in df else 0,
            "CurvatureMaxNonNullRows": int(df["CurvatureMax"].notna().sum()) if "CurvatureMax" in df else 0,
            "CurvaturePosNonNullRows": int(df["CurvaturePos"].notna().sum()) if "CurvaturePos" in df else 0,
        }
        rows.append(row)
        if row["Status"] != "pass":
            errors.append(f"group validation failed: {group_id}")
    return pd.DataFrame(rows), errors


def validate_step2_like_contract() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    reference_main = STEP2_FORMAL_ROOT / "车151HF/车151HF_t4_t7_real_well_main.csv"
    reference_context = STEP2_FORMAL_ROOT / "车151HF/车151HF_t4_t7_real_well_3x3_context.csv"
    ref_main_cols = pd.read_csv(reference_main, nrows=0).columns.tolist()
    ref_context_cols = pd.read_csv(reference_context, nrows=0).columns.tolist()

    for well_segment in OUTER_SEGMENTS:
        main_path = STEP2_OUTER_ROOT / well_segment / f"{well_segment}_t4_t7_real_well_main.csv"
        context_path = STEP2_OUTER_ROOT / well_segment / f"{well_segment}_t4_t7_real_well_3x3_context.csv"
        legacy_context_path = STEP2_OUTER_ROOT / well_segment / f"{well_segment}_t4_t7_real_well_seismic_window_context.csv"
        main_df = pd.read_csv(main_path)
        context_cols = pd.read_csv(context_path, nrows=0).columns.tolist()
        legacy_context_cols = pd.read_csv(legacy_context_path, nrows=0).columns.tolist() if legacy_context_path.exists() else []
        main_schema_ok = main_df.columns.tolist() == STEP2_MAIN_COLUMNS == ref_main_cols
        context_schema_ok = context_cols == CONTEXT_COLUMNS == ref_context_cols
        legacy_context_safe = legacy_context_cols == CONTEXT_COLUMNS
        seis_vc = pd.to_numeric(main_df["SeisAmpValidCount"], errors="coerce")
        row = {
            "WellSegment": well_segment,
            "MainPath": str(main_path),
            "ContextPath": str(context_path),
            "LegacyContextPath": str(legacy_context_path),
            "Status": "pass" if all([main_schema_ok, context_schema_ok, legacy_context_safe, seis_vc.eq(9).all()]) else "fail",
            "MainSchemaOK": bool_status(main_schema_ok),
            "ContextSchemaOK": bool_status(context_schema_ok),
            "LegacyContextSchemaSafe": bool_status(legacy_context_safe),
            "RowCount": int(len(main_df)),
            "SeisAmpValidCountUnique": ",".join(map(str, sorted(seis_vc.dropna().astype(int).unique().tolist()))),
            "CoherenceNonNullRows": int(main_df["Coherence"].notna().sum()),
            "AntTrackNonNullRows": int(main_df["AntTrack"].notna().sum()),
            "CurvatureMaxNonNullRows": int(main_df["CurvatureMax"].notna().sum()),
            "CurvaturePosNonNullRows": int(main_df["CurvaturePos"].notna().sum()),
        }
        rows.append(row)
        if row["Status"] != "pass":
            errors.append(f"outer step2-like contract failed: {well_segment}")
    return pd.DataFrame(rows), errors


def base_main_path(well_segment: str) -> Path:
    if well_segment in INNER_SEGMENTS:
        return Path(INNER_SEGMENTS[well_segment]["main"])
    return STEP2_OUTER_ROOT / well_segment / f"{well_segment}_t4_t7_real_well_main.csv"


def validate_gr_resistivity_contract(groups_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    manifest_path = STEP3_OUTPUT_ROOT / "sample_group_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    for well_segment, strata_name, _, _ in MANUAL_STRATA:
        group_id = f"{well_segment}_{strata_name}"
        group_path = groups_dir / f"{group_id}.csv"
        main_path = base_main_path(well_segment)
        enrichment_path = gr_resistivity_path(well_segment)
        if not group_path.exists() or not main_path.exists() or not enrichment_path.exists():
            errors.append(f"missing GR/resistivity contract input: {group_id}")
            rows.append({"GroupID": group_id, "Status": "missing"})
            continue
        group_df = pd.read_csv(group_path)
        main_df = pd.read_csv(main_path, usecols=["DEPT"])
        enrichment = pd.read_csv(enrichment_path)
        schema_ok = enrichment.columns.tolist() == GR_RESISTIVITY_TABLE_COLUMNS
        depth_ok = len(main_df) == len(enrichment) and pd.to_numeric(main_df["DEPT"], errors="coerce").equals(
            pd.to_numeric(enrichment["DEPT"], errors="coerce")
        )
        triple_ok = True
        for columns in GR_RESISTIVITY_PAIR_COLUMNS.values():
            count = enrichment[columns].notna().sum(axis=1) if set(columns).issubset(enrichment.columns) else pd.Series([-1])
            triple_ok = triple_ok and count.isin([0, len(columns)]).all()
        joined = group_df[["DEPT"]].merge(enrichment[["DEPT"]], on="DEPT", how="left", indicator=True)
        group_depth_ok = joined["_merge"].eq("both").all()
        actual = group_gr_resistivity_coverage(group_df, enrichment)
        manifest_row = manifest[manifest["GroupID"].astype(str).eq(group_id)] if "GroupID" in manifest else pd.DataFrame()
        manifest_ok = len(manifest_row) == 1 and all(
            int(manifest_row.iloc[0][column]) == int(value) for column, value in actual.items() if column in manifest_row.columns
        ) and all(column in manifest_row.columns for column in actual)
        checks = [schema_ok, depth_ok, triple_ok, group_depth_ok, manifest_ok]
        row = {
            "GroupID": group_id,
            "WellSegment": well_segment,
            "StrataName": strata_name,
            "MainPath": str(main_path),
            "GRResistivityPath": str(enrichment_path),
            "Status": "pass" if all(checks) else "fail",
            "SchemaOK": bool_status(schema_ok),
            "MainDepthContractOK": bool_status(depth_ok),
            "TripleCompletenessOK": bool_status(triple_ok),
            "GroupDepthLookupOK": bool_status(group_depth_ok),
            "ManifestCountsOK": bool_status(manifest_ok),
            **actual,
        }
        rows.append(row)
        if row["Status"] != "pass":
            errors.append(f"GR/resistivity validation failed: {group_id}")
    return pd.DataFrame(rows), errors


def write_contract_columns(output_root: Path) -> None:
    rows = []
    for idx, column in enumerate(STEP2_MAIN_COLUMNS):
        rows.append({"TableKind": "step2_like_main", "Order": idx + 1, "Column": column})
    for idx, column in enumerate(CONTEXT_COLUMNS):
        rows.append({"TableKind": "step2_like_3x3_context", "Order": idx + 1, "Column": column})
    for idx, column in enumerate(GR_RESISTIVITY_TABLE_COLUMNS):
        rows.append({"TableKind": "step2_gr_resistivity", "Order": idx + 1, "Column": column})
    for idx, column in enumerate(GROUP_COLUMNS):
        rows.append({"TableKind": "step3_group_csv", "Order": idx + 1, "Column": column})
    pd.DataFrame(rows).to_csv(output_root / "formal_data_contract_columns.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    output_root = STEP3_OUTPUT_ROOT
    groups_dir = output_root / "groups"
    output_root.mkdir(parents=True, exist_ok=True)

    group_qc, group_errors = validate_group_files(groups_dir)
    outer_qc, outer_errors = validate_step2_like_contract()
    gr_resistivity_qc, gr_resistivity_errors = validate_gr_resistivity_contract(groups_dir)
    write_contract_columns(output_root)

    group_qc.to_csv(output_root / "delivery_group_qc.csv", index=False, encoding="utf-8-sig")
    outer_qc.to_csv(output_root / "delivery_outer_step2_contract_qc.csv", index=False, encoding="utf-8-sig")
    gr_resistivity_qc.to_csv(output_root / "delivery_gr_resistivity_qc.csv", index=False, encoding="utf-8-sig")

    formal_rebuild_files = sorted(path.name for path in output_root.glob("*.csv"))
    summary = {
        "status": "pass" if not group_errors and not outer_errors and not gr_resistivity_errors else "fail",
        "expected_group_count": len(EXPECTED_GROUP_IDS),
        "actual_group_count": int(sum((groups_dir / f"{group_id}.csv").exists() for group_id in EXPECTED_GROUP_IDS)),
        "outer_step2_like_well_count": len(OUTER_SEGMENTS),
        "inner_formal_step2_wells": sorted(INNER_SEGMENTS.keys()),
        "group_errors": group_errors,
        "outer_step2_contract_errors": outer_errors,
        "gr_resistivity_errors": gr_resistivity_errors,
        "forbidden_group_columns": sorted(FORBIDDEN_GROUP_COLUMNS),
        "formal_rebuild_top_level_csvs": formal_rebuild_files,
        "no_redundant_total_training_table": "imaging_supervision_main.csv" not in formal_rebuild_files,
    }
    (output_root / "delivery_acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

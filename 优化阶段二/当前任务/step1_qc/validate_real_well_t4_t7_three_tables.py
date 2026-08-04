from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_CORE_COLUMNS = {
    "SampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
}

FORBIDDEN_CORE_COLUMNS = {
    "T4Time",
    "T5Time",
    "T6Time",
    "T7Time",
    "LayerGroup",
    "InTargetT4T7",
    "SurfaceOrderValid",
}

REQUIRED_CONTEXT_COLUMNS = {
    "SampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    "OffsetX",
    "OffsetY",
    "NeighborX",
    "NeighborY",
    "AttributeName",
    "AttributeValue",
}

REQUIRED_HORIZON_COLUMNS = {
    "SampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    "T4Time",
    "T5Time",
    "T6Time",
    "T7Time",
    "LayerGroup",
    "InTargetT4T7",
    "SurfaceOrderValid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real-well T4-T7 three-table demo outputs."
    )
    parser.add_argument("--demo-root", required=True, help="Directory containing per-well three-table outputs.")
    parser.add_argument("--summary-csv", required=True, help="Summary CSV for the demo batch.")
    parser.add_argument("--output-json", required=True, help="Validation report JSON path.")
    return parser.parse_args()


def validate_well(demo_root: Path, summary_row: pd.Series) -> dict:
    well = summary_row["WellName"]
    well_dir = demo_root / well
    core_path = well_dir / f"{well}_real_well_attribute_core.csv"
    context_path = well_dir / f"{well}_real_well_attribute_3x3_context.csv"
    horizon_path = well_dir / f"{well}_real_well_horizon_map.csv"

    result = {
        "WellName": well,
        "Status": summary_row["Status"],
        "SourceKind": summary_row.get("SourceKind"),
        "files_exist": core_path.exists() and context_path.exists() and horizon_path.exists(),
        "issues": [],
    }
    if not result["files_exist"]:
        result["issues"].append("missing_three_table_files")
        return result

    core = pd.read_csv(core_path)
    context = pd.read_csv(context_path)
    horizon = pd.read_csv(horizon_path)

    result["core_rows"] = int(len(core))
    result["context_rows"] = int(len(context))
    result["horizon_rows"] = int(len(horizon))
    result["core_columns"] = list(core.columns)
    result["context_columns"] = list(context.columns)
    result["horizon_columns"] = list(horizon.columns)

    missing_core = sorted(REQUIRED_CORE_COLUMNS - set(core.columns))
    missing_context = sorted(REQUIRED_CONTEXT_COLUMNS - set(context.columns))
    missing_horizon = sorted(REQUIRED_HORIZON_COLUMNS - set(horizon.columns))
    forbidden_core = sorted(FORBIDDEN_CORE_COLUMNS & set(core.columns))

    if missing_core:
        result["issues"].append(f"missing_core_columns:{','.join(missing_core)}")
    if missing_context:
        result["issues"].append(f"missing_context_columns:{','.join(missing_context)}")
    if missing_horizon:
        result["issues"].append(f"missing_horizon_columns:{','.join(missing_horizon)}")
    if forbidden_core:
        result["issues"].append(f"forbidden_core_columns:{','.join(forbidden_core)}")

    merged = core.merge(
        horizon[
            [
                "SampleID",
                "T4Time",
                "T5Time",
                "T6Time",
                "T7Time",
                "LayerGroup",
                "InTargetT4T7",
                "SurfaceOrderValid",
            ]
        ],
        on="SampleID",
        how="left",
        validate="one_to_one",
    )

    result["core_sample_ids_unique"] = bool(core["SampleID"].is_unique)
    result["horizon_sample_ids_unique"] = bool(horizon["SampleID"].is_unique)
    result["context_unique_sample_count"] = int(context["SampleID"].nunique())
    result["attribute_names"] = sorted(context["AttributeName"].dropna().unique().tolist())
    result["attribute_count"] = len(result["attribute_names"])
    result["offset_pairs"] = sorted(
        {(float(x), float(y)) for x, y in zip(context["OffsetX"], context["OffsetY"])}
    )
    result["offset_pair_count"] = len(result["offset_pairs"])

    grouped = (
        context.groupby("SampleID")
        .agg(
            row_count=("AttributeName", "size"),
            attr_count=("AttributeName", "nunique"),
            offset_count=("OffsetX", lambda s: len(set(zip(s, context.loc[s.index, "OffsetY"])))),
        )
        .reset_index()
    )
    expected_rows_per_sample = result["attribute_count"] * result["offset_pair_count"]
    result["expected_context_rows_per_sample"] = int(expected_rows_per_sample)
    result["context_rows_per_sample_unique"] = sorted(grouped["row_count"].unique().tolist())
    result["context_attr_count_per_sample_unique"] = sorted(grouped["attr_count"].unique().tolist())
    result["context_offset_count_per_sample_unique"] = sorted(grouped["offset_count"].unique().tolist())

    if not result["core_sample_ids_unique"]:
        result["issues"].append("duplicate_sampleid_in_core")
    if not result["horizon_sample_ids_unique"]:
        result["issues"].append("duplicate_sampleid_in_horizon")
    if result["context_unique_sample_count"] != len(core):
        result["issues"].append("context_sample_count_mismatch")
    if len(grouped["row_count"].unique()) != 1 or grouped["row_count"].iloc[0] != expected_rows_per_sample:
        result["issues"].append("context_rows_per_sample_mismatch")
    if len(grouped["attr_count"].unique()) != 1 or grouped["attr_count"].iloc[0] != result["attribute_count"]:
        result["issues"].append("context_attribute_count_mismatch")
    if len(grouped["offset_count"].unique()) != 1 or grouped["offset_count"].iloc[0] != result["offset_pair_count"]:
        result["issues"].append("context_offset_count_mismatch")

    result["time_ge_t4_all"] = bool((merged["TIME"] >= merged["T4Time"]).all())
    result["time_le_t7_all"] = bool((merged["TIME"] <= merged["T7Time"]).all())
    result["in_target_all"] = bool(merged["InTargetT4T7"].fillna(False).all())
    result["surface_order_all"] = bool(merged["SurfaceOrderValid"].fillna(False).all())
    result["layer_groups"] = merged["LayerGroup"].value_counts(dropna=False).to_dict()
    result["time_minus_t4_min"] = float((merged["TIME"] - merged["T4Time"]).min())
    result["t7_minus_time_min"] = float((merged["T7Time"] - merged["TIME"]).min())

    if not result["time_ge_t4_all"]:
        result["issues"].append("time_below_t4")
    if not result["time_le_t7_all"]:
        result["issues"].append("time_above_t7")
    if not result["in_target_all"]:
        result["issues"].append("contains_non_target_rows")
    if not result["surface_order_all"]:
        result["issues"].append("contains_invalid_surface_order")

    return result


def main() -> None:
    args = parse_args()
    demo_root = Path(args.demo_root).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    per_well = [validate_well(demo_root, row) for _, row in summary_df.iterrows()]
    report = {
        "demo_root": str(demo_root),
        "summary_csv": str(summary_csv),
        "well_count": len(per_well),
        "validated_ok_well_count": sum(1 for item in per_well if item["Status"] == "ok"),
        "wells": per_well,
    }
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()

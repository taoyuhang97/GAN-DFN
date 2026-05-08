from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LAYER_ORDER = [
    "TOP_1100MS->T1",
    "T1->T2",
    "T2->T3",
    "T3->T4",
    "T4->T5",
    "T5->T6",
    "T6->T7",
    "T7->BOTTOM_3800MS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize full_infer layer density control results across all units.")
    parser.add_argument("--units-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-summary-csv", type=Path)
    return parser.parse_args()


def sort_layers(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {name: idx for idx, name in enumerate(LAYER_ORDER)}
    work = df.copy()
    work["_layer_order"] = work["LayerSurfacePairKey"].map(order_map).fillna(9999)
    work = work.sort_values(["_layer_order", "LayerSurfacePairKey"], kind="stable").drop(columns=["_layer_order"])
    return work.reset_index(drop=True)


def load_density_control_frames(units_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(units_root.glob("*/unit_layer_density_control.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df["SourceCSV"] = str(csv_path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no unit_layer_density_control.csv found under: {units_root}")
    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("LayerSurfacePairKey", dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "UnitCount": int(g["UnitID"].nunique()),
                    "SegmentCount": int(len(g)),
                    "TotalThicknessMs": float(g["LayerThicknessMs"].sum()),
                    "TargetPatchCount": int(pd.to_numeric(g["TargetPatchCount"], errors="coerce").fillna(0).sum()),
                    "PredictedPatchCountAfter": int(pd.to_numeric(g["PredictedPatchCountAfter"], errors="coerce").fillna(0).sum()),
                    "TargetDensityWeighted": float(
                        pd.to_numeric(g["TargetPatchCount"], errors="coerce").fillna(0).sum()
                        / max(float(pd.to_numeric(g["LayerThicknessMs"], errors="coerce").fillna(0).sum()), 1e-9)
                    ),
                    "PredictedDensityWeighted": float(
                        pd.to_numeric(g["PredictedPatchCountAfter"], errors="coerce").fillna(0).sum()
                        / max(float(pd.to_numeric(g["LayerThicknessMs"], errors="coerce").fillna(0).sum()), 1e-9)
                    ),
                    "MeanTargetDensity": float(pd.to_numeric(g["TargetDensity"], errors="coerce").mean()),
                    "MeanPredictedDensityAfter": float(pd.to_numeric(g["PredictedDensityAfter"], errors="coerce").mean()),
                    "MedianPredictedDensityAfter": float(pd.to_numeric(g["PredictedDensityAfter"], errors="coerce").median()),
                    "MeanEffectiveDensityScale": float(pd.to_numeric(g["EffectiveDensityScale"], errors="coerce").mean()),
                    "ZeroOutputSegmentCount": int((pd.to_numeric(g["PredictedPatchCountAfter"], errors="coerce").fillna(0) <= 0).sum()),
                    "SuppressedSegmentCount": int(pd.to_numeric(g["SuppressedByActivationThreshold"], errors="coerce").fillna(0).astype(bool).sum()),
                }
            )
        )
        .reset_index()
    )
    grouped["DensityRatioPredOverTarget"] = grouped["PredictedDensityWeighted"] / grouped["TargetDensityWeighted"].replace(0.0, pd.NA)
    grouped["TargetMinusPredDensityWeighted"] = grouped["TargetDensityWeighted"] - grouped["PredictedDensityWeighted"]
    return sort_layers(grouped)


def main() -> None:
    args = parse_args()
    units_root = args.units_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_density_control_frames(units_root)
    summary_df = build_summary(raw_df)

    if args.prior_summary_csv is not None and args.prior_summary_csv.exists():
        prior_df = pd.read_csv(args.prior_summary_csv, encoding="utf-8-sig")
        keep_cols = [col for col in prior_df.columns if col in {"LayerSurfacePairKey", "RobustDensity", "AggregateDensity", "WeightedAggregateDensity"}]
        if keep_cols:
            summary_df = summary_df.merge(prior_df[keep_cols].drop_duplicates("LayerSurfacePairKey"), on="LayerSurfacePairKey", how="left")

    raw_csv = output_dir / "full_infer_layer_density_control_raw.csv"
    summary_csv = output_dir / "full_infer_layer_density_summary.csv"
    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"units_root={units_root}")
    print(f"unit_count={raw_df['UnitID'].nunique()}")
    print(f"segment_count={len(raw_df)}")
    print(f"raw_csv={raw_csv}")
    print(f"summary_csv={summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

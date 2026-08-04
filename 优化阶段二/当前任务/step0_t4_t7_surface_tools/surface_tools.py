from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SURFACE_CODES = ("T4", "T5", "T6", "T7")
UPDATED_SURFACE_HINTS = ("20240715", "DM_Sm", "Ato", "gljm", "di_zhen", "dizhen")
SURFACE_CODE_PATTERN = re.compile(r"^(T\d+)", flags=re.IGNORECASE)
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
DEFAULT_MIN_THICKNESS = 1.0


@dataclass(frozen=True)
class SurfaceFileChoice:
    surface_code: str
    surface_name: str
    surface_path: Path
    selection_reason: str


def read_csv_flexible(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to read csv: {path}") from last_error


def read_dat_surface(path: Path) -> pd.DataFrame:
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if len(rows) < 1:
        raise ValueError(f"surface file has no valid XYZ rows: {path}")
    return pd.DataFrame(rows, columns=["X", "Y", "Z"])


def _surface_name_score(surface_code: str, stem: str) -> tuple[int, int, int, str]:
    text = stem.lower()
    hint_count = sum(int(hint.lower() in text) for hint in UPDATED_SURFACE_HINTS)
    has_updated_hint = int(hint_count > 0)
    has_old_hint = int("old" in text or "jiu" in text or "legacy" in text)
    if surface_code == "T4":
        return (has_old_hint, -has_updated_hint, -hint_count, stem)
    return (has_updated_hint, hint_count, -has_old_hint, stem)


def choose_surface_files(layer_dir: Path) -> list[SurfaceFileChoice]:
    grouped: dict[str, list[Path]] = {code: [] for code in SURFACE_CODES}
    for path in sorted(layer_dir.glob("*.dat")):
        match = SURFACE_CODE_PATTERN.match(path.name)
        if match is None:
            continue
        code = match.group(1).upper()
        if code in grouped:
            grouped[code].append(path)

    choices: list[SurfaceFileChoice] = []
    for code in SURFACE_CODES:
        candidates = grouped.get(code, [])
        if not candidates:
            raise FileNotFoundError(f"missing required surface file for {code} under {layer_dir}")
        selected = sorted(candidates, key=lambda item: _surface_name_score(code, item.stem), reverse=True)[0]
        reason = "prefer_old_t4" if code == "T4" else "prefer_updated_t5_t7"
        choices.append(
            SurfaceFileChoice(
                surface_code=code,
                surface_name=selected.stem,
                surface_path=selected,
                selection_reason=reason,
            )
        )
    return choices


def load_surface_tables(layer_dir: Path) -> dict[str, dict[str, object]]:
    surfaces: dict[str, dict[str, object]] = {}
    for choice in choose_surface_files(layer_dir):
        table = read_dat_surface(choice.surface_path)
        surfaces[choice.surface_code] = {
            "choice": choice,
            "table": table,
        }
    return surfaces


def assign_surface_times(records_df: pd.DataFrame, surfaces: dict[str, dict[str, object]]) -> pd.DataFrame:
    work_df = records_df.copy()
    for required_col in ("X", "Y", "TIME"):
        if required_col not in work_df.columns:
            raise ValueError(f"records csv missing required column: {required_col}")
        work_df[required_col] = pd.to_numeric(work_df[required_col], errors="coerce")
    work_df = work_df.reset_index(drop=True)
    work_df["RecordID"] = np.arange(len(work_df), dtype=np.int64)

    valid_xy_mask = work_df["X"].notna() & work_df["Y"].notna()
    for code in SURFACE_CODES:
        work_df[f"{code}_TIME"] = np.nan
        work_df[f"{code}_NEAREST_X"] = np.nan
        work_df[f"{code}_NEAREST_Y"] = np.nan
        work_df[f"{code}_MANHATTAN_DISTANCE"] = np.nan
        if not valid_xy_mask.any():
            continue
        surface_df = surfaces[code]["table"]
        surface_xy = surface_df[["X", "Y"]].to_numpy(dtype=float)
        surface_z = surface_df["Z"].to_numpy(dtype=float)
        query_xy = work_df.loc[valid_xy_mask, ["X", "Y"]].to_numpy(dtype=float)
        tree = cKDTree(surface_xy)
        distances, nearest_idx = tree.query(query_xy, k=1, p=1)
        matched_rows = work_df.index[valid_xy_mask]
        work_df.loc[matched_rows, f"{code}_TIME"] = surface_z[nearest_idx]
        work_df.loc[matched_rows, f"{code}_NEAREST_X"] = surface_xy[nearest_idx, 0]
        work_df.loc[matched_rows, f"{code}_NEAREST_Y"] = surface_xy[nearest_idx, 1]
        work_df.loc[matched_rows, f"{code}_MANHATTAN_DISTANCE"] = distances
    return work_df


def classify_time_domain(records_df: pd.DataFrame) -> pd.DataFrame:
    work_df = records_df.copy()
    time_arr = pd.to_numeric(work_df["TIME"], errors="coerce").to_numpy(dtype=float)
    t4_arr = pd.to_numeric(work_df["T4_TIME"], errors="coerce").to_numpy(dtype=float)
    t6_arr = pd.to_numeric(work_df["T6_TIME"], errors="coerce").to_numpy(dtype=float)
    t7_arr = pd.to_numeric(work_df["T7_TIME"], errors="coerce").to_numpy(dtype=float)

    labels = np.full(len(work_df), "out_of_target", dtype=object)
    mask_sha3 = np.isfinite(time_arr) & np.isfinite(t4_arr) & np.isfinite(t6_arr) & (time_arr >= t4_arr) & (time_arr < t6_arr)
    mask_sha4 = np.isfinite(time_arr) & np.isfinite(t6_arr) & np.isfinite(t7_arr) & (time_arr >= t6_arr) & (time_arr < t7_arr)
    labels[mask_sha3] = "sha3_t4_t6"
    labels[mask_sha4] = "sha4_t6_t7"
    work_df["TimeDomainClass"] = labels
    return work_df


def validate_surface_order(records_df: pd.DataFrame, min_thickness: float) -> pd.DataFrame:
    work_df = records_df.copy()
    for code in SURFACE_CODES:
        work_df[f"{code}_TIME"] = pd.to_numeric(work_df[f"{code}_TIME"], errors="coerce")

    work_df["Thickness_T4_T5"] = work_df["T5_TIME"] - work_df["T4_TIME"]
    work_df["Thickness_T5_T6"] = work_df["T6_TIME"] - work_df["T5_TIME"]
    work_df["Thickness_T6_T7"] = work_df["T7_TIME"] - work_df["T6_TIME"]
    work_df["Thickness_T4_T6"] = work_df["T6_TIME"] - work_df["T4_TIME"]
    work_df["Thickness_T6_T7_Target"] = work_df["T7_TIME"] - work_df["T6_TIME"]

    work_df["Check_Order_T4_T5_T6_T7"] = (
        work_df["T4_TIME"].lt(work_df["T5_TIME"])
        & work_df["T5_TIME"].lt(work_df["T6_TIME"])
        & work_df["T6_TIME"].lt(work_df["T7_TIME"])
    )
    work_df["Check_MinThickness_T4_T5"] = work_df["Thickness_T4_T5"].ge(float(min_thickness))
    work_df["Check_MinThickness_T5_T6"] = work_df["Thickness_T5_T6"].ge(float(min_thickness))
    work_df["Check_MinThickness_T6_T7"] = work_df["Thickness_T6_T7"].ge(float(min_thickness))
    work_df["Check_All"] = (
        work_df["Check_Order_T4_T5_T6_T7"]
        & work_df["Check_MinThickness_T4_T5"]
        & work_df["Check_MinThickness_T5_T6"]
        & work_df["Check_MinThickness_T6_T7"]
    )
    return work_df


def build_summary(records_df: pd.DataFrame, surfaces: dict[str, dict[str, object]], records_path: Path, layer_dir: Path, encoding: str, min_thickness: float) -> dict[str, object]:
    class_counts = {
        str(key): int(value)
        for key, value in records_df["TimeDomainClass"].value_counts(dropna=False).sort_index().items()
    }
    selected_files = {}
    for code, payload in surfaces.items():
        choice: SurfaceFileChoice = payload["choice"]
        selected_files[code] = {
            "surface_name": choice.surface_name,
            "surface_file": str(choice.surface_path),
            "selection_reason": choice.selection_reason,
        }
    return {
        "records_csv": str(records_path),
        "records_encoding": encoding,
        "layer_dir": str(layer_dir),
        "selected_surface_files": selected_files,
        "num_records": int(len(records_df)),
        "num_valid_xy_records": int((records_df["X"].notna() & records_df["Y"].notna()).sum()),
        "num_all_checks_passed": int(records_df["Check_All"].fillna(False).sum()),
        "num_any_check_failed": int((~records_df["Check_All"].fillna(False)).sum()),
        "min_thickness": float(min_thickness),
        "time_domain_counts": class_counts,
    }


def run_pipeline(records_csv: Path, layer_dir: Path, output_csv: Path, summary_json: Path, min_thickness: float) -> dict[str, object]:
    records_df, encoding = read_csv_flexible(records_csv)
    surfaces = load_surface_tables(layer_dir)
    assigned_df = assign_surface_times(records_df=records_df, surfaces=surfaces)
    classified_df = classify_time_domain(assigned_df)
    validated_df = validate_surface_order(classified_df, min_thickness=min_thickness)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    validated_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = build_summary(
        records_df=validated_df,
        surfaces=surfaces,
        records_path=records_csv,
        layer_dir=layer_dir,
        encoding=encoding,
        min_thickness=min_thickness,
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign T4-T7 surface times to records using nearest Manhattan XY distance and classify time domain."
    )
    parser.add_argument("--records-csv", required=True, help="Input records csv with X,Y,TIME columns.")
    parser.add_argument("--layer-dir", required=True, help="Directory containing T4/T5/T6/T7 .dat surfaces.")
    parser.add_argument("--output-csv", required=True, help="Output csv path for enriched records.")
    parser.add_argument("--summary-json", required=True, help="Output summary json path.")
    parser.add_argument("--min-thickness", type=float, default=DEFAULT_MIN_THICKNESS, help="Minimum adjacent layer thickness in time domain.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_pipeline(
        records_csv=Path(args.records_csv),
        layer_dir=Path(args.layer_dir),
        output_csv=Path(args.output_csv),
        summary_json=Path(args.summary_json),
        min_thickness=float(args.min_thickness),
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

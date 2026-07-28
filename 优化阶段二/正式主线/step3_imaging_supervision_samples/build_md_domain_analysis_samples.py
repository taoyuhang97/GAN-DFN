"""Rebuild three imaging-well analysis datasets on an explicit MD axis."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd


DATA_ROOT = Path("/data/shared/project-oil/wx数据/砂砾岩")
CURRENT_DIR = Path(__file__).resolve().parent
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)


@dataclass(frozen=True)
class WellCase:
    well_name: str
    log_path: Path
    gr_curve: str
    deep_curve: str
    near_curve: str
    pair_type: str
    measurement_family: str
    density_path: Path
    density_kind: str
    density_column: int | None
    point_path: Path
    strata_intervals: tuple[tuple[str, float, float], ...]


CASES = (
    WellCase(
        well_name="车页1导眼",
        log_path=DATA_ROOT / "测井数据补充-20260723/LLD/车页1HF(导眼)@常规测井评价(2024-11-30)@1@综合.las",
        gr_curve="GR",
        deep_curve="LLD",
        near_curve="LLS",
        pair_type="LATERAL_LLD_LLS",
        measurement_family="lateral_resistivity",
        density_path=DATA_ROOT / "车镇成像测井/车页1HF（导眼）井-FMI/DLIS&LAS成果数据/车页1HF井裂缝密度、裂缝长度、裂缝孔隙度成果数据_3496.5-3755m.las",
        density_kind="fracture_density_las_p10",
        density_column=None,
        point_path=DATA_ROOT / "优化阶段一/研究内容一/成像测井/裂缝提取/车页1导眼_fractures.csv",
        strata_intervals=(("沙三段", 3500.0, 3734.0), ("沙四段", 3734.0, 3750.0)),
    ),
    WellCase(
        well_name="车662",
        log_path=DATA_ROOT / "成像测井-测井曲线/车662@常规测井评价(2006-08-19)@2.las",
        gr_curve="GR",
        deep_curve="RD",
        near_curve="RS",
        pair_type="LATERAL_RD_RS",
        measurement_family="lateral_resistivity",
        density_path=DATA_ROOT / "车镇成像测井/成像测井裂缝分布统计/che662-fracture-porosity.txt",
        density_kind="fracture_density_txt_fvdc",
        density_column=1,
        point_path=DATA_ROOT / "研究内容一/成像测井/裂缝标注/车662.xlsx",
        strata_intervals=(("沙三段", 3475.0, 3850.0), ("沙四段", 3850.0, 3973.0)),
    ),
    WellCase(
        well_name="车663",
        log_path=DATA_ROOT / "成像测井-测井曲线/车663@常规测井评价(2006-08-07)@1.las",
        gr_curve="GR",
        deep_curve="RD",
        near_curve="RS",
        pair_type="LATERAL_RD_RS",
        measurement_family="lateral_resistivity",
        density_path=DATA_ROOT / "车镇成像测井/成像测井裂缝分布统计/che663-fracture-porosity.txt",
        density_kind="fracture_density_txt_fvdc",
        density_column=3,
        point_path=DATA_ROOT / "研究内容一/成像测井/裂缝标注/车663.xlsx",
        strata_intervals=(("沙三段", 3879.0, 4220.0), ("沙四段", 4220.0, 4281.0)),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MD-domain imaging labels and same-pass GR/resistivity tables.")
    parser.add_argument("--output-dir", type=Path, default=CURRENT_DIR / "output/md_v2_analysis")
    parser.add_argument("--max-interpolation-gap-m", type=float, default=0.5)
    return parser.parse_args()


def clean_numeric(values: pd.Series, *, positive: bool = False) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    out = out.mask(out.abs() >= 9000.0)
    if positive:
        out = out.mask(out <= 0.0)
    return out


def read_las(path: Path) -> tuple[pd.DataFrame, object]:
    las = lasio.read(str(path), ignore_header_errors=True)
    frame = las.df().reset_index()
    frame = frame.rename(columns={frame.columns[0]: "MD"})
    frame.columns = [str(column).upper().strip() for column in frame.columns]
    frame["MD"] = pd.to_numeric(frame["MD"], errors="coerce")
    return frame.dropna(subset=["MD"]).sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True), las


def read_log_table(case: WellCase) -> pd.DataFrame:
    raw, _ = read_las(case.log_path)
    required = {case.gr_curve, case.deep_curve, case.near_curve}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"{case.well_name} log pass missing curves {sorted(missing)}: {case.log_path}")
    out = pd.DataFrame(
        {
            "WellName": case.well_name,
            "MD": raw["MD"],
            "GR": clean_numeric(raw[case.gr_curve]),
            "RDeep": clean_numeric(raw[case.deep_curve], positive=True),
            "RNear": clean_numeric(raw[case.near_curve], positive=True),
        }
    )
    out["GRSourceCurve"] = case.gr_curve
    out["DeepSourceCurve"] = case.deep_curve
    out["NearSourceCurve"] = case.near_curve
    out["PairType"] = case.pair_type
    out["MeasurementFamily"] = case.measurement_family
    out["LogSourcePath"] = str(case.log_path)
    return out


def read_density_las(path: Path) -> pd.DataFrame:
    frame, _ = read_las(path)
    p10_columns = [column for column in frame.columns if column.startswith("P10")]
    if not p10_columns:
        raise RuntimeError(f"no P10 fracture-density curve found: {path}")
    return pd.DataFrame({"MD": frame["MD"], "Density": clean_numeric(frame[p10_columns[0]])})


def read_density_txt(path: Path, density_column: int) -> pd.DataFrame:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            values = [float(item) for item in line.split()]
        except ValueError:
            continue
        if len(values) > density_column:
            rows.append(values)
    if not rows:
        raise RuntimeError(f"no numeric density rows found: {path}")
    out = pd.DataFrame({"MD": [row[0] for row in rows], "Density": [row[density_column] for row in rows]})
    out["Density"] = clean_numeric(out["Density"])
    return out.sort_values("MD").groupby("MD", as_index=False)["Density"].mean()


def read_density(case: WellCase) -> pd.DataFrame:
    if case.density_kind == "fracture_density_las_p10":
        return read_density_las(case.density_path)
    if case.density_column is None:
        raise RuntimeError(f"density column is required for {case.well_name}")
    return read_density_txt(case.density_path, case.density_column)


def find_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    for column in columns:
        normalized = str(column).lower().replace(" ", "")
        if any(candidate in normalized for candidate in candidates):
            return column
    raise RuntimeError(f"missing point column matching {candidates}: {columns}")


def read_points(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
    columns = [str(column) for column in raw.columns]
    md_col = find_column(columns, ("md",))
    dip_col = find_column(columns, ("dipangle", "angle(0~90)", "angle（０－９０）"))
    azimuth_col = find_column(columns, ("dipazimuth", "azimuth(0~360)", "azimuth（０－３６０）"))
    out = pd.DataFrame(
        {
            "MD": pd.to_numeric(raw[md_col], errors="coerce"),
            "FracDip": pd.to_numeric(raw[dip_col], errors="coerce"),
            "FracAzimuth": pd.to_numeric(raw[azimuth_col], errors="coerce"),
        }
    )
    return out.dropna(subset=["MD"]).sort_values("MD").reset_index(drop=True)


def assign_strata(md: pd.Series, intervals: tuple[tuple[str, float, float], ...]) -> pd.Series:
    out = pd.Series(pd.NA, index=md.index, dtype="string")
    for name, top, base in intervals:
        out.loc[md.ge(top) & md.lt(base)] = name
    return out


def interpolate_with_gap(source_md: pd.Series, source_value: pd.Series, target_md: pd.Series, max_gap_m: float) -> np.ndarray:
    x = pd.to_numeric(source_md, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(source_value, errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(target_md, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.full(len(target), np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate(([True], np.diff(x) > 1e-9))
    x = x[keep]
    y = y[keep]
    result = np.interp(target, x, y, left=np.nan, right=np.nan)
    right = np.searchsorted(x, target, side="left")
    left = np.clip(right - 1, 0, len(x) - 1)
    right = np.clip(right, 0, len(x) - 1)
    supported = (target >= x[0]) & (target <= x[-1]) & ((x[right] - x[left]) <= max_gap_m)
    result[~supported] = np.nan
    return result


def circular_mean_deg(values: list[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return np.nan
    angle = math.degrees(math.atan2(float(np.sin(np.deg2rad(arr)).mean()), float(np.cos(np.deg2rad(arr)).mean())))
    return angle + 360.0 if angle < 0 else angle


def attach_points(label_df: pd.DataFrame, points: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = label_df.copy()
    out["GT_POINT_FLAG"] = 0
    out["RawPointCount"] = 0
    out["FracDip"] = np.nan
    out["FracAzimuth"] = np.nan
    if out.empty or points.empty:
        return out, 0
    md = out["MD"].to_numpy(dtype=float)
    median_step = float(np.nanmedian(np.diff(md))) if len(md) > 1 else 1.0
    tolerance = max(0.1, median_step * 0.51)
    grouped: dict[int, list[tuple[float, float]]] = {}
    dropped = 0
    for row in points.itertuples(index=False):
        point_md = float(row.MD)
        pos = int(np.searchsorted(md, point_md))
        candidates = [idx for idx in (pos - 1, pos) if 0 <= idx < len(md)]
        idx = min(candidates, key=lambda item: abs(md[item] - point_md)) if candidates else -1
        if idx < 0 or abs(md[idx] - point_md) > tolerance:
            dropped += 1
            continue
        grouped.setdefault(idx, []).append((float(row.FracDip), float(row.FracAzimuth)))
    for idx, values in grouped.items():
        dips = [item[0] for item in values]
        azimuths = [item[1] for item in values]
        out.at[idx, "GT_POINT_FLAG"] = 1
        out.at[idx, "RawPointCount"] = len(values)
        out.at[idx, "FracDip"] = float(np.nanmean(dips))
        out.at[idx, "FracAzimuth"] = circular_mean_deg(azimuths)
    return out, dropped


def build_case(case: WellCase, output_dir: Path, max_gap_m: float) -> dict[str, Any]:
    logs = read_log_table(case)
    density = read_density(case)
    density["WellName"] = case.well_name
    density["StrataName"] = assign_strata(density["MD"], case.strata_intervals)
    density = density[density["StrataName"].notna()].copy().reset_index(drop=True)
    density["DensitySourceKind"] = case.density_kind
    density["DensitySourcePath"] = str(case.density_path)
    labels, dropped_points = attach_points(density, read_points(case.point_path))
    labels["PointSourcePath"] = str(case.point_path)
    labels["HasFractureDensity"] = labels["Density"].fillna(0.0).gt(0.0).astype(int)

    aligned = labels.copy()
    for column in ("GR", "RDeep", "RNear"):
        aligned[column] = interpolate_with_gap(logs["MD"], logs[column], aligned["MD"], max_gap_m)
    aligned["DEPT"] = aligned["MD"]
    aligned["GRSourceCurve"] = case.gr_curve
    aligned["DeepSourceCurve"] = case.deep_curve
    aligned["NearSourceCurve"] = case.near_curve
    aligned["PairType"] = case.pair_type
    aligned["MeasurementFamily"] = case.measurement_family
    aligned["LogSourcePath"] = str(case.log_path)
    aligned["CoordinateStatus"] = "md_domain_only"

    well_dir = output_dir / case.well_name
    well_dir.mkdir(parents=True, exist_ok=True)
    logs.to_csv(well_dir / f"{case.well_name}_md_logs.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(well_dir / f"{case.well_name}_md_imaging_labels.csv", index=False, encoding="utf-8-sig")
    aligned.to_csv(well_dir / f"{case.well_name}_md_analysis.csv", index=False, encoding="utf-8-sig")
    pair_valid = aligned["RDeep"].notna() & aligned["RNear"].notna()
    return {
        "WellName": case.well_name,
        "PairType": case.pair_type,
        "LogRows": int(len(logs)),
        "LabelRows": int(len(labels)),
        "DensityPositiveRows": int(labels["HasFractureDensity"].sum()),
        "RawPointRows": int(labels["RawPointCount"].sum()),
        "DroppedRawPointRows": int(dropped_points),
        "PairValidRows": int(pair_valid.sum()),
        "PairCoverage": float(pair_valid.mean()) if len(aligned) else 0.0,
        "MDMin": float(aligned["MD"].min()),
        "MDMax": float(aligned["MD"].max()),
        "LogSourcePath": str(case.log_path),
        "DensitySourcePath": str(case.density_path),
        "PointSourcePath": str(case.point_path),
        "AnalysisPath": str(well_dir / f"{case.well_name}_md_analysis.csv"),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [build_case(case, args.output_dir, args.max_interpolation_gap_m) for case in CASES]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_dir / "md_analysis_manifest.csv", index=False, encoding="utf-8-sig")
    contract = {
        "depth_axis": "MD",
        "dept_compatibility_alias": "DEPT=MD",
        "density_grid_policy": "preserve raw imaging-density MD samples inside configured strata intervals",
        "log_alignment_policy": f"linear interpolation with bracketing gap <= {args.max_interpolation_gap_m} m",
        "coordinate_policy": "MD-domain only; TVD/X/Y/TIME are not inferred in this product",
        "cases": [asdict(case) for case in CASES],
    }
    contract["cases"] = [
        {key: str(value) if isinstance(value, Path) else value for key, value in row.items()} for row in contract["cases"]
    ]
    (args.output_dir / "md_analysis_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/data/shared/project-oil/wx数据/砂砾岩")

STEP2_FORMAL_ROOT = REPO_ROOT / "优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_all_wells"
STEP2_OUTER_ROOT = REPO_ROOT / "优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_imaging_outer_wells"
STEP2_SCRIPT_ROOT = REPO_ROOT / "优化阶段二/正式主线/step2_real_well_t4_t7_samples"
STEP3_OUTPUT_ROOT = Path(__file__).resolve().parent / "output/formal_rebuild"
GROUP_OUTPUT_ROOT = STEP3_OUTPUT_ROOT / "groups"
if str(STEP2_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP2_SCRIPT_ROOT))

from build_gr_resistivity_samples import OUTPUT_COLUMNS as GR_RESISTIVITY_TABLE_COLUMNS  # noqa: E402
from build_gr_resistivity_samples import run_enrichment  # noqa: E402

AROUND_DIR = DATA_ROOT / "研究内容一/成像测井/测井-地震时窗"
AROUND_REBUILD_DIR = DATA_ROOT / "优化阶段一/研究内容一/成像测井/测井-地震时窗"
LOG_DIR = DATA_ROOT / "成像测井-测井曲线"
DENSITY_DIR = DATA_ROOT / "车镇成像测井/成像测井裂缝分布统计"
OUTER_POINT_DIR = DATA_ROOT / "研究内容一/成像测井/裂缝标注"
INNER_OLD_LABEL_DIR = DATA_ROOT / "优化阶段一/研究内容一/裂缝存在性预测/LSTM/地层划分验证/exp_strata_ac_gr_v1/labeled_datasets"
INNER_POINT_DIR = DATA_ROOT / "优化阶段一/研究内容一/成像测井/裂缝提取"

INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
STANDARD_LOG_COLUMNS = ["AC", "CAL", "CNL", "DEN", "GR", "RFOC", "RILD", "RILM", "SP"]
MAIN_ATTRIBUTE_COLUMNS = ["SeisAmp", "Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]
STAT_SUFFIXES = ["Mean", "Std", "Min", "Max", "ValidCount"]
GR_RESISTIVITY_PAIR_COLUMNS = {
    "LLD_LLS": ["GR_LLD_LLS", "LLD", "LLS"],
    "RD_RS": ["GR_RD_RS", "RD", "RS"],
    "RILD_RILM": ["GR_RILD_RILM", "RILD", "RILM"],
}
GRID_AXIS_LABELS = ("x0_y0", "x0_y1", "x0_y2", "x1_y0", "x1_y1", "x1_y2", "x2_y0", "x2_y1", "x2_y2")
CONTEXT_COLUMNS = [
    "SampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    *[f"{attr}_{label}" for attr in MAIN_ATTRIBUTE_COLUMNS for label in GRID_AXIS_LABELS],
]

GROUP_COLUMNS = [
    "SampleID",
    "WellSegment",
    "StrataName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
    *STANDARD_LOG_COLUMNS,
    *MAIN_ATTRIBUTE_COLUMNS,
    *[f"{attr}{suffix}" for attr in MAIN_ATTRIBUTE_COLUMNS for suffix in STAT_SUFFIXES],
    "Density",
    "HasFracture",
    "GT_POINT_FLAG",
    "Frac_Azimuth",
    "Frac_Dip",
    "SampleUsableForModel",
    "SampleUsableStatus",
]

STEP2_MAIN_COLUMNS = [
    "SampleID",
    "WellName",
    "X",
    "Y",
    "TIME",
    "TVD",
    "DEPT",
    *STANDARD_LOG_COLUMNS,
    *MAIN_ATTRIBUTE_COLUMNS,
    *[f"{attr}{suffix}" for attr in MAIN_ATTRIBUTE_COLUMNS for suffix in STAT_SUFFIXES],
    "SampleUsableForModel",
    "SampleUsableStatus",
]

LOG_SOURCE_CANDIDATES = {
    "AC": ["AC", "DT24"],
    "CAL": ["CAL", "CALI"],
    "CNL": ["CNL", "CNCF"],
    "DEN": ["DEN", "RHOB", "ZDEN"],
    "GR": ["GR", "GRAC", "GRCN", "GRSL", "GR1"],
    "RFOC": [],
    "RILD": [],
    "RILM": [],
    "SP": ["SP"],
}

OUTER_SEGMENTS = {
    "车660-1": {
        "around": AROUND_DIR / "车660_1_around_data.csv",
        "rebuild": AROUND_REBUILD_DIR / "车660_1_around_data_rebuild.csv",
        "las_prefix": "车660@",
        "density": DENSITY_DIR / "che660-1-fracture-porosity.txt",
        "density_col": 3,
        "points": OUTER_POINT_DIR / "车660_1.xlsx",
    },
    "车660-2": {
        "around": AROUND_DIR / "车660_2_around_data.csv",
        "rebuild": AROUND_REBUILD_DIR / "车660_2_around_data_rebuild.csv",
        "las_prefix": "车660@",
        "density": DENSITY_DIR / "che660-2-fracture-porosity.txt",
        "density_col": 7,
        "points": OUTER_POINT_DIR / "车660_2.xlsx",
    },
    "车662": {
        "around": AROUND_DIR / "车662_around_data.csv",
        "rebuild": AROUND_REBUILD_DIR / "车662_around_data_rebuild.csv",
        "las_prefix": "车662@",
        "density": DENSITY_DIR / "che662-fracture-porosity.txt",
        "density_col": 1,
        "points": OUTER_POINT_DIR / "车662.xlsx",
    },
    "车663": {
        "around": AROUND_DIR / "车663_around_data.csv",
        "rebuild": AROUND_REBUILD_DIR / "车663_around_data_rebuild.csv",
        "las_prefix": "车663@",
        "density": DENSITY_DIR / "che663-fracture-porosity.txt",
        "density_col": 3,
        "points": OUTER_POINT_DIR / "车663.xlsx",
    },
}

INNER_SEGMENTS = {
    "车151HF": {
        "main": STEP2_FORMAL_ROOT / "车151HF/车151HF_t4_t7_real_well_main.csv",
        "old_label": INNER_OLD_LABEL_DIR / "车151HF_sample_labeled.csv",
        "points": INNER_POINT_DIR / "车151HF_fractures.csv",
    },
    "车页1导眼": {
        "main": STEP2_FORMAL_ROOT / "车页1导眼/车页1导眼_t4_t7_real_well_main.csv",
        "old_label": INNER_OLD_LABEL_DIR / "车页1导眼_sample_labeled.csv",
        "points": INNER_POINT_DIR / "车页1导眼_fractures.csv",
    },
}

MANUAL_STRATA = [
    ("车151HF", "沙三段", 3655.0, 4935.0),
    ("车660-1", "沙三段", 3927.0, 4304.0),
    ("车660-2", "沙三段", 4299.0, 4321.0),
    ("车660-2", "沙四段", 4321.0, 4752.0),
    ("车662", "沙三段", 3475.0, 3850.0),
    ("车662", "沙四段", 3850.0, 3973.0),
    ("车663", "沙三段", 3879.0, 4220.0),
    ("车663", "沙四段", 4220.0, 4281.0),
    ("车页1导眼", "沙三段", 3500.0, 3734.0),
    ("车页1导眼", "沙四段", 3734.0, 3750.0),
]


@dataclass
class OuterBuildResult:
    well_segment: str
    main_path: Path
    context_path: Path
    source_kind: str
    source_path: Path
    row_count: int
    depth_min: float
    depth_max: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    out = out.mask(out.abs() >= 1e6)
    return out


def clean_coordinate(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def gr_resistivity_path(well_segment: str) -> Path:
    root = STEP2_FORMAL_ROOT if well_segment in INNER_SEGMENTS else STEP2_OUTER_ROOT
    return root / well_segment / f"{well_segment}_t4_t7_real_well_gr_resistivity.csv"


def load_aligned_gr_resistivity(base_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing GR/resistivity enrichment: {path}")
    enrichment = pd.read_csv(path)
    if enrichment.columns.tolist() != GR_RESISTIVITY_TABLE_COLUMNS:
        raise ValueError(f"unexpected GR/resistivity schema: {path}")
    base_depth = clean_numeric(base_df["DEPT"]).to_numpy(dtype=float)
    enrichment_depth = clean_numeric(enrichment["DEPT"]).to_numpy(dtype=float)
    if len(base_depth) != len(enrichment_depth) or not np.allclose(base_depth, enrichment_depth, atol=1.0e-9, rtol=0.0):
        raise ValueError(f"GR/resistivity DEPT does not match base main table: {path}")
    for columns in GR_RESISTIVITY_PAIR_COLUMNS.values():
        count = enrichment[columns].notna().sum(axis=1)
        if not count.isin([0, len(columns)]).all():
            raise ValueError(f"partial GR/resistivity triple found: {path} columns={columns}")
    return enrichment


def interpolate_series(source_depth: pd.Series, source_value: pd.Series, target_depth: pd.Series) -> pd.Series:
    work = pd.DataFrame({"DEPT": clean_numeric(source_depth), "VALUE": clean_numeric(source_value)}).dropna()
    work = work.groupby("DEPT", as_index=False)["VALUE"].mean().sort_values("DEPT")
    if len(work) < 2:
        return pd.Series(np.nan, index=target_depth.index)
    x = work["DEPT"].to_numpy(dtype=float)
    y = work["VALUE"].to_numpy(dtype=float)
    target = pd.to_numeric(target_depth, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.interp(target, x, y, left=np.nan, right=np.nan), index=target_depth.index)


def parse_las_table(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    columns: list[str] = []
    in_curve = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("~curve") or lower.startswith("~c"):
            in_curve = True
            continue
        if in_curve:
            if stripped.startswith("~") or lower.startswith("~parameter"):
                break
            if "." not in stripped:
                continue
            name = stripped.split(".", 1)[0].strip().upper()
            if not name or name == "#MNEM":
                continue
            columns.append("DEPT" if name in {"DEPT", "DEPTH", "MD", "#DEPTH"} else name)

    ascii_idx = next((idx for idx, line in enumerate(lines) if line.strip().lower().startswith("~ascii")), None)
    if ascii_idx is None or not columns:
        return pd.DataFrame()

    rows = []
    for line in lines[ascii_idx + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != len(columns):
            continue
        rows.append(parts)
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        df[col] = clean_numeric(df[col])
    if "DEPT" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["DEPT"]).sort_values("DEPT").reset_index(drop=True)


def load_las_tables(prefix: str) -> list[pd.DataFrame]:
    tables = []
    for path in sorted(LOG_DIR.glob(f"{prefix}*.las")):
        df = parse_las_table(path)
        if not df.empty:
            tables.append(df)
    return tables


def fill_log_curve_from_sources(target_df: pd.DataFrame, curve: str, las_tables: list[pd.DataFrame]) -> pd.Series:
    target_depth = target_df["DEPT"]
    candidates = LOG_SOURCE_CANDIDATES[curve]
    if not candidates:
        return pd.Series(np.nan, index=target_df.index)

    for col in candidates:
        if col in target_df.columns:
            existing = clean_numeric(target_df[col])
            if existing.notna().sum() >= max(2, int(len(existing) * 0.2)):
                return existing

    observations = []
    for las_df in las_tables:
        for col in candidates:
            if col not in las_df.columns:
                continue
            part = las_df[["DEPT", col]].rename(columns={col: "VALUE"}).copy()
            observations.append(part)
    if not observations:
        return pd.Series(np.nan, index=target_df.index)

    source = pd.concat(observations, ignore_index=True)
    return interpolate_series(source["DEPT"], source["VALUE"], target_depth)


def build_external_context_df(main_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    context_df = main_df[["SampleID", "WellName", "X", "Y", "TIME"]].copy()

    center_amp = clean_numeric(source_df["SEIS_TRUE"]) if "SEIS_TRUE" in source_df.columns else pd.Series(np.nan, index=source_df.index)
    for idx, label in enumerate(GRID_AXIS_LABELS):
        source_col = f"SEIS_{idx * 7 + 3}"
        context_df[f"SeisAmp_{label}"] = clean_numeric(source_df[source_col]) if source_col in source_df.columns else np.nan

    # Formal Step2 context uses the true X/Y/TIME sample at x1_y1, not a nearest-trace alias.
    context_df["SeisAmp_x1_y1"] = center_amp

    for attr in ["Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]:
        for label in GRID_AXIS_LABELS:
            context_df[f"{attr}_{label}"] = np.nan

    return context_df[CONTEXT_COLUMNS].copy()


def apply_context_stats(main_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    out = main_df.copy()
    for attr in MAIN_ATTRIBUTE_COLUMNS:
        context_cols = [f"{attr}_{label}" for label in GRID_AXIS_LABELS]
        values = context_df[context_cols].apply(clean_numeric)
        valid_count = values.notna().sum(axis=1)
        out[f"{attr}Mean"] = values.mean(axis=1, skipna=True)
        out[f"{attr}Std"] = values.std(axis=1, skipna=True, ddof=0)
        out[f"{attr}Min"] = values.min(axis=1, skipna=True)
        out[f"{attr}Max"] = values.max(axis=1, skipna=True)
        out[f"{attr}ValidCount"] = valid_count.astype(int)
    return out


def build_outer_step2_like_samples() -> list[OuterBuildResult]:
    ensure_dir(STEP2_OUTER_ROOT)
    results: list[OuterBuildResult] = []

    for well_segment, cfg in OUTER_SEGMENTS.items():
        source_path = cfg["rebuild"] if cfg["rebuild"].exists() else cfg["around"]
        source_kind = "rebuild_around_data" if cfg["rebuild"].exists() else "around_data_plus_las_logs"
        source_df = pd.read_csv(source_path)
        source_df = source_df.copy()
        source_df["DEPT"] = clean_numeric(source_df["TVD"])
        source_df["TVD"] = clean_numeric(source_df["TVD"])
        source_df["TIME"] = clean_numeric(source_df["TIME"])
        source_df["X"] = clean_coordinate(source_df["X"])
        source_df["Y"] = clean_coordinate(source_df["Y"])
        source_df = source_df.dropna(subset=["DEPT", "TVD", "TIME", "X", "Y"]).sort_values("DEPT").reset_index(drop=True)

        las_tables = load_las_tables(str(cfg["las_prefix"]))
        main_df = pd.DataFrame(index=source_df.index)
        main_df["SampleID"] = [f"{well_segment}_{idx}" for idx in range(len(source_df))]
        main_df["WellName"] = well_segment
        for col in ["X", "Y", "TIME", "TVD", "DEPT"]:
            main_df[col] = source_df[col]

        for curve in STANDARD_LOG_COLUMNS:
            main_df[curve] = fill_log_curve_from_sources(source_df, curve, las_tables)

        main_df["SeisAmp"] = clean_numeric(source_df["SEIS_TRUE"]) if "SEIS_TRUE" in source_df.columns else np.nan
        for attr in ["Coherence", "AntTrack", "CurvatureMax", "CurvaturePos"]:
            main_df[attr] = np.nan

        context_df = build_external_context_df(main_df, source_df)
        main_df = apply_context_stats(main_df, context_df)

        main_df["SampleUsableForModel"] = main_df["SeisAmp"].notna() & pd.to_numeric(main_df["SeisAmpValidCount"], errors="coerce").ge(5)
        main_df["SampleUsableStatus"] = np.where(
            main_df["SampleUsableForModel"],
            "usable_seisamp_3x3_missing_optional_volumes",
            "missing_center_seisamp",
        )
        main_df = main_df[STEP2_MAIN_COLUMNS].copy()

        out_dir = STEP2_OUTER_ROOT / well_segment
        ensure_dir(out_dir)
        main_path = out_dir / f"{well_segment}_t4_t7_real_well_main.csv"
        context_path = out_dir / f"{well_segment}_t4_t7_real_well_3x3_context.csv"
        legacy_context_path = out_dir / f"{well_segment}_t4_t7_real_well_seismic_window_context.csv"
        main_df.to_csv(main_path, index=False, encoding="utf-8-sig")

        context_df.to_csv(context_path, index=False, encoding="utf-8-sig")
        context_df.to_csv(legacy_context_path, index=False, encoding="utf-8-sig")

        results.append(
            OuterBuildResult(
                well_segment=well_segment,
                main_path=main_path,
                context_path=context_path,
                source_kind=source_kind,
                source_path=source_path,
                row_count=int(len(main_df)),
                depth_min=float(main_df["DEPT"].min()),
                depth_max=float(main_df["DEPT"].max()),
            )
        )

    pd.DataFrame([result.__dict__ for result in results]).to_csv(
        STEP2_OUTER_ROOT / "outer_imaging_step2_like_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return results


def read_density_table(path: Path, density_col: int) -> pd.DataFrame:
    rows = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.split()
        if len(parts) < 2:
            continue
        try:
            depth = float(parts[0])
        except ValueError:
            continue
        values = []
        for item in parts[1:]:
            try:
                value = float(item)
            except ValueError:
                value = np.nan
            values.append(value)
        if not values:
            continue
        rows.append([depth, *values])
    if not rows:
        raise ValueError(f"no numeric density rows found: {path}")

    max_width = max(len(row) for row in rows)
    normalized = [row + [np.nan] * (max_width - len(row)) for row in rows]
    raw_df = pd.DataFrame(normalized)
    raw_df = raw_df.rename(columns={0: "DEPTH"})
    for col in raw_df.columns:
        raw_df[col] = clean_numeric(raw_df[col])

    # Use FVDC only. PHIT/porosity-like columns are not fracture density labels.
    if density_col not in raw_df.columns:
        raise ValueError(f"FVDC density column {density_col} not found in {path}")
    out = raw_df[["DEPTH", density_col]].rename(columns={density_col: "Density"}).dropna(subset=["DEPTH"])
    out = out.sort_values("DEPTH").groupby("DEPTH", as_index=False)["Density"].mean()
    return out


def load_inner_density_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "DEPTH": clean_numeric(df["DEPT"] if "DEPT" in df.columns else df["TVD"]),
            "Density": clean_numeric(df["P10"]),
        }
    )
    return out.dropna(subset=["DEPTH"]).sort_values("DEPTH").groupby("DEPTH", as_index=False)["Density"].mean()


def attach_density(sample_df: pd.DataFrame, density_df: pd.DataFrame) -> pd.DataFrame:
    out = sample_df.copy()
    out["Density"] = interpolate_series(density_df["DEPTH"], density_df["Density"], out["DEPT"])
    return out


def circular_mean_deg(values: Iterable[float]) -> float:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan")
    radians = np.deg2rad(arr)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return float(np.mean(arr))
    angle = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)))
    return angle + 360.0 if angle < 0 else angle


def normalize_point_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        raw_df = pd.read_excel(path)
    else:
        raw_df = pd.read_csv(path)
    lower_map = {str(col).lower(): col for col in raw_df.columns}

    def find_col(candidates: list[str]) -> str:
        for col in raw_df.columns:
            text = str(col).lower()
            if any(candidate in text for candidate in candidates):
                return col
        raise ValueError(f"missing point column {candidates}: {path}")

    md_col = find_col(["md"])
    dip_col = find_col(["dip angle", "angle"])
    az_col = find_col(["dip azimuth", "azimuth"])
    out = pd.DataFrame(
        {
            "MD": clean_numeric(raw_df[md_col]),
            "Frac_Dip": clean_numeric(raw_df[dip_col]),
            "Frac_Azimuth": clean_numeric(raw_df[az_col]),
        }
    )
    return out.dropna(subset=["MD"]).sort_values("MD").drop_duplicates(subset=["MD"], keep="first").reset_index(drop=True)


def attach_point_labels(sample_df: pd.DataFrame, point_df: pd.DataFrame) -> pd.DataFrame:
    out = sample_df.copy().sort_values("DEPT").reset_index(drop=True)
    out["GT_POINT_FLAG"] = 0
    out["Frac_Azimuth"] = np.nan
    out["Frac_Dip"] = np.nan

    sample_depth = pd.to_numeric(out["DEPT"], errors="coerce").to_numpy(dtype=float)
    if out.empty or point_df.empty or not np.isfinite(sample_depth).any():
        return out
    min_depth = float(np.nanmin(sample_depth))
    max_depth = float(np.nanmax(sample_depth))
    mapped: dict[int, list[tuple[float, float]]] = {}
    for row in point_df.itertuples(index=False):
        point_depth = finite_or_nan(getattr(row, "MD"))
        if not math.isfinite(point_depth) or point_depth < min_depth or point_depth > max_depth:
            continue
        insert_idx = int(np.searchsorted(sample_depth, point_depth, side="left"))
        if insert_idx <= 0:
            nearest_idx = 0
        elif insert_idx >= len(sample_depth):
            nearest_idx = len(sample_depth) - 1
        else:
            left_idx = insert_idx - 1
            right_idx = insert_idx
            nearest_idx = left_idx if abs(sample_depth[left_idx] - point_depth) <= abs(sample_depth[right_idx] - point_depth) else right_idx
        mapped.setdefault(nearest_idx, []).append((finite_or_nan(getattr(row, "Frac_Azimuth")), finite_or_nan(getattr(row, "Frac_Dip"))))

    for idx, values in mapped.items():
        azimuth_values = [item[0] for item in values if math.isfinite(item[0])]
        dip_values = [item[1] for item in values if math.isfinite(item[1])]
        out.at[idx, "GT_POINT_FLAG"] = 1
        out.at[idx, "Frac_Azimuth"] = circular_mean_deg(azimuth_values)
        out.at[idx, "Frac_Dip"] = float(np.mean(dip_values)) if dip_values else np.nan
    return out


def load_base_samples(
    outer_results: list[OuterBuildResult],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[dict[str, object]]]:
    samples: dict[str, pd.DataFrame] = {}
    gr_resistivity_tables: dict[str, pd.DataFrame] = {}
    source_rows: list[dict[str, object]] = []

    for well_segment, cfg in INNER_SEGMENTS.items():
        df = pd.read_csv(cfg["main"])
        density = load_inner_density_source(cfg["old_label"])
        points = normalize_point_table(cfg["points"])
        df = attach_density(df, density)
        df = attach_point_labels(df, points)
        df["HasFracture"] = clean_numeric(df["Density"]).fillna(0.0).gt(0.0).astype(int)
        samples[well_segment] = df
        enrichment_path = gr_resistivity_path(well_segment)
        gr_resistivity_tables[well_segment] = load_aligned_gr_resistivity(df, enrichment_path)
        source_rows.append(
            {
                "WellSegment": well_segment,
                "BaseKind": "formal_step2_real_well_main",
                "BasePath": str(cfg["main"]),
                "DensityKind": "old_labeled_sample_P10_only",
                "DensityPath": str(cfg["old_label"]),
                "PointPath": str(cfg["points"]),
                "GRResistivityPath": str(enrichment_path),
                "GRResistivitySourceKind": "formal_step2_enrichment",
            }
        )

    outer_by_segment = {result.well_segment: result for result in outer_results}
    for well_segment, cfg in OUTER_SEGMENTS.items():
        result = outer_by_segment[well_segment]
        df = pd.read_csv(result.main_path)
        density = read_density_table(cfg["density"], int(cfg["density_col"]))
        points = normalize_point_table(cfg["points"])
        df = attach_density(df, density)
        df = attach_point_labels(df, points)
        df["HasFracture"] = clean_numeric(df["Density"]).fillna(0.0).gt(0.0).astype(int)
        samples[well_segment] = df
        enrichment_path = gr_resistivity_path(well_segment)
        gr_resistivity_tables[well_segment] = load_aligned_gr_resistivity(df, enrichment_path)
        source_rows.append(
            {
                "WellSegment": well_segment,
                "BaseKind": result.source_kind,
                "BasePath": str(result.main_path),
                "DensityKind": "fracture_density_txt_FVDC",
                "DensityPath": str(cfg["density"]),
                "PointPath": str(cfg["points"]),
                "GRResistivityPath": str(enrichment_path),
                "GRResistivitySourceKind": "outer_step2_enrichment",
            }
        )
    return samples, gr_resistivity_tables, source_rows


def group_gr_resistivity_coverage(group_df: pd.DataFrame, enrichment: pd.DataFrame) -> dict[str, int]:
    joined = group_df[["DEPT", "HasFracture"]].merge(
        enrichment,
        on="DEPT",
        how="left",
        validate="many_to_one",
    )
    result: dict[str, int] = {}
    any_pair = pd.Series(False, index=joined.index)
    for pair_key, columns in GR_RESISTIVITY_PAIR_COLUMNS.items():
        valid = joined[columns].notna().all(axis=1)
        result[f"{pair_key}_ValidRows"] = int(valid.sum())
        any_pair |= valid
    result["GRResistivityValidRows"] = int(any_pair.sum())
    result["GRResistivityPositiveRows"] = int((any_pair & joined["HasFracture"].gt(0)).sum())
    return result


def write_group_samples(samples: dict[str, pd.DataFrame], gr_resistivity_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ensure_dir(GROUP_OUTPUT_ROOT)
    manifest_rows = []
    for well_segment, strata_name, top_depth, base_depth in MANUAL_STRATA:
        df = samples[well_segment].copy()
        depth = clean_numeric(df["DEPT"])
        group_df = df[depth.ge(top_depth) & depth.lt(base_depth)].copy().reset_index(drop=True)
        group_df["WellSegment"] = well_segment
        group_df["StrataName"] = strata_name
        group_df["SampleID"] = [f"{well_segment}_{strata_name}_{idx}" for idx in range(len(group_df))]

        group_df["Density"] = clean_numeric(group_df["Density"])
        group_df = group_df[group_df["Density"].notna()].copy().reset_index(drop=True)
        group_df["HasFracture"] = group_df["Density"].fillna(0.0).gt(0.0).astype(int)
        group_df["GT_POINT_FLAG"] = clean_numeric(group_df["GT_POINT_FLAG"]).fillna(0).astype(int)
        group_df["SampleUsableForModel"] = group_df["Density"].notna() & group_df["SeisAmp"].notna()
        group_df["SampleUsableStatus"] = np.where(group_df["SampleUsableForModel"], "usable", "missing_density_or_seisamp")

        for col in GROUP_COLUMNS:
            if col not in group_df.columns:
                group_df[col] = np.nan
        group_df = group_df[GROUP_COLUMNS].copy()
        gr_coverage = group_gr_resistivity_coverage(group_df, gr_resistivity_tables[well_segment])

        group_id = f"{well_segment}_{strata_name}"
        out_path = GROUP_OUTPUT_ROOT / f"{group_id}.csv"
        group_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        manifest_rows.append(
            {
                "GroupID": group_id,
                "WellSegment": well_segment,
                "StrataName": strata_name,
                "TopDEPT": top_depth,
                "BaseDEPT": base_depth,
                "Path": str(out_path),
                "RowCount": int(len(group_df)),
                "DensityNonNullRows": int(group_df["Density"].notna().sum()),
                "DensityPositiveRows": int(group_df["HasFracture"].sum()),
                "GTPointRows": int(group_df["GT_POINT_FLAG"].sum()),
                "SeisAmpNonNullRows": int(group_df["SeisAmp"].notna().sum()),
                "CoherenceNonNullRows": int(group_df["Coherence"].notna().sum()),
                "AntTrackNonNullRows": int(group_df["AntTrack"].notna().sum()),
                "CurvatureMaxNonNullRows": int(group_df["CurvatureMax"].notna().sum()),
                "CurvaturePosNonNullRows": int(group_df["CurvaturePos"].notna().sum()),
                "AllStandardLogNonNullRows": int(group_df[STANDARD_LOG_COLUMNS].notna().all(axis=1).sum()),
                **gr_coverage,
            }
        )
    return pd.DataFrame(manifest_rows)


def main() -> None:
    ensure_dir(STEP3_OUTPUT_ROOT)
    outer_results = build_outer_step2_like_samples()
    outer_enrichment_summary, _ = run_enrichment(
        output_root=STEP2_OUTER_ROOT,
        selected_wells=set(OUTER_SEGMENTS),
    )
    if outer_enrichment_summary["Status"].eq("failed").any():
        raise RuntimeError("outer imaging GR/resistivity enrichment failed")
    samples, gr_resistivity_tables, source_rows = load_base_samples(outer_results)

    group_manifest = write_group_samples(samples, gr_resistivity_tables)
    group_manifest.to_csv(STEP3_OUTPUT_ROOT / "sample_group_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(source_rows).to_csv(STEP3_OUTPUT_ROOT / "source_manifest.csv", index=False, encoding="utf-8-sig")

    print(f"outer_step2_like_samples={len(outer_results)}")
    print(f"group_samples={len(group_manifest)}")
    print(f"output_root={STEP3_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()

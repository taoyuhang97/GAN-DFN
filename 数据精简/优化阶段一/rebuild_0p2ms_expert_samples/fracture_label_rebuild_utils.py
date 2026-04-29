from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")
P_COLS = ["P10", "P21", "P33"]
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)


@dataclass
class SampleBuildStats:
    well_name: str
    input_row_count: int
    filtered_row_count: int
    gt_label_count: int
    gt_point_count: int
    mapped_raw_point_count: int
    dropped_raw_point_count: int
    density_source_kind: str


def read_csv_flexible(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read csv: {csv_path}") from last_error


def read_fracture_density_las(las_path: Path) -> pd.DataFrame:
    ascii_start = None
    with las_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for idx, line in enumerate(file_obj):
            if line.strip().lower().startswith("~ascii"):
                ascii_start = idx + 1
                break

    if ascii_start is None:
        raise ValueError(f"~Ascii section not found in LAS: {las_path}")

    df = pd.read_csv(
        las_path,
        sep=r"\s+",
        engine="python",
        skiprows=ascii_start,
        names=["DEPTH", "P10", "P21", "P33"],
        na_values=["-999.25", "-9999", "-99999", "nan"],
    )
    return df


def excel_ref_to_col_idx(cell_ref: str) -> int:
    col = 0
    for ch in str(cell_ref):
        if not ch.isalpha():
            break
        col = (col * 26) + (ord(ch.upper()) - ord("A") + 1)
    return max(col - 1, 0)


def load_shared_strings(xlsx_file: ZipFile) -> list[str]:
    try:
        xml_bytes = xlsx_file.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml_bytes)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    values = []
    for item in root.findall(f"{ns}si"):
        text_parts = []
        for text_node in item.iter(f"{ns}t"):
            text_parts.append(text_node.text or "")
        values.append("".join(text_parts))
    return values


def read_xlsx_cell_value(cell, shared_strings: list[str]) -> str:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        text_node = cell.find(f"{ns}is/{ns}t")
        return "" if text_node is None else str(text_node.text or "")
    value_node = cell.find(f"{ns}v")
    if value_node is None:
        return ""
    raw_text = "" if value_node.text is None else str(value_node.text)
    if cell_type == "s":
        try:
            idx = int(float(raw_text))
        except ValueError:
            return raw_text
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    return raw_text


def read_simple_xlsx(path: Path, names: list[str]) -> pd.DataFrame:
    with ZipFile(path) as xlsx_file:
        shared_strings = load_shared_strings(xlsx_file)
        sheet_xml = xlsx_file.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows = []
    for row_node in root.iter(f"{ns}row"):
        current_row = {}
        for cell in row_node.findall(f"{ns}c"):
            col_idx = excel_ref_to_col_idx(cell.attrib.get("r", ""))
            current_row[col_idx] = read_xlsx_cell_value(cell, shared_strings)
        if current_row:
            width = max(current_row.keys()) + 1
            rows.append([current_row.get(idx, "") for idx in range(width)])
    if not rows:
        return pd.DataFrame(columns=names)
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    df = pd.DataFrame(normalized)
    if df.shape[1] < len(names):
        for _ in range(len(names) - df.shape[1]):
            df[df.shape[1]] = ""
    df = df.iloc[:, : len(names)].copy()
    df.columns = names
    if not df.empty:
        first_value = pd.to_numeric(df[names[0]], errors="coerce").iloc[0]
        if not np.isfinite(first_value):
            df = df.iloc[1:].reset_index(drop=True)
    return df


def load_raw_point_table(raw_path: Path) -> pd.DataFrame:
    suffix = raw_path.suffix.lower()
    if suffix == ".csv":
        raw_df = read_csv_flexible(raw_path).copy()
    elif suffix == ".xlsx":
        raw_df = read_simple_xlsx(raw_path, ["MD", "Angle(0~90)", "Azimuth(0~360)"]).copy()
    else:
        raise ValueError(f"Unsupported raw point file: {raw_path}")

    rename_map = {
        "Angle(0~90)": "Frac_Dip",
        "Azimuth(0~360)": "Frac_Azimuth",
    }
    raw_df = raw_df.rename(columns=rename_map)
    required_cols = ["MD", "Frac_Dip", "Frac_Azimuth"]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Raw point file missing columns {missing_cols}: {raw_path}")

    out = raw_df[required_cols].copy()
    for col in required_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["MD"]).sort_values("MD").drop_duplicates(subset=["MD"], keep="first").reset_index(drop=True)
    return out


def prepare_density_source_df(source_kind: str, source_path: Path) -> pd.DataFrame:
    if source_kind == "las":
        df = read_fracture_density_las(source_path).copy()
        depth_col = "DEPTH"
    elif source_kind == "sample_csv":
        df = read_csv_flexible(source_path).copy()
        depth_col = "TVD" if "TVD" in df.columns else "DEPT"
        if depth_col not in df.columns:
            raise ValueError(f"No depth column found in sample density source: {source_path}")
    else:
        raise ValueError(f"Unsupported density source kind: {source_kind}")

    out = pd.DataFrame()
    out["DEPTH"] = pd.to_numeric(df[depth_col], errors="coerce")
    for col in P_COLS:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan

    out = out.dropna(subset=["DEPTH"]).copy()
    for col in P_COLS:
        out.loc[out[col].isin(INVALID_SENTINELS), col] = np.nan
    out = out.sort_values("DEPTH").reset_index(drop=True)
    out = out.groupby("DEPTH", as_index=False).agg({col: "mean" for col in P_COLS})
    return out


def interpolate_density_to_sample_df(
    sample_df: pd.DataFrame,
    density_df: pd.DataFrame,
    depth_col: str = "TVD",
) -> pd.DataFrame:
    if depth_col not in sample_df.columns:
        raise ValueError(f"Missing depth column {depth_col} in sample df")

    out = sample_df.copy()
    target_depth = pd.to_numeric(out[depth_col], errors="coerce").to_numpy(dtype=np.float64)
    source_depth = pd.to_numeric(density_df["DEPTH"], errors="coerce").to_numpy(dtype=np.float64)
    valid_depth_mask = np.isfinite(source_depth)
    source_depth = source_depth[valid_depth_mask]
    if source_depth.size < 2:
        raise ValueError("Density source has fewer than 2 valid depth samples")

    for col in P_COLS:
        source_values = pd.to_numeric(density_df[col], errors="coerce").to_numpy(dtype=np.float64)
        source_values = source_values[valid_depth_mask]
        valid_mask = np.isfinite(source_values)
        if np.count_nonzero(valid_mask) < 2:
            out[col] = np.nan
            continue
        x = source_depth[valid_mask]
        y = source_values[valid_mask]
        out[col] = np.interp(target_depth, x, y, left=np.nan, right=np.nan)
    return out


def filter_to_density_covered_rows(df: pd.DataFrame, key_col: str = "P10") -> pd.DataFrame:
    if key_col not in df.columns:
        raise ValueError(f"Missing density key col: {key_col}")
    out = df.copy()
    out[key_col] = pd.to_numeric(out[key_col], errors="coerce")
    return out[out[key_col].notna()].reset_index(drop=True)


def circular_mean_deg(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    radians = np.deg2rad(values)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return float(np.mean(values))
    angle = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    if angle < 0:
        angle += 360.0
    return float(angle)


def map_raw_points_to_sample_grid(
    sample_df: pd.DataFrame,
    raw_point_df: pd.DataFrame,
    depth_col: str = "TVD",
) -> tuple[pd.DataFrame, int, int]:
    out = sample_df.copy()
    out["GT_POINT_FLAG"] = 0
    out["RAW_POINT_COUNT"] = 0
    out["Frac_Azimuth"] = np.nan
    out["Frac_Dip"] = np.nan

    if raw_point_df.empty or out.empty:
        return out, 0, int(len(raw_point_df))

    sample_depth = pd.to_numeric(out[depth_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_sample_mask = np.isfinite(sample_depth)
    if not np.all(valid_sample_mask):
        out = out.loc[valid_sample_mask].reset_index(drop=True)
        sample_depth = pd.to_numeric(out[depth_col], errors="coerce").to_numpy(dtype=np.float64)

    grouped_points: dict[int, list[dict[str, float]]] = {}
    mapped_count = 0
    dropped_count = 0
    min_depth = float(sample_depth[0])
    max_depth = float(sample_depth[-1])

    for row in raw_point_df.itertuples(index=False):
        point_depth = float(getattr(row, "MD"))
        if not np.isfinite(point_depth) or point_depth < min_depth or point_depth > max_depth:
            dropped_count += 1
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

        grouped_points.setdefault(nearest_idx, []).append(
            {
                "Frac_Azimuth": float(getattr(row, "Frac_Azimuth")),
                "Frac_Dip": float(getattr(row, "Frac_Dip")),
            }
        )
        mapped_count += 1

    for row_idx, point_rows in grouped_points.items():
        azimuth_values = np.asarray([item["Frac_Azimuth"] for item in point_rows], dtype=np.float64)
        dip_values = np.asarray([item["Frac_Dip"] for item in point_rows], dtype=np.float64)
        azimuth_values = azimuth_values[np.isfinite(azimuth_values)]
        dip_values = dip_values[np.isfinite(dip_values)]

        out.at[row_idx, "GT_POINT_FLAG"] = 1
        out.at[row_idx, "RAW_POINT_COUNT"] = int(len(point_rows))
        out.at[row_idx, "Frac_Azimuth"] = circular_mean_deg(azimuth_values)
        out.at[row_idx, "Frac_Dip"] = float(np.mean(dip_values)) if dip_values.size > 0 else np.nan

    return out, mapped_count, dropped_count


def assign_gt_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    density_values = out[P_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    density_mask = density_values.max(axis=1).gt(0.0)
    point_mask = pd.to_numeric(out["GT_POINT_FLAG"], errors="coerce").fillna(0).ge(1)
    out["GT_LABEL"] = (density_mask | point_mask).astype(np.int64)
    out["GT_POINT_FLAG"] = pd.to_numeric(out["GT_POINT_FLAG"], errors="coerce").fillna(0).astype(np.int64)
    out["RAW_POINT_COUNT"] = pd.to_numeric(out["RAW_POINT_COUNT"], errors="coerce").fillna(0).astype(np.int64)
    return out


def build_uniform_sample(
    well_name: str,
    around_csv: Path,
    density_source_kind: str,
    density_source_path: Path,
    raw_point_path: Path,
    output_csv: Path,
    key_col: str = "P10",
) -> SampleBuildStats:
    around_df = read_csv_flexible(around_csv)
    density_df = prepare_density_source_df(density_source_kind, density_source_path)
    sample_df = interpolate_density_to_sample_df(around_df, density_df, depth_col="TVD")
    input_row_count = int(len(sample_df))
    sample_df = filter_to_density_covered_rows(sample_df, key_col=key_col)

    raw_point_df = load_raw_point_table(raw_point_path)
    sample_df, mapped_count, dropped_count = map_raw_points_to_sample_grid(
        sample_df=sample_df,
        raw_point_df=raw_point_df,
        depth_col="TVD",
    )
    sample_df = assign_gt_label(sample_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return SampleBuildStats(
        well_name=well_name,
        input_row_count=input_row_count,
        filtered_row_count=int(len(sample_df)),
        gt_label_count=int(pd.to_numeric(sample_df["GT_LABEL"], errors="coerce").fillna(0).sum()),
        gt_point_count=int(pd.to_numeric(sample_df["GT_POINT_FLAG"], errors="coerce").fillna(0).sum()),
        mapped_raw_point_count=mapped_count,
        dropped_raw_point_count=dropped_count,
        density_source_kind=density_source_kind,
    )

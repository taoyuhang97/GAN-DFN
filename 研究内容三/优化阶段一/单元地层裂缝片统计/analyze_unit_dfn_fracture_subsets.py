from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATCH_CSV_CANDIDATES = [
    "predicted_unit_patches.csv",
    "amplified_patch_variance.csv",
    "predicted_window_concat_patches.csv",
]

REQUIRED_PATCH_COLUMNS = ["CenterTIME", "Azimuth", "Dip"]
LAYER_OUTPUT_COLUMNS = ["GeoIntervalKey", "StrataName", "TopSurfaceCode", "BaseSurfaceCode"]


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按地层或时间段统计单元 DFN 裂缝片数量，并绘制倾向-倾角分布图。"
    )
    parser.add_argument("--dfn-dir", required=True, help="单元 DFN 目录，例如 units\\BX75_BY21")
    parser.add_argument("--patch-csv", default="", help="可选，显式指定裂缝片 CSV 文件路径")
    parser.add_argument(
        "--output-dir",
        default="",
        help="可选，输出目录。默认保存到 <dfn-dir>\\fracture_subset_statistics",
    )
    parser.add_argument(
        "--strata",
        action="append",
        default=[],
        help="需要统计的地层，可重复传入，例如 --strata T6-T7",
    )
    parser.add_argument(
        "--depth-range",
        action="append",
        nargs=2,
        metavar=("TOP_MS", "BASE_MS"),
        default=[],
        help="需要统计的时间段，可重复传入，例如 --depth-range 2747 2866.6",
    )
    return parser.parse_args()


def find_patch_csv(dfn_dir: Path, explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"指定的裂缝片文件不存在: {path}")
        return path

    for file_name in PATCH_CSV_CANDIDATES:
        candidate = dfn_dir / file_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"在 {dfn_dir} 下未找到可用裂缝片文件，已尝试: {', '.join(PATCH_CSV_CANDIDATES)}"
    )


def load_patch_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_PATCH_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"裂缝片文件缺少必要字段: {missing}")

    for col in ["CenterTIME", "Azimuth", "Dip", "PatchLength", "PatchHeight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_layer_df_from_unit_layers(unit_layers_path: Path) -> pd.DataFrame:
    layer_df = pd.read_csv(unit_layers_path).copy()
    required = {"TopTime", "BaseTime", "TopSurfaceCode", "BaseSurfaceCode"}
    missing = required.difference(layer_df.columns)
    if missing:
        raise ValueError(f"{unit_layers_path} 缺少字段: {sorted(missing)}")

    for col in ["TopTime", "BaseTime", "TopDepth", "BaseDepth"]:
        if col in layer_df.columns:
            layer_df[col] = pd.to_numeric(layer_df[col], errors="coerce")

    if "StrataName" not in layer_df.columns:
        layer_df["StrataName"] = (
            layer_df["TopSurfaceCode"].astype(str).str.strip()
            + "->"
            + layer_df["BaseSurfaceCode"].astype(str).str.strip()
        )
    if "GeoIntervalKey" not in layer_df.columns:
        layer_df["GeoIntervalKey"] = ""

    layer_df = layer_df.sort_values(["TopTime", "BaseTime"], kind="stable").reset_index(drop=True)
    return layer_df


def build_layer_df_from_resolved_surfaces(resolved_path: Path) -> pd.DataFrame:
    surface_df = pd.read_csv(resolved_path).copy()
    time_col = "SnappedTime" if "SnappedTime" in surface_df.columns else "RawTime"
    required = {"SurfaceCode", time_col}
    missing = required.difference(surface_df.columns)
    if missing:
        raise ValueError(f"{resolved_path} 缺少字段: {sorted(missing)}")

    surface_df[time_col] = pd.to_numeric(surface_df[time_col], errors="coerce")
    surface_df = surface_df.dropna(subset=[time_col]).sort_values(time_col, kind="stable").reset_index(drop=True)

    records: list[dict[str, object]] = []
    for idx in range(len(surface_df) - 1):
        top_row = surface_df.iloc[idx]
        base_row = surface_df.iloc[idx + 1]
        top_surface = str(top_row["SurfaceCode"]).strip()
        base_surface = str(base_row["SurfaceCode"]).strip()
        records.append(
            {
                "GeoIntervalKey": f"{idx + 1:03d}_interval_{idx + 1:03d}",
                "StrataName": f"{top_surface}->{base_surface}",
                "TopSurfaceCode": top_surface,
                "BaseSurfaceCode": base_surface,
                "TopTime": float(top_row[time_col]),
                "BaseTime": float(base_row[time_col]),
            }
        )
    return pd.DataFrame.from_records(records)


def load_layer_df(dfn_dir: Path) -> pd.DataFrame:
    unit_layers_path = dfn_dir / "unit_layers_input.csv"
    if unit_layers_path.exists():
        return build_layer_df_from_unit_layers(unit_layers_path)

    resolved_path = dfn_dir / "resolved_layer_surfaces.csv"
    if resolved_path.exists():
        return build_layer_df_from_resolved_surfaces(resolved_path)

    return pd.DataFrame(columns=["GeoIntervalKey", "StrataName", "TopSurfaceCode", "BaseSurfaceCode", "TopTime", "BaseTime"])


def assign_layers_by_time(patch_df: pd.DataFrame, layer_df: pd.DataFrame) -> pd.DataFrame:
    work = patch_df.copy()
    for col in LAYER_OUTPUT_COLUMNS:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("")

    if layer_df.empty:
        return work

    missing_mask = (
        work["StrataName"].astype(str).str.strip().eq("")
        | work["TopSurfaceCode"].astype(str).str.strip().eq("")
        | work["BaseSurfaceCode"].astype(str).str.strip().eq("")
    )
    if not missing_mask.any():
        return work

    layer_df = layer_df.copy().reset_index(drop=True)
    center_time = pd.to_numeric(work["CenterTIME"], errors="coerce").to_numpy(dtype=float)
    assigned = np.zeros(len(work), dtype=bool)

    for idx, layer_row in layer_df.iterrows():
        top_time = float(layer_row["TopTime"])
        base_time = float(layer_row["BaseTime"])
        is_last = idx == len(layer_df) - 1
        if is_last:
            mask = missing_mask.to_numpy() & (center_time >= top_time) & (center_time <= base_time)
        else:
            mask = missing_mask.to_numpy() & (center_time >= top_time) & (center_time < base_time)
        if not mask.any():
            continue
        assigned |= mask
        work.loc[mask, "GeoIntervalKey"] = str(layer_row.get("GeoIntervalKey", ""))
        work.loc[mask, "StrataName"] = str(layer_row.get("StrataName", ""))
        work.loc[mask, "TopSurfaceCode"] = str(layer_row.get("TopSurfaceCode", ""))
        work.loc[mask, "BaseSurfaceCode"] = str(layer_row.get("BaseSurfaceCode", ""))

    return work


def normalize_strata_query(text: str) -> str:
    value = str(text).strip()
    if not value:
        return value
    value = value.replace(" ", "")
    separators = ["->", "-", ">", "_to_", "_TO_", "_To_"]
    for separator in separators:
        if separator not in value:
            continue
        left, right = value.split(separator, 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            return f"{left}->{right}"
    return value


def parse_strata_pair(text: str) -> tuple[str, str] | tuple[None, None]:
    normalized = normalize_strata_query(text)
    if "->" not in normalized:
        return None, None
    top_surface, base_surface = normalized.split("->", 1)
    return top_surface.strip(), base_surface.strip()


def make_safe_name(text: str) -> str:
    safe = (
        str(text)
        .strip()
        .replace("->", "_to_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    safe = re.sub(r"[^0-9A-Za-z_\-.]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("._")
    return safe or "subset"


def format_ms(value: float) -> str:
    return f"{value:g}ms"


def filter_by_strata(patch_df: pd.DataFrame, strata_query: str) -> tuple[pd.DataFrame, str]:
    normalized = normalize_strata_query(strata_query)
    top_surface, base_surface = parse_strata_pair(strata_query)

    mask = patch_df["StrataName"].astype(str).str.strip().eq(normalized)
    if top_surface and base_surface:
        mask = mask | (
            patch_df["TopSurfaceCode"].astype(str).str.strip().eq(top_surface)
            & patch_df["BaseSurfaceCode"].astype(str).str.strip().eq(base_surface)
        )
    subset = patch_df[mask].copy()
    display_name = normalized.replace("->", "-")
    return subset, display_name


def filter_by_depth_range(patch_df: pd.DataFrame, top_ms: float, base_ms: float) -> tuple[pd.DataFrame, str]:
    top_value = float(min(top_ms, base_ms))
    base_value = float(max(top_ms, base_ms))
    mask = (patch_df["CenterTIME"] >= top_value) & (patch_df["CenterTIME"] < base_value)
    subset = patch_df[mask].copy()
    display_name = f"{format_ms(top_value)}-{format_ms(base_value)}"
    return subset, display_name


def summarize_subset(
    subset_df: pd.DataFrame,
    selection_type: str,
    selection_name: str,
    filter_desc: str,
) -> dict[str, object]:
    azimuth_series = pd.to_numeric(subset_df.get("Azimuth"), errors="coerce")
    dip_series = pd.to_numeric(subset_df.get("Dip"), errors="coerce")
    time_series = pd.to_numeric(subset_df.get("CenterTIME"), errors="coerce")
    length_series = pd.to_numeric(subset_df.get("PatchLength"), errors="coerce")
    height_series = pd.to_numeric(subset_df.get("PatchHeight"), errors="coerce")

    fracture_count = int(len(subset_df))
    unique_patch_id_count = (
        int(subset_df["PatchID"].astype(str).nunique()) if "PatchID" in subset_df.columns else fracture_count
    )

    record: dict[str, object] = {
        "selection_type": selection_type,
        "selection_name": selection_name,
        "filter_description": filter_desc,
        "fracture_count": fracture_count,
        "unique_patch_id_count": unique_patch_id_count,
        "row_count": fracture_count,
        "time_min_ms": time_series.min() if len(subset_df) else np.nan,
        "time_max_ms": time_series.max() if len(subset_df) else np.nan,
        "azimuth_mean_deg": azimuth_series.mean() if len(subset_df) else np.nan,
        "azimuth_std_deg": azimuth_series.std(ddof=0) if len(subset_df) else np.nan,
        "dip_mean_deg": dip_series.mean() if len(subset_df) else np.nan,
        "dip_std_deg": dip_series.std(ddof=0) if len(subset_df) else np.nan,
        "patch_length_mean": length_series.mean() if len(subset_df) else np.nan,
        "patch_height_mean": height_series.mean() if len(subset_df) else np.nan,
    }

    if len(subset_df):
        top_surface_modes = subset_df.get("TopSurfaceCode", pd.Series(dtype=str)).astype(str).mode(dropna=False)
        base_surface_modes = subset_df.get("BaseSurfaceCode", pd.Series(dtype=str)).astype(str).mode(dropna=False)
        strata_modes = subset_df.get("StrataName", pd.Series(dtype=str)).astype(str).mode(dropna=False)
        record["dominant_top_surface"] = top_surface_modes.iloc[0] if not top_surface_modes.empty else ""
        record["dominant_base_surface"] = base_surface_modes.iloc[0] if not base_surface_modes.empty else ""
        record["dominant_strata_name"] = strata_modes.iloc[0] if not strata_modes.empty else ""
    else:
        record["dominant_top_surface"] = ""
        record["dominant_base_surface"] = ""
        record["dominant_strata_name"] = ""
    return record


def plot_azimuth_dip_polar(subset_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    if subset_df.empty:
        ax.text(0.5, 0.5, "该筛选范围内无裂缝片", transform=ax.transAxes, ha="center", va="center", fontsize=13)
        ax.set_axis_off()
    else:
        azimuth_rad = np.deg2rad(pd.to_numeric(subset_df["Azimuth"], errors="coerce"))
        dip_value = pd.to_numeric(subset_df["Dip"], errors="coerce")
        valid_mask = (~np.isnan(azimuth_rad)) & (~np.isnan(dip_value))

        ax.scatter(
            azimuth_rad[valid_mask],
            dip_value[valid_mask],
            c="steelblue",
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rmax(90)
        ax.set_rticks([30, 60, 90])
        ax.set_rlabel_position(135)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            f"裂缝数: {len(subset_df)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_count_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(summary_df["selection_name"], summary_df["fracture_count"], color="steelblue", alpha=0.85)
    ax.set_xlabel("筛选对象")
    ax.set_ylabel("裂缝数量")
    ax.set_title("不同筛选对象裂缝数量对比")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")

    for idx, value in enumerate(summary_df["fracture_count"]):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_subset_outputs(
    subset_df: pd.DataFrame,
    output_dir: Path,
    selection_type: str,
    selection_name: str,
    filter_desc: str,
) -> dict[str, object]:
    safe_name = make_safe_name(f"{selection_type}_{selection_name}")
    subset_csv_path = output_dir / f"{safe_name}_patches.csv"
    figure_path = output_dir / f"{safe_name}_azimuth_dip.png"

    subset_df.to_csv(subset_csv_path, index=False, encoding="utf-8-sig")
    plot_azimuth_dip_polar(subset_df, f"{selection_name} 倾向-倾角分布", figure_path)

    summary_record = summarize_subset(
        subset_df=subset_df,
        selection_type=selection_type,
        selection_name=selection_name,
        filter_desc=filter_desc,
    )
    summary_record["subset_csv"] = str(subset_csv_path)
    summary_record["azimuth_dip_figure"] = str(figure_path)
    return summary_record


def build_selection_specs(args: argparse.Namespace) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []

    for strata_query in args.strata:
        specs.append({"selection_type": "strata", "query": strata_query})

    for top_ms_text, base_ms_text in args.depth_range:
        specs.append(
            {
                "selection_type": "depth_range",
                "top_ms": float(top_ms_text),
                "base_ms": float(base_ms_text),
            }
        )
    return specs


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    dfn_dir = Path(args.dfn_dir)
    if not dfn_dir.exists():
        raise FileNotFoundError(f"DFN 目录不存在: {dfn_dir}")

    selection_specs = build_selection_specs(args)
    if not selection_specs:
        raise ValueError("至少需要指定一个 --strata 或 --depth-range")

    output_dir = Path(args.output_dir) if args.output_dir else dfn_dir / "fracture_subset_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_csv_path = find_patch_csv(dfn_dir, args.patch_csv)
    patch_df = load_patch_df(patch_csv_path)
    layer_df = load_layer_df(dfn_dir)
    patch_df = assign_layers_by_time(patch_df, layer_df)

    reference_layer_path = output_dir / "layer_reference.csv"
    if not layer_df.empty:
        layer_df.to_csv(reference_layer_path, index=False, encoding="utf-8-sig")

    summary_records: list[dict[str, object]] = []
    for spec in selection_specs:
        if spec["selection_type"] == "strata":
            subset_df, selection_name = filter_by_strata(patch_df, str(spec["query"]))
            filter_desc = f"按地层筛选: {normalize_strata_query(str(spec['query']))}"
        else:
            subset_df, selection_name = filter_by_depth_range(
                patch_df,
                float(spec["top_ms"]),
                float(spec["base_ms"]),
            )
            filter_desc = f"按时间段筛选: {selection_name}, 规则为 [top, base)"

        summary_record = save_subset_outputs(
            subset_df=subset_df,
            output_dir=output_dir,
            selection_type=str(spec["selection_type"]),
            selection_name=selection_name,
            filter_desc=filter_desc,
        )
        summary_records.append(summary_record)
        print(f"{selection_name}: {summary_record['fracture_count']} 条裂缝片")

    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = output_dir / "fracture_subset_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"汇总统计已保存: {summary_csv_path}")

    if not summary_df.empty:
        count_figure_path = output_dir / "fracture_subset_count_comparison.png"
        plot_count_bar(summary_df, count_figure_path)
        print(f"数量对比图已保存: {count_figure_path}")


if __name__ == "__main__":
    main()

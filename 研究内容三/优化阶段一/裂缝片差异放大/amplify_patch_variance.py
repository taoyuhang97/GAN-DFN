from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


DEFAULT_INPUT_CSV = Path(
    r"/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/单元DFN批量生成/bx33_35_by33_35_surface_scaled_x5_20260331/units/BX33_BY33/predicted_unit_patches.csv"
)

VERTEX_COLUMNS = [
    "V1X", "V1Y", "V1Z",
    "V2X", "V2Y", "V2Z",
    "V3X", "V3Y", "V3Z",
    "V4X", "V4Y", "V4Z",
]
LAYER_ATTRIBUTE_COLUMNS = ["GeoIntervalKey", "StrataName", "TopSurfaceCode", "BaseSurfaceCode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Amplify DFN patch size variance with robust quantile mapping and export CSV/VTK results."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mapping-power", type=float, default=0.85)
    parser.add_argument("--power", type=float, default=None, help="Deprecated alias for --mapping-power.")
    parser.add_argument("--base-scale", type=float, default=1.10)
    parser.add_argument("--length-low-quantile", type=float, default=0.05)
    parser.add_argument("--length-high-quantile", type=float, default=0.99)
    parser.add_argument("--height-low-quantile", type=float, default=0.05)
    parser.add_argument("--height-high-quantile", type=float, default=0.99)
    parser.add_argument("--length-gain-min", type=float, default=0.95)
    parser.add_argument("--length-gain-max", type=float, default=1.75)
    parser.add_argument("--height-gain-min", type=float, default=0.95)
    parser.add_argument("--height-gain-max", type=float, default=1.55)
    parser.add_argument("--low-anchor", type=float, default=0.12)
    parser.add_argument("--high-anchor", type=float, default=0.88)
    parser.add_argument("--low-tail-power", type=float, default=0.90)
    parser.add_argument("--high-tail-log-alpha", type=float, default=10.0)
    parser.add_argument("--random-scale-min", type=float, default=0.97)
    parser.add_argument("--random-scale-max", type=float, default=1.03)
    parser.add_argument("--dip-noise-deg", type=float, default=0.0)
    parser.add_argument("--color-clip-quantile", type=float, default=0.99)
    parser.add_argument("--display-z-scale", type=float, default=5.0)
    parser.add_argument("--vtk-no-invert-time", action="store_true")
    parser.add_argument(
        "--no-preserve-means",
        action="store_true",
        help="Disable the global mean-preserving correction for PatchLength and PatchHeight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_vtk_polydata(
    output_path: Path,
    points: list[list[float]],
    polygons: list[list[int]],
    cell_scalars: dict[str, list[float]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_points = len(points)
    num_polys = len(polygons)
    poly_list_size = num_polys * 5

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("Amplified DFN Patches\n")
        handle.write("ASCII\n")
        handle.write("DATASET POLYDATA\n")
        handle.write(f"POINTS {num_points} float\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

        handle.write(f"POLYGONS {num_polys} {poly_list_size}\n")
        for polygon in polygons:
            handle.write(f"4 {polygon[0]} {polygon[1]} {polygon[2]} {polygon[3]}\n")

        if cell_scalars:
            handle.write(f"CELL_DATA {num_polys}\n")
            for scalar_name, scalar_values in cell_scalars.items():
                handle.write(f"SCALARS {scalar_name} float 1\n")
                handle.write("LOOKUP_TABLE default\n")
                for value in scalar_values:
                    handle.write(f"{float(value):.6f}\n")


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return vector / norm


def normal_to_azimuth_dip(normal_vec: np.ndarray) -> tuple[float, float]:
    normal_vec = normalize(normal_vec)
    if normal_vec[2] < 0:
        normal_vec = -normal_vec
    dip = math.degrees(math.acos(float(np.clip(normal_vec[2], -1.0, 1.0))))
    azimuth = math.degrees(math.atan2(float(normal_vec[0]), float(normal_vec[1])))
    return float(azimuth % 360.0), float(dip)


def safe_numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value):
        return float(default)
    return float(value)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_category_code_map(series: pd.Series) -> tuple[pd.Series, dict[str, dict[int, str]]]:
    normalized = series.map(normalize_text)
    unique_values = [value for value in pd.unique(normalized) if value]
    code_by_value = {value: idx + 1 for idx, value in enumerate(unique_values)}
    value_by_code = {idx + 1: value for idx, value in enumerate(unique_values)}
    codes = normalized.map(code_by_value).fillna(-9999).astype(int)
    return codes, {"codes": value_by_code}


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {"CenterX", "CenterY", "CenterTIME", "PatchLength", "PatchHeight", *VERTEX_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input csv is missing required columns: {missing}")


def validate_quantile_pair(low_quantile: float, high_quantile: float, name: str) -> None:
    if not (0.0 <= low_quantile < high_quantile <= 1.0):
        raise ValueError(f"{name} quantiles must satisfy 0 <= low < high <= 1")


def validate_gain_pair(min_gain: float, max_gain: float, name: str) -> None:
    if min_gain <= 0 or max_gain <= 0:
        raise ValueError(f"{name} gains must be > 0")
    if min_gain > max_gain:
        raise ValueError(f"{name} gain min must be <= gain max")


def validate_anchor_pair(low_anchor: float, high_anchor: float) -> None:
    if not (0.0 <= low_anchor < high_anchor <= 1.0):
        raise ValueError("anchors must satisfy 0 <= low-anchor < high-anchor <= 1")


def build_display_point(point: np.ndarray, display_z_scale: float, invert_time: bool) -> list[float]:
    z_value = float(point[2])
    if invert_time:
        z_value = -z_value
    z_value *= float(display_z_scale)
    return [float(point[0]), float(point[1]), z_value]


def build_metric_profile(series: pd.Series, low_quantile: float, high_quantile: float, name: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"failed to compute metric profile for {name}")
    q_low = float(values.quantile(low_quantile))
    q_mid = float(values.quantile(0.50))
    q_high = float(values.quantile(high_quantile))
    if not np.isfinite(q_low) or not np.isfinite(q_mid) or not np.isfinite(q_high):
        raise ValueError(f"invalid quantiles for {name}")
    if q_high <= q_low:
        q_low = float(values.min())
        q_high = float(values.max())
        if q_high <= q_low:
            q_high = q_low + 1e-6
    return {
        "name": name,
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "q_low": float(q_low),
        "q_mid": float(q_mid),
        "q_high": float(q_high),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def order_preserving_map_metric(
    raw_value: float,
    profile: dict[str, float],
    gain_min: float,
    gain_max: float,
    mapping_power: float,
    base_scale: float,
    low_anchor: float,
    high_anchor: float,
    low_tail_power: float,
    high_tail_log_alpha: float,
) -> tuple[float, float, float, float, str]:
    raw_value = float(raw_value)
    value_min = float(profile["min"])
    value_low = float(profile["q_low"])
    value_high = float(profile["q_high"])
    value_max = float(profile["max"])

    if raw_value <= value_low:
        if value_low - value_min <= 1e-8:
            local_ratio = 1.0
        else:
            local_ratio = (raw_value - value_min) / (value_low - value_min)
        local_ratio = float(np.clip(local_ratio, 0.0, 1.0))
        shaped_local_ratio = float(local_ratio ** low_tail_power)
        mapped_ratio = float(low_anchor * shaped_local_ratio)
        zone = "low_tail"
    elif raw_value <= value_high:
        if value_high - value_low <= 1e-8:
            local_ratio = 0.5
        else:
            local_ratio = (raw_value - value_low) / (value_high - value_low)
        local_ratio = float(np.clip(local_ratio, 0.0, 1.0))
        shaped_local_ratio = float(local_ratio ** mapping_power)
        mapped_ratio = float(low_anchor + (high_anchor - low_anchor) * shaped_local_ratio)
        zone = "body"
    else:
        if value_max - value_high <= 1e-8:
            local_ratio = 1.0
        else:
            local_ratio = (raw_value - value_high) / (value_max - value_high)
        local_ratio = float(np.clip(local_ratio, 0.0, 1.0))
        if high_tail_log_alpha > 0:
            shaped_local_ratio = float(
                np.log1p(high_tail_log_alpha * local_ratio) / np.log1p(high_tail_log_alpha)
            )
        else:
            shaped_local_ratio = local_ratio
        mapped_ratio = float(high_anchor + (1.0 - high_anchor) * shaped_local_ratio)
        zone = "high_tail"

    target_min = max(float(value_low) * float(gain_min) * float(base_scale), 1e-6)
    target_max = max(float(value_high) * float(gain_max) * float(base_scale), target_min + 1e-6)
    mapped = float(target_min + (target_max - target_min) * mapped_ratio)
    gain = float(mapped / raw_value) if abs(raw_value) > 1e-8 else 1.0
    return mapped, mapped_ratio, local_ratio, gain, zone


def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True).fillna(0.0)


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or abs(maximum - minimum) <= 1e-12:
        return pd.Series(np.full(len(numeric), 0.5, dtype=float), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def build_patch_vertices(center: np.ndarray, vec_u: np.ndarray, vec_v: np.ndarray) -> list[np.ndarray]:
    return [
        center - vec_u / 2.0 - vec_v / 2.0,
        center + vec_u / 2.0 - vec_v / 2.0,
        center + vec_u / 2.0 + vec_v / 2.0,
        center - vec_u / 2.0 + vec_v / 2.0,
    ]


def recompute_patch(
    row: pd.Series,
    length_profile: dict[str, float],
    height_profile: dict[str, float],
    length_gain_min: float,
    length_gain_max: float,
    height_gain_min: float,
    height_gain_max: float,
    mapping_power: float,
    base_scale: float,
    low_anchor: float,
    high_anchor: float,
    low_tail_power: float,
    high_tail_log_alpha: float,
    random_scale_min: float,
    random_scale_max: float,
    dip_noise_deg: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    v1 = np.array([row["V1X"], row["V1Y"], row["V1Z"]], dtype=float)
    v2 = np.array([row["V2X"], row["V2Y"], row["V2Z"]], dtype=float)
    v4 = np.array([row["V4X"], row["V4Y"], row["V4Z"]], dtype=float)
    center = np.array([row["CenterX"], row["CenterY"], row["CenterTIME"]], dtype=float)

    vec_u = v2 - v1
    vec_v = v4 - v1
    orig_l = float(np.linalg.norm(vec_u))
    orig_h = float(np.linalg.norm(vec_v))

    if not np.isfinite(orig_l) or not np.isfinite(orig_h) or orig_l <= 1e-8 or orig_h <= 1e-8:
        raise ValueError("encountered patch with invalid edge length or height")

    target_l, mapped_ratio_l, local_ratio_l, gain_l, zone_l = order_preserving_map_metric(
        raw_value=orig_l,
        profile=length_profile,
        gain_min=length_gain_min,
        gain_max=length_gain_max,
        mapping_power=mapping_power,
        base_scale=base_scale,
        low_anchor=low_anchor,
        high_anchor=high_anchor,
        low_tail_power=low_tail_power,
        high_tail_log_alpha=high_tail_log_alpha,
    )
    target_h, mapped_ratio_h, local_ratio_h, gain_h, zone_h = order_preserving_map_metric(
        raw_value=orig_h,
        profile=height_profile,
        gain_min=height_gain_min,
        gain_max=height_gain_max,
        mapping_power=mapping_power,
        base_scale=base_scale,
        low_anchor=low_anchor,
        high_anchor=high_anchor,
        low_tail_power=low_tail_power,
        high_tail_log_alpha=high_tail_log_alpha,
    )

    scale_u = target_l / orig_l
    scale_v = target_h / orig_h
    rand_l = float(rng.uniform(random_scale_min, random_scale_max))
    rand_h = float(rng.uniform(random_scale_min, random_scale_max))

    vec_u_scaled = vec_u * (scale_u * rand_l)
    vec_v_scaled = vec_v * (scale_v * rand_h)

    if dip_noise_deg > 0:
        normal = normalize(np.cross(vec_u_scaled, vec_v_scaled))
        if np.linalg.norm(normal) > 1e-8:
            strike = normalize(np.cross(np.array([0.0, 0.0, 1.0], dtype=float), normal))
            if np.linalg.norm(strike) > 1e-8:
                random_angle = math.radians(float(rng.uniform(-dip_noise_deg, dip_noise_deg)))
                rot = Rotation.from_rotvec(random_angle * strike)
                vec_u_scaled = rot.apply(vec_u_scaled)
                vec_v_scaled = rot.apply(vec_v_scaled)

    mapped_l = float(np.linalg.norm(vec_u_scaled))
    mapped_h = float(np.linalg.norm(vec_v_scaled))
    # When only patch size is adjusted, preserve the original orientation attributes.
    normal_vec = np.array(
        [
            safe_numeric(row, "NormalX", 0.0),
            safe_numeric(row, "NormalY", 0.0),
            safe_numeric(row, "NormalZ", 1.0),
        ],
        dtype=float,
    )
    if np.linalg.norm(normal_vec) <= 1e-8:
        normal_vec = normalize(np.cross(vec_u, vec_v))
    else:
        normal_vec = normalize(normal_vec)

    azimuth = safe_numeric(row, "Azimuth", 0.0)
    dip = safe_numeric(row, "Dip", 0.0)

    return {
        "center": center,
        "vec_u_scaled": vec_u_scaled,
        "vec_v_scaled": vec_v_scaled,
        "mapped_patch_length": mapped_l,
        "mapped_patch_height": mapped_h,
        "mapped_patch_area": mapped_l * mapped_h,
        "normal": normal_vec,
        "azimuth": azimuth,
        "dip": dip,
        "orig_length": orig_l,
        "orig_height": orig_h,
        "orig_area": orig_l * orig_h,
        "length_display_ratio": mapped_ratio_l,
        "height_display_ratio": mapped_ratio_h,
        "length_local_ratio": local_ratio_l,
        "height_local_ratio": local_ratio_h,
        "length_mapping_zone": zone_l,
        "height_mapping_zone": zone_h,
        "length_gain_used": gain_l * rand_l,
        "height_gain_used": gain_h * rand_h,
    }


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_csv.parent / "patch_variance_amplified"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_power = float(args.power) if args.power is not None else float(args.mapping_power)
    if mapping_power <= 0:
        raise ValueError("mapping power must be > 0")
    if args.base_scale <= 0:
        raise ValueError("base-scale must be > 0")
    if args.random_scale_min <= 0 or args.random_scale_max <= 0:
        raise ValueError("random scale bounds must be > 0")
    if args.random_scale_min > args.random_scale_max:
        raise ValueError("random-scale-min must be <= random-scale-max")
    if not (0.0 < float(args.color_clip_quantile) <= 1.0):
        raise ValueError("color-clip-quantile must satisfy 0 < q <= 1")

    validate_quantile_pair(float(args.length_low_quantile), float(args.length_high_quantile), "length")
    validate_quantile_pair(float(args.height_low_quantile), float(args.height_high_quantile), "height")
    validate_gain_pair(float(args.length_gain_min), float(args.length_gain_max), "length")
    validate_gain_pair(float(args.height_gain_min), float(args.height_gain_max), "height")
    validate_anchor_pair(float(args.low_anchor), float(args.high_anchor))
    if float(args.low_tail_power) <= 0:
        raise ValueError("low-tail-power must be > 0")

    df = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    validate_input_columns(df)
    for column in LAYER_ATTRIBUTE_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_text)

    length_profile = build_metric_profile(
        series=df["PatchLength"],
        low_quantile=float(args.length_low_quantile),
        high_quantile=float(args.length_high_quantile),
        name="PatchLength",
    )
    height_profile = build_metric_profile(
        series=df["PatchHeight"],
        low_quantile=float(args.height_low_quantile),
        high_quantile=float(args.height_high_quantile),
        name="PatchHeight",
    )

    rng = np.random.default_rng(args.seed)
    invert_time = not bool(args.vtk_no_invert_time)

    all_points_raw: list[list[float]] = []
    all_points_display: list[list[float]] = []
    all_polygons: list[list[int]] = []
    point_offset = 0

    df["OrigPatchLength"] = pd.to_numeric(df["PatchLength"], errors="coerce")
    df["OrigPatchHeight"] = pd.to_numeric(df["PatchHeight"], errors="coerce")
    df["OrigPatchArea"] = df["OrigPatchLength"] * df["OrigPatchHeight"]

    patch_results: list[dict[str, object]] = []
    for idx, row in df.iterrows():
        patch = recompute_patch(
            row=row,
            length_profile=length_profile,
            height_profile=height_profile,
            length_gain_min=float(args.length_gain_min),
            length_gain_max=float(args.length_gain_max),
            height_gain_min=float(args.height_gain_min),
            height_gain_max=float(args.height_gain_max),
            mapping_power=float(mapping_power),
            base_scale=float(args.base_scale),
            low_anchor=float(args.low_anchor),
            high_anchor=float(args.high_anchor),
            low_tail_power=float(args.low_tail_power),
            high_tail_log_alpha=float(args.high_tail_log_alpha),
            random_scale_min=float(args.random_scale_min),
            random_scale_max=float(args.random_scale_max),
            dip_noise_deg=float(args.dip_noise_deg),
            rng=rng,
        )
        patch_results.append(patch)

    mean_length_before = float(pd.to_numeric(df["OrigPatchLength"], errors="coerce").dropna().mean())
    mean_height_before = float(pd.to_numeric(df["OrigPatchHeight"], errors="coerce").dropna().mean())
    mean_area_before = float(pd.to_numeric(df["OrigPatchArea"], errors="coerce").dropna().mean())
    mean_length_after_mapping = float(np.mean([float(item["mapped_patch_length"]) for item in patch_results]))
    mean_height_after_mapping = float(np.mean([float(item["mapped_patch_height"]) for item in patch_results]))

    preserve_means = not bool(args.no_preserve_means)
    if preserve_means and mean_length_after_mapping > 1e-12:
        length_mean_correction_factor = mean_length_before / mean_length_after_mapping
    else:
        length_mean_correction_factor = 1.0
    if preserve_means and mean_height_after_mapping > 1e-12:
        height_mean_correction_factor = mean_height_before / mean_height_after_mapping
    else:
        height_mean_correction_factor = 1.0

    df["LengthMeanCorrectionFactor"] = float(length_mean_correction_factor)
    df["HeightMeanCorrectionFactor"] = float(height_mean_correction_factor)

    for idx, patch in enumerate(patch_results):
        center = np.asarray(patch["center"], dtype=float)
        vec_u_scaled = np.asarray(patch["vec_u_scaled"], dtype=float) * float(length_mean_correction_factor)
        vec_v_scaled = np.asarray(patch["vec_v_scaled"], dtype=float) * float(height_mean_correction_factor)
        vertices = build_patch_vertices(center=center, vec_u=vec_u_scaled, vec_v=vec_v_scaled)
        patch_length = float(np.linalg.norm(vec_u_scaled))
        patch_height = float(np.linalg.norm(vec_v_scaled))
        patch_area = float(patch_length * patch_height)
        normal_vec = np.asarray(patch["normal"], dtype=float)
        azimuth = float(patch["azimuth"])
        dip = float(patch["dip"])

        df.at[idx, "PatchLength"] = patch_length
        df.at[idx, "PatchHeight"] = patch_height
        df.at[idx, "PatchArea"] = patch_area
        df.at[idx, "LengthDisplayRatio"] = float(patch["length_display_ratio"])
        df.at[idx, "HeightDisplayRatio"] = float(patch["height_display_ratio"])
        df.at[idx, "LengthLocalRatio"] = float(patch["length_local_ratio"])
        df.at[idx, "HeightLocalRatio"] = float(patch["height_local_ratio"])
        df.at[idx, "LengthMappingZone"] = str(patch["length_mapping_zone"])
        df.at[idx, "HeightMappingZone"] = str(patch["height_mapping_zone"])
        df.at[idx, "LengthGainUsed"] = float(patch["length_gain_used"]) * float(length_mean_correction_factor)
        df.at[idx, "HeightGainUsed"] = float(patch["height_gain_used"]) * float(height_mean_correction_factor)

        for vertex_index, vertex in enumerate(vertices, start=1):
            df.at[idx, f"V{vertex_index}X"] = float(vertex[0])
            df.at[idx, f"V{vertex_index}Y"] = float(vertex[1])
            df.at[idx, f"V{vertex_index}Z"] = float(vertex[2])
            all_points_raw.append([float(vertex[0]), float(vertex[1]), float(vertex[2])])
            all_points_display.append(
                build_display_point(
                    point=vertex,
                    display_z_scale=float(args.display_z_scale),
                    invert_time=invert_time,
                )
            )

        all_polygons.append([point_offset, point_offset + 1, point_offset + 2, point_offset + 3])
        point_offset += 4

    df["PatchArea"] = pd.to_numeric(df["PatchLength"], errors="coerce") * pd.to_numeric(df["PatchHeight"], errors="coerce")
    df["PatchAreaLog"] = np.log1p(pd.to_numeric(df["PatchArea"], errors="coerce"))
    color_clip_upper = float(pd.to_numeric(df["PatchArea"], errors="coerce").quantile(float(args.color_clip_quantile)))
    df["PatchAreaClip"] = pd.to_numeric(df["PatchArea"], errors="coerce").clip(upper=color_clip_upper)
    df["PatchAreaRank"] = percentile_rank(df["PatchArea"])
    df["PatchAreaClipNorm"] = normalize_series(df["PatchAreaClip"])
    df["PatchAreaLogNorm"] = normalize_series(df["PatchAreaLog"])
    df["PatchLengthRank"] = percentile_rank(df["PatchLength"])
    df["PatchHeightRank"] = percentile_rank(df["PatchHeight"])

    output_csv = output_dir / "amplified_patch_variance.csv"
    output_vtk_raw = output_dir / "amplified_patch_variance_raw_time.vtk"
    output_vtk_display = output_dir / "amplified_patch_variance_display.vtk"
    output_summary = output_dir / "amplified_patch_variance_summary.json"
    output_vtk_mappings = output_dir / "amplified_patch_variance_vtk_mappings.json"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    vtk_mappings: dict[str, dict[int, str]] = {}
    for column in LAYER_ATTRIBUTE_COLUMNS:
        if column not in df.columns:
            continue
        codes, mapping_obj = build_category_code_map(df[column])
        scalar_name = f"{column}Int"
        df[scalar_name] = codes.astype(int)
        vtk_mappings[scalar_name] = mapping_obj["codes"]

    cell_scalars = {
        "PatchArea": pd.to_numeric(df["PatchArea"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchAreaLog": pd.to_numeric(df["PatchAreaLog"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchAreaClip": pd.to_numeric(df["PatchAreaClip"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchAreaRank": pd.to_numeric(df["PatchAreaRank"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchAreaClipNorm": pd.to_numeric(df["PatchAreaClipNorm"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchAreaLogNorm": pd.to_numeric(df["PatchAreaLogNorm"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchLength": pd.to_numeric(df["PatchLength"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchHeight": pd.to_numeric(df["PatchHeight"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchLengthRank": pd.to_numeric(df["PatchLengthRank"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PatchHeightRank": pd.to_numeric(df["PatchHeightRank"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "LengthGainUsed": pd.to_numeric(df["LengthGainUsed"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "HeightGainUsed": pd.to_numeric(df["HeightGainUsed"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "Azimuth": pd.to_numeric(df["Azimuth"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "Dip": pd.to_numeric(df["Dip"], errors="coerce").fillna(0.0).astype(float).tolist(),
    }
    for scalar_name in ["GeoIntervalKeyInt", "StrataNameInt", "TopSurfaceCodeInt", "BaseSurfaceCodeInt"]:
        if scalar_name in df.columns:
            cell_scalars[scalar_name] = pd.to_numeric(df[scalar_name], errors="coerce").fillna(-9999).astype(float).tolist()

    write_vtk_polydata(output_vtk_raw, all_points_raw, all_polygons, cell_scalars=cell_scalars)
    write_vtk_polydata(output_vtk_display, all_points_display, all_polygons, cell_scalars=cell_scalars)
    output_vtk_mappings.write_text(json.dumps(vtk_mappings, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_vtk_raw_time": str(output_vtk_raw),
        "output_vtk_display": str(output_vtk_display),
        "output_vtk_mappings": str(output_vtk_mappings),
        "patch_count": int(len(df)),
        "seed": int(args.seed),
        "mapping_power": float(mapping_power),
        "base_scale": float(args.base_scale),
        "length_profile": length_profile,
        "height_profile": height_profile,
        "length_gain_min": float(args.length_gain_min),
        "length_gain_max": float(args.length_gain_max),
        "height_gain_min": float(args.height_gain_min),
        "height_gain_max": float(args.height_gain_max),
        "low_anchor": float(args.low_anchor),
        "high_anchor": float(args.high_anchor),
        "low_tail_power": float(args.low_tail_power),
        "high_tail_log_alpha": float(args.high_tail_log_alpha),
        "random_scale_min": float(args.random_scale_min),
        "random_scale_max": float(args.random_scale_max),
        "dip_noise_deg": float(args.dip_noise_deg),
        "display_z_scale": float(args.display_z_scale),
        "invert_time": bool(invert_time),
        "color_clip_quantile": float(args.color_clip_quantile),
        "color_clip_upper": float(color_clip_upper),
        "preserve_means_enabled": bool(preserve_means),
        "length_mean_correction_factor": float(length_mean_correction_factor),
        "height_mean_correction_factor": float(height_mean_correction_factor),
        "mean_length_before": mean_length_before,
        "mean_height_before": mean_height_before,
        "mean_area_before": mean_area_before,
        "mean_length_after_mapping_before_correction": mean_length_after_mapping,
        "mean_height_after_mapping_before_correction": mean_height_after_mapping,
        "mean_length_after": float(pd.to_numeric(df["PatchLength"], errors="coerce").dropna().mean()),
        "mean_height_after": float(pd.to_numeric(df["PatchHeight"], errors="coerce").dropna().mean()),
        "mean_area_after": float(pd.to_numeric(df["PatchArea"], errors="coerce").dropna().mean()),
        "mapping_mode": "piecewise_monotonic_quantile_mapping",
        "long_tail_handling": "tails are compressed by monotonic piecewise mapping and then globally mean-corrected, so order is preserved",
        "recommended_paraview_scalar": "PatchAreaRank",
        "alternative_paraview_scalars": ["PatchAreaLogNorm", "PatchAreaClipNorm", "PatchLengthRank", "PatchHeightRank"],
        "preserved_layer_columns": [column for column in LAYER_ATTRIBUTE_COLUMNS if column in df.columns],
        "vtk_layer_scalar_names": list(vtk_mappings.keys()),
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_csv={input_csv}")
    print(f"patch_count={len(df)}")
    print(f"output_csv={output_csv}")
    print(f"output_vtk_raw_time={output_vtk_raw}")
    print(f"output_vtk_display={output_vtk_display}")
    print(f"output_vtk_mappings={output_vtk_mappings}")
    print(f"output_summary={output_summary}")
    print(f"preserve_means_enabled={preserve_means}")
    print(f"length_mean_correction_factor={length_mean_correction_factor:.6f}")
    print(f"height_mean_correction_factor={height_mean_correction_factor:.6f}")
    print("recommended_paraview_scalar=PatchAreaRank")
    print("alternative_paraview_scalars=PatchAreaLogNorm,PatchAreaClipNorm,PatchLengthRank,PatchHeightRank")


if __name__ == "__main__":
    main()

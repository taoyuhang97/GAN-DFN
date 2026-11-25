#!/usr/bin/env python3
"""Predict fractures for new sand wells using the existing multilog model library."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT_PATH.exists():
    font_manager.fontManager.addfont(str(_CJK_FONT_PATH))
    plt.rcParams["font.sans-serif"] = [
        font_manager.FontProperties(fname=str(_CJK_FONT_PATH)).get_name(),
        "DejaVu Sans",
    ]
else:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CURRENT_DIR = Path(__file__).resolve().parent
STEP2_DIR = CURRENT_DIR.parent / "step2_real_well_t4_t7_samples"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(STEP2_DIR) not in sys.path:
    sys.path.insert(0, str(STEP2_DIR))

import build_real_well_t4_t7_samples as step2  # noqa: E402
import train_and_predict_gr_resistivity_strata_library as v2  # noqa: E402
import train_and_predict_six_expert_library as six  # noqa: E402


TARGET_STRATA = ("沙三段", "沙四段")
PAIR_COLUMNS = {
    "LLD_LLS": ("LLD", "LLS"),
    "RD_RS": ("RD", "RS"),
    "RILD_RILM": ("RILD", "RILM"),
}
TIME_CLASS_TO_LAYER = dict(step2.TIME_CLASS_TO_LAYER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replace-existing-output", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_deviation(path: Path, kind: str) -> pd.DataFrame:
    if kind == "csv_cn":
        raw = pd.read_csv(path, encoding="gb18030")
        md_col = next((column for column in raw.columns if str(column).strip() == "测点深度"), None)
        dev_col = next((column for column in raw.columns if str(column).strip() == "井斜角"), None)
        az_col = next((column for column in raw.columns if str(column).strip() == "方位角"), None)
        if md_col is None or dev_col is None or az_col is None:
            raise ValueError(f"cannot locate Chinese deviation columns in {path}")
        frame = pd.DataFrame(
            {
                "MD": pd.to_numeric(raw[md_col], errors="coerce"),
                "DEV": pd.to_numeric(raw[dev_col], errors="coerce"),
                "AZIM": pd.to_numeric(raw[az_col], errors="coerce"),
            }
        )
        for source_name, target_name in (
            ("累计垂深", "ReferenceTVD"),
            ("东西位移", "ReferenceEast"),
            ("南北位移", "ReferenceNorth"),
        ):
            if source_name in raw.columns:
                frame[target_name] = pd.to_numeric(raw[source_name], errors="coerce")
    else:
        rows: list[list[float]] = []
        started = False
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if values and values[0].lower() in {"depth", "md"}:
                started = True
                continue
            if not started or len(values) < 3:
                continue
            try:
                rows.append([float(values[0]), float(values[1]), float(values[2])])
            except ValueError:
                continue
        frame = pd.DataFrame(rows, columns=["MD", "DEV", "AZIM"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame["MD"].ge(0.0) & frame["DEV"].between(0.0, 180.0)]
    frame["AZIM"] = frame["AZIM"].mod(360.0)
    frame = frame.sort_values("MD").drop_duplicates("MD").reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError(f"not enough deviation rows: {path}")
    return frame


def trajectory_reference_qc(
    deviation: pd.DataFrame,
    track: pd.DataFrame,
    md_min: float,
    md_max: float,
    wellhead_x: float,
    wellhead_y: float,
) -> dict[str, float | int | None]:
    if "ReferenceTVD" not in deviation.columns:
        return {
            "ReferenceRows": 0,
            "ReferenceMaxAbsTVDM": None,
            "ReferenceMaxAbsEastM": None,
            "ReferenceMaxAbsNorthM": None,
        }
    mask = deviation.MD.between(md_min, md_max) & deviation.ReferenceTVD.notna()
    subset = deviation.loc[mask]
    if subset.empty:
        return {
            "ReferenceRows": 0,
            "ReferenceMaxAbsTVDM": None,
            "ReferenceMaxAbsEastM": None,
            "ReferenceMaxAbsNorthM": None,
        }
    md = subset.MD.to_numpy(dtype=float)
    tvd = np.interp(md, track.MD, track.TVD)
    result: dict[str, float | int | None] = {
        "ReferenceRows": int(len(subset)),
        "ReferenceMaxAbsTVDM": float(np.max(np.abs(tvd - subset.ReferenceTVD.to_numpy(dtype=float)))),
        "ReferenceMaxAbsEastM": None,
        "ReferenceMaxAbsNorthM": None,
    }
    if "ReferenceEast" in subset.columns:
        east = np.interp(md, track.MD, track.X) - float(wellhead_x)
        result["ReferenceMaxAbsEastM"] = float(
            np.max(np.abs(east - subset.ReferenceEast.to_numpy(dtype=float)))
        )
    if "ReferenceNorth" in subset.columns:
        north = np.interp(md, track.MD, track.Y) - float(wellhead_y)
        result["ReferenceMaxAbsNorthM"] = float(
            np.max(np.abs(north - subset.ReferenceNorth.to_numpy(dtype=float)))
        )
    return result


def build_track(
    deviation: pd.DataFrame,
    wellhead_x: float,
    wellhead_y: float,
) -> pd.DataFrame:
    """Integrate a deviation survey with the minimum-curvature method."""
    md = deviation["MD"].to_numpy(dtype=float)
    inc = np.radians(deviation["DEV"].to_numpy(dtype=float))
    azi = np.radians(deviation["AZIM"].to_numpy(dtype=float))
    tvdx = np.zeros(len(md), dtype=float)
    east = np.zeros(len(md), dtype=float)
    north = np.zeros(len(md), dtype=float)
    east[0] = float(wellhead_x)
    north[0] = float(wellhead_y)
    for index in range(1, len(md)):
        delta_md = float(md[index] - md[index - 1])
        cos_dogleg = (
            math.cos(inc[index - 1]) * math.cos(inc[index])
            + math.sin(inc[index - 1])
            * math.sin(inc[index])
            * math.cos(azi[index] - azi[index - 1])
        )
        dogleg = math.acos(float(np.clip(cos_dogleg, -1.0, 1.0)))
        ratio_factor = 1.0 if abs(dogleg) < 1.0e-8 else 2.0 * math.tan(dogleg / 2.0) / dogleg
        north_delta = (
            delta_md
            / 2.0
            * (math.sin(inc[index - 1]) * math.cos(azi[index - 1])
               + math.sin(inc[index]) * math.cos(azi[index]))
            * ratio_factor
        )
        east_delta = (
            delta_md
            / 2.0
            * (math.sin(inc[index - 1]) * math.sin(azi[index - 1])
               + math.sin(inc[index]) * math.sin(azi[index]))
            * ratio_factor
        )
        tvd_delta = delta_md / 2.0 * (math.cos(inc[index - 1]) + math.cos(inc[index])) * ratio_factor
        north[index] = north[index - 1] + north_delta
        east[index] = east[index - 1] + east_delta
        tvdx[index] = tvdx[index - 1] + tvd_delta
    return pd.DataFrame(
        {
            "MD": md,
            "TVD": tvdx,
            "X": east,
            "Y": north,
            "DEV": deviation["DEV"].to_numpy(dtype=float),
            "AZIM": deviation["AZIM"].to_numpy(dtype=float),
        }
    )


def interpolate_track(track: pd.DataFrame, md: np.ndarray) -> pd.DataFrame:
    if md.min() < track.MD.min() - 1.0e-6 or md.max() > track.MD.max() + 1.0e-6:
        raise ValueError(
            f"deviation does not cover log MD: log={md.min():.3f}-{md.max():.3f}, "
            f"survey={track.MD.min():.3f}-{track.MD.max():.3f}"
        )
    return pd.DataFrame(
        {
            "MD": md,
            "TVD": np.interp(md, track.MD, track.TVD),
            "X": np.interp(md, track.MD, track.X),
            "Y": np.interp(md, track.MD, track.Y),
            "DEV": np.interp(md, track.MD, track.DEV),
            "AZIM": np.interp(md, track.MD, track.AZIM),
        }
    )


def read_time_depth(path: Path) -> tuple[pd.DataFrame, float | None]:
    lines = Path(path).read_text(encoding="gb18030", errors="ignore").splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if "TIME" in line.upper() and "TVD" in line.upper() and "MD" in line.upper()
    )
    rows: list[list[float]] = []
    for raw in lines[header_index + 1 :]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 4:
            continue
        try:
            rows.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
        except ValueError:
            continue
    frame = pd.DataFrame(rows, columns=["TIME", "TVDSS", "TVD", "MD"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["TIME", "TVD", "MD"])
    frame = frame.sort_values("TVD").drop_duplicates("TVD")
    if len(frame) < 2:
        raise ValueError(f"invalid time-depth file: {path}")
    kb: float | None = None
    for raw in lines:
        if raw.strip().upper().startswith("# KB"):
            try:
                kb = float(raw.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                pass
            break
    return frame, kb


def existing_well_catalog(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*_t4_t7_real_well_main.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["WellName", "X", "Y"])
        if frame.empty:
            continue
        rows.append(
            {
                "WellName": str(frame.WellName.iloc[0]),
                "X": float(pd.to_numeric(frame.X, errors="coerce").median()),
                "Y": float(pd.to_numeric(frame.Y, errors="coerce").median()),
                "MainPath": str(path),
            }
        )
    return rows


def choose_time_depth_source(
    well: dict[str, Any],
    target_track: pd.DataFrame,
    log_md_min: float,
    log_md_max: float,
    existing_wells: list[dict[str, Any]],
    time_depth_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target_tvd_min = float(target_track.TVD.min())
    target_tvd_max = float(target_track.TVD.max())
    rows: list[dict[str, Any]] = []
    for candidate in existing_wells:
        source_path = time_depth_dir / f"{candidate['WellName']}.dat"
        if not source_path.exists():
            continue
        try:
            source_td, source_kb = read_time_depth(source_path)
        except Exception as exc:
            rows.append(
                {
                    "TargetWell": well["well_name"],
                    "SourceWell": candidate["WellName"],
                    "DistanceM": float(
                        math.hypot(
                            candidate["X"] - float(well["wellhead_x"]),
                            candidate["Y"] - float(well["wellhead_y"]),
                        )
                    ),
                    "Usable": False,
                    "Reason": str(exc),
                }
            )
            continue
        source_md_min = float(source_td.MD.min())
        source_md_max = float(source_td.MD.max())
        source_tvd_min = float(source_td.TVD.min())
        source_tvd_max = float(source_td.TVD.max())
        distance = float(
            math.hypot(
                candidate["X"] - float(well["wellhead_x"]),
                candidate["Y"] - float(well["wellhead_y"]),
            )
        )
        rows.append(
            {
                "TargetWell": well["well_name"],
                "SourceWell": candidate["WellName"],
                "DistanceM": distance,
                "SourcePath": str(source_path),
                "SourceKB": source_kb,
                "SourceMDMin": source_md_min,
                "SourceMDMax": source_md_max,
                "SourceTVDMin": source_tvd_min,
                "SourceTVDMax": source_tvd_max,
                "TargetMDMin": log_md_min,
                "TargetMDMax": log_md_max,
                "TargetTVDMin": target_tvd_min,
                "TargetTVDMax": target_tvd_max,
                "MDCovered": source_md_min <= log_md_min and source_md_max >= log_md_max,
                "TVDCovered": source_tvd_min <= target_tvd_min and source_tvd_max >= target_tvd_max,
            }
        )
        rows[-1]["Usable"] = bool(rows[-1]["MDCovered"] and rows[-1]["TVDCovered"])
        rows[-1]["Reason"] = (
            "ok"
            if rows[-1]["Usable"]
            else "source does not cover full target MD+TVD range"
        )
    candidates = pd.DataFrame(rows).sort_values("DistanceM").reset_index(drop=True)
    usable = candidates[candidates.Usable.astype(bool)]
    if usable.empty:
        raise RuntimeError(f"no complete time-depth source for {well['well_name']}")
    selected_well = str(usable.iloc[0].SourceWell)
    selected_path = Path(str(usable.iloc[0].SourcePath))
    selected_td, selected_kb = read_time_depth(selected_path)
    selection = usable.iloc[0].to_dict()
    selection["TargetWellName"] = str(well["well_name"])
    selection["SourceWellName"] = selected_well
    selection["SourcePath"] = str(selected_path)
    selection["SourceKB"] = selected_kb
    selection["TimeMappingAxis"] = "TVD"
    selection["TimeDepthTable"] = selected_td
    return selection, candidates


def load_las_curves(path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = step2.read_las_ascii_block(path)
    curve_audit = [str(column) for column in raw.columns]
    frame = step2.normalize_las_curves(raw)
    for curve in ("LLD", "LLS", "RD", "RS", "RILD", "RILM"):
        if curve in frame.columns:
            frame[curve] = step2.clean_numeric_series(frame[curve])
        else:
            frame[curve] = np.nan
    for curve in ("AC", "DEN"):
        frame.loc[pd.to_numeric(frame[curve], errors="coerce").le(0.0), curve] = np.nan
    return frame, curve_audit


def build_well_input(
    well: dict[str, Any],
    output_root: Path,
    existing_wells: list[dict[str, Any]],
    time_depth_dir: Path,
    layer_dir: Path,
    surface_tables,
) -> dict[str, Any]:
    well_name = str(well["well_name"])
    source_las = Path(well["source_las"])
    curves, curve_audit = load_las_curves(source_las)
    curves = curves.dropna(subset=["DEPT"]).sort_values("DEPT").drop_duplicates("DEPT").reset_index(drop=True)
    if curves.empty:
        raise RuntimeError(f"empty LAS curves: {source_las}")

    deviation = read_deviation(Path(well["deviation_path"]), str(well["deviation_format"]))
    track = build_track(deviation, float(well["wellhead_x"]), float(well["wellhead_y"]))
    trajectory_qc = trajectory_reference_qc(
        deviation,
        track,
        float(curves.DEPT.min()),
        float(curves.DEPT.max()),
        float(well["wellhead_x"]),
        float(well["wellhead_y"]),
    )
    curve_track = interpolate_track(track, curves.DEPT.to_numpy(dtype=float))
    curves["MD"] = curve_track.MD
    curves["TVD"] = curve_track.TVD
    curves["X"] = curve_track.X
    curves["Y"] = curve_track.Y
    curves["DEV"] = curve_track.DEV
    curves["AZIM"] = curve_track.AZIM

    selection, candidates = choose_time_depth_source(
        well,
        curve_track,
        float(curves.DEPT.min()),
        float(curves.DEPT.max()),
        existing_wells,
        time_depth_dir,
    )
    source_td: pd.DataFrame = selection.pop("TimeDepthTable")
    time = np.interp(
        curve_track.TVD.to_numpy(dtype=float),
        source_td.TVD.to_numpy(dtype=float),
        source_td.TIME.to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )
    curves["TIME"] = time

    records = curves[["DEPT", "TVD", "X", "Y", "TIME"]].copy()
    records = step2.assign_surface_times(records, surface_tables)
    records = step2.classify_time_domain(records)
    records = step2.validate_surface_order(records, min_thickness=1.0)
    records["SurfaceOrderValid"] = records["Check_Order_T4_T5_T6_T7"].fillna(False)
    records["StrataName"] = records["TimeDomainClass"].map(TIME_CLASS_TO_LAYER).fillna("")
    records["WellName"] = well_name
    records["SampleID"] = [f"{well_name}_{index:06d}" for index in range(len(records))]
    for column in ("GR", "AC", "CNL", "DEN", "LLD", "LLS", "RD", "RS", "RILD", "RILM"):
        records[column] = curves[column].to_numpy()

    pair_type = str(well["input_pair_type"])
    deep_curve, reference_curve = PAIR_COLUMNS[pair_type]
    required = ["GR", "AC", "CNL", "DEN", deep_curve, reference_curve]
    valid_six = records[required].notna().all(axis=1)
    valid_target = valid_six & records.StrataName.isin(TARGET_STRATA) & records.TIME.notna()

    def pair_gr(expected_pair: str) -> pd.Series:
        if pair_type == expected_pair:
            return records.GR
        return pd.Series(np.nan, index=records.index)

    enrichment = pd.DataFrame(
        {
            "DEPT": records.DEPT,
            "GR": records.GR,
            "GR_LLD_LLS": pair_gr("LLD_LLS"),
            "LLD": records.LLD,
            "LLS": records.LLS,
            "GR_RD_RS": pair_gr("RD_RS"),
            "RD": records.RD,
            "RS": records.RS,
            "GR_RILD_RILM": pair_gr("RILD_RILM"),
            "RILD": records.RILD,
            "RILM": records.RILM,
            "AC": records.AC,
            "CNL": records.CNL,
            "DEN": records.DEN,
        }
    )

    well_dir = output_root / "step2_like" / well_name
    well_dir.mkdir(parents=True, exist_ok=True)
    curve_table = curves[["DEPT", "GR", "AC", "CNL", "DEN", "LLD", "LLS", "RD", "RS", "RILD", "RILM"]]
    curve_table.to_csv(
        well_dir / f"{well_name}_t4_t7_real_well_conventional_curves.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_columns = [
        "SampleID", "WellName", "X", "Y", "TIME", "TVD", "DEPT", "StrataName",
        "T4_TIME", "T5_TIME", "T6_TIME", "T7_TIME", "SurfaceOrderValid",
        "GR", "AC", "CNL", "DEN", "LLD", "LLS", "RD", "RS", "RILD", "RILM",
    ]
    records[main_columns].to_csv(
        well_dir / f"{well_name}_t4_t7_real_well_main.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curve_track.rename(columns={"MD": "DEPT"}).to_csv(
        well_dir / f"{well_name}_well_track.csv",
        index=False,
        encoding="utf-8-sig",
    )
    interval_rows = []
    for strata_name, group in records[records.StrataName.isin(TARGET_STRATA)].groupby("StrataName"):
        interval_rows.append(
            {
                "WellName": well_name,
                "StrataName": strata_name,
                "TopDEPT": float(group.DEPT.min()),
                "BaseDEPT": float(group.DEPT.max()),
                "TopTVD": float(group.TVD.min()),
                "BaseTVD": float(group.TVD.max()),
                "TopTIME": float(group.TIME.min()),
                "BaseTIME": float(group.TIME.max()),
                "Rows": int(len(group)),
            }
        )
    interval = pd.DataFrame(interval_rows)
    interval.to_csv(
        well_dir / f"{well_name}_t4_t7_real_well_interval.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qc = {
        "WellName": well_name,
        "SourceLAS": str(source_las),
        "SourceCurveHeaders": curve_audit,
        "InputPairType": pair_type,
        "DepthMin": float(curves.DEPT.min()),
        "DepthMax": float(curves.DEPT.max()),
        "Rows": int(len(curves)),
        "RequiredCurves": required,
        "SixCurveValidRows": int(valid_six.sum()),
        "SixCurveValidMDMin": float(records.loc[valid_six, "DEPT"].min()) if valid_six.any() else None,
        "SixCurveValidMDMax": float(records.loc[valid_six, "DEPT"].max()) if valid_six.any() else None,
        "TargetValidRows": int(valid_target.sum()),
        "StrataCounts": {str(k): int(v) for k, v in records.loc[valid_target, "StrataName"].value_counts().items()},
        "TrajectoryMDMin": float(track.MD.min()),
        "TrajectoryMDMax": float(track.MD.max()),
        "TrajectoryTVDMin": float(track.TVD.min()),
        "TrajectoryTVDMax": float(track.TVD.max()),
        "TrajectoryReferenceQC": trajectory_qc,
        "TimeDepthSelection": selection,
        "WellheadX": float(well["wellhead_x"]),
        "WellheadY": float(well["wellhead_y"]),
        "WellheadKB": float(well["wellhead_kb_m"]),
        "CoordinateNote": str(well.get("coordinate_note", "")),
    }
    return {
        "WellName": well_name,
        "InputPairType": pair_type,
        "Records": records,
        "Enrichment": enrichment,
        "QC": qc,
        "Candidates": candidates,
        "Interval": interval,
    }


def load_direct_models(model_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, float]]:
    registry = pd.read_csv(model_root / "model_library" / "six_expert_registry.csv", encoding="utf-8-sig")
    models: dict[tuple[str, str], dict[str, Any]] = {}
    for row in registry.itertuples(index=False):
        if str(row.TrainingMode) != "direct_imaging_supervision":
            continue
        key = (str(row.StrataName), str(row.PairType))
        models[key] = {
            "kind": "direct",
            "stage1": joblib.load(model_root / str(row.Stage1ModelPath)),
            "stage2": joblib.load(model_root / str(row.Stage2ModelPath)),
            "threshold": float(row.DeploymentStage1Threshold),
            "density_scale": float(row.OOFDensityScale),
            "expert_id": str(row.ExpertID),
            "model_id": str(row.ModelID),
        }
    if not models:
        raise RuntimeError("no direct expert models loaded")
    training = pd.read_csv(
        model_root / "training_feature_table.csv",
        encoding="utf-8-sig",
        usecols=["WellName", "StrataName", "DEPT", "Density", "GT_POINT_FLAG"],
    )
    return models, six.point_count_scales(training)


def plot_well(
    well_name: str,
    curves: pd.DataFrame,
    predictions: pd.DataFrame,
    points: pd.DataFrame,
    interval: pd.DataFrame,
    output_path: Path,
) -> None:
    frame = curves.merge(
        predictions[
            ["DEPT", "PredFractureProb", "PredConditionalDensity", "PredDensity", "PredHasFracture"]
        ],
        on="DEPT",
        how="left",
    )
    frame = frame[frame.PredFractureProb.notna()].sort_values("DEPT")
    if frame.empty:
        raise RuntimeError(f"no valid prediction depth to plot for {well_name}")
    fig, axes = plt.subplots(1, 6, figsize=(17, 12), sharey=True, constrained_layout=True)
    colors = {"沙三段": "#ff7a00", "沙四段": "#00a6ff"}
    for ax in axes:
        ax.grid(alpha=0.2)
        for row in interval.itertuples(index=False):
            ax.axhspan(row.TopDEPT, row.BaseDEPT, color=colors.get(str(row.StrataName), "#cccccc"), alpha=0.08)
    axes[0].plot(frame.GR, frame.DEPT, color="#2e7d32", lw=0.8)
    axes[0].set_xlabel("GR")
    axes[1].plot(frame.AC, frame.DEPT, color="#6a1b9a", lw=0.8)
    axes[1].set_xlabel("AC")
    axes[2].plot(frame.CNL, frame.DEPT, color="#00838f", lw=0.8)
    axes[2].set_xlabel("CNL")
    axes[3].plot(frame.DEN, frame.DEPT, color="#8d6e63", lw=0.8)
    axes[3].set_xlabel("DEN")
    for curve, color in (("LLD", "#1e88e5"), ("LLS", "#64b5f6"), ("RD", "#ef6c00"), ("RS", "#ffb74d")):
        if frame[curve].notna().any():
            axes[4].plot(frame[curve].abs().clip(lower=1e-3), frame.DEPT, color=color, lw=0.8, label=curve)
            axes[4].set_xscale("log")
    axes[4].set_xlabel("Resistivity (abs)")
    axes[4].legend(fontsize=7, loc="upper right")
    axes[5].plot(frame.PredFractureProb, frame.DEPT, color="#c62828", lw=0.9, label="Prob")
    density = pd.to_numeric(frame.PredDensity, errors="coerce")
    density_norm = density / max(float(density.max(skipna=True) or 0.0), 1.0e-9)
    axes[5].plot(density_norm, frame.DEPT, color="#ad1457", lw=0.8, label="Density / max")
    if not points.empty:
        axes[5].scatter(
            np.ones(len(points)) * 1.04,
            points.DEPT,
            marker="_",
            s=130,
            color="#d50000",
            label="Predicted fracture",
        )
    axes[5].set_xlim(-0.02, 1.08)
    axes[5].set_xlabel("Probability / normalized density")
    axes[5].legend(fontsize=7, loc="upper right")
    axes[0].set_ylim(float(frame.DEPT.max()) + 10.0, float(frame.DEPT.min()) - 10.0)
    axes[0].set_ylabel("MD (m)")
    fig.suptitle(
        f"{well_name} conventional-log fracture prediction\n"
        "orange: Sha-3, blue: Sha-4; existing Step4 multilog v2 model"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace_existing_output:
            raise RuntimeError(f"output exists; pass --replace-existing-output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    six.REQUIRE_MULTILOG_CURVES = True
    six.FEATURE_COLUMNS = list(six.MULTILOG_FEATURE_COLUMNS)
    model_root = Path(config["model_root"]).resolve()
    models, point_scales = load_direct_models(model_root)
    existing_wells = existing_well_catalog(Path(config["existing_wells_root"]).resolve())
    time_depth_dir = Path(config["time_depth_dir"]).resolve()
    surface_tables = step2.load_surface_tables(Path(config["layer_dir"]).resolve())

    qc_rows: list[dict[str, Any]] = []
    candidate_tables: list[pd.DataFrame] = []
    detail_parts: list[pd.DataFrame] = []
    merged_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []
    well_summaries: list[dict[str, Any]] = []
    for well in config["wells"]:
        built = build_well_input(
            well,
            output_dir,
            existing_wells,
            time_depth_dir,
            Path(config["layer_dir"]).resolve(),
            surface_tables,
        )
        well_name = built["WellName"]
        records = built["Records"]
        enrichment = built["Enrichment"]
        deep_curve, reference_curve = PAIR_COLUMNS[built["InputPairType"]]
        required_curves = ["GR", "AC", "CNL", "DEN", deep_curve, reference_curve]
        base = records[
            records.StrataName.isin(TARGET_STRATA)
            & records[required_curves].notna().all(axis=1)
            & records[["X", "Y", "TIME", "TVD", "DEPT"]].notna().all(axis=1)
        ].copy()
        detail = six.build_prediction_rows(
            base,
            enrichment,
            models,
            float(config["max_support_gap_m"]),
        )
        if detail.empty:
            raise RuntimeError(f"no predictions for {well_name}")
        merged = six.merge_expert_predictions(detail, float(config["max_support_gap_m"]))
        points = six.refine_points(merged, point_scales)
        if merged.duplicated(["WellName", "DEPT"]).any():
            raise RuntimeError(f"duplicate merged prediction key for {well_name}")
        detail_parts.append(detail)
        merged_parts.append(merged)
        point_parts.append(points)
        qc_rows.append(built["QC"])
        candidate_tables.append(built["Candidates"])

        well_pred_dir = output_dir / "predictions" / "per_well" / well_name
        provenance_dir = output_dir / "provenance" / "per_well"
        six.write_main_and_provenance(
            detail,
            well_pred_dir / f"{well_name}_expert_detail_prediction.csv",
            provenance_dir / f"{well_name}_expert_detail_provenance.csv",
            six.PREDICTION_MAIN_COLUMNS,
            six.PREDICTION_PROVENANCE_COLUMNS,
        )
        six.write_main_and_provenance(
            merged,
            well_pred_dir / f"{well_name}_merged_density_prediction.csv",
            provenance_dir / f"{well_name}_merged_density_provenance.csv",
            six.PREDICTION_MAIN_COLUMNS,
            six.PREDICTION_PROVENANCE_COLUMNS,
        )
        six.write_main_and_provenance(
            points,
            well_pred_dir / f"{well_name}_fracture_points.csv",
            provenance_dir / f"{well_name}_fracture_points_provenance.csv",
            six.POINT_MAIN_COLUMNS,
            six.POINT_PROVENANCE_COLUMNS,
        )
        plot_well(
            well_name,
            records,
            merged,
            points,
            built["Interval"],
            output_dir / "plots" / f"{well_name}_fracture_prediction.png",
        )
        well_summaries.append(
            {
                "WellName": well_name,
                "InputPairType": built["InputPairType"],
                "TimeDepthSource": built["QC"]["TimeDepthSelection"]["SourceWellName"],
                "SourceDistanceM": built["QC"]["TimeDepthSelection"]["DistanceM"],
                "SixCurveValidRows": built["QC"]["SixCurveValidRows"],
                "TargetValidRows": built["QC"]["TargetValidRows"],
                "StrataCounts": built["QC"]["StrataCounts"],
                "ExpertPredictionRows": int(len(detail)),
                "MergedPredictionRows": int(len(merged)),
                "PredictedPositiveRows": int(merged.PredHasFracture.sum()),
                "PredictedPositiveFraction": float(merged.PredHasFracture.mean()),
                "FracturePoints": int(len(points)),
                "Plot": str(output_dir / "plots" / f"{well_name}_fracture_prediction.png"),
            }
        )

    detail_all = pd.concat(detail_parts, ignore_index=True)
    merged_all = pd.concat(merged_parts, ignore_index=True)
    points_all = pd.concat(point_parts, ignore_index=True)
    six.write_main_and_provenance(
        detail_all,
        output_dir / "predictions" / "all_new_wells_expert_detail_prediction.csv",
        output_dir / "provenance" / "all_new_wells_expert_detail_provenance.csv",
        six.PREDICTION_MAIN_COLUMNS,
        six.PREDICTION_PROVENANCE_COLUMNS,
    )
    six.write_main_and_provenance(
        merged_all,
        output_dir / "predictions" / "all_new_wells_merged_density_prediction.csv",
        output_dir / "provenance" / "all_new_wells_merged_density_provenance.csv",
        six.PREDICTION_MAIN_COLUMNS,
        six.PREDICTION_PROVENANCE_COLUMNS,
    )
    six.write_main_and_provenance(
        points_all,
        output_dir / "predictions" / "all_new_wells_fracture_points.csv",
        output_dir / "provenance" / "all_new_wells_fracture_points_provenance.csv",
        six.POINT_MAIN_COLUMNS,
        six.POINT_PROVENANCE_COLUMNS,
    )
    tdir = output_dir / "input_qc"
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(tdir / "new_well_curve_inventory.csv", index=False, encoding="utf-8-sig")
    pd.concat(candidate_tables, ignore_index=True).to_csv(
        tdir / "time_depth_source_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_time_depth = pd.DataFrame(
        [
            {
                "WellName": row["WellName"],
                **{
                    key: value
                    for key, value in row["TimeDepthSelection"].items()
                    if key not in {"TimeDepthTable"}
                },
            }
            for row in qc_rows
        ]
    )
    selected_time_depth.to_csv(
        tdir / "selected_time_depth_sources.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "WellName": row["WellName"],
                **row["TrajectoryReferenceQC"],
            }
            for row in qc_rows
        ]
    ).to_csv(
        tdir / "trajectory_reference_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage = (
        merged_all.groupby(["WellName", "StrataName"], as_index=False)
        .agg(
            Rows=("DEPT", "size"),
            MDMin=("DEPT", "min"),
            MDMax=("DEPT", "max"),
            TVDMin=("TVD", "min"),
            TVDMax=("TVD", "max"),
            TIMEMin=("TIME", "min"),
            TIMEMax=("TIME", "max"),
            PredictedPositiveRows=("PredHasFracture", "sum"),
            MeanProbability=("PredFractureProb", "mean"),
            MeanPredDensity=("PredDensity", "mean"),
        )
        .merge(
            points_all.groupby(["WellName", "StrataName"], as_index=False)
            .size()
            .rename(columns={"size": "FracturePoints"}),
            on=["WellName", "StrataName"],
            how="left",
            validate="one_to_one",
        )
    )
    coverage["PredictedPositiveFraction"] = (
        coverage.PredictedPositiveRows / coverage.Rows
    )
    coverage["FracturePoints"] = coverage.FracturePoints.fillna(0).astype(int)
    coverage.to_csv(
        tdir / "prediction_coverage_by_well_strata.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_columns = set(
        pd.read_csv(
            output_dir / "predictions" / "all_new_wells_merged_density_prediction.csv",
            nrows=0,
        ).columns
    )
    forbidden_main_fields = {
        "SourceSampleID",
        "SourceGRCurve",
        "SourceDeepCurve",
        "SourceReferenceCurve",
        "SupportSegmentID",
        "MergeRule",
    }
    main_missing_required = sorted(set(six.PREDICTION_MAIN_COLUMNS) - main_columns)
    main_source_fields = sorted(forbidden_main_fields & main_columns)
    checks = {
        "all_wells_predicted": len(well_summaries) == len(config["wells"]),
        "merged_keys_unique": not merged_all.duplicated(["WellName", "DEPT"]).any(),
        "main_required_columns_present": not main_missing_required,
        "main_has_no_source_fields": not main_source_fields,
        "all_predictions_have_time": bool(merged_all["TIME"].notna().all()),
        "all_predictions_have_strata": bool(merged_all["StrataName"].isin(TARGET_STRATA).all()),
    }
    status = "pass" if all(checks.values()) else "fail"
    summary = {
        "version": str(config["version"]),
        "status": status,
        "model_root": str(model_root),
        "model_version": "formal_multilog_expert_library_v2",
        "training_performed": False,
        "time_depth_strategy": config["time_depth_borrow"],
        "well_count": len(well_summaries),
        "wells": well_summaries,
        "merged_prediction_rows": int(len(merged_all)),
        "fracture_points": int(len(points_all)),
        "checks": checks,
        "main_missing_required_columns": main_missing_required,
        "main_source_fields": main_source_fields,
        "caveat": (
            "No imaging fracture truth exists for these two wells. Prediction coverage and "
            "reasonableness can be checked, but accuracy/AUC cannot be computed on the new wells."
        ),
    }
    write_json(output_dir / "new_well_prediction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

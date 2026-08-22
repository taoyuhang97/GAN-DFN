#!/usr/bin/env python3
"""Build TaiGuJie Step1 regular-time and imaging-MD strata contracts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[2]
INVALID_SENTINELS = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
SURFACE_ORDER = ("top", "middle", "bottom")
STRATA_UPPER = "上部复合层"
STRATA_LOWER = "太古界风化壳"
# Curve families required for the first-version GR/RD/RS prediction contract.
# A LAS is only considered usable when it declares at least one mnemonic from
# every family.  Keep this in sync with Step2's read_las_required aliases.
GR_CURVE_FAMILY = ("GR", "GR1", "GRSL")
RD_CURVE_FAMILY = ("RD",)
RS_CURVE_FAMILY = ("RS",)
REQUIRED_CURVE_FAMILIES = (GR_CURVE_FAMILY, RD_CURVE_FAMILY, RS_CURVE_FAMILY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TaiGuJie Step1 strata contracts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("configs") / "taigu_step1_contracts.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def clean_numeric(values: pd.Series | np.ndarray) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    for sentinel in INVALID_SENTINELS:
        out = out.mask(np.isclose(out, sentinel, equal_nan=False))
    return out


def code_to_well_name(code: str) -> str:
    code = code.strip().upper()
    for prefix, chinese in (("CBGX", "埕北古斜"), ("CBG", "埕北古"), ("CB", "埕北")):
        if code.startswith(prefix):
            return chinese + code[len(prefix) :]
    return code


def parse_wellheads(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            rows.append(
                {
                    "WellCode": parts[0].upper(),
                    "WellName": code_to_well_name(parts[0]),
                    "WellX": float(parts[1]),
                    "WellY": float(parts[2]),
                    "KB": float(parts[3]),
                    "TotalDepth": float(parts[4]),
                    "BottomX": float(parts[5]),
                    "BottomY": float(parts[6]),
                }
            )
        except ValueError:
            continue
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"no wellhead rows read from {path}")
    return out.drop_duplicates("WellCode", keep="last").sort_values("WellCode").reset_index(drop=True)


def read_time_depth(path: Path) -> pd.DataFrame:
    rows: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(parts[0]), float(parts[3])])
        except ValueError:
            continue
    out = pd.DataFrame(rows, columns=["TIME", "MD"])
    out["TIME"] = clean_numeric(out["TIME"])
    out["MD"] = clean_numeric(out["MD"])
    out = out.dropna().sort_values("MD").groupby("MD", as_index=False)["TIME"].mean()
    if len(out) < 2:
        raise RuntimeError(f"insufficient MD-TIME rows: {path}")
    return out


def read_surface_nearest(path: Path, query_xy: np.ndarray) -> pd.DataFrame:
    # Layer files are fixed five-column rows: inline, xline, X, Y, TIME(ms).
    raw = np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2, 3, 4))
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise RuntimeError(f"invalid five-column surface: {path}")
    tree = cKDTree(raw[:, 2:4])
    distance, index = tree.query(query_xy, k=1)
    selected = raw[np.asarray(index, dtype=int)]
    return pd.DataFrame(
        {
            "SurfaceInline": selected[:, 0].astype(int),
            "SurfaceXline": selected[:, 1].astype(int),
            "SurfaceX": selected[:, 2],
            "SurfaceY": selected[:, 3],
            "SurfaceTime": selected[:, 4],
            "SurfaceMatchDistance": np.asarray(distance, dtype=float),
        }
    )


def index_las_files(roots: list[Path]) -> dict[str, list[str]]:
    pattern = re.compile(r"^(埕北古斜|埕北古|埕北)(\d+)")
    result: dict[str, list[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.las")):
            match = pattern.match(path.name)
            if match is None:
                continue
            name = f"{match.group(1)}{match.group(2)}"
            result.setdefault(name, []).append(str(path))
    return result


def read_las_curve_mnemonics(path: Path) -> set[str]:
    """Return upper-case mnemonics declared in the LAS ~Curve section (header only)."""
    mnemonics: set[str] = set()
    in_curve = False
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        low = raw_line.strip().lower()
        if low.startswith("~curve") or low == "~c":
            in_curve = True
            continue
        if low.startswith("~"):
            in_curve = False
            if low.startswith("~ascii") or low == "~a":
                break
            continue
        if in_curve and raw_line.strip() and not raw_line.strip().startswith("#"):
            mnemonics.add(raw_line.strip().split()[0].split(".")[0].upper())
    return mnemonics


def has_required_curve_families(mnemonics: set[str]) -> bool:
    return all(bool(mnemonics & set(family)) for family in REQUIRED_CURVE_FAMILIES)


def classify_time(time: pd.Series, top: float, middle: float, bottom: float) -> pd.Series:
    out = pd.Series("OUT_OF_TARGET", index=time.index, dtype="object")
    out.loc[time.ge(top) & time.lt(middle)] = STRATA_UPPER
    out.loc[time.ge(middle) & time.lt(bottom)] = STRATA_LOWER
    return out


def build_regular_contracts(config: dict[str, Any], config_path: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    wellheads = parse_wellheads(resolve_path(config["wellhead_path"], config_path))
    td_dir = resolve_path(config["time_depth_dir"], config_path)
    surface_paths = {name: resolve_path(value, config_path) for name, value in config["surface_paths"].items()}
    las_roots = [resolve_path(item, config_path) for item in config.get("regular_las_roots", [])]
    las_by_well = index_las_files(las_roots)
    max_distance = float(config.get("max_surface_match_distance_m", 100.0))

    query_xy = wellheads[["WellX", "WellY"]].to_numpy(dtype=float)
    surface_results: dict[str, pd.DataFrame] = {}
    for name in SURFACE_ORDER:
        print(f"[Step1] loading {name} surface", flush=True)
        surface_results[name] = read_surface_nearest(surface_paths[name], query_xy)

    summary_rows: list[dict[str, object]] = []
    point_frames: list[pd.DataFrame] = []
    td_manifest_rows: list[dict[str, object]] = []
    for row_index, well in wellheads.iterrows():
        code = str(well["WellCode"])
        name = str(well["WellName"])
        td_path = td_dir / f"{code}.dat"
        top = surface_results["top"].iloc[row_index]
        middle = surface_results["middle"].iloc[row_index]
        bottom = surface_results["bottom"].iloc[row_index]
        horizon_times = [float(top["SurfaceTime"]), float(middle["SurfaceTime"]), float(bottom["SurfaceTime"])]
        distances = [float(top["SurfaceMatchDistance"]), float(middle["SurfaceMatchDistance"]), float(bottom["SurfaceMatchDistance"])]
        order_valid = horizon_times[0] < horizon_times[1] < horizon_times[2]
        distance_valid = max(distances) <= max_distance
        las_paths = las_by_well.get(name, [])
        las_gr_rd_rs_ready = any(has_required_curve_families(read_las_curve_mnemonics(Path(path))) for path in las_paths)
        status = "eligible"
        error = ""
        td = pd.DataFrame(columns=["MD", "TIME"])
        if not td_path.exists():
            status, error = "missing_time_depth", "time-depth file not found"
        else:
            try:
                td = read_time_depth(td_path)
            except Exception as exc:
                status, error = "invalid_time_depth", str(exc)
        if status == "eligible" and not order_valid:
            status, error = "invalid_surface_order", "top/middle/bottom time order failed"
        if status == "eligible" and not distance_valid:
            status, error = "surface_match_too_far", f"max distance {max(distances):.3f}m"
        if status == "eligible" and not las_paths:
            status, error = "missing_regular_las", "no matching LAS in configured roots"
        if status == "eligible" and not las_gr_rd_rs_ready:
            status, error = "missing_gr_rd_rs_curves", "no matched LAS declares GR/RD/RS curve families"
        time_monotonic = bool(td["TIME"].is_monotonic_increasing) if not td.empty else False
        if status == "eligible" and not time_monotonic:
            status, error = "non_monotonic_time_depth", "TIME is not monotonic with MD"

        if not td.empty:
            td = td.copy()
            td["WellCode"] = code
            td["WellName"] = name
            td["TopTime"] = horizon_times[0]
            td["MiddleTime"] = horizon_times[1]
            td["BottomTime"] = horizon_times[2]
            td["StrataName"] = classify_time(td["TIME"], *horizon_times) if order_valid else "OUT_OF_TARGET"
            td["InTarget"] = td["StrataName"].ne("OUT_OF_TARGET")
            td["ContractStatus"] = status
            point_frames.append(td)
        td_manifest_rows.append(
            {
                "WellCode": code,
                "WellName": name,
                "TimeDepthPath": str(td_path),
                "MDMin": float(td["MD"].min()) if not td.empty else np.nan,
                "MDMax": float(td["MD"].max()) if not td.empty else np.nan,
                "TimeMin": float(td["TIME"].min()) if not td.empty else np.nan,
                "TimeMax": float(td["TIME"].max()) if not td.empty else np.nan,
                "TimeDepthRows": int(len(td)),
                "TimeMonotonic": time_monotonic,
            }
        )
        summary_rows.append(
            {
                **well.to_dict(),
                "TimeDepthPath": str(td_path),
                "RegularLasCount": len(las_paths),
                "RegularLasPaths": "|".join(las_paths),
                "RegularLasGrRdRsReady": bool(las_gr_rd_rs_ready),
                "TopTime": horizon_times[0],
                "MiddleTime": horizon_times[1],
                "BottomTime": horizon_times[2],
                "TopSurfaceInline": int(top["SurfaceInline"]),
                "TopSurfaceXline": int(top["SurfaceXline"]),
                "TopSurfaceDistance": distances[0],
                "MiddleSurfaceInline": int(middle["SurfaceInline"]),
                "MiddleSurfaceXline": int(middle["SurfaceXline"]),
                "MiddleSurfaceDistance": distances[1],
                "BottomSurfaceInline": int(bottom["SurfaceInline"]),
                "BottomSurfaceXline": int(bottom["SurfaceXline"]),
                "BottomSurfaceDistance": distances[2],
                "SurfaceOrderValid": order_valid,
                "SurfaceDistanceValid": distance_valid,
                "TimeDepthMonotonic": time_monotonic,
                "UpperCompositeRows": int(td["StrataName"].eq(STRATA_UPPER).sum()) if not td.empty else 0,
                "WeatheredCrustRows": int(td["StrataName"].eq(STRATA_LOWER).sum()) if not td.empty else 0,
                "ContractStatus": status,
                "StatusDetail": error,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values("WellCode")
    td_manifest = pd.DataFrame(td_manifest_rows).sort_values("WellCode")
    points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    summary.to_csv(output_dir / "regular_well_time_strata_contract.csv", index=False, encoding="utf-8-sig")
    td_manifest.to_csv(output_dir / "regular_time_depth_manifest.csv", index=False, encoding="utf-8-sig")
    points.to_csv(output_dir / "regular_time_depth_strata_points.csv", index=False, encoding="utf-8-sig")
    return summary, points


def build_imaging_contracts(config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    for item in config.get("imaging_tvd_contracts", []):
        name = str(item["well_name"])
        status = str(item.get("status", "candidate_supervision"))
        source_kind = str(item.get("source_kind", "imaging_interpretation_manual"))
        intervals = [[float(a), float(b)] for a, b in item.get("interpreted_tvd_intervals", [])]
        boundary = float(item["boundary_tvd"]) if "boundary_tvd" in item else np.nan
        single = str(item["single_strata"]) if "single_strata" in item else ""
        upper = str(item.get("upper_strata", ""))
        lower = str(item.get("lower_strata", ""))
        has_bounds = np.isfinite(boundary)
        strata_names = single or (f"{upper}|{lower}" if upper and lower else "")
        rows.append(
            {
                "WellName": name,
                "InterpretedTVDMin": min(a for a, _ in intervals) if intervals else np.nan,
                "InterpretedTVDMax": max(b for _, b in intervals) if intervals else np.nan,
                "InterpretedTVDIntervals": json.dumps(intervals, ensure_ascii=False) if intervals else "",
                "BoundaryTVD": boundary,
                "StrataNames": strata_names,
                "HasExplicitMdBounds": bool(has_bounds),
                "StrataEvidence": "imaging_interpretation_document_6_2",
                "SourceKind": source_kind,
                "ContractStatus": status,
            }
        )
        qc_rows.append(
            {
                "WellName": name,
                "BoundaryTVD": boundary,
                "BoundaryValid": has_bounds or not intervals,
                "IntervalCount": len(intervals),
                "ContractStatus": status,
                "HasExplicitMdBounds": bool(has_bounds),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "imaging_tvd_strata_contract.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(qc_rows).to_csv(output_dir / "imaging_tvd_strata_qc.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    output_dir = args.output_dir or resolve_path(str(config["output_dir"]), config_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace_output:
            raise RuntimeError(f"output exists; pass --replace-output: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regular_summary, regular_points = build_regular_contracts(config, config_path, output_dir)
    imaging_contract = build_imaging_contracts(config, output_dir)
    acceptance = {
        "regular_well_count": int(len(regular_summary)),
        "regular_eligible_wells": int(regular_summary["ContractStatus"].eq("eligible").sum()),
        "regular_gr_rd_rs_ready_wells": int(regular_summary["RegularLasGrRdRsReady"].sum()),
        "regular_status_counts": {str(k): int(v) for k, v in regular_summary["ContractStatus"].value_counts().items()},
        "regular_time_depth_points": int(len(regular_points)),
        "imaging_contract_rows": int(len(imaging_contract)),
        "imaging_well_count": int(imaging_contract["WellName"].nunique()),
        "strata_names": [STRATA_UPPER, STRATA_LOWER],
        "coordinate_policy": "wellhead_xy_nearest_surface_grid",
        "regular_depth_policy": "own_well_md_to_time_only",
        "imaging_depth_policy": "interpretation_tvd_only",
        "single_strata_wells_have_explicit_md_bounds": False,
        "interpretation_depth_semantics": "tvd_true_vertical_depth",
    }
    (output_dir / "step1_acceptance_summary.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

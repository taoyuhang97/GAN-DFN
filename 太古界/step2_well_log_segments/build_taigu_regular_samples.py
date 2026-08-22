#!/usr/bin/env python3
"""Build TaiGuJie Step2 regular-log segment bottoms (v3).

v3 changes (2026-08-12):
1. Depth semantics: interpretation products (intervals, boundaries, density and
   point depths) are TVD (true vertical depth); LAS DEPT is measured depth (MD).
   Every row therefore gets MD, TVD, X, Y and TIME.
2. Per-well trajectory: deviation file > LAS DEV/AZIM integration > vertical
   assumption. Imaging wells have no deviation files and use LAS DEV/AZIM
   (first valid row assumed vertical above, start at wellhead X/Y).
3. Imaging wells: strata/filtering are applied in TVD space using the Step1
   imaging TVD contract (InterpretedTVDIntervals + BoundaryTVD) and TVD windows.
4. Regular wells: strata are classified per row by querying the three surfaces
   at each row's X/Y (per-row horizon attachment, same as the glutenite line).
5. Segment manifest carries completeness ratio (RawRowsRead/CompleteRows) and
   trajectory provenance; the per-row data files stay lean.
"""

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


INVALID = (-999.25, -9999.0, -99999.0, 9999.0, 99999.0)
CURVES = ("GR", "RD", "RS")
GR_PRIORITY = ("GR", "GR1", "GRSL")
DECISION_DATE = "2026-08-12"
STRATA_UPPER = "上部复合层"
STRATA_LOWER = "太古界风化壳"
OUT_OF_TARGET = "OUT_OF_TARGET"

SEGMENT_COLUMNS = ["MD", "TVD", "X", "Y", "TIME", "StrataName", "GR", "RD", "RS"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TaiGuJie Step2 MD/TVD log segments (v3)")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "taigu_step2_contracts_v3.json")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--replace-output", action="store_true")
    return p.parse_args()


def clean(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    for value in INVALID:
        out = out.mask(np.isclose(out, value, equal_nan=False))
    return out


def read_las_required(path: Path) -> tuple[pd.DataFrame, dict[str, str | None], int]:
    """Read MD/GR/RD/RS columns; return (complete frame, used-curve audit, raw row count)."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_curve = False
    columns: list[str] = []
    data_start = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        low = line.lower()
        if low.startswith("~curve") or low == "~c":
            in_curve = True
            continue
        if low.startswith("~"):
            in_curve = False
            if low.startswith("~ascii") or low == "~a":
                data_start = i + 1
                break
            continue
        if in_curve and line and not line.startswith("#"):
            columns.append(line.split()[0].split(".")[0].upper())
    if data_start is None or not columns:
        raise ValueError(f"invalid LAS sections: {path}")

    md_col = next((c for c in columns if c in ("DEPT", "DEPTH")), None)
    gr_col = next((c for c in GR_PRIORITY if c in columns), None)
    rd_col = "RD" if "RD" in columns else None
    rs_col = "RS" if "RS" in columns else None
    if md_col is None:
        raise ValueError(f"LAS has no depth column: {path}")

    wanted = {"MD": md_col, "GR": gr_col, "RD": rd_col, "RS": rs_col}
    indices = {name: columns.index(col) for name, col in wanted.items() if col is not None}
    max_index = max(indices.values()) if indices else -1

    rows: list[list[float]] = []
    for raw in lines[data_start:]:
        if not raw.strip() or raw.lstrip().startswith("~"):
            continue
        values = raw.split()
        if len(values) <= max_index:
            continue
        row: list[float] = []
        try:
            for name in ("MD", *CURVES):
                idx = indices.get(name)
                row.append(float(values[idx]) if idx is not None else np.nan)
        except (ValueError, IndexError):
            continue
        rows.append(row)

    raw_rows = len(rows)
    out = pd.DataFrame(rows, columns=("MD", *CURVES))
    for c in out.columns:
        out[c] = clean(out[c])
    out = out.dropna(subset=["MD"]).sort_values("MD").drop_duplicates("MD", keep="last").reset_index(drop=True)
    out = out.dropna(subset=list(CURVES)).reset_index(drop=True)
    meta = {"GRSourceCurve": gr_col, "RDSourceCurve": rd_col, "RSSourceCurve": rs_col}
    return out, meta, raw_rows


def index_las_files(roots: list[Path], known_wells: set[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {well: [] for well in known_wells}
    ordered_wells = sorted(known_wells, key=len, reverse=True)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".las":
                continue
            for well in ordered_wells:
                if not path.name.startswith(well):
                    continue
                next_char = path.name[len(well) : len(well) + 1]
                if next_char and next_char.isdigit():
                    continue
                result[well].append(path)
                break
    return result


def interpolate_time_depth(source: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    source = source.dropna(subset=["MD", "TIME"]).sort_values("MD").drop_duplicates("MD")
    x, y = source["MD"].to_numpy(float), source["TIME"].to_numpy(float)
    if len(x) < 2:
        return np.full(len(target), np.nan)
    return np.interp(target, x, y, left=np.nan, right=np.nan)


def read_deviation_file(path: Path) -> pd.DataFrame:
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 8:
            continue
        try:
            rows.append([float(x) for x in p[1:7]])  # MD, TVD, X, Y, XOff, YOff
        except ValueError:
            continue
    out = pd.DataFrame(rows, columns=["MD", "TVD", "X", "Y", "XOff", "YOff"])
    return out.sort_values("MD").drop_duplicates("MD").reset_index(drop=True)


def read_las_dev_azim(path: Path) -> np.ndarray | None:
    cols, in_c, start = [], False, None
    for i, raw in enumerate(Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()):
        low = raw.strip().lower()
        if low.startswith("~curve") or low == "~c":
            in_c = True
            continue
        if low.startswith("~"):
            in_c = False
            if low.startswith("~ascii") or low == "~a":
                start = i + 1
                break
            continue
        if in_c and raw.strip() and not raw.strip().startswith("#"):
            cols.append(raw.strip().split()[0].split(".")[0].upper())
    idx = {n: cols.index(n) for n in ("DEPT", "DEV", "AZIM") if n in cols}
    if len(idx) < 3:
        return None
    rows = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()[start:]:
        v = raw.split()
        if len(v) <= max(idx.values()):
            continue
        try:
            rows.append([float(v[idx[n]]) for n in ("DEPT", "DEV", "AZIM")])
        except (ValueError, IndexError):
            continue
    return np.array(rows) if rows else None


def build_trajectory(well: str, well_code: str, las_files: list[Path], deviation_dir: Path,
                     wellhead_xy: tuple[float, float], bottom_xy: tuple[float, float] | None = None) -> dict[str, Any]:
    """Return MD->(TVD,X,Y) mapping with provenance; vertical fallback is exact for vertical wells."""
    if bottom_xy is not None:
        lateral = np.hypot(bottom_xy[0] - wellhead_xy[0], bottom_xy[1] - wellhead_xy[1])
        if lateral <= 5.0:
            return {
                "source": "vertical_assumption",
                "md": np.array([], dtype=float),
                "tvd": np.array([], dtype=float),
                "x": np.array([], dtype=float),
                "y": np.array([], dtype=float),
                "start_md": np.nan,
                "inc_median_deg": 0.0,
                "note": f"wellhead_bottom_lateral_offset_{lateral:.2f}m_le_5m",
            }
    if well_code:
        dev_file = deviation_dir / f"{well_code}.dat"
        if dev_file.exists():
            track = read_deviation_file(dev_file)
            if len(track) >= 2:
                inc = np.degrees(np.arccos(np.clip(np.diff(track.TVD.to_numpy()) / np.maximum(np.diff(track.MD.to_numpy()), 1e-9), -1, 1)))
                return {
                    "source": "deviation_file",
                    "md": track.MD.to_numpy(float),
                    "tvd": track.TVD.to_numpy(float),
                    "x": track.X.to_numpy(float),
                    "y": track.Y.to_numpy(float),
                    "start_md": float(track.MD.min()),
                    "inc_median_deg": float(np.median(inc)) if len(inc) else np.nan,
                }
    frames = []
    for f in las_files:
        a = read_las_dev_azim(f)
        if a is None:
            continue
        ok = np.isfinite(a[:, 1]) & np.isfinite(a[:, 2]) & (a[:, 1] >= 0) & (a[:, 1] < 90)
        if ok.sum() < 50:
            continue
        frames.append(a[ok])
    if frames:
        a = np.concatenate(frames)
        a = a[np.argsort(a[:, 0], kind="stable")]
        _, first = np.unique(a[:, 0], return_index=True)
        a = a[first]
        dep, dev, az = a[:, 0], np.radians(a[:, 1]), np.radians(a[:, 2])
        d = np.diff(dep)
        tvd = np.concatenate([[dep[0]], dep[0] + np.cumsum(np.cos(dev[1:]) * d)])
        x0, y0 = wellhead_xy
        x = np.concatenate([[x0], x0 + np.cumsum(np.sin(dev[1:]) * np.cos(az[1:]) * d)])
        y = np.concatenate([[y0], y0 + np.cumsum(np.sin(dev[1:]) * np.sin(az[1:]) * d)])
        return {
            "source": "las_dev_integration",
            "md": dep,
            "tvd": tvd,
            "x": x,
            "y": y,
            "start_md": float(dep[0]),
            "inc_median_deg": float(np.degrees(np.median(dev[1:]))),
        }
    return {
        "source": "vertical_assumption",
        "md": np.array([], dtype=float),
        "tvd": np.array([], dtype=float),
        "x": np.array([], dtype=float),
        "y": np.array([], dtype=float),
        "start_md": np.nan,
        "inc_median_deg": np.nan,
    }


def apply_trajectory(df: pd.DataFrame, traj: dict[str, Any], wellhead_xy: tuple[float, float]) -> pd.DataFrame:
    md = df["MD"].to_numpy(float)
    if traj["source"] == "vertical_assumption":
        df["TVD"] = md
        df["X"] = wellhead_xy[0]
        df["Y"] = wellhead_xy[1]
    else:
        t_md = traj["md"]
        df["TVD"] = np.interp(md, t_md, traj["tvd"], left=np.nan, right=np.nan)
        df["X"] = np.interp(md, t_md, traj["x"], left=np.nan, right=np.nan)
        df["Y"] = np.interp(md, t_md, traj["y"], left=np.nan, right=np.nan)
    return df


def classify_time(df: pd.DataFrame, top: np.ndarray, middle: np.ndarray, bottom: np.ndarray) -> pd.Series:
    result = pd.Series(OUT_OF_TARGET, index=df.index, dtype="object")
    result.loc[df["TIME"].ge(top) & df["TIME"].lt(middle)] = STRATA_UPPER
    result.loc[df["TIME"].ge(middle) & df["TIME"].lt(bottom)] = STRATA_LOWER
    return result


def imaging_strata_mask(df: pd.DataFrame, contract_row: pd.Series) -> pd.Series:
    result = pd.Series(OUT_OF_TARGET, index=df.index, dtype="object")
    boundary = contract_row.BoundaryTVD
    if pd.notna(boundary):
        names = str(contract_row.StrataNames).split("|")
        upper = names[0] if names else STRATA_UPPER
        lower = names[1] if len(names) > 1 else STRATA_LOWER
        result.loc[df["TVD"].lt(float(boundary))] = upper
        result.loc[df["TVD"].ge(float(boundary))] = lower
    else:
        result.loc[:] = str(contract_row.StrataNames)
    inside = pd.Series(False, index=df.index)
    intervals = json.loads(contract_row.InterpretedTVDIntervals or "[]")
    for lo, hi in intervals:
        inside |= df["TVD"].ge(float(lo)) & df["TVD"].le(float(hi))
    result.loc[~inside] = OUT_OF_TARGET
    return result


def source_intervals(frame: pd.DataFrame, gap_factor: float) -> list[tuple[float, float]]:
    md = frame["MD"].to_numpy(dtype=float)
    if len(md) == 0:
        return []
    if len(md) == 1:
        return [(float(md[0]), float(md[0]))]
    step = float(np.median(np.diff(md)))
    split = np.diff(md) > max(step * gap_factor, 1.0e-8)
    starts = np.r_[0, np.flatnonzero(split) + 1]
    ends = np.r_[np.flatnonzero(split), len(md) - 1]
    return [(float(md[start]), float(md[end])) for start, end in zip(starts, ends)]


def log_date_from_name(name: str) -> str:
    match = re.search(r"\((\d{4}-\d{2}-\d{2})\)", name)
    return match.group(1) if match else ""


def load_surface(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2, 3, 4))
    if raw.ndim != 2 or raw.shape[1] != 5:
        raise RuntimeError(f"invalid five-column surface: {path}")
    return raw


def query_surface(raw: np.ndarray, query_xy: np.ndarray) -> pd.DataFrame:
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


def apply_tvd_windows(log: pd.DataFrame, windows: list[list[float]]) -> pd.DataFrame:
    if not windows:
        return log
    keep = pd.Series(False, index=log.index)
    for lo, hi in windows:
        keep |= log.TVD.ge(float(lo)) & log.TVD.le(float(hi))
    return log[keep].copy()


def window_id(well: str, tvd: float, windows: dict[str, list[list[float]]]) -> str:
    for i, (lo, hi) in enumerate(windows.get(well, []), start=1):
        if float(lo) <= tvd <= float(hi):
            return f"w{i}"
    return ""


def make_segment(well: str, source_kind: str, seg_seq: int, source_path: Path, log: pd.DataFrame,
                 meta: dict[str, str | None], td_source: str, borrowed_from: str, borrow_distance: float,
                 borrow_coverage: str, window_id_str: str, traj_source: str, raw_rows: int,
                 well_dir: Path) -> tuple[dict[str, object], str]:
    seg_id = f"{well}_seg_{seg_seq:03d}"
    out_path = well_dir / f"{seg_id}.csv"
    well_dir.mkdir(parents=True, exist_ok=True)
    log[SEGMENT_COLUMNS].to_csv(out_path, index=False, encoding="utf-8-sig")
    strata_names = "|".join(sorted(log.StrataName.drop_duplicates().astype(str)))
    complete_rows = int(len(log))
    row = {
        "SegmentID": seg_id,
        "WellName": well,
        "SourceKind": source_kind,
        "SourcePath": str(source_path),
        "LogDate": log_date_from_name(source_path.name),
        "GRSourceCurve": meta.get("GRSourceCurve") or "",
        "RDSourceCurve": meta.get("RDSourceCurve") or "",
        "RSSourceCurve": meta.get("RSSourceCurve") or "",
        "WindowID": window_id_str,
        "MDMin": float(log.MD.min()),
        "MDMax": float(log.MD.max()),
        "TVDMin": float(log.TVD.min()),
        "TVDMax": float(log.TVD.max()),
        "Rows": complete_rows,
        "GRCompleteRows": complete_rows,
        "RDCompleteRows": complete_rows,
        "RSCompleteRows": complete_rows,
        "RawRowsRead": int(raw_rows),
        "CompleteRatio": float(complete_rows / raw_rows) if raw_rows else np.nan,
        "TIMEValidRows": int(log.TIME.notna().sum()),
        "XYValidRows": int((log.X.notna() & log.Y.notna()).sum()),
        "StrataNames": strata_names,
        "InTarget": True,
        "TimeDepthSource": td_source,
        "BorrowedFrom": borrowed_from,
        "BorrowDistanceM": borrow_distance,
        "BorrowCoverageStatus": borrow_coverage,
        "TrajectorySource": traj_source,
        "OutputFilePath": str(out_path),
    }
    return row, seg_id


def process_well_segments(well: str, source_kind: str, candidates: list[tuple[Path, pd.DataFrame, dict[str, str | None], int]],
                          gap_factor: float, windows: dict[str, list[list[float]]], td_source: str,
                          borrowed_from: str, borrow_distance: float, borrow_coverage: str,
                          traj_source: str, out_dir: Path, well_dir: Path) -> tuple[list[dict[str, object]], int, int]:
    segment_rows: list[dict[str, object]] = []
    seg_seq = 0
    for path, log, meta, raw_rows in candidates:
        for lo, hi in source_intervals(log, gap_factor):
            piece = log[(log.MD >= lo) & (log.MD <= hi)].copy().reset_index(drop=True)
            if piece.empty:
                continue
            seg_seq += 1
            wid = window_id(well, float(piece.TVD.median()), windows)
            row, _ = make_segment(well, source_kind, seg_seq, path, piece, meta, td_source,
                                  borrowed_from, borrow_distance, borrow_coverage, wid, traj_source,
                                  raw_rows, well_dir)
            segment_rows.append(row)
    rows = sum(int(r["Rows"]) for r in segment_rows)
    return segment_rows, seg_seq, rows


def main() -> int:
    ns = parse_args()
    cfg = json.loads(ns.config.read_text(encoding="utf-8"))
    out_dir = ns.output_dir or Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parents[2] / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not ns.replace_output:
            raise RuntimeError(f"output exists; pass --replace-output: {out_dir}")
        for child in out_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    step1_dir = Path(cfg["step1_output_dir"])
    if not step1_dir.is_absolute():
        step1_dir = Path(__file__).resolve().parents[2] / step1_dir
    regular_contract = pd.read_csv(step1_dir / "regular_well_time_strata_contract.csv", encoding="utf-8-sig")
    imaging_contract = pd.read_csv(step1_dir / "imaging_tvd_strata_contract.csv", encoding="utf-8-sig")
    td_points = pd.read_csv(step1_dir / "regular_time_depth_strata_points.csv", encoding="utf-8-sig")
    td_by_well = {w: g[["MD", "TIME"]].copy() for w, g in td_points.groupby("WellName")}
    regular_by_well = {str(row.WellName): row for _, row in regular_contract.iterrows()}
    code_to_name = {str(row.WellCode): str(row.WellName) for _, row in regular_contract.iterrows()}
    imaging_by_well = {well: group.iloc[0] for well, group in imaging_contract.groupby("WellName")}

    surfaces = {name: load_surface(Path(path)) for name, path in cfg["surface_paths"].items()}
    imaging_coords = {str(k): (float(v[0]), float(v[1])) for k, v in cfg.get("imaging_well_coords", {}).items()}
    borrow = {str(k): str(v) for k, v in cfg.get("imaging_time_depth_borrow", {}).items()}
    deviation_dir = Path(cfg["deviation_dir"])

    eligible_regular = {w: r for w, r in regular_by_well.items() if r.ContractStatus == "eligible"}
    regular_files = index_las_files([Path(x) for x in cfg["regular_las_roots"]], set(eligible_regular))
    imaging_files = index_las_files([Path(x) for x in cfg["imaging_las_roots"]], set(imaging_by_well))

    gap_factor = float(cfg.get("source_internal_gap_factor", 3.0))
    windows = {str(k): [[float(a), float(b)] for a, b in v] for k, v in cfg.get("well_tvd_windows", {}).items()}

    segment_rows: list[dict[str, object]] = []
    well_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    borrow_audit: dict[str, object] = {}
    trajectory_audit: dict[str, object] = {}

    def add_rejected(well: str, well_type: str, reason: str, evidence: str, related: str) -> None:
        rejected_rows.append({
            "WellName": well, "WellType": well_type, "Reason": reason,
            "EvidenceSummary": evidence, "RelatedFiles": related, "DecisionDate": DECISION_DATE,
        })

    # ---- regular wells ----
    for well, c in sorted(eligible_regular.items()):
        wellhead_xy = (float(c.WellX), float(c.WellY))
        bottom_xy = (float(c.BottomX), float(c.BottomY)) if pd.notna(c.BottomX) and pd.notna(c.BottomY) else None
        traj = build_trajectory(well, str(c.WellCode), regular_files.get(well, []), deviation_dir, wellhead_xy, bottom_xy)
        trajectory_audit[well] = {"Source": traj["source"], "StartMD": traj["start_md"], "InclinationMedianDeg": traj["inc_median_deg"]}
        matched = regular_files.get(well, [])
        parsed, complete, candidates = 0, 0, []
        for path in matched:
            try:
                log, meta, raw_rows = read_las_required(path)
                parsed += 1
            except Exception:
                continue
            if log.empty:
                continue
            complete += 1
            log = apply_trajectory(log, traj, wellhead_xy)
            log["TIME"] = interpolate_time_depth(td_by_well[well], log["MD"].to_numpy(float))
            log["TopTime"] = np.nan
            log["MiddleTime"] = np.nan
            log["BottomTime"] = np.nan
            pos = log.TIME.notna() & log.TVD.notna() & log.X.notna() & log.Y.notna()
            q = log.loc[pos, ["X", "Y"]].to_numpy(float)
            if len(q):
                top = query_surface(surfaces["top"], q)["SurfaceTime"].to_numpy(float)
                mid = query_surface(surfaces["middle"], q)["SurfaceTime"].to_numpy(float)
                bot = query_surface(surfaces["bottom"], q)["SurfaceTime"].to_numpy(float)
                log.loc[pos, "TopTime"] = top
                log.loc[pos, "MiddleTime"] = mid
                log.loc[pos, "BottomTime"] = bot
            log = log.loc[pos & log.TopTime.notna()].copy()
            log["StrataName"] = classify_time(log, log.TopTime.to_numpy(), log.MiddleTime.to_numpy(), log.BottomTime.to_numpy())
            log = log[log.StrataName.ne(OUT_OF_TARGET)].copy()
            log = apply_tvd_windows(log, windows.get(well, []))
            if not log.empty:
                candidates.append((path, log, meta, raw_rows))
        well_dir = out_dir / well
        segs, seg_count, total_rows = process_well_segments(
            well, "regular", candidates, gap_factor, windows, "own", "", np.nan, "", traj["source"], out_dir, well_dir
        )
        segment_rows.extend(segs)
        reason = ""
        if not segs:
            if not matched:
                reason = "missing_regular_las"
            elif not parsed:
                reason = "missing_gr_rd_rs_curves"
            elif not complete:
                reason = "no_complete_gr_rd_rs_coverage"
            else:
                reason = "no_target_interval"
            add_rejected(well, "regular", reason,
                         f"LAS files={len(matched)}, parsed={parsed}, complete={complete}, usable={len(candidates)}",
                         "|".join(str(p) for p in matched))
        well_rows.append({
            "WellName": well, "WellCode": str(c.WellCode), "WellType": "regular",
            "WellX": float(c.WellX), "WellY": float(c.WellY),
            "ContractStatus": str(c.ContractStatus),
            "TimeDepthSource": "own", "TimeDepthPath": str(c.TimeDepthPath),
            "BorrowedFrom": "", "BorrowDistanceM": np.nan, "BorrowCoverageStatus": "",
            "TopTime": float(c.TopTime), "MiddleTime": float(c.MiddleTime), "BottomTime": float(c.BottomTime),
            "InterpretedTVDMin": np.nan, "InterpretedTVDMax": np.nan, "InterpretedTVDIntervals": "",
            "BoundaryTVD": np.nan, "StrataNames": "", "StrataEvidence": "", "HasExplicitMdBounds": "",
            "TrajectorySource": traj["source"], "TrajectoryStartMD": traj["start_md"],
            "InclinationMedianDeg": traj["inc_median_deg"],
            "Step2Status": "eligible" if segs else "rejected",
            "RejectReason": reason, "SegmentCount": seg_count, "TotalRows": total_rows,
        })

    # ---- imaging wells ----
    for well in sorted(imaging_by_well):
        ic = imaging_by_well[well]
        coords = imaging_coords.get(well)
        borrow_code = borrow.get(well)
        neighbor_name = code_to_name.get(borrow_code, "") if borrow_code else ""
        neighbor_td = td_by_well.get(neighbor_name)
        intervals = json.loads(ic.InterpretedTVDIntervals or "[]")
        horizon_rows = {
            name: query_surface(surfaces[name], np.array([[coords[0], coords[1]]], dtype=float)).iloc[0]
            for name in ("top", "middle", "bottom")
        } if coords else {}

        well_dir = out_dir / well

        if coords is None or neighbor_td is None or neighbor_td.empty or not intervals:
            reason = "neighbor_td_inapplicable"
            add_rejected(well, "imaging", reason,
                         f"coords={'missing' if coords is None else 'ok'}, borrow={'missing' if not borrow_code else 'ok'}, "
                         f"neighbor_td={'missing' if neighbor_td is None or neighbor_td.empty else 'ok'}, intervals={len(intervals)}",
                         "")
            well_rows.append({
                "WellName": well, "WellCode": "", "WellType": "imaging",
                "WellX": coords[0] if coords else np.nan, "WellY": coords[1] if coords else np.nan,
                "ContractStatus": str(ic.ContractStatus), "TimeDepthSource": "borrowed_neighbor",
                "TimeDepthPath": "", "BorrowedFrom": borrow_code or "", "BorrowDistanceM": np.nan,
                "BorrowCoverageStatus": "none", "TopTime": np.nan, "MiddleTime": np.nan, "BottomTime": np.nan,
                "InterpretedTVDMin": ic.InterpretedTVDMin, "InterpretedTVDMax": ic.InterpretedTVDMax,
                "InterpretedTVDIntervals": ic.InterpretedTVDIntervals, "BoundaryTVD": ic.BoundaryTVD,
                "StrataNames": ic.StrataNames, "StrataEvidence": ic.StrataEvidence,
                "HasExplicitMdBounds": ic.HasExplicitMdBounds, "TrajectorySource": "",
                "TrajectoryStartMD": np.nan, "InclinationMedianDeg": np.nan,
                "Step2Status": "rejected", "RejectReason": reason, "SegmentCount": 0, "TotalRows": 0,
            })
            continue

        neighbor_row = regular_by_well.get(neighbor_name)
        dist = float(np.hypot(neighbor_row.WellX - coords[0], neighbor_row.WellY - coords[1])) if neighbor_row is not None else np.nan
        traj = build_trajectory(well, "", imaging_files.get(well, []), deviation_dir, coords)
        trajectory_audit[well] = {"Source": traj["source"], "StartMD": traj["start_md"], "InclinationMedianDeg": traj["inc_median_deg"]}

        matched = imaging_files.get(well, [])
        parsed, complete, candidates = 0, 0, []
        for path in matched:
            try:
                log, meta, raw_rows = read_las_required(path)
                parsed += 1
            except Exception:
                continue
            if log.empty:
                continue
            complete += 1
            log = apply_trajectory(log, traj, coords)
            log["TIME"] = interpolate_time_depth(neighbor_td, log["MD"].to_numpy(float))
            log = log[log.TVD.notna()].copy()
            log["StrataName"] = imaging_strata_mask(log, ic)
            log = log[log.StrataName.ne(OUT_OF_TARGET)].copy()
            log = apply_tvd_windows(log, windows.get(well, []))
            if not log.empty:
                candidates.append((path, log, meta, raw_rows))

        sel_md = np.concatenate([cand[1].MD.to_numpy() for cand in candidates]) if candidates else np.array([])
        if len(sel_md):
            td_lo, td_hi = float(neighbor_td.MD.min()), float(neighbor_td.MD.max())
            coverage = "full" if td_lo <= sel_md.min() and td_hi >= sel_md.max() else ("partial" if (td_lo <= sel_md.max() and td_hi >= sel_md.min()) else "none")
        else:
            coverage = "none"
        borrow_audit[well] = {"BorrowedFrom": borrow_code, "DistanceM": dist, "CoverageStatus": coverage}

        segs, seg_count, total_rows = process_well_segments(
            well, "imaging", candidates, gap_factor, windows, "borrowed_neighbor",
            borrow_code or "", dist, coverage, traj["source"], out_dir, well_dir
        )
        segment_rows.extend(segs)
        reason = ""
        if not segs:
            if not matched:
                reason = "missing_imaging_las"
            elif not parsed:
                reason = "missing_gr_rd_rs_curves"
            elif not complete:
                reason = "no_complete_gr_rd_rs_coverage"
            else:
                reason = "no_target_interval"
            add_rejected(well, "imaging", reason,
                         f"LAS files={len(matched)}, parsed={parsed}, complete={complete}, usable={len(candidates)}, "
                         f"borrowed={borrow_code}({coverage})",
                         "|".join(str(p) for p in matched))
        well_rows.append({
            "WellName": well, "WellCode": "", "WellType": "imaging",
            "WellX": coords[0], "WellY": coords[1], "ContractStatus": str(ic.ContractStatus),
            "TimeDepthSource": "borrowed_neighbor", "TimeDepthPath": f"{borrow_code}.dat",
            "BorrowedFrom": borrow_code or "", "BorrowDistanceM": dist, "BorrowCoverageStatus": coverage,
            "TopTime": float(horizon_rows["top"].SurfaceTime), "MiddleTime": float(horizon_rows["middle"].SurfaceTime),
            "BottomTime": float(horizon_rows["bottom"].SurfaceTime),
            "InterpretedTVDMin": ic.InterpretedTVDMin, "InterpretedTVDMax": ic.InterpretedTVDMax,
            "InterpretedTVDIntervals": ic.InterpretedTVDIntervals, "BoundaryTVD": ic.BoundaryTVD,
            "StrataNames": ic.StrataNames, "StrataEvidence": ic.StrataEvidence,
            "HasExplicitMdBounds": ic.HasExplicitMdBounds, "TrajectorySource": traj["source"],
            "TrajectoryStartMD": traj["start_md"], "InclinationMedianDeg": traj["inc_median_deg"],
            "Step2Status": "eligible" if segs else "rejected",
            "RejectReason": reason, "SegmentCount": seg_count, "TotalRows": total_rows,
        })

    # ---- rejected regular wells from the Step1 contract ----
    for well, c in sorted(regular_by_well.items()):
        if c.ContractStatus == "eligible":
            continue
        add_rejected(well, "regular", str(c.ContractStatus),
                     str(c.StatusDetail) if pd.notna(c.StatusDetail) else "",
                     str(c.RegularLasPaths) if pd.notna(c.RegularLasPaths) else "")
        well_rows.append({
            "WellName": well, "WellCode": str(c.WellCode), "WellType": "regular",
            "WellX": float(c.WellX), "WellY": float(c.WellY), "ContractStatus": str(c.ContractStatus),
            "TimeDepthSource": "own", "TimeDepthPath": str(c.TimeDepthPath),
            "BorrowedFrom": "", "BorrowDistanceM": np.nan, "BorrowCoverageStatus": "",
            "TopTime": float(c.TopTime), "MiddleTime": float(c.MiddleTime), "BottomTime": float(c.BottomTime),
            "InterpretedTVDMin": np.nan, "InterpretedTVDMax": np.nan, "InterpretedTVDIntervals": "",
            "BoundaryTVD": np.nan, "StrataNames": "", "StrataEvidence": "", "HasExplicitMdBounds": "",
            "TrajectorySource": "", "TrajectoryStartMD": np.nan, "InclinationMedianDeg": np.nan,
            "Step2Status": "rejected", "RejectReason": str(c.ContractStatus),
            "SegmentCount": 0, "TotalRows": 0,
        })

    manifest_df = pd.DataFrame(segment_rows)
    well_df = pd.DataFrame(well_rows)
    rejected_df = pd.DataFrame(rejected_rows)
    manifest_df.to_csv(out_dir / "taigu_step2_segment_manifest.csv", index=False, encoding="utf-8-sig")
    well_df.to_csv(out_dir / "taigu_step2_well_metadata.csv", index=False, encoding="utf-8-sig")
    rejected_df.to_csv(out_dir / "taigu_step2_rejected_wells.csv", index=False, encoding="utf-8-sig")

    rejected_counts = {str(k): int(v) for k, v in rejected_df.Reason.value_counts().items()} if not rejected_df.empty else {}
    summary = {
        "version": "taigu_step2_regular_v3",
        "decision_date": DECISION_DATE,
        "wells_configured": int(len(well_df)),
        "wells_eligible": int(well_df.Step2Status.eq("eligible").sum()) if not well_df.empty else 0,
        "wells_rejected": int(well_df.Step2Status.eq("rejected").sum()) if not well_df.empty else 0,
        "rejected_by_reason": rejected_counts,
        "segments": int(len(manifest_df)) if not manifest_df.empty else 0,
        "rows": int(manifest_df.Rows.sum()) if not manifest_df.empty else 0,
        "trajectory_sources": {str(k): int(v) for k, v in well_df.TrajectorySource.value_counts().items()} if not well_df.empty else {},
        "imaging_borrow": borrow_audit,
        "trajectory_audit": trajectory_audit,
        "well_tvd_windows": windows,
        "data_layout": "one_segment_per_las_source; per-row MD/TVD/X/Y/TIME; metadata separated",
        "depth_semantics": "interpretation=tvd; las_depth=md; per-row md_to_tvd via trajectory",
    }
    (out_dir / "taigu_step2_acceptance_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

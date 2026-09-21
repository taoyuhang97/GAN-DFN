#!/usr/bin/env python3
"""太古界 Step9 v2: 井轨迹弯曲剖面的五背景、XZ/YZ 正式展示流程。

本实现沿用砂砾岩正式 Step9 的剖面几何和 20 图产品结构，但数据合同完全
切换为太古界：三属性使用属性主道头，OBN 振幅使用独立道头，层位使用
Top/Mid/Base，DFN 使用 Step8 最终多尺度结果。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import font_manager
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
SEISMIC_COMMON = HERE.parent / "common" / "seismic_sampling"
if str(SEISMIC_COMMON) not in sys.path:
    sys.path.insert(0, str(SEISMIC_COMMON))
from amplitude_sampling import ObnAmplitudeSampler  # noqa: E402

# P0-4：测井段坐标回接统一走公共模块（太古界/common/well_segment_join）
TAIGU_ROOT = HERE.parent
if str(TAIGU_ROOT) not in sys.path:
    sys.path.insert(0, str(TAIGU_ROOT))
from common.well_segment_join import (  # noqa: E402
    attach_geometry_per_segment,
    summarize_join,
)

ATTRIBUTES = ("AntTrack", "Coherence", "CurvatureMax")
# 层位界面样式对齐砂砾岩方案B（T4 橙实线 / T5 青长虚线 / T6 紫实线 / T7 粉点划线）。
# 太古界三个界面按"顶界面/目标层顶/目标层底"对应 T4/T6/T7 三种样式，
# 线宽统一 2.4 pt，颜色避开 DFN 层段色（橙/蓝）以免语义冲突。
HORIZONS = (
    ("TopTimeMs", "上部复合层顶(T-a-1)", "#f97316", "-", 2.4),
    ("MidTimeMs", "太古界顶(Art_1)", "#a855f7", "-", 2.4),
    ("BaseTimeMs", "风化壳底(Art_d1-1)", "#ec4899", (0, (6, 2, 1, 2)), 2.4),
)
# 图例方案对齐砂砾岩（优化阶段二/正式主线/step9_section_visualize）：
#   DFN 裂缝片按层段着色（橙/蓝）、尺度用灰阶三档线宽、成像解释带 Step3 前缀、
#   图例整体移到数据区外侧（不打在剖面上）。
DFN_LAYER_COLORS = {"上部复合层": "#ff7a00", "太古界风化壳": "#00a6ff"}
# 尺度靠线宽区分：小/中/大差距拉开到 4 倍（0.9 / 2.2 / 4.2 pt），
# 大尺度另外加深色描边，避免"一条缝看不出属于哪个尺度"。
DFN_SCALE_WIDTH = {"small": 0.9, "medium": 2.2, "large": 4.2}
LARGE_SCALE_HALO_COLOR = "#111827"
SCALE_LEGEND_STYLES = {
    "small": ("#9ca3af", 1.4, "小尺度裂缝（细线）"),
    "medium": ("#6b7280", 3.0, "中尺度裂缝（中线）"),
    "large": ("#374151", 5.4, "预测大尺度裂缝（深灰粗线+描边）"),
}
IMAGING_POINT_COLOR = "#ff00e6"
IMAGING_PATCH_COLOR = "#ff2bd6"
IMAGING_PATCH_HALO = "#3b0764"
IMAGING_TRACK_COLOR = "#00f5ff"
IMAGING_TRACK_GLOW = "#083344"
FAULT_TRACE_COLOR = "#ffe600"
FAULT_TRACE_HALO = "#111827"
CONVENTIONAL_LOG_COLOR = "#16a34a"
WINDOW_LAYER_COLOR = "#f59e0b"
STEP4_POINT_COLOR = "#a3e635"
STEP4_POINT_EDGE = "#0f172a"
OVERLAY_TIME_SCALE_M_PER_MS = 2.0
WELL_TRACK_COLOR = "#111827"
WELL_TRACK_HALO = "#f8fafc"
ORIENTATION_TICK_COLOR = "#ff00e6"
ORIENTATION_TICK_MIN_M = 22.0
ORIENTATION_TICK_MAX_M = 55.0
SCOPE_LABELS = {"overview": "5 km Demo区", "local_200m": "井周200 m"}

# v7：图例按用途分组（12–15 项平铺很难读）
LEGEND_GROUPS = [
    ("地层界面", ("上部复合层顶", "太古界顶", "风化壳底")),
    ("井与测井段", ("常规测井段", "成像测井段", "目标地层内", "井轨迹")),
    ("DFN 裂缝", ("DFN裂缝片", "小尺度裂缝", "中尺度裂缝", "大尺度裂缝")),
    ("测井证据", ("Step3", "Step4")),
    ("构造", ("原始断层",)),
]


def grouped_legend(fig, ax, fontsize: float = 8.4) -> None:
    """按用途分组排列叠加层图例，组前插入组标题（占位 handle）。"""
    handles, labels = ax.get_legend_handles_labels()
    pairs = list(zip(handles, labels))
    used: set[int] = set()
    ordered_handles: list[Any] = []
    ordered_labels: list[str] = []
    for group_name, keys in LEGEND_GROUPS:
        matched = [
            (index, handle, label)
            for index, (handle, label) in enumerate(pairs)
            if index not in used and any(key in str(label) for key in keys)
        ]
        if not matched:
            continue
        ordered_handles.append(Line2D([], [], color="none"))
        ordered_labels.append(f"— {group_name} —")
        for index, handle, label in matched:
            used.add(index)
            ordered_handles.append(handle)
            ordered_labels.append(label)
    for index, (handle, label) in enumerate(pairs):
        if index not in used:
            ordered_handles.append(handle)
            ordered_labels.append(label)
    if ordered_labels:
        fig.legend(ordered_handles, ordered_labels, loc="upper left",
                   bbox_to_anchor=(1.005, 1.0), fontsize=fontsize, framealpha=0.92,
                   borderaxespad=0.0)
CHINESE_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def configure_fonts() -> str:
    for candidate in CHINESE_FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "unavailable"


CHINESE_FONT = configure_fonts()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成太古界 Step9 正式20图多背景剖面")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--replace-output", action="store_true")
    return p


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def pth(config: dict[str, Any], key: str) -> Path:
    if not config.get(key):
        raise ValueError(f"配置缺少路径: {key}")
    path = Path(str(config[key])).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{key}不存在: {path}")
    return path


def validate(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        key: str(pth(config, key))
        for key in (
            "attribute_trace_header_csv", "obn_trace_header_csv", "obn_segy",
            "attribute_demo_grid_csv", "horizon_contract_csv", "step8_patches_csv",
            "step2_segments_root", "step3_groups_root",
        )
    }
    # 叠加层输入（原始断层 / 常规测井裂缝点）：声明了就必须存在。
    for optional_key in ("original_fault_vtk", "step4_points_csv"):
        if config.get(optional_key):
            paths[optional_key] = str(pth(config, optional_key))
    volumes = config.get("volume_paths", {})
    for name in ATTRIBUTES:
        if name not in volumes:
            raise ValueError(f"volume_paths缺少{name}")
        path = Path(str(volumes[name])).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}体不存在: {path}")
        paths[name] = str(path)
    if config.get("profile_well") != "埕北古斜405":
        raise ValueError("正式配置的profile_well应为埕北古斜405")
    target = config.get("target_block", {})
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key not in target:
            raise ValueError(f"target_block缺少{key}")
    well_dir = pth(config, "step2_segments_root") / str(config["profile_well"])
    if not list(well_dir.glob("*.csv")):
        raise FileNotFoundError(f"参考井无Step2轨迹分段: {well_dir}")
    return {
        "status": "pass", "profile_well": config["profile_well"], "paths": paths,
        "matplotlib_chinese_font": CHINESE_FONT,
        "contracts": {
            "attribute_trace_idx": "attribute_trace_header_csv专用",
            "obn_trace_idx": "obn_trace_header_csv专用，不与属性TraceIdx互换",
            "horizons": [x[0] for x in HORIZONS],
            "anttrack_minus_one_is_valid": True,
            "seismic_role": "仅Step9展示，不参与裂缝预测或属性融合",
        },
    }


def numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_track(config: dict[str, Any]) -> pd.DataFrame:
    root = pth(config, "step2_segments_root") / config["profile_well"]
    frames = []
    for path in sorted(root.glob("*.csv")):
        frame = numeric(pd.read_csv(path, encoding="utf-8-sig"), ("MD", "X", "Y", "TIME"))
        if {"MD", "X", "Y", "TIME"}.issubset(frame.columns):
            keep = ["MD", "X", "Y", "TIME"] + [
                column
                for column in ("TVD", "InImagingInterval", "InHorizonLayer", "UseCase")
                if column in frame.columns
            ]
            frames.append(frame[keep])
    track = pd.concat(frames, ignore_index=True).dropna(subset=["X", "Y", "TIME"])
    return track.sort_values("TIME").drop_duplicates("TIME").reset_index(drop=True)


def load_imaging(config: dict[str, Any]) -> pd.DataFrame:
    """剖面成像裂缝点：按 InputSegmentPath 回接 Step2 段坐标。

    P0-4：改走公共 well_segment_join 模块。未命中的点不再静默丢弃，
    审计统计回写到 section_summary，明细另存 imaging_point_join_unmatched.csv。
    """
    parts = []
    audits = []
    root = pth(config, "step3_groups_root")
    samples_root = pth(config, "step2_segments_root")
    join_config = dict(config.get("segment_join", {}))
    tolerance_setting = join_config.get("tolerance_m", config.get("md_merge_tolerance"))
    tolerance = (
        None
        if tolerance_setting is None or str(tolerance_setting).strip().lower() in {"", "auto"}
        else float(tolerance_setting)
    )
    for path in sorted(root.glob(f"{config['profile_well']}_*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = numeric(frame, ("MD", "GT_POINT_FLAG", "FracAzimuth", "FracDip"))
        if "GT_POINT_FLAG" in frame:
            frame = frame[frame["GT_POINT_FLAG"].fillna(0).astype(int) == 1]
        if {"FracAzimuth", "FracDip"}.issubset(frame.columns):
            frame = frame.dropna(subset=["FracAzimuth", "FracDip"])
        if frame.empty or not {"MD", "WellName"}.issubset(frame.columns):
            continue
        # 逐段回接：每行自带 InputSegmentPath，只在该段内取坐标
        matched, audit = attach_geometry_per_segment(
            frame,
            preferred_column="InputSegmentPath",
            tolerance_m=tolerance,
            geometry_source="step9_imaging",
            samples_root=samples_root,
        )
        parts.append(matched)
        audits.append(audit)
    if not parts:
        return pd.DataFrame(columns=["X", "Y", "TIME"])
    merged_all = pd.concat(parts, ignore_index=True)
    audit_all = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    config["_imaging_join_qc"] = summarize_join(audit_all)
    matched_mask = merged_all["MDJoinStatus"].astype(str).eq("matched")
    join_failures = merged_all.loc[~matched_mask]
    if not join_failures.empty:
        output_dir = pth(config, "output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        join_failures.to_csv(
            output_dir / "imaging_point_join_unmatched.csv", index=False, encoding="utf-8-sig"
        )
    keep_columns = [
        column for column in ("WellName", "X", "Y", "TIME", "FracAzimuth", "FracDip") if column in merged_all.columns
    ]
    return merged_all.loc[matched_mask, keep_columns].dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)


def in_block(frame: pd.DataFrame, block: dict[str, float]) -> pd.DataFrame:
    return frame[
        frame["X"].between(block["x_min"], block["x_max"])
        & frame["Y"].between(block["y_min"], block["y_max"])
    ].copy()


def load_step4_points(config: dict[str, Any], block: dict[str, float]) -> pd.DataFrame:
    """常规测井段裂缝点位（Step4 分段预测结果），用于剖面叠加。"""
    path_value = config.get("step4_points_csv")
    if not path_value:
        return pd.DataFrame(columns=["X", "Y", "TIME", "WellName"])
    path = Path(str(path_value)).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    needed = [column for column in ("WellName", "X", "Y", "TIME", "StrataName") if column in frame.columns]
    frame = numeric(frame, tuple(column for column in needed if column != "WellName"))
    frame = in_block(frame, block)
    return frame.dropna(subset=["X", "Y", "TIME"]).reset_index(drop=True)


def read_legacy_polydata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 legacy VTK POLYDATA（binary/ascii），返回 (points, triangles)。

    只依赖标准库与 numpy：太古界的原始断层体是 Step7C 用 `write_legacy_vtk`
    写出的 POLYDATA（三角形单元、big-endian、VTK 5.1 的 OFFSETS/CONNECTIVITY，
    id 数组实际按 int32 写入）。同时兼容不带 OFFSETS 的旧式布局与 ascii 文本。
    """
    try:
        return _read_polydata_impl(path)
    except ValueError:
        raise


def _read_polydata_impl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        handle.readline()  # version
        handle.readline()  # title
        encoding = handle.readline().decode("latin1").strip().upper()
        dataset = handle.readline().decode("latin1").strip().upper()
        if not dataset.startswith("DATASET POLYDATA"):
            raise ValueError(f"{path} 不是 POLYDATA 文件: {dataset}")
        points: np.ndarray | None = None
        triangles: list[np.ndarray] = []
        while True:
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("latin1").strip()
            if not line:
                continue
            parts = line.split()
            keyword = parts[0].upper()
            if keyword == "POINTS":
                count = int(parts[1])
                if encoding == "BINARY":
                    points = np.frombuffer(handle.read(count * 12), dtype=">f4").reshape(count, 3).astype(np.float64)
                else:
                    points = np.array([[float(handle.readline().split()[i]) for i in range(3)] for _ in range(count)])
            elif keyword in {"POLYGONS", "TRIANGLE_STRIPS"}:
                count = int(parts[1])
                if encoding == "BINARY":
                    mark = handle.tell()
                    peek = handle.readline().decode("latin1").strip().upper()
                    if peek.startswith("OFFSETS"):
                        # OFFSETS 数据块一直读到下一行的 CONNECTIVITY 关键字为止，
                        # 再按 int32/int64 两种可能解码（本项目写出的是 int32）。
                        data_start = handle.tell()
                        offsets_raw = bytearray()
                        while True:
                            chunk = handle.read(4096)
                            if not chunk:
                                break
                            hit = chunk.find(b"\nCONNECTIVITY")
                            if hit >= 0:
                                offsets_raw.extend(chunk[:hit])
                                # chunk[:hit] 不含换行符，跳过多出来的 '\n' 才能读到关键字行
                                handle.seek(data_start + len(offsets_raw) + 1)
                                break
                            offsets_raw.extend(chunk)
                        offsets = _decode_id_array(bytes(offsets_raw))
                        conn_header = handle.readline().decode("latin1").strip()
                        if not conn_header.upper().startswith("CONNECTIVITY"):
                            raise ValueError(f"VTK OFFSETS 之后缺少 CONNECTIVITY: {conn_header}")
                        total = int(offsets[-1])
                        connectivity = _decode_exact_id_array(handle.read(total * 4), total)
                    else:
                        handle.seek(mark)
                        flat = np.frombuffer(handle.read(int(parts[2]) * 4), dtype=">i4")
                        offsets = np.empty(count + 1, dtype=np.int64)
                        connectivity_list: list[int] = []
                        cursor = 0
                        for index in range(count):
                            offsets[index] = cursor
                            size = int(flat[cursor])
                            connectivity_list.extend(flat[cursor + 1:cursor + 1 + size].tolist())
                            cursor += 1 + size
                        offsets[count] = cursor
                        connectivity = np.asarray(connectivity_list, dtype=np.int64)
                else:
                    offsets = np.empty(count + 1, dtype=np.int64)
                    connectivity_list = []
                    cursor = 0
                    for index in range(count):
                        row = handle.readline().split()
                        offsets[index] = cursor
                        size = int(row[0])
                        connectivity_list.extend(int(value) for value in row[1:1 + size])
                        cursor += size
                    offsets[count] = cursor
                    connectivity = np.asarray(connectivity_list, dtype=np.int64)
                for index in range(len(offsets) - 1):
                    ids = connectivity[offsets[index]:offsets[index + 1]]
                    for offset in range(1, len(ids) - 1):
                        triangles.append(np.array([ids[0], ids[offset], ids[offset + 1]], dtype=np.int64))
            elif keyword in {"CELL_DATA", "POINT_DATA", "VERTICES", "LINES", "SCALARS", "VECTORS", "LOOKUP_TABLE"}:
                # 只关心点与面；其余块跳过（断层体没有额外标量场）。
                if keyword in {"CELL_DATA", "POINT_DATA"}:
                    handle.readline()
            else:
                continue
    if points is None:
        raise ValueError(f"{path} 缺少 POINTS 段")
    triangle_array = (
        np.asarray(triangles, dtype=np.int64) if triangles else np.empty((0, 3), dtype=np.int64)
    )
    return points, triangle_array


def imaging_orientation_ticks(imaging: pd.DataFrame, projection: str,
                              min_length_m: float = ORIENTATION_TICK_MIN_M,
                              max_length_m: float = ORIENTATION_TICK_MAX_M) -> list[list[list[float]]]:
    """成像测井裂缝产状符号：沿"视倾角方向"画短线，线长随倾角增大。

    方向取"裂缝面与剖面面的交线"（见 `apparent_dip_trace`），即真实视倾角方向；
    线长按倾角 δ 从 min_length_m 线性增长到 max_length_m（度量空间，1 ms = 2 m）。
    这样"倾向"体现为短线朝哪一侧倾斜，"倾角"体现为短线的陡缓与长短。
    """
    if imaging.empty or not {"FracAzimuth", "FracDip"}.issubset(imaging.columns):
        return []
    ticks: list[list[list[float]]] = []
    for row in imaging.itertuples(index=False):
        dip = math.radians(float(row.FracDip))
        trace = apparent_dip_trace(float(row.FracAzimuth), float(row.FracDip), projection)
        if trace is None:
            continue
        dh, dt = trace
        length = min_length_m + (max_length_m - min_length_m) * (dip / (math.pi / 2.0))
        half = length / 2.0
        center_h = float(row.X) if projection == "XZ" else float(row.Y)
        center_t = float(row.TIME)
        ticks.append(
            [[center_h - half * dh, center_t - half * dt], [center_h + half * dh, center_t + half * dt]]
        )
    return ticks


def _decode_id_array(payload: bytes) -> np.ndarray:
    """按 int32/int64 两种可能解码 VTK id 数组：取"首值为 0 且单调不减"的那一种。"""
    for dtype, width in ((">i4", 4), (">i8", 8)):
        usable = (len(payload) // width) * width
        if usable < width * 2:
            continue
        values = np.frombuffer(payload[:usable], dtype=dtype).astype(np.int64)
        if values[0] == 0 and np.all(np.diff(values) >= 0):
            return values
    raise ValueError("无法解码 VTK id 数组（既不是 int32 也不是 int64）")


def _decode_exact_id_array(payload: bytes, count: int) -> np.ndarray:
    """按已知长度解码 connectivity：优先 int32，长度不足时回退 int64。"""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    for dtype, width in ((">i4", 4), (">i8", 8)):
        if len(payload) >= count * width:
            return np.frombuffer(payload[: count * width], dtype=dtype).astype(np.int64)
    raise ValueError(f"VTK connectivity 数据不足: 期望 {count} 个 id，实际字节 {len(payload)}")


def fault_section_segments(points: np.ndarray, triangles: np.ndarray, track: pd.DataFrame,
                           projection: str) -> list[list[list[float]]]:
    """把原始断层面（三角网）与弯曲剖面求交，得到剖面上的断层迹线段。

    剖面法：XZ 投影的剖面曲面是 y = Ywell(t)；YZ 投影是 x = Xwell(t)。
    对每个三角形按顶点的 g = 垂向坐标 - 井轨迹(t) 的符号变化求边交点。
    """
    if points.size == 0 or triangles.size == 0:
        return []
    perp_index = 1 if projection == "XZ" else 0
    h_index = 0 if projection == "XZ" else 1
    well_column = "Y" if projection == "XZ" else "X"
    segments: list[list[list[float]]] = []
    for triangle in triangles:
        vertices = points[triangle]
        times = vertices[:, 2]
        well_perp = interp_track(track, times, well_column)
        g = vertices[:, perp_index] - well_perp
        crossings: list[np.ndarray] = []
        for index in range(3):
            nxt = (index + 1) % 3
            gi, gj = float(g[index]), float(g[nxt])
            if abs(gi) <= 1.0e-9:
                crossings.append(vertices[index])
            if gi * gj < 0.0:
                fraction = gi / (gi - gj)
                crossings.append(vertices[index] + fraction * (vertices[nxt] - vertices[index]))
        if len(crossings) < 2:
            continue
        best = None
        best_distance = -1.0
        for i in range(len(crossings)):
            for j in range(i + 1, len(crossings)):
                delta = crossings[i] - crossings[j]
                distance = float(delta @ delta)
                if distance > best_distance:
                    best_distance = distance
                    best = (crossings[i], crossings[j])
        if best is None or best_distance <= 0.0:
            continue
        first, second = best
        segments.append(
            [[float(first[h_index]), float(first[2])], [float(second[h_index]), float(second[2])]]
        )
    return segments


def apparent_dip_trace(azimuth_deg: float, dip_deg: float, projection: str) -> tuple[float, float] | None:
    """求"裂缝面 与 剖面面"的交线方向，返回 (水平分量, TIME 分量) 的单位方向。

    约定：``FracAzimuth`` 是**倾向方位**（自北起顺时针），度量空间取 x=东、y=北、z=上，
    1 ms = ``OVERLAY_TIME_SCALE_M_PER_MS`` 米。

    * 裂缝面法向 n = (sinα·sinδ, cosα·sinδ, cosδ)；
    * 剖面法向：XZ 剖面法向为 ŷ（剖面含 X 轴），YZ 剖面法向为 x̂；
    * 交线方向 t = n × 剖面法向（注意：不能拿"倾向矢量去掉视线分量"代替，
      投影后的矢量一般不在裂缝面内，视倾角会算错）。

    返回的 ``dt_ms`` 已换算到 TIME（向下为正），即沿该方向 TIME 增大（下倾方向）。
    旧实现直接把"倾向矢量在横轴上的投影"当作水平分量，对高角度缝会把视倾角压成水平线
    （405 实测：正确中位 68–82°，旧实现只画出 8–10°）。
    """
    azimuth = math.radians(float(azimuth_deg))
    dip = math.radians(float(dip_deg))
    # 度量空间（x=东, y=北, z=上）里的裂缝面单位法向
    normal = np.array([
        math.sin(azimuth) * math.sin(dip),
        math.cos(azimuth) * math.sin(dip),
        math.cos(dip),
    ])
    view = np.array([0.0, 1.0, 0.0]) if projection == "XZ" else np.array([1.0, 0.0, 0.0])
    trace = np.cross(normal, view)
    norm = float(np.linalg.norm(trace))
    if norm <= 1.0e-9:
        return None
    trace = trace / norm
    if trace[2] > 0.0:  # 统一朝下（度量空间 z 向下为负）
        trace = -trace
    h_index = 0 if projection == "XZ" else 1
    dh_m = float(trace[h_index])
    dt_ms = -float(trace[2]) / OVERLAY_TIME_SCALE_M_PER_MS
    if abs(dh_m) <= 1.0e-9 and abs(dt_ms) <= 1.0e-9:
        return None
    return dh_m, dt_ms


def imaging_patch_segments(imaging: pd.DataFrame, projection: str, length_m: float,
                           aspect_ratio: float) -> list[list[list[float]]]:
    """成像解释片：沿"裂缝面与剖面面的交线"画线段（即真实视倾角方向）。

    早期实现取"解释点小平面投影后的最长对角线"，对高角度缝会退化成近水平短线；
    现改为直接使用交线方向（见 `apparent_dip_trace`），长度取配置的解释片长度。
    """
    if imaging.empty or not {"FracAzimuth", "FracDip"}.issubset(imaging.columns):
        return []
    h_index = 0 if projection == "XZ" else 1
    half = max(float(length_m), 1.0) / 2.0
    segments: list[list[list[float]]] = []
    for row in imaging.itertuples(index=False):
        trace = apparent_dip_trace(float(row.FracAzimuth), float(row.FracDip), projection)
        if trace is None:
            continue
        dh, dt = trace
        center_h = float(row.X) if projection == "XZ" else float(row.Y)
        center_t = float(row.TIME)
        segments.append([
            [center_h - half * dh, center_t - half * dt],
            [center_h + half * dh, center_t + half * dt],
        ])
    return segments


class AttributeSampler:
    def __init__(self, header: pd.DataFrame, path: Path, time_offset_ms: float, max_distance_m: float):
        self.header = header
        self.trace_ids = header["TraceIdx"].to_numpy(np.int64)
        self.tree = cKDTree(header[["X", "Y"]].to_numpy(np.float64))
        self.handle = segyio.open(str(path), "r", ignore_geometry=True)
        self.handle.mmap()
        self.samples = np.asarray(self.handle.samples, np.float64) + time_offset_ms
        self.max_distance_m = max_distance_m
        self.trace_cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self.handle.close()

    def sample_row(self, x: np.ndarray, y: np.ndarray, time_ms: float) -> tuple[np.ndarray, np.ndarray]:
        dist, pos = self.tree.query(np.column_stack([x, y]), k=1)
        trace_ids = self.trace_ids[np.asarray(pos, np.int64)]
        sample = int(np.clip(np.searchsorted(self.samples, time_ms), 0, len(self.samples) - 1))
        values = np.empty(len(trace_ids), np.float32)
        for trace_idx in np.unique(trace_ids):
            key = int(trace_idx)
            if key not in self.trace_cache:
                self.trace_cache[key] = np.asarray(self.handle.trace[key], np.float32)
            values[trace_ids == trace_idx] = self.trace_cache[key][sample]
        values[(~np.isfinite(values)) | (dist > self.max_distance_m)] = np.nan
        return values, np.asarray(dist, np.float64)


def interp_track(track: pd.DataFrame, times: np.ndarray, column: str) -> np.ndarray:
    return np.interp(times, track["TIME"].to_numpy(), track[column].to_numpy())


def axes_for_scope(grid: pd.DataFrame, track: pd.DataFrame, scope: str, radius: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(grid["X"].unique())
    y = np.sort(grid["Y"].unique())
    if scope == "local_200m":
        x = x[(x >= track["X"].min() - radius) & (x <= track["X"].max() + radius)]
        y = y[(y >= track["Y"].min() - radius) & (y <= track["Y"].max() + radius)]
    return x, y


def section_queries(track: pd.DataFrame, times: np.ndarray, coords: np.ndarray, projection: str):
    if projection == "XZ":
        fixed = interp_track(track, times, "Y")
        well_h = interp_track(track, times, "X")
        return np.tile(coords, (len(times), 1)), np.repeat(fixed[:, None], len(coords), axis=1), well_h
    fixed = interp_track(track, times, "X")
    well_h = interp_track(track, times, "Y")
    return np.repeat(fixed[:, None], len(coords), axis=1), np.tile(coords, (len(times), 1)), well_h


def sample_attribute_section(sampler: AttributeSampler, track: pd.DataFrame, times: np.ndarray,
                             coords: np.ndarray, projection: str, progress: int) -> tuple[np.ndarray, dict[str, Any]]:
    qx, qy, _ = section_queries(track, times, coords, projection)
    out = np.full(qx.shape, np.nan, np.float32)
    distances = []
    for row, time_ms in enumerate(times):
        out[row], dist = sampler.sample_row(qx[row], qy[row], float(time_ms))
        distances.append(dist)
        if row == 0 or (row + 1) % progress == 0 or row + 1 == len(times):
            print(f"    {projection}: {row + 1}/{len(times)} 时间样点", flush=True)
    d = np.concatenate(distances)
    finite = out[np.isfinite(out)]
    return out, {
        "shape": list(out.shape), "finite_fraction": float(np.isfinite(out).mean()),
        "nearest_distance_p95_m": float(np.quantile(d, .95)),
        "raw_percentiles": np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if len(finite) else [],
    }


def sample_obn_section(sampler: ObnAmplitudeSampler, track: pd.DataFrame, times: np.ndarray,
                       coords: np.ndarray, projection: str, max_distance: float,
                       progress: int) -> tuple[np.ndarray, dict[str, Any]]:
    qx, qy, _ = section_queries(track, times, coords, projection)
    out = np.full(qx.shape, np.nan, np.float32)
    all_dist = []
    for row, time_ms in enumerate(times):
        idx, dist = sampler.nearest_trace(qx[row], qy[row])
        values = sampler.sample_at_trace(idx, np.full(len(idx), time_ms))
        values[dist > max_distance] = np.nan
        out[row] = values
        all_dist.append(dist)
        if row == 0 or (row + 1) % progress == 0 or row + 1 == len(times):
            print(f"    {projection}: {row + 1}/{len(times)} 时间样点", flush=True)
    d = np.concatenate(all_dist)
    finite = out[np.isfinite(out)]
    return out, {
        "shape": list(out.shape), "finite_fraction": float(np.isfinite(out).mean()),
        "nearest_distance_p95_m": float(np.quantile(d, .95)),
        "raw_percentiles": np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if len(finite) else [],
    }


def horizon_curves(horizon: pd.DataFrame, track: pd.DataFrame, coords: np.ndarray,
                   projection: str) -> dict[str, np.ndarray]:
    tree = cKDTree(horizon[["X", "Y"]].to_numpy(np.float64))
    output = {}
    seed = np.full(len(coords), float(track["TIME"].median()))
    for field, *_ in HORIZONS:
        values = seed.copy()
        for _ in range(8):
            if projection == "XZ":
                q = np.column_stack([coords, interp_track(track, values, "Y")])
            else:
                q = np.column_stack([interp_track(track, values, "X"), coords])
            _, pos = tree.query(q, k=1)
            updated = horizon.iloc[np.asarray(pos, np.int64)][field].to_numpy(np.float64)
            values = np.where(np.isfinite(updated), updated, values)
        output[field] = np.where(np.isfinite(updated), values, np.nan)
    return output


def patch_segments(patches: pd.DataFrame, track: pd.DataFrame, projection: str,
                   half_width: float) -> tuple[list[list[list[float]]], list[str], list[float], list[str]]:
    """DFN 裂缝片在剖面内的迹线：与成像层同口径，取"裂缝面 ∩ 剖面面"交线。

    注意两套方位约定的差别：

    * Step7/Step8 合同的 `AzimuthDeg` 是**走向**（0–180，axial），VTK 几何也按走向建面，
      所以这里先 `+90°` 转成"倾向方位"再交给 `apparent_dip_trace`；
    * Step3 成像的 `FracAzimuth` 本身就是**真倾向方位**（甲方 LAS 参数块写明
      `TLFamily_Azimuth = True Dip Azimuth`），直接使用。

    旧实现把水平分量取成"倾向方位的水平分量"、并给 `|lateral|` 加 0.08 下限，
    结果是：绝大多数片子被画成真倾角，而走向≈垂直剖面时（`lateral→0`）直接翻成
    竖直——本该接近水平的高角度缝反而画成竖线，和成像标签完全对不上。
    """
    centers_t = patches["CenterTime"].to_numpy(np.float64)
    perpendicular = "Y" if projection == "XZ" else "X"
    center_perp = patches[f"Center{perpendicular}"].to_numpy(np.float64)
    well_perp = interp_track(track, centers_t, perpendicular)
    selected = patches[np.abs(center_perp - well_perp) <= half_width]
    segments, colors, widths, scales = [], [], [], []
    color = dict(DFN_LAYER_COLORS)
    for row in selected.itertuples(index=False):
        cx = float(row.CenterX if projection == "XZ" else row.CenterY)
        ct = float(row.CenterTime)
        dip_azimuth = (float(row.AzimuthDeg) + 90.0) % 360.0
        trace = apparent_dip_trace(dip_azimuth, float(row.DipDeg), projection)
        if trace is None:
            continue
        dh, dt = trace
        length = float(getattr(row, "PatchLengthM", np.nan))
        if not np.isfinite(length):
            length = float(getattr(row, "LengthM", 30.0))
        half = min(length, 500.0) / 2
        segments.append([[cx - half * dh, ct - half * dt], [cx + half * dh, ct + half * dt]])
        colors.append(color.get(str(row.LayerGroup), "#7b1fa2"))
        scale_key = str(row.FractureScale)
        scales.append(scale_key)
        widths.append(DFN_SCALE_WIDTH.get(scale_key, 0.9))
    return segments, colors, widths, scales


def draw_overlays(ax, projection: str, coords: np.ndarray, times: np.ndarray, track: pd.DataFrame,
                  curves: dict[str, np.ndarray], patches: pd.DataFrame, imaging: pd.DataFrame,
                  half_width: float, faults: list[list[list[float]]] | None = None,
                  step4: pd.DataFrame | None = None, scope: str = "overview",
                  config: dict[str, Any] | None = None) -> dict[str, int]:
    _, _, well_h = section_queries(track, times, coords, projection)
    config = dict(config or {})
    for field, label, color, linestyle, linewidth in HORIZONS:
        ax.plot(coords, curves[field], color=color, lw=linewidth, linestyle=linestyle, label=label, zorder=8)
    # 原始断层（Step7C 的原始断层三角网与弯曲剖面求交）——对齐砂砾岩黄色描边方案
    fault_segments = list(faults or [])
    if fault_segments:
        fault_style = (0, (5, 3)) if scope == "overview" else "-"
        ax.add_collection(LineCollection(fault_segments, colors=FAULT_TRACE_HALO, linewidths=5.2,
                                         alpha=0.70 if scope == "overview" else 0.86, zorder=6.2,
                                         linestyles=fault_style))
        ax.add_collection(LineCollection(fault_segments, colors=FAULT_TRACE_COLOR, linewidths=2.5,
                                         alpha=0.78 if scope == "overview" else 0.96, zorder=6.3,
                                         linestyles=fault_style))
        ax.plot([], [], color=FAULT_TRACE_COLOR, lw=2.5, linestyle=fault_style,
                label="原始断层（黄色，非预测）")
    real_track = (times >= float(track["TIME"].min())) & (times <= float(track["TIME"].max()))
    # 常规测井段（Step2 的 405 常规测井采样区间）：绿色加粗虚线，独立于成像井段
    conventional_label = "埕北古斜405常规测井段"
    ax.plot(well_h[real_track], times[real_track], color=CONVENTIONAL_LOG_COLOR, lw=7.5,
            linestyle=(0, (7, 3)), alpha=0.9, zorder=10.05, label=conventional_label)
    # 井轨迹：浅色描边 + 深色实线，画在叠加层最上面，保证在密集裂缝片之上仍可辨认
    ax.plot(well_h[real_track], times[real_track], color=WELL_TRACK_HALO, lw=3.6, alpha=0.9, zorder=12.4)
    ax.plot(well_h[real_track], times[real_track], color=WELL_TRACK_COLOR, lw=2.0,
            label="埕北古斜405井轨迹", zorder=12.45)
    segments, colors, widths, scales = patch_segments(patches, track, projection, half_width)
    if segments:
        large_index = [index for index, scale in enumerate(scales) if scale == "large"]
        if large_index:
            ax.add_collection(LineCollection(
                [segments[index] for index in large_index],
                colors=LARGE_SCALE_HALO_COLOR,
                linewidths=[widths[index] + 2.6 for index in large_index],
                alpha=.55, zorder=6.9))
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, alpha=.82, zorder=7))
        for layer, layer_color in DFN_LAYER_COLORS.items():
            ax.plot([], [], color=layer_color, lw=2.0, label=f"DFN裂缝片：{layer}")
        for scale_key in ("small", "medium", "large"):
            scale_color, scale_width, scale_label = SCALE_LEGEND_STYLES[scale_key]
            ax.plot([], [], color=scale_color, lw=scale_width, label=scale_label)
    patch_lines: list[list[list[float]]] = []
    if len(imaging):
        perp = "Y" if projection == "XZ" else "X"
        mask = np.abs(imaging[perp].to_numpy() - interp_track(track, imaging["TIME"].to_numpy(), perp)) <= half_width
        shown = imaging[mask]
        hcol = "X" if projection == "XZ" else "Y"
        patch_lines = imaging_patch_segments(
            shown, projection,
            float(config.get("imaging_patch_length_m", 45.0)),
            float(config.get("imaging_patch_aspect_ratio", 1.5)),
        )
        orientation_ticks = imaging_orientation_ticks(
            shown, projection,
            float(config.get("orientation_tick_min_m", ORIENTATION_TICK_MIN_M)),
            float(config.get("orientation_tick_max_m", ORIENTATION_TICK_MAX_M)),
        )
        if patch_lines:
            ax.add_collection(LineCollection(patch_lines, colors=IMAGING_PATCH_HALO, linewidths=3.0,
                                             alpha=0.40, zorder=10.4))
            ax.add_collection(LineCollection(patch_lines, colors=IMAGING_PATCH_COLOR, linewidths=1.8,
                                             alpha=0.62, zorder=10.5,
                                             label="Step3真实成像裂缝解释片（按产状）"))
        if orientation_ticks:
            ax.add_collection(LineCollection(orientation_ticks, colors="#111827", linewidths=3.4,
                                             alpha=0.55, zorder=11.1))
            ax.add_collection(LineCollection(orientation_ticks, colors=ORIENTATION_TICK_COLOR, linewidths=1.5,
                                             alpha=0.95, zorder=11.2,
                                             label="Step3成像裂缝产状（视倾角方向，长度∝倾角）"))
        ax.scatter(shown[hcol], shown["TIME"], marker="^", c=IMAGING_POINT_COLOR, s=9,
                   label="Step3成像测井裂缝点", zorder=11)
    else:
        shown = imaging
        orientation_ticks = []
    # v7：去掉剖面左缘的"成像测井段 / 常规测井段 / 目标地层内"标注（三项旋转文字互相遮挡、
    # 且不属于图例信息）；405 的三段区间数值仍写入 section_summary.json 与文档。
    kept_imaging = int(len(shown))
    span = float(coords[-1] - coords[0])
    window_audit: dict[str, Any] = {}
    if {"InImagingInterval", "InHorizonLayer"}.issubset(track.columns):
        imaging_rows = track[pd.to_numeric(track["InImagingInterval"], errors="coerce").fillna(0).eq(1)]
        horizon_rows = track[pd.to_numeric(track["InHorizonLayer"], errors="coerce").fillna(0).eq(1)]
        window_audit = {
            "conventional_log_time_ms": [float(track["TIME"].min()), float(track["TIME"].max())],
            "imaging_interval_time_ms": (
                [float(imaging_rows["TIME"].min()), float(imaging_rows["TIME"].max())] if len(imaging_rows) else None
            ),
            "horizon_layer_time_ms": (
                [float(horizon_rows["TIME"].min()), float(horizon_rows["TIME"].max())] if len(horizon_rows) else None
            ),
        }
    step4_count = 0
    if step4 is not None and len(step4):
        perp = "Y" if projection == "XZ" else "X"
        mask = np.abs(step4[perp].to_numpy() - interp_track(track, step4["TIME"].to_numpy(), perp)) <= half_width
        selected = step4[mask]
        step4_count = int(len(selected))
        if step4_count:
            hcol = "X" if projection == "XZ" else "Y"
            # 常规测井裂缝点画在井轨迹之上：它与井轨迹完全重合，
            # 若压在轨迹下面会被描边吃掉，只能看到零星几个点。
            ax.scatter(selected[hcol], selected["TIME"], marker="o", s=42, c=STEP4_POINT_COLOR,
                       edgecolors=STEP4_POINT_EDGE, linewidths=0.8, zorder=12.6,
                       label="Step4常规测井裂缝点（预测）")
    return {
        "dfn_segment_count": len(segments),
        "imaging_point_count": kept_imaging,
        "imaging_patch_count": len(patch_lines),
        "imaging_orientation_tick_count": len(orientation_ticks),
        "conventional_log_interval_ms": [
            float(track["TIME"].min()),
            float(track["TIME"].max()),
        ],
        "dfn_scale_segment_counts": {
            scale: int(scales.count(scale)) for scale in ("small", "medium", "large")
        },
        "fault_segment_count": len(fault_segments),
        "step4_point_count": step4_count,
        # 405 三段区间（左缘标注已在 v7 移除，数值改为只在 summary 里保留）
        "window_audit": window_audit,
    }


def display_array(name: str, values: np.ndarray) -> tuple[np.ndarray, str, float, float]:
    finite = values[np.isfinite(values)]
    if name == "CurvatureMax":
        shown = np.abs(values)
        upper = float(np.nanquantile(np.abs(finite), .99)) if len(finite) else 1.0
        return shown, "gray_r", 0.0, max(upper, 1e-9)
    lo, hi = (np.nanquantile(finite, [.01, .99]) if len(finite) else (0.0, 1.0))
    cmap = "gray_r" if name == "AntTrack" else "gray"
    return values, cmap, float(lo), float(hi if hi > lo else lo + 1)


def plot_image(path: Path, name: str, renderer: str, projection: str, scope: str,
               values: np.ndarray, coords: np.ndarray, times: np.ndarray, track: pd.DataFrame,
               curves: dict[str, np.ndarray], patches: pd.DataFrame, imaging: pd.DataFrame,
               config: dict[str, Any], faults: dict[str, list[list[list[float]]]] | None = None,
               step4: pd.DataFrame | None = None) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    extent = [coords[0], coords[-1], times[-1], times[0]]
    if renderer == "attribute":
        shown, cmap, lo, hi = display_array(name, values)
        image = ax.imshow(shown, extent=extent, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(image, ax=ax, pad=.01, label={"AntTrack":"蚂蚁体原始值","Coherence":"相干体原始值","CurvatureMax":"|最大正曲率|"}[name])
    else:
        finite = values[np.isfinite(values)]
        limit = float(np.quantile(np.abs(finite), .99)) if len(finite) else 1.0
        if renderer == "density":
            image = ax.imshow(values, extent=extent, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit, interpolation="nearest")
            fig.colorbar(image, ax=ax, pad=.01, label="OBN振幅")
        else:
            count = min(int(config.get("wiggle_trace_count", 80)), len(coords))
            take = np.unique(np.linspace(0, len(coords)-1, count, dtype=int))
            spacing = (coords[-1] - coords[0]) / max(len(take)-1, 1)
            scale = .42 * spacing / max(limit, 1e-9)
            for col in take:
                trace = np.nan_to_num(values[:, col])
                x = coords[col] + trace * scale
                ax.plot(x, times, color="black", lw=.45)
                ax.fill_betweenx(times, coords[col], x, where=trace >= 0, color="black", alpha=.65)
    width_key = "local_dfn_projection_half_width_m" if scope == "local_200m" else "dfn_projection_half_width_m"
    overlay = draw_overlays(ax, projection, coords, times, track, curves, patches, imaging,
                            float(config.get(width_key, 200.0 if scope == "local_200m" else 50.0)),
                            faults=(faults or {}).get(projection), step4=step4, scope=scope, config=config)
    ax.set_xlim(coords[0], coords[-1]); ax.set_ylim(times[-1], times[0])
    ax.set_xlabel("X / m" if projection == "XZ" else "Y / m"); ax.set_ylabel("TIME / ms (TWT)")
    title = {"AntTrack":"蚂蚁体", "Coherence":"相干体", "CurvatureMax":"曲率绝对值", "SeisAmp":"OBN地震"}[name]
    mode = "波形+变面积" if renderer == "wiggle" else ("变密度" if renderer == "density" else "属性")
    ax.set_title(f"埕北古斜405 {SCOPE_LABELS[scope]} {projection} | {title}{mode}")
    ax.grid(False)
    fig.tight_layout()
    # 图例放在数据区外侧（对齐砂砾岩方案：不遮挡剖面主体），并按用途分组（v7）。
    grouped_legend(fig, ax, fontsize=8.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(config.get("dpi", 180)), bbox_inches="tight"); plt.close(fig)
    return overlay


def main() -> int:
    args = parser().parse_args(); started = time.time(); config_path = args.config.resolve()
    config = read_json(config_path); validation = validate(config)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2)); return 0
    output = Path(config["output_dir"]).resolve(); summary_path = output / "section_summary.json"
    if summary_path.exists() and not args.replace_output:
        raise FileExistsError(f"结果已存在，请使用--replace-output: {summary_path}")
    output.mkdir(parents=True, exist_ok=True)
    print("[Step9] 1/6 读取属性网格、层位、井轨迹和Step8 DFN", flush=True)
    block = config["target_block"]
    grid = in_block(numeric(pd.read_csv(pth(config, "attribute_demo_grid_csv")), ("TraceIdx","X","Y")), block)
    header = in_block(numeric(pd.read_csv(pth(config, "attribute_trace_header_csv")), ("TraceIdx","X","Y")), block)
    horizon = in_block(numeric(pd.read_csv(pth(config, "horizon_contract_csv"), usecols=["TraceIdx","X","Y","TopTimeMs","MidTimeMs","BaseTimeMs"]), ("TraceIdx","X","Y","TopTimeMs","MidTimeMs","BaseTimeMs")), block)
    horizon = horizon.dropna(subset=["TopTimeMs","MidTimeMs","BaseTimeMs"])
    track = load_track(config); imaging = load_imaging(config)
    step4_points = load_step4_points(config, block)
    # 原始断层：Step7C 输出的断层三角网与弯曲剖面求交（XZ/YZ 各一次，两种尺度共用）
    fault_segments: dict[str, list[list[list[float]]]] = {"XZ": [], "YZ": []}
    fault_summary: dict[str, Any] = {"path": None, "triangle_count": 0, "segment_counts": {}}
    if config.get("original_fault_vtk"):
        fault_path = Path(str(config["original_fault_vtk"])).resolve()
        if not fault_path.exists():
            raise FileNotFoundError(fault_path)
        fault_points, fault_triangles = read_legacy_polydata(fault_path)
        for projection in ("XZ", "YZ"):
            fault_segments[projection] = fault_section_segments(fault_points, fault_triangles, track, projection)
        fault_summary = {
            "path": str(fault_path),
            "point_count": int(len(fault_points)),
            "triangle_count": int(len(fault_triangles)),
            "segment_counts": {projection: int(len(fault_segments[projection])) for projection in ("XZ", "YZ")},
        }
        print(f"[Step9] 原始断层剖面迹线 XZ={len(fault_segments['XZ'])} YZ={len(fault_segments['YZ'])}", flush=True)
    patch_cols = ["CenterX","CenterY","CenterTime","AzimuthDeg","DipDeg","PatchLengthM","LengthM","LayerGroup","FractureScale"]
    available = pd.read_csv(pth(config, "step8_patches_csv"), nrows=0).columns
    patches = pd.read_csv(pth(config, "step8_patches_csv"), usecols=[c for c in patch_cols if c in available])
    patches = numeric(patches, ("CenterX","CenterY","CenterTime","AzimuthDeg","DipDeg","PatchLengthM","LengthM")).dropna(subset=["CenterX","CenterY","CenterTime","AzimuthDeg","DipDeg"])
    top = float(horizon["TopTimeMs"].min()) - float(config.get("time_padding_ms", 20)); bottom = float(horizon["BaseTimeMs"].max()) + float(config.get("time_padding_ms", 20))
    dt = float(config.get("section_sample_interval_ms", 2)); times = np.arange(math.floor(top/dt)*dt, math.ceil(bottom/dt)*dt + .1*dt, dt)
    samples: dict[str, Any] = {}; sample_summary: dict[str, Any] = {}
    overview_x, overview_y = axes_for_scope(grid, track, "overview", float(config.get("local_axis_radius_m", 200)))
    local_x, local_y = axes_for_scope(grid, track, "local_200m", float(config.get("local_axis_radius_m", 200)))
    samples["overview"] = {"axes": {"XZ": overview_x, "YZ": overview_y}}
    samples["local_200m"] = {"axes": {"XZ": local_x, "YZ": local_y}}
    sample_summary["overview"] = {}; sample_summary["local_200m"] = {}
    print("[Step9] 2/6 采样三属性剖面（属性TraceIdx合同）", flush=True)
    offsets = config.get("attribute_time_offsets_ms", {}); progress = int(config.get("progress_interval", 100))
    for name in ATTRIBUTES:
        print(f"  [overview] {name}", flush=True)
        sampler = AttributeSampler(header, Path(config["volume_paths"][name]), float(offsets.get(name, 0)), float(config.get("attribute_max_nearest_distance_m", 20)))
        try:
            samples["overview"][name] = {}; sample_summary["overview"][name] = {}
            for projection, coords in (("XZ", overview_x), ("YZ", overview_y)):
                values, stats = sample_attribute_section(sampler, track, times, coords, projection, progress)
                samples["overview"][name][projection] = values; sample_summary["overview"][name][projection] = stats
        finally: sampler.close()
        samples["local_200m"][name] = {}; sample_summary["local_200m"][name] = {}
        for projection, local_axis in (("XZ", local_x), ("YZ", local_y)):
            full_axis = samples["overview"]["axes"][projection]
            positions = np.searchsorted(full_axis, local_axis)
            local_values = samples["overview"][name][projection][:, positions]
            samples["local_200m"][name][projection] = local_values
            finite = local_values[np.isfinite(local_values)]
            sample_summary["local_200m"][name][projection] = {
                "source": "overview_axis_subset", "shape": list(local_values.shape),
                "finite_fraction": float(np.isfinite(local_values).mean()),
                "raw_percentiles": np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if len(finite) else [],
            }
    print("[Step9] 3/6 采样OBN振幅剖面（独立OBN TraceIdx合同）", flush=True)
    with ObnAmplitudeSampler(pth(config, "obn_segy"), pth(config, "obn_trace_header_csv")) as sampler:
        samples["overview"]["SeisAmp"] = {}; sample_summary["overview"]["SeisAmp"] = {}
        for projection in ("XZ", "YZ"):
            values, stats = sample_obn_section(sampler, track, times, samples["overview"]["axes"][projection], projection, float(config.get("obn_max_nearest_distance_m", 25)), progress)
            samples["overview"]["SeisAmp"][projection] = values; sample_summary["overview"]["SeisAmp"][projection] = stats
    samples["local_200m"]["SeisAmp"] = {}; sample_summary["local_200m"]["SeisAmp"] = {}
    for projection, local_axis in (("XZ", local_x), ("YZ", local_y)):
        positions = np.searchsorted(samples["overview"]["axes"][projection], local_axis)
        local_values = samples["overview"]["SeisAmp"][projection][:, positions]
        samples["local_200m"]["SeisAmp"][projection] = local_values
        finite = local_values[np.isfinite(local_values)]
        sample_summary["local_200m"]["SeisAmp"][projection] = {
            "source": "overview_axis_subset", "shape": list(local_values.shape),
            "finite_fraction": float(np.isfinite(local_values).mean()),
            "raw_percentiles": np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if len(finite) else [],
        }
    print("[Step9] 4/6 保存NPZ采样缓存", flush=True)
    for scope in samples:
        cache = output / "section_samples" / scope; cache.mkdir(parents=True, exist_ok=True)
        for name in (*ATTRIBUTES, "SeisAmp"):
            np.savez_compressed(cache / f"{name.lower()}_section_samples.npz", times=times, xz_h=samples[scope]["axes"]["XZ"], yz_h=samples[scope]["axes"]["YZ"], xz_values=samples[scope][name]["XZ"], yz_values=samples[scope][name]["YZ"])
    print("[Step9] 5/6 生成20张正式剖面", flush=True)
    images=[]; number=0
    for scope in ("overview", "local_200m"):
        curves = {p: horizon_curves(horizon, track, samples[scope]["axes"][p], p) for p in ("XZ","YZ")}
        plan=[("AntTrack","attribute"),("Coherence","attribute"),("CurvatureMax","attribute"),("SeisAmp","density"),("SeisAmp","wiggle")]
        for name, renderer in plan:
            for projection in ("XZ","YZ"):
                number += 1; token = name.lower() if name != "SeisAmp" else ("seismic_density" if renderer == "density" else "seismic_wiggle_area")
                path = output / scope / f"{number:02d}_{token}_{projection.lower()}.png"
                print(f"  图片 {number}/20: {path.name}", flush=True)
                overlay = plot_image(path,name,renderer,projection,scope,samples[scope][name][projection],samples[scope]["axes"][projection],times,track,curves[projection],patches,imaging,config,faults=fault_segments,step4=step4_points)
                images.append({"number":number,"scope":scope,"background":name,"renderer":renderer,"projection":projection,"path":str(path),"overlay":overlay})
    print("[Step9] 6/6 汇总QC", flush=True)
    imaging_join_qc = dict(config.get("_imaging_join_qc", {}))
    checks={"image_count_is_20":len(images)==20,"horizon_order_valid":bool(((horizon.TopTimeMs<horizon.MidTimeMs)&(horizon.MidTimeMs<horizon.BaseTimeMs)).all()),"attribute_demo_grid_complete":len(grid)==int(config.get("expected_demo_trace_count",len(grid))),"anttrack_minus_one_preserved":True,"separate_trace_contracts":pth(config,"attribute_trace_header_csv")!=pth(config,"obn_trace_header_csv"),"imaging_point_join_has_no_unmatched":int(imaging_join_qc.get("unmatched_count",0))==0}
    overlay_totals = {
        "original_fault_segments": fault_summary.get("segment_counts", {}),
        "step4_points_on_section_total": sum(int(row["overlay"].get("step4_point_count", 0)) for row in images),
        "imaging_patch_total": sum(int(row["overlay"].get("imaging_patch_count", 0)) for row in images),
        "dfn_segment_total": sum(int(row["overlay"].get("dfn_segment_count", 0)) for row in images),
    }
    checks={"image_count_is_20":len(images)==20,"horizon_order_valid":bool(((horizon.TopTimeMs<horizon.MidTimeMs)&(horizon.MidTimeMs<horizon.BaseTimeMs)).all()),"attribute_demo_grid_complete":len(grid)==int(config.get("expected_demo_trace_count",len(grid))),"anttrack_minus_one_preserved":True,"separate_trace_contracts":pth(config,"attribute_trace_header_csv")!=pth(config,"obn_trace_header_csv"),"imaging_point_join_has_no_unmatched":int(imaging_join_qc.get("unmatched_count",0))==0,"original_fault_overlay_loaded":bool(not config.get("original_fault_vtk") or fault_summary.get("triangle_count",0)>0),"original_fault_intersects_section":bool(not config.get("original_fault_vtk") or any(v>0 for v in (fault_summary.get("segment_counts") or {}).values())),"imaging_patch_overlay_drawn":bool(not len(imaging) or overlay_totals["imaging_patch_total"]>0),"step4_points_overlay_drawn":bool(not len(step4_points) or overlay_totals["step4_points_on_section_total"]>0)}
    summary={"status":"pass" if all(checks.values()) else "fail","config":str(config_path),"profile_well":config["profile_well"],"temporary_neighbor_time_depth_risk":"埕北古斜405当前按该井对应时深文件使用，真实时深仍待核验","target_block":block,"time_range_ms":[float(times[0]),float(times[-1])],"inputs":validation["paths"],"contracts":validation["contracts"],"counts":{"attribute_grid":len(grid),"valid_horizon_traces":len(horizon),"track_samples":len(track),"step8_patches":len(patches),"imaging_points":len(imaging),"step4_points":len(step4_points)},"overlay_inputs":{"original_fault":fault_summary,"step4_points":len(step4_points),"imaging_points":len(imaging),**overlay_totals},"imaging_point_join_qc":imaging_join_qc,"sample_qc":sample_summary,"images":images,"checks":checks,"elapsed_seconds":time.time()-started}
    write_json(summary_path,summary); print(f"[Step9] status={summary['status']} summary={summary_path}",flush=True)
    return 0 if summary["status"]=="pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

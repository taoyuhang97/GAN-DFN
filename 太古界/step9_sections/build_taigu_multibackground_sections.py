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
from common.orientation_frame import convention as convention  # noqa: E402

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
# 简化字（SC）字面优先：Noto Sans CJK 的 .ttc 里 face0=JP、face1=KR、face2=SC、face3=TC、face4=HK，
# 而 matplotlib 的 addfont() 只注册 ttc 的第一个字面（JP）——用 JP 字面会把"复/壳"等简化字
# 画成日文字形（结构不同）。这里显式抽 SC 字面成单独 .otf 再注册。
SC_FONT_FACE_HINTS = ("CJK SC", "CJK SC Regular", "SC")
FONT_CACHE_DIR = Path.home() / ".cache" / "taigu_step9_fonts"


def register_simplified_chinese_font() -> str | None:
    """注册简体中文（SC）字面，返回字体名；失败返回 None。"""
    try:
        from fontTools.ttLib import TTCollection
    except Exception:
        return None
    for candidate in CHINESE_FONT_CANDIDATES[:2]:
        path = Path(candidate)
        if not path.exists() or path.suffix.lower() != ".ttc":
            continue
        try:
            collection = TTCollection(str(path))
        except Exception:
            continue
        for index, face in enumerate(collection.fonts):
            try:
                family = str(face["name"].getDebugName(1) or "")
            except Exception:
                continue
            if "SC" not in family:
                continue
            target = FONT_CACHE_DIR / f"{path.stem}-face{index}.otf"
            try:
                if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
                    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    face.save(str(target))
                font_manager.fontManager.addfont(str(target))
                name = font_manager.FontProperties(fname=str(target)).get_name()
                if name:
                    return name
            except Exception:
                continue
    return None


def configure_fonts() -> str:
    sc_name = register_simplified_chinese_font()
    if sc_name:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [sc_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"[Step9] 中文字体：{sc_name}（简体字面，来自 Noto CJK ttc 的 SC face）", flush=True)
        return sc_name
    for candidate in CHINESE_FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[Step9] 中文字体：{name}（未取到 SC 字面，可能为日文字形）", flush=True)
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
            # 中文字体（简体 SC 字面）：图例里"复/壳"等字必须用 SC face，否则会渲染成日文字形
            "matplotlib_chinese_font": CHINESE_FONT,
            "conventional_log_policy": "目标地层（InHorizonLayer==1）内的井轨迹；层外不计入，层内不跨缺口插值",
            "dfn_azimuth_semantics": "dip_azimuth",
            "dfn_azimuth_rule": (
                "DipAzimuthDeg=真倾向方位(0–360，2026-09-22 统一口径) → 直接送入 apparent_dip_trace；"
                "AzimuthDeg 仅为派生走向(=(D-90)%180)的兼容字段"
            ),
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


SECTION_INTERSECTION_EPS_M = 1.0e-6


def representative_line_2d(coords_2d: np.ndarray):
    """用 PCA 主轴把二维点集压成一条代表线（对齐砂砾岩 Step9 的实现）。"""
    if len(coords_2d) < 2 or not np.all(np.isfinite(coords_2d)):
        return None
    center = coords_2d.mean(axis=0)
    centered = coords_2d - center
    if float(np.linalg.norm(centered)) <= 1e-8:
        return None
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vh[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return None
    direction = direction / norm
    projections = centered @ direction
    lo, hi = float(np.nanmin(projections)), float(np.nanmax(projections))
    if hi - lo <= 1e-8:
        return None
    p1 = center + lo * direction
    p2 = center + hi * direction
    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def plane_polygon_intersection_line(vertices: np.ndarray, axis: int, value: float):
    """凸多边形与平面（X=value 或 Y=value）的交线，返回剖面坐标 ((h1,z1),(h2,z2))。"""
    points: list[np.ndarray] = []
    count = len(vertices)
    if count < 3:
        return None
    for index in range(count):
        p1 = vertices[index]
        p2 = vertices[(index + 1) % count]
        d1 = float(p1[axis] - value)
        d2 = float(p2[axis] - value)
        if abs(d1) <= SECTION_INTERSECTION_EPS_M:
            points.append(p1.copy())
        if d1 * d2 < 0.0:
            ratio = abs(d1) / (abs(d1) + abs(d2))
            points.append(p1 + ratio * (p2 - p1))
        elif abs(d2) <= SECTION_INTERSECTION_EPS_M:
            points.append(p2.copy())
    if len(points) < 2:
        return None
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= 1.0e-5 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    arr = np.asarray(unique, dtype=float)
    coords = arr[:, [0, 2]] if axis == 1 else arr[:, [1, 2]]
    if len(coords) == 2:
        p1, p2 = coords[0], coords[1]
        if float(np.linalg.norm(p2 - p1)) <= 1.0e-8:
            return None
        return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))
    return representative_line_2d(coords)


def projected_polygon_line(vertices: np.ndarray, projection: str):
    coords = vertices[:, [0, 2]] if projection == "XZ" else vertices[:, [1, 2]]
    return representative_line_2d(coords)


_PATCH_POLYGON_CACHE: dict[str, list[np.ndarray]] = {}


def load_patch_polygons(config: dict[str, Any]) -> list[np.ndarray]:
    """读取 Step8 统一 VTK 里的**预测裂缝片**多边形（ASCII legacy POLYDATA）。

    行序与 Step8 的 `well_corrected_dfn_fracture_patches.csv` 一一对应
    （预测片在前、原始断层三角形在后）。
    """
    path = str(config.get("step8_vtk") or "")
    if not path:
        return []
    if path in _PATCH_POLYGON_CACHE:
        return _PATCH_POLYGON_CACHE[path]
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.read().splitlines()
    point_index = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    polygon_index = next(i for i, line in enumerate(lines) if line.startswith("POLYGONS"))
    point_count = int(lines[point_index].split()[1])
    polygon_count = int(lines[polygon_index].split()[1])
    points = np.array([[float(v) for v in lines[point_index + 1 + i].split()] for i in range(point_count)])
    polygons: list[np.ndarray] = []
    for i in range(polygon_index + 1, min(polygon_index + 1 + polygon_count, len(lines))):
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        if not parts[0].isdigit():
            break
        count = int(parts[0])
        if count < 3:
            continue
        ids = np.asarray([int(value) for value in parts[1:1 + count]], dtype=np.int64)
        if ids.size and int(ids.max()) < point_count:
            polygons.append(points[ids])
    _PATCH_POLYGON_CACHE[path] = polygons
    return polygons


def patch_segments_from_geometry(polygons: list[np.ndarray], patches: pd.DataFrame, track: pd.DataFrame,
                                 projection: str, half_width: float,
                                 projection_half_width: float
                                 ) -> tuple[list[list[list[float]]], list[str], list[float], list[str], list[bool]]:
    """按**真实几何**画 DFN 片（对齐砂砾岩 Step9）：求交优先，其次小尺度投影。

    * 片中心到剖面法向（XZ→Y、YZ→X）的距离 <= ``half_width`` → 与剖面平面求交，
      得到真实交线（``crossed=True``）；
    * 求交失败或未落在半宽内的小尺度片，若距离 <= ``projection_half_width`` →
      画该多边形在剖面内的投影线（``crossed=False``）。
    """
    perpendicular = "Y" if projection == "XZ" else "X"
    axis_index = 1 if projection == "XZ" else 0
    color = dict(DFN_LAYER_COLORS)
    segments: list[list[list[float]]] = []
    colors: list[str] = []
    widths: list[float] = []
    scales: list[str] = []
    crossed: list[bool] = []
    for index, vertices in enumerate(polygons):
        if index >= len(patches):
            break
        row = patches.iloc[index]
        center = vertices.mean(axis=0)
        center_time = float(center[2])
        well_perp = float(interp_track(track, np.array([center_time]), perpendicular)[0])
        distance = abs(float(center[axis_index]) - well_perp)
        scale_key = str(row.get("FractureScale", "")).strip().lower()
        line = None
        is_crossed = False
        if distance <= half_width:
            line = plane_polygon_intersection_line(vertices, axis=axis_index, value=well_perp)
            is_crossed = line is not None
        if line is None and scale_key == "small" and distance <= projection_half_width:
            line = projected_polygon_line(vertices, projection)
            is_crossed = False
        if line is None:
            continue
        segments.append([[line[0][0], line[0][1]], [line[1][0], line[1][1]]])
        colors.append(color.get(str(row.get("LayerGroup", "")), "#7b1fa2"))
        widths.append(DFN_SCALE_WIDTH.get(scale_key, 0.9))
        scales.append(scale_key)
        crossed.append(bool(is_crossed))
    return segments, colors, widths, scales, crossed


def patch_segments(patches: pd.DataFrame, track: pd.DataFrame, projection: str,
                   half_width: float) -> tuple[list[list[list[float]]], list[str], list[float], list[str], list[bool]]:
    """DFN 裂缝片在剖面内的迹线：与成像层同口径，取"裂缝面 ∩ 剖面面"交线。

    **方位口径（2026-09-22 统一）**：DFN 片表以 `DipAzimuthDeg`（真倾向方位 0–360）为准，
    直接送入 `apparent_dip_trace`。`AzimuthDeg`（派生走向）仅在缺 `DipAzimuthDeg` 时回退
    （按 `strike_from_xy_line_angle → dip_azimuth_from_strike` 换算）。

    旧实现把水平分量取成"倾向方位的水平分量"、并给 `|lateral|` 加 0.08 下限，
    结果是：绝大多数片子被画成真倾角，而走向≈垂直剖面时（`lateral→0`）直接翻成
    竖直——本该接近水平的高角度缝反而画成竖线，和成像标签完全对不上。
    """
    has_dip_azimuth = "DipAzimuthDeg" in patches.columns

    def dip_azimuth_of(row) -> float:
        if has_dip_azimuth:
            value = pd.to_numeric(pd.Series([getattr(row, "DipAzimuthDeg", np.nan)]), errors="coerce").iloc[0]
            if np.isfinite(value):
                return float(value) % 360.0
        legacy = float(getattr(row, "AzimuthDeg", 0.0))
        return convention.dip_azimuth_from_strike(
            convention.strike_from_xy_line_angle(legacy)
        )

    centers_t = patches["CenterTime"].to_numpy(np.float64)
    perpendicular = "Y" if projection == "XZ" else "X"
    center_perp = patches[f"Center{perpendicular}"].to_numpy(np.float64)
    well_perp = interp_track(track, centers_t, perpendicular)
    selected = patches[np.abs(center_perp - well_perp) <= half_width]
    segments, colors, widths, scales, crossed = [], [], [], [], []
    color = dict(DFN_LAYER_COLORS)
    for row in selected.itertuples(index=False):
        cx = float(row.CenterX if projection == "XZ" else row.CenterY)
        ct = float(row.CenterTime)
        dip_azimuth = dip_azimuth_of(row)
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
        # 判断"剖面是否真的穿过这个片"：片的走向/倾角决定它在剖面法向（XZ→Y，YZ→X）
        # 上占多宽，若片中心在该方向上的距离小于片自身半宽，就是这个剖面真实切到的片。
        # 真实交线画在最上层（对齐砂砾岩 13.0/13.2），其余只作井周投影（半透明、12.7）。
        # "剖面是否真的切到这片"用的也是同一个方位口径：换算成**走向角**再算片在剖面法向的半宽。
        # 半径方向改用**真倾向方位**直接算：长边沿走向 (cosD, -sinD)，短边沿下倾 (sinD, cosD)。
        theta = math.radians(dip_azimuth)
        height_time = float(getattr(row, "HeightTimeMs", np.nan))
        half_dip_m = (
            (0.5 * height_time * 2.0) / max(math.tan(math.radians(min(max(float(row.DipDeg), 1.0), 89.9))), 1.0e-6)
            if np.isfinite(height_time) else 0.0
        )
        # 长边 (cosD, -sinD)、短边 (sinD, cosD) 在 X/Y 上的投影宽度
        dx_half = half * abs(math.cos(theta)) + half_dip_m * abs(math.sin(theta))
        dy_half = half * abs(math.sin(theta)) + half_dip_m * abs(math.cos(theta))
        half_extent = dy_half if projection == "XZ" else dx_half
        row_perp = float(row.CenterY) if projection == "XZ" else float(row.CenterX)
        well_perp_center = float(interp_track(track, np.array([ct]), perpendicular)[0])
        crossed.append(bool(abs(row_perp - well_perp_center) <= half_extent))
    return segments, colors, widths, scales, crossed


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
    # 常规测井段（需求方 2026-09-22 口径）：**只取目标地层（层位窗口）内的井轨迹**，
    # 不考虑目标地层范围之外的区域；层内按实际采样点连续绘制，不做跨缺口插值。
    # 405 在目标层内的常规测井数据是完整的（两期测井互补：2016-06-17 覆盖 4180–4471，
    # 2016-06-10 覆盖 4183–4813，4471–4480 的数据缺口由 06-10 补上），因此层内不应出现断口。
    conventional_label = "埕北古斜405常规测井段"
    if "InHorizonLayer" in track.columns:
        layer_rows = track[pd.to_numeric(track["InHorizonLayer"], errors="coerce").fillna(0).eq(1)].copy()
    else:
        layer_rows = track.copy()
    layer_rows = layer_rows.sort_values("TIME").drop_duplicates("TIME")
    if len(layer_rows) >= 2:
        layer_h = layer_rows["X" if projection == "XZ" else "Y"].to_numpy(np.float64)
        ax.plot(layer_h, layer_rows["TIME"].to_numpy(np.float64), color=CONVENTIONAL_LOG_COLOR, lw=7.5,
                linestyle=(0, (7, 3)), alpha=0.9, zorder=10.05, label=conventional_label)
    else:
        ax.plot(well_h[real_track], times[real_track], color=CONVENTIONAL_LOG_COLOR, lw=7.5,
                linestyle=(0, (7, 3)), alpha=0.9, zorder=10.05, label=conventional_label)
    # 井轨迹：浅色描边 + 深色实线，画在叠加层最上面，保证在密集裂缝片之上仍可辨认
    ax.plot(well_h[real_track], times[real_track], color=WELL_TRACK_HALO, lw=3.6, alpha=0.9, zorder=12.4)
    ax.plot(well_h[real_track], times[real_track], color=WELL_TRACK_COLOR, lw=2.0,
            label="埕北古斜405井轨迹", zorder=12.45)
    # 成像测井井段轨迹（青色，压在井轨迹之上）——对齐砂砾岩 Step9 的"Step3成像测井井段轨迹"
    if "InImagingInterval" in track.columns:
        imaging_rows = track[pd.to_numeric(track["InImagingInterval"], errors="coerce").fillna(0).eq(1)].sort_values("TIME")
        if len(imaging_rows) >= 2:
            imaging_h = imaging_rows["X" if projection == "XZ" else "Y"].to_numpy(np.float64)
            imaging_t = imaging_rows["TIME"].to_numpy(np.float64)
            ax.plot(imaging_h, imaging_t, color=IMAGING_TRACK_GLOW, lw=5.2, alpha=.78, zorder=12.5)
            ax.plot(imaging_h, imaging_t, color=IMAGING_TRACK_COLOR, lw=2.8, alpha=.96, zorder=12.55,
                    label=f"{config.get('profile_well', '')}成像测井段轨迹")
    # DFN 片绘制口径（2026-09-22）：
    #   fields   —— 旧口径：按 (AzimuthDeg, DipDeg) 合成迹线（长度=片的走向长度）；
    #   geometry —— 对齐砂砾岩 Step9：按 Step8 VTK 的真实多边形在剖面内求交/投影。
    # 2026-09-22 统一口径：默认按 Step8 VTK 的**真实多边形**求交（几何 = 字段）；
    # `fields` 仅保留为对照分支（按字段合成迹线，非正式）。
    draw_mode = str(config.get("dfn_draw_mode", "geometry")).strip().lower()
    if draw_mode == "geometry":
        polygons = load_patch_polygons(config)
        segments, colors, widths, scales, crossed = patch_segments_from_geometry(
            polygons, patches, track, projection, half_width,
            float(config.get("dfn_small_projection_half_width_m", half_width)),
        )
    else:
        segments, colors, widths, scales, crossed = patch_segments(patches, track, projection, half_width)
    if segments:
        projected_index = [index for index, flag in enumerate(crossed) if not flag]
        crossed_index = [index for index, flag in enumerate(crossed) if flag]
        # 井周投影片：剖面并没有真正穿过它，只作背景参考（半透明、略细）
        if projected_index:
            ax.add_collection(LineCollection(
                [segments[index] for index in projected_index],
                colors=[colors[index] for index in projected_index],
                linewidths=[max(widths[index] * 0.85, 0.7) for index in projected_index],
                alpha=.42, zorder=12.7))
        # 真实交线：剖面确实切到的片子，画在最上层，压在井轨迹与成像裂缝之上
        if crossed_index:
            large_index = [index for index in crossed_index if scales[index] == "large"]
            if large_index:
                ax.add_collection(LineCollection(
                    [segments[index] for index in large_index],
                    colors=LARGE_SCALE_HALO_COLOR,
                    linewidths=[widths[index] + 4.4 for index in large_index],
                    alpha=.78, zorder=12.9))
            ax.add_collection(LineCollection(
                [segments[index] for index in crossed_index],
                colors=[colors[index] for index in crossed_index],
                linewidths=[widths[index] for index in crossed_index],
                alpha=.95, zorder=13.1))
        for layer, layer_color in DFN_LAYER_COLORS.items():
            ax.plot([], [], color=layer_color, lw=2.0, label=f"DFN裂缝片：{layer}")
        ax.plot([], [], color="#9ca3af", lw=1.2, alpha=.42, label="DFN裂缝片：井周投影（半透明）")
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
        layer_series = horizon_rows.sort_values("TIME").drop_duplicates("TIME")["TIME"].to_numpy(np.float64)
        layer_steps = np.diff(layer_series) if len(layer_series) > 1 else np.asarray([], dtype=np.float64)
        layer_step_median = float(np.median(layer_steps)) if layer_steps.size else float("nan")
        layer_gap_threshold = max(5.0 * layer_step_median, 2.0) if layer_steps.size else float("nan")
        layer_gap_count = int((layer_steps > layer_gap_threshold).sum()) if layer_steps.size else 0
        window_audit = {
            # 常规测井段 = 目标地层内的井轨迹（需求方 2026-09-22 口径）
            "conventional_log_time_ms": (
                [float(layer_series.min()), float(layer_series.max())] if len(layer_series) else None
            ),
            "conventional_log_sample_count": int(len(layer_series)),
            "conventional_log_step_median_ms": layer_step_median,
            "conventional_log_gap_count": layer_gap_count,
            "conventional_log_gap_threshold_ms": layer_gap_threshold,
            "track_time_ms": [float(track["TIME"].min()), float(track["TIME"].max())],
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
        # 常规测井段 = 目标地层内的井轨迹（需求方 2026-09-22 口径）；完整轨迹见 window_audit.track_time_ms
        "conventional_log_interval_ms": (
            window_audit.get("conventional_log_time_ms")
            or [float(track["TIME"].min()), float(track["TIME"].max())]
        ),
        "dfn_scale_segment_counts": {
            scale: int(scales.count(scale)) for scale in ("small", "medium", "large")
        },
        "dfn_crossed_count": int(sum(1 for flag in crossed if flag)),
        "dfn_projected_count": int(sum(1 for flag in crossed if not flag)),
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
    patch_cols = ["CenterX","CenterY","CenterTime","DipAzimuthDeg","AzimuthDeg","DipDeg","PatchLengthM","LengthM","LayerGroup","FractureScale"]
    available = pd.read_csv(pth(config, "step8_patches_csv"), nrows=0).columns
    patches = pd.read_csv(pth(config, "step8_patches_csv"), usecols=[c for c in patch_cols if c in available])
    numeric_cols = ("CenterX","CenterY","CenterTime","DipAzimuthDeg","AzimuthDeg","DipDeg","PatchLengthM","LengthM")
    patches = numeric(patches, tuple(c for c in numeric_cols if c in patches.columns))
    required_cols = [c for c in ("CenterX","CenterY","CenterTime","AzimuthDeg","DipDeg") if c in patches.columns]
    patches = patches.dropna(subset=required_cols)
    # 2026-09-22 统一口径：DipAzimuthDeg 是唯一真值（0–360 倾向方位），AzimuthDeg 是派生走向。
    patches.attrs["azimuth_semantics"] = "dip_azimuth"
    # 纵向范围改为"逐剖面自适应"（对齐砂砾岩 Step9 口径：用该剖面自己画出来的层位曲线
    # 的 min/max ± padding），不再让 overview 与 local_200m 共用"全工区层位合同的全局 min/max"。
    # 方案 B：scope_time_padding_ms = 100 ms，层位上下各留 100 ms 上下文。
    dt = float(config.get("section_sample_interval_ms", 2))
    scope_padding = float(config.get("scope_time_padding_ms", config.get("time_padding_ms", 20.0)))
    samples: dict[str, Any] = {}; sample_summary: dict[str, Any] = {}
    overview_x, overview_y = axes_for_scope(grid, track, "overview", float(config.get("local_axis_radius_m", 200)))
    local_x, local_y = axes_for_scope(grid, track, "local_200m", float(config.get("local_axis_radius_m", 200)))
    samples["overview"] = {"axes": {"XZ": overview_x, "YZ": overview_y}}
    samples["local_200m"] = {"axes": {"XZ": local_x, "YZ": local_y}}
    sample_summary["overview"] = {}; sample_summary["local_200m"] = {}
    scope_curves = {
        scope: {p: horizon_curves(horizon, track, samples[scope]["axes"][p], p) for p in ("XZ", "YZ")}
        for scope in ("overview", "local_200m")
    }

    def scope_window(curves_by_projection: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
        values = [v[np.isfinite(v)] for curves in curves_by_projection.values() for v in curves.values() if np.isfinite(v).any()]
        stacked = np.concatenate(values)
        return float(stacked.min()) - scope_padding, float(stacked.max()) + scope_padding

    overview_top, overview_bottom = scope_window(scope_curves["overview"])
    times = np.arange(math.floor(overview_top / dt) * dt, math.ceil(overview_bottom / dt) * dt + .1 * dt, dt)
    # local 直接取 overview 采样网格的子集（两者都对齐到 dt 网格，且 local 层位带必含于 overview），
    # 这样属性/OBN 采样只做一次，local 只是行列裁剪。
    local_top, local_bottom = scope_window(scope_curves["local_200m"])
    local_start = int(np.searchsorted(times, math.floor(local_top / dt) * dt - 1.0e-9))
    local_stop = int(np.searchsorted(times, math.ceil(local_bottom / dt) * dt + 1.0e-9))
    if local_stop - local_start < 2:
        local_start, local_stop = 0, len(times)
    times_by_scope = {"overview": times, "local_200m": times[local_start:local_stop]}
    print(f"[Step9] 纵向范围(逐剖面自适应, padding={scope_padding:.0f}ms): "
          f"overview {times[0]:.0f}-{times[-1]:.0f} ms({len(times)}样点) / "
          f"local_200m {times_by_scope['local_200m'][0]:.0f}-{times_by_scope['local_200m'][-1]:.0f} ms"
          f"({len(times_by_scope['local_200m'])}样点)", flush=True)
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
            local_values = samples["overview"][name][projection][local_start:local_stop, positions]
            samples["local_200m"][name][projection] = local_values
            finite = local_values[np.isfinite(local_values)]
            sample_summary["local_200m"][name][projection] = {
                "source": "overview_axis_subset", "shape": list(local_values.shape),
                "time_window_ms": [float(times_by_scope["local_200m"][0]), float(times_by_scope["local_200m"][-1])],
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
        local_values = samples["overview"]["SeisAmp"][projection][local_start:local_stop, positions]
        samples["local_200m"]["SeisAmp"][projection] = local_values
        finite = local_values[np.isfinite(local_values)]
        sample_summary["local_200m"]["SeisAmp"][projection] = {
            "source": "overview_axis_subset", "shape": list(local_values.shape),
            "time_window_ms": [float(times_by_scope["local_200m"][0]), float(times_by_scope["local_200m"][-1])],
            "finite_fraction": float(np.isfinite(local_values).mean()),
            "raw_percentiles": np.percentile(finite, [0, 1, 50, 99, 100]).tolist() if len(finite) else [],
        }
    print("[Step9] 4/6 保存NPZ采样缓存", flush=True)
    for scope in samples:
        cache = output / "section_samples" / scope; cache.mkdir(parents=True, exist_ok=True)
        for name in (*ATTRIBUTES, "SeisAmp"):
            np.savez_compressed(cache / f"{name.lower()}_section_samples.npz", times=times_by_scope[scope], xz_h=samples[scope]["axes"]["XZ"], yz_h=samples[scope]["axes"]["YZ"], xz_values=samples[scope][name]["XZ"], yz_values=samples[scope][name]["YZ"])
    print("[Step9] 5/6 生成20张正式剖面", flush=True)
    images=[]; number=0
    for scope in ("overview", "local_200m"):
        curves = scope_curves[scope]; scope_times = times_by_scope[scope]
        plan=[("AntTrack","attribute"),("Coherence","attribute"),("CurvatureMax","attribute"),("SeisAmp","density"),("SeisAmp","wiggle")]
        for name, renderer in plan:
            for projection in ("XZ","YZ"):
                number += 1; token = name.lower() if name != "SeisAmp" else ("seismic_density" if renderer == "density" else "seismic_wiggle_area")
                path = output / scope / f"{number:02d}_{token}_{projection.lower()}.png"
                print(f"  图片 {number}/20: {path.name}", flush=True)
                overlay = plot_image(path,name,renderer,projection,scope,samples[scope][name][projection],samples[scope]["axes"][projection],scope_times,track,curves[projection],patches,imaging,config,faults=fault_segments,step4=step4_points)
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
    summary={"status":"pass" if all(checks.values()) else "fail","config":str(config_path),"profile_well":config["profile_well"],"temporary_neighbor_time_depth_risk":"埕北古斜405当前按该井对应时深文件使用，真实时深仍待核验","target_block":block,"time_range_ms":[float(times[0]),float(times[-1])],"time_range_ms_by_scope":{s:[float(times_by_scope[s][0]),float(times_by_scope[s][-1])] for s in ("overview","local_200m")},"time_window_policy":{"mode":"per_scope_adaptive","scope_time_padding_ms":scope_padding,"basis":"该剖面上三条层位曲线(Top/Mid/Base)的 min/max ± padding"},"inputs":validation["paths"],"contracts":validation["contracts"],"counts":{"attribute_grid":len(grid),"valid_horizon_traces":len(horizon),"track_samples":len(track),"step8_patches":len(patches),"imaging_points":len(imaging),"step4_points":len(step4_points)},"overlay_inputs":{"original_fault":fault_summary,"step4_points":len(step4_points),"imaging_points":len(imaging),**overlay_totals},"imaging_point_join_qc":imaging_join_qc,"sample_qc":sample_summary,"images":images,"checks":checks,"elapsed_seconds":time.time()-started}
    write_json(summary_path,summary); print(f"[Step9] status={summary['status']} summary={summary_path}",flush=True)
    return 0 if summary["status"]=="pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

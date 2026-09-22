"""太古界产状口径：**倾向方位 0–360（自北顺时针）+ 倾角**。

本模块是太古界 DFN 链条（Step6B/6C → Step7A/B/C/D → Step8 → Step9）里
**唯一**允许做方位换算的地方。业务脚本不得再自行 ``atan2`` 算方位。

## 坐标与符号约定

* 度量空间：``x = 东``、``y = 北``、``z = 上``（UTM：X≈66.6 万、Y≈424.2 万，已核对）。
* ``DipAzimuthDeg``（真倾向方位，``[0,360)``）：缝面往哪边倒，自正北起顺时针。
* ``DipDeg``（倾角，``(0,90]``）：缝面与水平面的夹角。
* 派生走向 ``StrikeDeg = (DipAzimuthDeg - 90) mod 180``（axial，``[0,180)``）。
* ``TIME`` 轴向下为正；1 ms = :data:`TIME_SCALE_M_PER_MS` 米。

## 三套历史口径与换算关系

本仓历史上同时存在三套"方位角"写法，混用是 2026-09-22 那轮剖面异常的根因：

1. **真倾向方位**（甲方 ``Dip Azimuth`` 列；Step3 ``Frac_Azimuth``）——唯一真值口径；
2. **罗盘走向**（Step6B/6C 的 ``atan2(main[0], main[1])``）——``S = (D-90)%180``；
3. **XY 平面角**（Step7A 脊线 / Step7B 带 PCA / 所有 VTK 顶点
   ``strike=[cos A, sin A]``）——记为 ``A``。

关键换算（已数值验证，见 ``test_convention.py``）：

* 给定真值 ``D``，旧式顶点角应取 ``A = (180 - D) % 180 = (90 - S) % 180``；
* 反过来，历史产物里的 ``A`` 换算成罗盘走向是 ``S = (90 - A) % 180``。

注意常见的错误写法：把真倾向方位直接当走向（``D % 180``）或只做
``(D + 90) % 180``（那是**罗盘走向**，不是顶点角 ``A``），两者分别差 90° 与 55°。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

#: 度量空间坐标帧（x=东, y=北, z=上）。
FRAME: tuple[str, str, str] = ("east", "north", "up")

#: TIME 轴换算：1 ms = 2 m。
TIME_SCALE_M_PER_MS: float = 2.0

#: 走向/轴向量的周期。
AXIAL_PERIOD_DEG: float = 180.0

_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 基础规范化
# --------------------------------------------------------------------------- #
def wrap_dip_azimuth(value: float) -> float:
    """倾向方位规范化到 ``[0, 360)``。``NaN`` 原样返回。"""
    value = float(value)
    if not math.isfinite(value):
        return value
    return value % 360.0


def wrap_strike(value: float) -> float:
    """走向规范化到 ``[0, 180)``。``NaN`` 原样返回。"""
    value = float(value)
    if not math.isfinite(value):
        return value
    return value % 180.0


def clip_dip(value: float, *, minimum: float = 1.0, maximum: float = 89.0) -> float:
    """倾角裁剪到 ``[minimum, maximum]``（默认 ``[1, 89]``，避免退化几何）。"""
    return float(np.clip(float(value), float(minimum), float(maximum)))


# --------------------------------------------------------------------------- #
# 倾向方位 ⇄ 走向
# --------------------------------------------------------------------------- #
def strike_from_dip_azimuth(dip_azimuth_deg: float) -> float:
    """真倾向方位 → 罗盘走向：``(D - 90) mod 180``。"""
    return wrap_strike(wrap_dip_azimuth(dip_azimuth_deg) - 90.0)


def dip_azimuth_from_strike(strike_deg: float) -> float:
    """罗盘走向 → 真倾向方位：``(S + 90) mod 360``。"""
    return wrap_dip_azimuth(wrap_strike(strike_deg) + 90.0)


def xy_line_angle_from_strike(strike_deg: float) -> float:
    """罗盘走向 → 旧式"长边方向角"``A``（``strike=[cosA, sinA]``，XY 平面）：``(90 - S) mod 180``。"""
    return wrap_strike(90.0 - wrap_strike(strike_deg))


def strike_from_xy_line_angle(xy_line_angle_deg: float) -> float:
    """旧式"长边方向角"``A`` → 罗盘走向：``(90 - A) mod 180``（历史产物换算用）。"""
    return wrap_strike(90.0 - wrap_strike(xy_line_angle_deg))


# --------------------------------------------------------------------------- #
# 面法向
# --------------------------------------------------------------------------- #
def normal_from_dip_azimuth_dip(dip_azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """(真倾向方位, 倾角) → 单位面法向（``z`` 分量为正，即取向上法向）。

    ``n = (sinD·sinδ, cosD·sinδ, cosδ)``

    该法向与"下倾方向"``u = (sinD·cosδ, cosD·cosδ, -sinδ)`` 正交，且 ``n_z = cosδ > 0``；
    水平分量 ``(n_x, n_y)`` 与下倾方向的水平投影同向，因此
    ``D = atan2(n_x, n_y)``（罗盘口径）。
    """
    dip_azimuth = math.radians(wrap_dip_azimuth(dip_azimuth_deg))
    dip = math.radians(float(dip_deg))
    normal = np.array(
        [
            math.sin(dip_azimuth) * math.sin(dip),
            math.cos(dip_azimuth) * math.sin(dip),
            math.cos(dip),
        ],
        dtype=np.float64,
    )
    return _normalize(normal)


def dip_azimuth_dip_from_normal(normal: Sequence[float]) -> tuple[float, float]:
    """单位面法向 → ``(真倾向方位[0,360), 倾角[0,90])``。

    法向正负随意；结果取"向上法向"对应的下倾方向，与
    :func:`normal_from_dip_azimuth_dip` 互逆。
    """
    arr = _normalize(np.asarray(normal, dtype=np.float64))
    if float(np.dot(arr, _UP)) < 0.0:
        arr = -arr
    dip = float(math.degrees(math.acos(float(np.clip(abs(arr[2]), 0.0, 1.0)))))
    horizontal = np.array([arr[0], arr[1]], dtype=np.float64)
    if float(np.linalg.norm(horizontal)) < 1e-12:
        return 0.0, dip
    dip_azimuth = math.degrees(math.atan2(float(horizontal[0]), float(horizontal[1])))
    return wrap_dip_azimuth(dip_azimuth), dip


def normal_from_dip_azimuth_dip_depth(dip_azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """``(倾向方位, 倾角)`` → 单位面法向，**第三维为"向下"**的帧 ``(E, N, Down)``。

    专供 Step6B/6C 这类直接用 ``(x, y, TIME*scale)``（TIME 向下为正）做 SVD 的环节，
    避免把"向上/向下"口径又混一次。等价于 :func:`normal_from_dip_azimuth_dip` 把 z 取反。
    """
    normal = normal_from_dip_azimuth_dip(dip_azimuth_deg, dip_deg)
    return np.array([normal[0], normal[1], -normal[2]], dtype=np.float64)


def dip_azimuth_dip_from_normal_depth(normal: Sequence[float]) -> tuple[float, float]:
    """:func:`normal_from_dip_azimuth_dip_depth` 的逆：``(E, N, Down)`` 帧 → ``(倾向方位, 倾角)``。"""
    arr = np.asarray(normal, dtype=np.float64)
    return dip_azimuth_dip_from_normal(np.array([arr[0], arr[1], -arr[2]], dtype=np.float64))


# --------------------------------------------------------------------------- #
# 剖面交线（裂缝面 ∩ 剖面面）
# --------------------------------------------------------------------------- #
def trace_on_section(
    dip_azimuth_deg: float,
    dip_deg: float,
    projection: str,
    *,
    time_scale_m_per_ms: float = TIME_SCALE_M_PER_MS,
) -> tuple[float, float] | None:
    """求"裂缝面 ∩ 剖面面"的交线方向，返回 ``(水平分量, TIME 分量)``（未归一化的单位向量分量）。

    ``projection`` 取 ``"XZ"``（剖面含 X 轴，法向 ``ŷ``）或 ``"YZ"``（剖面含 Y 轴，法向 ``x̂``）。
    TIME 分量已换算成"向下为正"的 ms 方向，即沿该方向 TIME 增大（下倾方向）。

    与 ``太古界/step9_sections`` 里已用 405 真值验证过的 ``apparent_dip_trace`` 等价
    （见 ``test_convention.py`` 的闭式公式比对）。
    """
    key = str(projection).strip().upper()
    if key == "XZ":
        view = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        horizontal_index = 0
    elif key == "YZ":
        view = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        horizontal_index = 1
    else:
        raise ValueError(f"unsupported projection: {projection!r} (expect 'XZ' or 'YZ')")

    normal = normal_from_dip_azimuth_dip(dip_azimuth_deg, dip_deg)
    trace = np.cross(normal, view)
    norm = float(np.linalg.norm(trace))
    if norm <= 1e-9:
        return None
    trace = trace / norm
    if trace[2] > 0.0:
        # 度量空间 z 向上 => trace[2] > 0 表示交线朝上，翻成朝下（TIME 增大）。
        trace = -trace
    dh_m = float(trace[horizontal_index])
    dt_ms = -float(trace[2]) / max(float(time_scale_m_per_ms), 1e-9)
    if abs(dh_m) <= 1e-12 and abs(dt_ms) <= 1e-12:
        return None
    return dh_m, dt_ms


def section_apparent_dip_deg(
    dip_azimuth_deg: float,
    dip_deg: float,
    projection: str,
    *,
    time_scale_m_per_ms: float = TIME_SCALE_M_PER_MS,
) -> float | None:
    """剖面上的视倾角（度），由 :func:`trace_on_section` 换算；退化时返回 ``None``。"""
    trace = trace_on_section(
        dip_azimuth_deg, dip_deg, projection, time_scale_m_per_ms=time_scale_m_per_ms
    )
    if trace is None:
        return None
    dh_m, dt_ms = trace
    vertical_m = abs(dt_ms) * float(time_scale_m_per_ms)
    if abs(dh_m) <= 1e-12:
        return 90.0
    return float(math.degrees(math.atan2(vertical_m, abs(dh_m))))


# --------------------------------------------------------------------------- #
# 统计辅助
# --------------------------------------------------------------------------- #
def circular_mean_dip_azimuth(
    values: Iterable[float], weights: Sequence[float] | None = None
) -> float:
    """倾向方位的全圆向量平均（``[0,360)``）；全零向量时返回 0。"""
    arr = np.asarray([wrap_dip_azimuth(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    rad = np.deg2rad(arr)
    if weights is None:
        weight = np.ones_like(arr)
    else:
        weight = np.asarray(weights, dtype=np.float64)
        if weight.size != arr.size:
            raise ValueError("weights length mismatch")
    vector = np.sum(weight * np.exp(1j * rad))
    if abs(vector) <= 1e-15:
        return 0.0
    return wrap_dip_azimuth(math.degrees(float(np.angle(vector))))


def axial_distance_deg(a: float, b: float) -> float:
    """两条走向/轴线的最小夹角，``[0, 90]``。"""
    delta = abs((wrap_strike(a) - wrap_strike(b)) % AXIAL_PERIOD_DEG)
    return float(min(delta, AXIAL_PERIOD_DEG - delta))


def angular_distance_deg(a: float, b: float) -> float:
    """两个倾向方位的最小夹角，``[0, 180]``。"""
    delta = abs((wrap_dip_azimuth(a) - wrap_dip_azimuth(b)) % 360.0)
    return float(min(delta, 360.0 - delta))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm

"""S1 验收单测：产状口径换算的正确性。

运行：``python3 太古界/common/orientation_frame/test_convention.py``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.orientation_frame import convention as cv  # noqa: E402

TOL = 1e-9
DIP_AZIMUTHS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.9, 145.0, 221.5]
DIPS = [1.0, 15.0, 45.0, 70.0, 89.0]


def test_normal_roundtrip() -> None:
    for dip_azimuth in DIP_AZIMUTHS:
        for dip in DIPS:
            normal = cv.normal_from_dip_azimuth_dip(dip_azimuth, dip)
            assert abs(float(np.linalg.norm(normal)) - 1.0) <= TOL, (dip_azimuth, dip)
            got_azimuth, got_dip = cv.dip_azimuth_dip_from_normal(normal)
            assert cv.angular_distance_deg(got_azimuth, dip_azimuth) <= 1e-7, (
                dip_azimuth,
                dip,
                got_azimuth,
            )
            assert abs(got_dip - dip) <= 1e-7, (dip_azimuth, dip, got_dip)
            # 向下三维帧（Step6B/6C 的 SVD 帧）同样往返一致
            normal_depth = cv.normal_from_dip_azimuth_dip_depth(dip_azimuth, dip)
            assert normal_depth[2] < 0.0, (dip_azimuth, dip)
            got_azimuth_d, got_dip_d = cv.dip_azimuth_dip_from_normal_depth(normal_depth)
            assert cv.angular_distance_deg(got_azimuth_d, dip_azimuth) <= 1e-7
            assert abs(got_dip_d - dip) <= 1e-7
    print("[pass] normal <-> (dip_azimuth, dip) 往返一致")


def test_axiom_vertex_angle() -> None:
    """A=(180-D)%180 与 A=(90-S)%180 等价，且造出的长边方位 = 真走向。"""
    for dip_azimuth in DIP_AZIMUTHS:
        for dip in DIPS:
            normal = cv.normal_from_dip_azimuth_dip(dip_azimuth, dip)
            true_strike = cv.strike_from_dip_azimuth(dip_azimuth)
            from_direct = cv.wrap_strike(180.0 - dip_azimuth)
            from_strike = cv.xy_line_angle_from_strike(true_strike)
            assert cv.axial_distance_deg(from_direct, from_strike) <= 1e-9, (
                dip_azimuth,
                from_direct,
                from_strike,
            )
            # 旧式几何：长边 = (cos A, sin A)；罗盘方位 = atan2(东, 北) = atan2(cos A, sin A)
            angle = math.radians(from_strike)
            bearing = math.degrees(math.atan2(math.cos(angle), math.sin(angle))) % 180.0
            assert cv.axial_distance_deg(bearing, true_strike) <= 1e-7, (
                dip_azimuth,
                bearing,
                true_strike,
            )
            # 历史产物反算
            assert cv.axial_distance_deg(
                cv.strike_from_xy_line_angle(from_strike), true_strike
            ) <= 1e-9
            assert abs(float(np.dot(normal, normal)) - 1.0) <= TOL
    print("[pass] A=(180-D)%180 ≡ (90-S)%180，长边方位 = 真走向")


def test_trace_matches_closed_form() -> None:
    """交线视倾角 = 闭式公式：XZ 为 tanβ=|sinα|·tanδ；YZ 为 |cosα|·tanδ。"""
    worst_xz = 0.0
    worst_yz = 0.0
    for dip_azimuth in DIP_AZIMUTHS:
        for dip in DIPS:
            alpha = math.radians(dip_azimuth)
            delta = math.radians(dip)
            for projection, expect in (
                ("XZ", abs(math.sin(alpha)) * math.tan(delta)),
                ("YZ", abs(math.cos(alpha)) * math.tan(delta)),
            ):
                apparent = cv.section_apparent_dip_deg(dip_azimuth, dip, projection)
                assert apparent is not None, (dip_azimuth, dip, projection)
                if expect <= 1e-12:
                    got = abs(math.tan(math.radians(apparent)))
                    err = got
                else:
                    err = abs(math.tan(math.radians(apparent)) - expect)
                if projection == "XZ":
                    worst_xz = max(worst_xz, err)
                else:
                    worst_yz = max(worst_yz, err)
    assert worst_xz <= 1e-6, worst_xz
    assert worst_yz <= 1e-6, worst_yz
    print(f"[pass] 交线视倾角 = 闭式公式（XZ 最大差 {worst_xz:.2e}，YZ 最大差 {worst_yz:.2e}）")


def test_trace_orientation_semantics() -> None:
    """交线方向：水平分量指向下倾一侧，TIME 分量恒为"向下"（正）。"""
    for dip in (30.0, 60.0):
        # 返回始终是 (水平分量, TIME 分量)：XZ 的水平分量是 X，YZ 的水平分量是 Y。
        trace_xz = cv.trace_on_section(90.0, dip, "XZ")
        assert trace_xz is not None and abs(trace_xz[0]) > 0.5, trace_xz
        # 倾向方位 90°（正东）时 XZ 剖面内水平分量指向 +X（东）
        assert trace_xz[0] > 0.0, trace_xz
        trace_yz = cv.trace_on_section(180.0, dip, "YZ")
        assert trace_yz is not None and abs(trace_yz[0]) > 0.5, trace_yz
        # 倾向方位 180°（正南）时 YZ 剖面内水平分量指向 -Y（南）
        assert trace_yz[0] < 0.0, trace_yz
        # TIME 分量（下倾方向）恒为正
        assert trace_xz[1] > 0.0 and trace_yz[1] > 0.0
    print("[pass] 交线水平/时间分量符号符合'下倾方向'语义")


def test_strike_helpers() -> None:
    for strike in (0.0, 35.0, 55.0, 90.0, 125.0, 145.0, 179.0):
        dip_azimuth = cv.dip_azimuth_from_strike(strike)
        assert cv.axial_distance_deg(cv.strike_from_dip_azimuth(dip_azimuth), strike) <= 1e-9
        assert cv.axial_distance_deg(
            cv.strike_from_xy_line_angle(cv.xy_line_angle_from_strike(strike)), strike
        ) <= 1e-9
    assert cv.axial_distance_deg(0.0, 179.0) <= 1.0 + 1e-9
    assert cv.angular_distance_deg(350.0, 10.0) <= 20.0 + 1e-9
    assert cv.angular_distance_deg(cv.circular_mean_dip_azimuth([350.0, 10.0]), 0.0) <= 1e-6
    assert cv.angular_distance_deg(cv.circular_mean_dip_azimuth([10.0, 350.0]), 0.0) <= 1e-6
    print("[pass] 走向/倾向方位换算与环形统计")


def test_documented_table() -> None:
    """复现施工方案 §1.2 的验证表。"""
    rows = [
        (145.0, 70.0, 55.0, 35.0),
        (128.0, 35.0, 38.0, 52.0),
        (221.5, 76.1, 131.5, 138.5),
        (0.0, 60.0, 90.0, 0.0),
    ]
    for dip_azimuth, dip, expect_strike, expect_a in rows:
        assert abs(cv.strike_from_dip_azimuth(dip_azimuth) - expect_strike) <= 1e-6
        assert abs(cv.wrap_strike(180.0 - dip_azimuth) - expect_a) <= 1e-6
        normal = cv.normal_from_dip_azimuth_dip(dip_azimuth, dip)
        got_strike = cv.strike_from_dip_azimuth(cv.dip_azimuth_dip_from_normal(normal)[0])
        assert cv.axial_distance_deg(got_strike, expect_strike) <= 1e-6
    print("[pass] 施工方案 §1.2 验证表复现")


def main() -> int:
    tests = [
        test_normal_roundtrip,
        test_axiom_vertex_angle,
        test_trace_matches_closed_form,
        test_trace_orientation_semantics,
        test_strike_helpers,
        test_documented_table,
    ]
    for test in tests:
        test()
    print(f"\nall {len(tests)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

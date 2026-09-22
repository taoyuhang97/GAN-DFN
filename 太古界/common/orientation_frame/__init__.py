"""太古界产状口径的唯一实现（倾向方位 0–360 + 倾角）。"""

from .convention import (  # noqa: F401
    AXIAL_PERIOD_DEG,
    FRAME,
    TIME_SCALE_M_PER_MS,
    axial_distance_deg,
    circular_mean_dip_azimuth,
    dip_azimuth_dip_from_normal,
    dip_azimuth_dip_from_normal_depth,
    dip_azimuth_from_strike,
    normal_from_dip_azimuth_dip,
    normal_from_dip_azimuth_dip_depth,
    strike_from_dip_azimuth,
    strike_from_xy_line_angle,
    trace_on_section,
    xy_line_angle_from_strike,
)

__all__ = [
    "AXIAL_PERIOD_DEG",
    "FRAME",
    "TIME_SCALE_M_PER_MS",
    "axial_distance_deg",
    "circular_mean_dip_azimuth",
    "dip_azimuth_dip_from_normal",
    "dip_azimuth_dip_from_normal_depth",
    "dip_azimuth_from_strike",
    "normal_from_dip_azimuth_dip",
    "normal_from_dip_azimuth_dip_depth",
    "strike_from_dip_azimuth",
    "strike_from_xy_line_angle",
    "trace_on_section",
    "xy_line_angle_from_strike",
]

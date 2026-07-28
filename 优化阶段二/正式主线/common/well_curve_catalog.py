"""Curated well-log mnemonic catalog used by the formal well-data rebuild.

The source workbook is a naming reference, not an executable authority: short
mnemonics may be reused by different logging products.  This module therefore
keeps only the reviewed mappings needed by the formal workflow and requires
the LAS pass/header to supply the final provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CurveDefinition:
    mnemonic: str
    chinese_name: str
    family: str
    radial_role: str | None = None


@dataclass(frozen=True)
class ResistivityPairDefinition:
    pair_type: str
    measurement_family: str
    deep_mnemonic: str
    near_mnemonic: str
    chinese_name: str


CURVE_DEFINITIONS = (
    CurveDefinition("GR", "自然伽马", "gamma_ray"),
    CurveDefinition("GRSL", "总自然伽马", "gamma_ray"),
    CurveDefinition("LLD", "深侧向电阻率", "lateral_resistivity", "deep"),
    CurveDefinition("LLS", "浅侧向电阻率", "lateral_resistivity", "shallow"),
    CurveDefinition("RD", "深侧向电阻率", "lateral_resistivity", "deep"),
    CurveDefinition("RS", "浅侧向电阻率", "lateral_resistivity", "shallow"),
    CurveDefinition("RILD", "深感应电阻率", "induction_resistivity", "deep"),
    CurveDefinition("RILM", "中感应电阻率", "induction_resistivity", "medium"),
    CurveDefinition("HRID", "高分辨率深感应电阻率", "high_resolution_induction", "deep"),
    CurveDefinition("HRIM", "高分辨率中感应电阻率", "high_resolution_induction", "medium"),
    CurveDefinition("HDRS", "高分辨率深感应电阻率", "high_resolution_induction", "deep"),
    CurveDefinition("HMRS", "高分辨率中感应电阻率", "high_resolution_induction", "medium"),
    CurveDefinition("R25", "2.5米底部梯度电阻率", "gradient_resistivity", "2.5m_bottom"),
    CurveDefinition("R4", "4米底部梯度电阻率", "gradient_resistivity", "4m_bottom"),
    CurveDefinition("RPRX", "邻近侧向电阻率", "lateral_resistivity", "proximal"),
    CurveDefinition("CON1", "感应电导率(转换)", "conductivity"),
    CurveDefinition("DEV", "井斜角", "well_deviation"),
    CurveDefinition("AZIM", "井眼方位角", "well_azimuth"),
)

RESISTIVITY_PAIRS = (
    ResistivityPairDefinition("LATERAL_LLD_LLS", "lateral_resistivity", "LLD", "LLS", "深浅侧向电阻率"),
    ResistivityPairDefinition("LATERAL_RD_RS", "lateral_resistivity", "RD", "RS", "深浅侧向电阻率"),
    ResistivityPairDefinition("INDUCTION_RILD_RILM", "induction_resistivity", "RILD", "RILM", "深中感应电阻率"),
    ResistivityPairDefinition("HR_INDUCTION_HRID_HRIM", "high_resolution_induction", "HRID", "HRIM", "高分辨率深中感应电阻率"),
    ResistivityPairDefinition("HR_INDUCTION_HDRS_HMRS", "high_resolution_induction", "HDRS", "HMRS", "高分辨率深中感应电阻率"),
)


def curve_catalog_rows() -> list[dict[str, str | None]]:
    return [asdict(item) for item in CURVE_DEFINITIONS]


def pair_catalog_rows() -> list[dict[str, str]]:
    return [asdict(item) for item in RESISTIVITY_PAIRS]


def find_resistivity_pairs(columns: set[str]) -> list[ResistivityPairDefinition]:
    upper = {str(column).upper().strip() for column in columns}
    return [item for item in RESISTIVITY_PAIRS if {item.deep_mnemonic, item.near_mnemonic}.issubset(upper)]

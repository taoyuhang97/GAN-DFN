#!/usr/bin/env python3
"""Audit GR/AC/CNL/DEN/resistivity pairs in original sandstone and Taigu LAS."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/shared/project-oil/wx数据")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "reports/原始测井属性审计_20261123"

REGIONS = {
    "砂砾岩": {
        "conventional_roots": [
            ROOT / "砂砾岩/测井",
            ROOT / "砂砾岩/成像测井-测井曲线",
            ROOT / "砂砾岩/测井数据补充-20260723",
        ],
        "imaging_wells": {
            "车151HF": "che151HF_20240415XRMI",
            "车660": "che660_FMI / che660_FMI_1",
            "车662": "che662_601_FMI / che662_686_FMI",
            "车663": "che663_687_FMI",
            "车古25": "chegu25_FMI",
            "车页1导眼": "车页1HF（导眼）井-FMI",
        },
    },
    "太古界": {
        "conventional_roots": [
            ROOT / "埕岛太古界/侧向电阻率曲线",
            ROOT / "埕岛太古界/补充井位曲线-20260709",
            ROOT / "埕岛太古界/补充井位曲线-20260722",
        ],
        "imaging_wells": {
            "埕北古斜23": "埕北古斜23_FMI",
            "埕北39": "chengbei39_FMI",
            "埕北古25": "埕北古25_FMI-LS / 埕北古25电成像-岩性扫描",
            "桩海102": "桩海102",
            "埕北305": "Chengbei305fmi",
            "老292": "老292",
            "埕北816": "中石化_胜利油田_埕北816井",
            "桩斜169": "中国石化胜利油田桩斜169井测井解释成果",
            "埕北313": "埕北313井FMI_ELAN解释成果数据",
            "埕北310": "埕北310井xrmi成果归档",
            "埕北古7": "胜利_埕北古7井FMI_Petrophysics_递交资料",
            "埕北古斜405": "中国石化胜利油田埕北古斜405井FMI成像测井解释成果",
        },
    },
}

GR_ALIASES = {
    "GR", "GRSL", "GR1", "GR2", "GR3", "1GR", "GR_1", "GR0",
    "HSGR", "SGR", "HGR", "CGR", "NGR", "BGR", "ZGR", "GRZJ",
    "GJGR", "GRY", "GRMAC", "GR11", "GR25", "GR57", "GR01", "GR02",
}
AC_ALIASES = {
    "AC", "AC1", "AC3", "1AC", "ACG", "ACIN", "ACSZH",
    "DT", "DTC", "DTC1", "DT24", "DTL", "SDT2",
}
CNL_ALIASES = {
    "CNL", "CNL3", "1CNL", "2CNL", "CNLSP", "CNLY",
    "CN", "CNC", "CNCF", "NPHI", "NLL", "TNPH", "NPOR",
}
DEN_ALIASES = {
    "DEN", "DEN3", "1DEN", "DENY", "RHOB", "ZDEN",
    "1ZDEN", "2ZDEN", "3ZDEN", "ZD", "BULK",
}

RESISTIVITY_FAMILIES = {
    "RD_RS": ({"RD", "1RD", "0RD", "RD1", "RD3", "VRD"}, {"RS", "1RS", "0RS", "RS1", "RS3", "VRS"}),
    "LLD_LLS": ({"LLD", "1LLD", "0LLD", "LLD1", "VLLD"}, {"LLS", "1LLS", "0LLS", "VRMSL", "VLLS"}),
    "RILD_RILM": ({"RILD", "1RILD", "VRILD"}, {"RILM", "1RILM", "VRILM"}),
}

WELL_NAME_MAP = {
    "车页1HF": "车页1导眼",
    "车页1HF导眼": "车页1导眼",
    "Che660": "车660",
    "CHE660": "车660",
    "Chengbei305": "埕北305",
    "Chengbei310": "埕北310",
    "Chengbei313": "埕北313",
    "Chengbei816": "埕北816",
    "Chengbeigu7": "埕北古7",
    "ZhuangX169": "桩斜169",
    "老292": "老292",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def canonical_well_name(path: Path) -> str:
    stem = path.stem.split("@")[0]
    stem = re.sub(r"[\(（].*?[\)）]", "", stem).strip()
    stem = re.sub(r"(井)?(FMI|XRMI|HRMI)$", "", stem, flags=re.IGNORECASE)
    return WELL_NAME_MAP.get(stem, stem)


def finite_count(values: np.ndarray) -> int:
    array = np.asarray(values, dtype=float).reshape(-1)
    return int(np.isfinite(array).sum())


def classify_curve(mnemonic: str) -> str | None:
    if mnemonic in GR_ALIASES:
        return "GR"
    if mnemonic in AC_ALIASES:
        return "AC"
    if mnemonic in CNL_ALIASES:
        return "CNL"
    if mnemonic in DEN_ALIASES:
        return "DEN"
    return None


def read_las(path: Path) -> dict[str, Any]:
    import lasio

    las = lasio.read(str(path))
    curves = {str(curve.mnemonic).strip().upper(): np.asarray(las[curve.mnemonic]) for curve in las.curves}
    counts_by_curve = {name: finite_count(values) for name, values in curves.items()}
    strong_curves = {
        name: values
        for name, values in counts_by_curve.items()
        if values > 0
    }
    attributes: dict[str, list[str]] = defaultdict(list)
    for name, count in strong_curves.items():
        attr = classify_curve(name)
        if attr is not None:
            attributes[attr].append(f"{name}:{count}")
    pair_hits: list[str] = []
    for pair_name, (deep_aliases, shallow_aliases) in RESISTIVITY_FAMILIES.items():
        deep = sorted(alias for alias in deep_aliases if strong_curves.get(alias, 0) > 0)
        shallow = sorted(alias for alias in shallow_aliases if strong_curves.get(alias, 0) > 0)
        if deep and shallow:
            pair_hits.append(
                f"{pair_name}({deep[0]}:{strong_curves[deep[0]]},"
                f"{shallow[0]}:{strong_curves[shallow[0]]})"
            )
    return {
        "attributes": dict(attributes),
        "resistivity_pairs": pair_hits,
        "curve_count": len(curves),
        "valid_curve_count": int(sum(count > 0 for count in counts_by_curve.values())),
    }


def scan_region(region: str, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    well_files: dict[str, list[Path]] = defaultdict(list)
    for root in spec["conventional_roots"]:
        for path in Path(root).rglob("*"):
            if path.is_file() and path.suffix.lower() == ".las":
                well_files[canonical_well_name(path)].append(path)

    inventory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for well, paths in sorted(well_files.items()):
        well_attrs: dict[str, set[str]] = {
            "GR": set(),
            "AC": set(),
            "CNL": set(),
            "DEN": set(),
        }
        pair_hits: set[str] = set()
        valid_files = 0
        for path in sorted(paths):
            try:
                parsed = read_las(path)
                valid_files += 1
            except Exception as exc:
                errors.append(
                    {"region": region, "well": well, "path": str(path), "error": repr(exc)}
                )
                continue
            for attr, names in parsed["attributes"].items():
                well_attrs[attr].update(names)
            pair_hits.update(parsed["resistivity_pairs"])
            inventory_rows.append(
                {
                    "Region": region,
                    "WellName": well,
                    "Path": str(path),
                    "CurveCount": parsed["curve_count"],
                    "ValidCurveCount": parsed["valid_curve_count"],
                    "GR_Curves": ";".join(parsed["attributes"].get("GR", [])),
                    "AC_Curves": ";".join(parsed["attributes"].get("AC", [])),
                    "CNL_Curves": ";".join(parsed["attributes"].get("CNL", [])),
                    "DEN_Curves": ";".join(parsed["attributes"].get("DEN", [])),
                    "ResistivityPairs": ";".join(parsed["resistivity_pairs"]),
                }
            )
        row = {
            "Region": region,
            "WellName": well,
            "LASFileCount": len(paths),
            "ValidLASFileCount": valid_files,
            "HasGR": int(bool(well_attrs["GR"])),
            "HasAC": int(bool(well_attrs["AC"])),
            "HasCNL": int(bool(well_attrs["CNL"])),
            "HasDEN": int(bool(well_attrs["DEN"])),
            "HasResistivityPair": int(bool(pair_hits)),
            "ResistivityPairTypes": ";".join(sorted(pair_hits)),
            "GR_SourceCurves": ";".join(sorted(well_attrs["GR"])),
            "AC_SourceCurves": ";".join(sorted(well_attrs["AC"])),
            "CNL_SourceCurves": ";".join(sorted(well_attrs["CNL"])),
            "DEN_SourceCurves": ";".join(sorted(well_attrs["DEN"])),
            "SourceFiles": " | ".join(str(path) for path in sorted(paths)),
        }
        row["HasAllFive"] = int(
            all(row[key] == 1 for key in ("HasGR", "HasAC", "HasCNL", "HasDEN", "HasResistivityPair"))
        )
        summary_rows.append(row)
    return (
        pd.DataFrame(inventory_rows),
        pd.DataFrame(summary_rows).sort_values("WellName").reset_index(drop=True),
        errors,
    )


def well_list(frame: pd.DataFrame, column: str | None = None) -> list[str]:
    if column is None:
        selected = frame
    else:
        selected = frame[frame[column].eq(1)]
    return sorted(selected["WellName"].astype(str).unique().tolist())


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_inventory: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    all_errors: list[dict[str, str]] = []
    region_stats: dict[str, Any] = {}

    for region, spec in REGIONS.items():
        inventory, conventional, errors = scan_region(region, spec)
        all_inventory.append(inventory)
        all_summary.append(conventional.assign(WellScope="conventional"))
        all_errors.extend(errors)

        imaging_names = sorted(spec["imaging_wells"])
        conventional_by_well = conventional.set_index("WellName")
        imaging_rows = []
        for well in imaging_names:
            if well in conventional_by_well.index:
                row = conventional_by_well.loc[well].to_dict()
                row["WellName"] = well
                row["ConventionalDataFound"] = 1
            else:
                row = {
                    "Region": region,
                    "WellName": well,
                    "LASFileCount": 0,
                    "ValidLASFileCount": 0,
                    "HasGR": 0,
                    "HasAC": 0,
                    "HasCNL": 0,
                    "HasDEN": 0,
                    "HasResistivityPair": 0,
                    "ResistivityPairTypes": "",
                    "GR_SourceCurves": "",
                    "AC_SourceCurves": "",
                    "CNL_SourceCurves": "",
                    "DEN_SourceCurves": "",
                    "SourceFiles": "",
                    "HasAllFive": 0,
                    "ConventionalDataFound": 0,
                }
            row["ImagingSourceDirectory"] = spec["imaging_wells"][well]
            imaging_rows.append(row)
        imaging = pd.DataFrame(imaging_rows).sort_values("WellName").reset_index(drop=True)
        all_summary.append(imaging.assign(WellScope="imaging"))

        region_stats[region] = {
            "conventional_well_count": int(len(conventional)),
            "imaging_well_count": int(len(imaging)),
            "imaging_wells_with_conventional_data": int(
                imaging["ConventionalDataFound"].sum()
            ) if "ConventionalDataFound" in imaging else 0,
            "conventional_all_five_count": int(conventional["HasAllFive"].sum()),
            "conventional_attribute_counts": {
                attr: int(conventional[column].sum())
                for attr, column in (
                    ("GR", "HasGR"),
                    ("AC", "HasAC"),
                    ("CNL", "HasCNL"),
                    ("DEN", "HasDEN"),
                    ("resistivity_pair", "HasResistivityPair"),
                )
            },
            "conventional_attribute_wells": {
                attr: well_list(conventional, column)
                for attr, column in (
                    ("GR", "HasGR"),
                    ("AC", "HasAC"),
                    ("CNL", "HasCNL"),
                    ("DEN", "HasDEN"),
                    ("resistivity_pair", "HasResistivityPair"),
                )
            },
            "conventional_all_five_wells": well_list(conventional, "HasAllFive"),
            "imaging_attribute_counts": {
                attr: int(imaging[column].sum())
                for attr, column in (
                    ("GR", "HasGR"),
                    ("AC", "HasAC"),
                    ("CNL", "HasCNL"),
                    ("DEN", "HasDEN"),
                    ("resistivity_pair", "HasResistivityPair"),
                )
            },
            "imaging_attribute_wells": {
                attr: well_list(imaging, column)
                for attr, column in (
                    ("GR", "HasGR"),
                    ("AC", "HasAC"),
                    ("CNL", "HasCNL"),
                    ("DEN", "HasDEN"),
                    ("resistivity_pair", "HasResistivityPair"),
                )
            },
            "imaging_all_five_wells": well_list(imaging, "HasAllFive"),
        }

    inventory_all = pd.concat(all_inventory, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    inventory_all.to_csv(
        output_dir / "well_curve_inventory.csv", index=False, encoding="utf-8-sig"
    )
    summary_all.to_csv(
        output_dir / "well_attribute_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        all_errors,
        columns=["region", "well", "path", "error"],
    ).to_csv(
        output_dir / "las_read_errors.csv", index=False, encoding="utf-8-sig"
    )
    payload = {
        "status": "pass" if not all_errors else "pass_with_read_errors",
        "semantics": {
            "attribute_present": "至少一个 LAS 文件存在至少一个有限样点",
            "well_level_union": "同一井多次测井取并集",
            "resistivity_pair": "同一 LAS 文件内深、浅/参考曲线同时有有限值",
            "pairs_checked": ["LLD/LLS", "RD/RS", "RILD/RILM"],
        },
        "regions": region_stats,
        "outputs": {
            "well_attribute_summary_csv": str(output_dir / "well_attribute_summary.csv"),
            "well_curve_inventory_csv": str(output_dir / "well_curve_inventory.csv"),
            "las_read_errors_csv": str(output_dir / "las_read_errors.csv"),
        },
    }
    (output_dir / "well_log_attribute_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

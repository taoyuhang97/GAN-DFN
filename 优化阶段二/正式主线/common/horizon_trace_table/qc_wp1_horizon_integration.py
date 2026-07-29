from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import segyio


CURRENT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = CURRENT_DIR.parents[1]
if str(FORMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(FORMAL_ROOT))

from common.horizon_trace_table.horizon_contract import (  # noqa: E402
    apply_window_inplace,
    build_spatial_lookup,
    contract_summary,
    load_contract_for_mapping,
    trace_window_mask,
    validate_window_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the shared WP1 horizon contract integration.")
    parser.add_argument("--master-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    master_path = args.master_config.resolve()
    master = read_json(master_path)
    mapping_path = Path(master["trace_mapping_npz"]).resolve()
    with np.load(mapping_path) as payload:
        mapping = {key: payload[key] for key in payload.files}
    contract = load_contract_for_mapping(master, mapping)

    with segyio.open(str(Path(master["input_density_sgy"]).resolve()), "r", ignore_geometry=True) as handle:
        samples_2ms = np.asarray(handle.samples, dtype=np.float64)
    axis_2ms_qc = validate_window_contract(master, contract, samples_2ms)
    samples_10ms = np.arange(float(samples_2ms[0]), float(samples_2ms[-1]) + 1.0e-6, 10.0, dtype=np.float64)
    axis_10ms_qc = validate_window_contract(master, contract, samples_10ms)

    subset_count = min(512, contract.trace_count)
    subset = replace(
        contract,
        trace_idx=contract.trace_idx[:subset_count],
        t4=contract.t4[:subset_count],
        t5=contract.t5[:subset_count],
        t6=contract.t6[:subset_count],
        t7=contract.t7[:subset_count],
        surface_order_valid=contract.surface_order_valid[:subset_count],
        shasan_present=contract.shasan_present[:subset_count],
        shasi_present=contract.shasi_present[:subset_count],
        correction_code=contract.correction_code[:subset_count],
    )
    synthetic = np.ones((subset_count, len(samples_2ms)), dtype=np.uint8)
    mask_qc = apply_window_inplace(synthetic, subset, samples_2ms, fill_value=0)
    expected = trace_window_mask(subset, samples_2ms, 0, subset_count)
    synthetic_mask_pass = bool(np.array_equal(synthetic.astype(bool), expected))

    generated_configs = {
        "step6": FORMAL_ROOT / "step6b_demo_density_volume_3d/configs/formal_demo_10km_multiscale_flow_v2_expanded.json",
        "step7a": FORMAL_ROOT / "step7a_small_scale_dfn/configs/formal_demo_10km_multiscale_flow_v2.json",
        "step7b": FORMAL_ROOT / "step7b_multiscale_initial_dfn/configs/formal_demo_10km_multiscale_flow_v2.json",
        "step7c": FORMAL_ROOT / "step7c_large_fault_dfn/configs/formal_demo_10km_multiscale_flow_v2.json",
        "step8": FORMAL_ROOT / "step8_dfn_well_correction/configs/formal_demo_10km_multiscale_flow_v2.json",
        "step9": FORMAL_ROOT / "step9_section_visualize/configs/formal_demo_10km_multiscale_flow_v2.json",
    }
    expected_table = str(Path(master["horizon_contract_table"]).resolve())
    config_rows: dict[str, Any] = {}
    config_pass = True
    for stage, path in generated_configs.items():
        payload = read_json(path)
        same_table = str(Path(payload["horizon_contract_table"]).resolve()) == expected_table
        windows = dict(payload.get("horizon_window_contracts", {}))
        has_windows = set(windows) == {"2ms", "10ms"} and all(Path(value).exists() for value in windows.values())
        has_trace_header = bool(payload.get("trace_header_csv")) and Path(payload["trace_header_csv"]).exists()
        passed = bool(same_table and has_windows and has_trace_header)
        config_pass &= passed
        config_rows[stage] = {
            "path": str(path),
            "same_horizon_table": same_table,
            "window_contracts_valid": has_windows,
            "trace_header_valid": has_trace_header,
            "status": "pass" if passed else "fail",
        }

    lookup = build_spatial_lookup(master)
    lookup_rows = []
    for name, x, y, time in [
        ("Che571", 571252.0, 4202608.0, 2863.0),
        ("Che572", 571812.0, 4203818.0, 2855.0),
        ("Central", 576000.0, 4200000.0, 2950.0),
    ]:
        local = lookup.query(x, y)
        lookup_rows.append({"name": name, **local, "QueryTime": time, "Interval": lookup.interval(x, y, time)})

    step7c_source = (FORMAL_ROOT / "step7c_large_fault_dfn/build_large_fault_dfn.py").read_text(encoding="utf-8")
    step9_source = (FORMAL_ROOT / "step9_section_visualize/build_cheye1_dfn_coherence_sections.py").read_text(
        encoding="utf-8"
    )
    source_checks = {
        "step7c_has_no_fixed_shasi_split": "shasi_time_split_ms" not in step7c_source,
        "step9_has_no_center_vs_well_time_prefilter": "center_time < well_time_min" not in step9_source,
        "step9_has_no_fault_time_between_prefilter": 'fault_df["TIME"].between(well_time_min, well_time_max)' not in step9_source,
    }

    checks = {
        "axis_2ms": axis_2ms_qc["status"] == "pass",
        "axis_10ms": axis_10ms_qc["status"] == "pass",
        "synthetic_mask": synthetic_mask_pass,
        "generated_configs": config_pass,
        **source_checks,
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "master_config": str(master_path),
        "horizon_contract": contract_summary(contract),
        "axis_2ms_qc": axis_2ms_qc,
        "axis_10ms_qc": axis_10ms_qc,
        "synthetic_mask_qc": {**mask_qc, "exact_match": synthetic_mask_pass},
        "generated_configs": config_rows,
        "lookup_samples": lookup_rows,
        "checks": checks,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

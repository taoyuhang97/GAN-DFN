from __future__ import annotations

import argparse

from fracture_label_rebuild_utils import build_uniform_sample
from sample_source_config import DEFAULT_WELL_CONFIGS, OUTPUT_SAMPLE_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-col", default="P10")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    OUTPUT_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    failed_count = 0
    for config in DEFAULT_WELL_CONFIGS:
        try:
            stats = build_uniform_sample(
                well_name=config.well_name,
                around_csv=config.around_csv,
                density_source_kind=config.density_source_kind,
                density_source_path=config.density_source_path,
                raw_point_path=config.raw_point_path,
                output_csv=config.output_csv,
                key_col=str(args.key_col),
            )
            ok_count += 1
            print(
                "[OK] "
                f"{stats.well_name}: "
                f"rows {stats.input_row_count} -> {stats.filtered_row_count}, "
                f"gt_label={stats.gt_label_count}, "
                f"gt_point={stats.gt_point_count}, "
                f"mapped_raw={stats.mapped_raw_point_count}, "
                f"dropped_raw={stats.dropped_raw_point_count}, "
                f"density_source={stats.density_source_kind}"
            )
        except Exception as exc:
            failed_count += 1
            print(f"[FAIL] {config.well_name}: {exc}")

    print(f"Done: ok={ok_count}, failed={failed_count}, total={len(DEFAULT_WELL_CONFIGS)}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import build_medium_scale_dfn_v4 as medium


def synthetic_group() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IX": [0, 1, 2, 10, 11, 12],
            "IY": [0, 0, 0, 0, 0, 0],
            "IT": [0, 0, 0, 0, 0, 0],
            "CenterTime": [2000.0] * 6,
            "SamplingWeight": [1.0, 0.95, 0.90, 0.70, 0.65, 0.60],
            "AntTrackScore": [1.0, 0.98, 0.96, 0.55, 0.50, 0.45],
            "MediumScore": [1.0, 0.95, 0.90, 0.70, 0.65, 0.60],
        }
    )


class MediumSelectionOrderTest(unittest.TestCase):
    def test_hybrid_reserves_spatial_slot(self) -> None:
        order, summary, sources = medium.medium_selection_order(
            synthetic_group(),
            {
                "component_selection_mode": "hybrid_anttrack_spatial",
                "anttrack_primary_fraction": 0.60,
                "spatial_candidate_multiplier": 4,
                "anttrack_ridge_quantile": 0.72,
                "anttrack_ridge_min_score": 0.45,
                "anttrack_ridge_nms_radius_cells": 0,
            },
            target=4,
            time_scale=2.0,
        )
        self.assertEqual(summary["selection_mode"], "hybrid_anttrack_spatial")
        self.assertEqual(summary["anttrack_primary_quota"], 2)
        self.assertTrue(all(sources[int(idx)] == "anttrack_ridge_primary" for idx in order[:2]))
        self.assertIn("spatial_coverage", {sources[int(idx)] for idx in order[2:]})

    def test_two_patch_target_keeps_one_spatial_slot(self) -> None:
        _order, summary, _sources = medium.medium_selection_order(
            synthetic_group(),
            {
                "component_selection_mode": "hybrid_anttrack_spatial",
                "anttrack_primary_fraction": 0.60,
                "anttrack_ridge_nms_radius_cells": 0,
            },
            target=2,
            time_scale=2.0,
        )
        self.assertEqual(summary["anttrack_primary_quota"], 1)

    def test_spatial_mode_starts_with_spatial_candidates(self) -> None:
        order, summary, sources = medium.medium_selection_order(
            synthetic_group(),
            {
                "component_selection_mode": "spatial_farthest",
                "anttrack_ridge_nms_radius_cells": 0,
            },
            target=3,
            time_scale=2.0,
        )
        self.assertEqual(summary["selection_mode"], "spatial_farthest")
        self.assertEqual(sources[int(order[0])], "spatial_coverage")

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported component_selection_mode"):
            medium.medium_selection_order(
                synthetic_group(),
                {"component_selection_mode": "unknown"},
                target=3,
                time_scale=2.0,
            )


if __name__ == "__main__":
    unittest.main()

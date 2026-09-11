"""Small, reusable value-semantics contract for Taigu seismic attributes.

This module intentionally creates no full-volume score cube.  Callers score
only the samples or blocks they are already processing.
"""
from __future__ import annotations

from typing import Any

import numpy as np


POLARITY = {
    "AntTrack": "high_is_fracture_evidence",
    "CurvatureMax": "high_is_fracture_evidence",
    "Coherence": "low_is_discontinuity_evidence",
}


def robust_limits(values: np.ndarray, low_quantile: float = 0.02, high_quantile: float = 0.98) -> dict[str, float]:
    """Return finite-value quantile limits; AntTrack=-1 remains finite."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("cannot construct normalization limits from no finite values")
    low = float(np.quantile(finite, low_quantile))
    high = float(np.quantile(finite, high_quantile))
    if high <= low:
        high = low + 1.0e-9
    return {"clip_low": low, "clip_high": high}


def score_attribute(values: np.ndarray, attribute: str, limits: dict[str, Any]) -> np.ndarray:
    """Map raw values to [0,1], preserving true NaN as NaN.

    AntTrack=-1 is deliberately assigned a low score, never converted to NaN.
    """
    if attribute not in POLARITY:
        raise KeyError(f"unsupported attribute: {attribute}")
    raw = np.asarray(values, dtype=np.float64)
    low, high = float(limits["clip_low"]), float(limits["clip_high"])
    if high <= low:
        raise ValueError(f"invalid normalization limits for {attribute}")
    score = np.clip((raw - low) / (high - low), 0.0, 1.0)
    if POLARITY[attribute].startswith("low_is"):
        score = 1.0 - score
    score[~np.isfinite(raw)] = np.nan
    return score.astype(np.float32)


def layer_score(values: np.ndarray, attribute: str, normalization_contract: dict[str, Any], layer: str) -> np.ndarray:
    """Score values with the saved small normalization contract."""
    return score_attribute(values, attribute, normalization_contract["layers"][layer][attribute])

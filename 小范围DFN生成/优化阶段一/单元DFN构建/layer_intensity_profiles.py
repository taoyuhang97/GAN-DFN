from __future__ import annotations


DEFAULT_LAYER_INTENSITY_PROFILE_NAME = "demo_v1"


LAYER_INTENSITY_PROFILES: dict[str, dict[str, float]] = {
    "uniform": {
        "TOP_1100MS->T1": 1.0,
        "T1->T2": 1.0,
        "T2->T3": 1.0,
        "T3->T4": 1.0,
        "T4->T5": 1.0,
        "T5->T6": 1.0,
        "T6->T7": 1.0,
        "T7->BOTTOM_3800MS": 1.0,
    },
    "demo_v1": {
        "TOP_1100MS->T1": 0.70,
        "T1->T2": 3.00,
        "T2->T3": 1.50,
        "T3->T4": 0.85,
        "T4->T5": 1.50,
        "T5->T6": 2.75,
        "T6->T7": 1.10,
        "T7->BOTTOM_3800MS": 0.75,
    },
}


def build_layer_surface_pair_key(top_surface_code: object, base_surface_code: object) -> str:
    top_text = str(top_surface_code or "").strip()
    base_text = str(base_surface_code or "").strip()
    return f"{top_text}->{base_text}"


def available_layer_intensity_profiles() -> list[str]:
    return sorted(LAYER_INTENSITY_PROFILES.keys())


def get_layer_intensity_profile(profile_name: str) -> dict[str, float]:
    normalized = str(profile_name or "").strip()
    if not normalized:
        normalized = DEFAULT_LAYER_INTENSITY_PROFILE_NAME
    if normalized not in LAYER_INTENSITY_PROFILES:
        available = ", ".join(available_layer_intensity_profiles())
        raise KeyError(f"unknown layer intensity profile: {normalized}; available={available}")
    return dict(LAYER_INTENSITY_PROFILES[normalized])

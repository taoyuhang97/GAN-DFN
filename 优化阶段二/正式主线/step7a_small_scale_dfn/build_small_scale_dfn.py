from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
LEGACY_STEP7B_DIR = CURRENT_DIR.parent / "step7b_initial_dfn_3d"
DEFAULT_CONFIG = CURRENT_DIR / "configs/formal_candidate_cheye1_step7a_small_v1.json"
if str(LEGACY_STEP7B_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_STEP7B_DIR))

import build_initial_dfn_from_3d_density_sgy as legacy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7A small-scale/background DFN from Step6A small density.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to JSON config.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "dfn_csv": output_dir / "small_dfn_patches.csv",
        "raw_vtk": output_dir / "small_dfn_raw_time.vtk",
        "audit_csv": output_dir / "small_dfn_generation_audit.csv",
        "summary_json": output_dir / "small_dfn_summary.json",
    }


def build_small_candidates(
    grid: dict[str, Any],
    surfaces: dict[str, np.ndarray],
    config: dict[str, Any],
    domain_name: str = "background",
    domain_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates, layer_summary = legacy.build_candidate_voxels(
        density=grid["density"],
        samples=grid["samples"],
        surfaces=surfaces,
        source_trace_idx=grid["source_trace_idx"],
        config=config,
        coherence=None,
    )
    candidates["FractureScale"] = "small"
    candidates["FractureScaleCode"] = 1
    candidates["SmallDomain"] = domain_name
    candidates["SmallDomainCode"] = int((domain_cfg or {}).get("domain_code", 1))
    candidates["BandID"] = ""
    candidates["BandPatchOrdinal"] = 0
    candidates["BandContinuityMode"] = str((domain_cfg or {}).get("band_continuity_mode", f"small_{domain_name}_density_sampling"))
    candidates["BandVoxelCount"] = candidates["ComponentVoxelCount"].astype(int)
    candidates["BandLengthM"] = 0.0
    candidates["BandTimeExtentMs"] = 0.0
    candidates["BandPatchSpacingM"] = 0.0
    candidates["BandMeanDensity"] = candidates["SourceDensity"].astype(float)
    return candidates, layer_summary


def merged_domain_config(base: dict[str, Any], domain_cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in domain_cfg.items():
        if key in {"density_sgy", "domain_code", "source_type", "structural_relation", "band_continuity_mode", "action_reason"}:
            continue
        out[key] = value
    return out


def normalize_patch_confidence(patch_df: pd.DataFrame) -> pd.Series:
    confidence = np.clip(pd.to_numeric(patch_df["SamplingWeight"], errors="coerce").fillna(0.0), 0.0, None)
    if len(confidence) == 0:
        return confidence
    conf_max = float(confidence.quantile(0.95))
    return (confidence / max(conf_max, 1.0e-9)).clip(0.05, 1.0)


def domain_layer_param(config: dict[str, Any], key: str, layer: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, dict):
        return float(value.get(layer, default))
    return float(value)


def sample_candidate_voxels_by_intensity(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sample Step7A small fractures from local density intensity, not a fixed output count."""
    if candidates.empty:
        raise RuntimeError("no candidates available for Step7A intensity sampling")

    basis_column = str(config.get("intensity_basis_column", "SamplingWeight"))
    if basis_column not in candidates.columns:
        basis_column = "SourceDensity"
    basis = np.clip(pd.to_numeric(candidates[basis_column], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, None)
    if float(basis.sum()) <= 0.0:
        raise RuntimeError(f"candidate intensity basis has no positive mass: {basis_column}")

    work = candidates.copy()
    work["_IntensityBasis"] = basis
    density_power = float(config.get("density_power", 1.0))
    reference_quantile = float(config.get("density_reference_quantile", 0.95))
    min_density_norm = float(config.get("min_density_norm", 0.0))
    max_density_norm = float(config.get("max_intensity_density_norm", config.get("max_density_norm", 2.0)))
    max_lambda_per_voxel = float(config.get("max_lambda_per_voxel", 0.35))
    max_poisson_count_per_voxel = int(config.get("max_poisson_count_per_voxel", 1))
    fail_fast_patch_count_limit = int(config.get("fail_fast_patch_count_limit", config.get("safety_max_patch_count", 250000)))

    expected = np.zeros(len(work), dtype=float)
    layer_summary: dict[str, Any] = {}
    for layer, index in work.groupby("LayerGroup", dropna=False).groups.items():
        idx = np.asarray(list(index), dtype=int)
        layer_name = str(layer)
        values = work.loc[idx, "_IntensityBasis"].to_numpy(dtype=float)
        positive = values[values > 0.0]
        if positive.size == 0:
            continue
        reference = max(float(np.nanquantile(positive, reference_quantile)), 1.0e-9)
        density_norm = np.clip(values / reference, min_density_norm, max_density_norm)
        raw_occurrence_rate = config.get("occurrence_rate", 0.01)
        default_occurrence_rate = 0.01 if isinstance(raw_occurrence_rate, dict) else float(raw_occurrence_rate)
        occurrence_rate = domain_layer_param(config, "occurrence_rate", layer_name, default_occurrence_rate)
        lam = occurrence_rate * np.power(np.clip(density_norm, 0.0, None), density_power)
        lam = np.clip(lam, 0.0, max_lambda_per_voxel)
        expected[idx] = lam
        layer_summary[layer_name] = {
            "candidate_count": int(len(idx)),
            "basis_reference_quantile": reference_quantile,
            "basis_reference_value": float(reference),
            "occurrence_rate": float(occurrence_rate),
            "expected_patch_count": float(lam.sum()),
            "lambda_stats": legacy.finite_stats(lam),
            "basis_stats": legacy.finite_stats(values),
        }

    if max_poisson_count_per_voxel <= 1:
        probability = 1.0 - np.exp(-expected)
        patch_counts = (rng.random(len(work)) < probability).astype(int)
        expected_for_cell = probability
        sampling_replacement = False
        sampling_distribution = "bernoulli_from_poisson_intensity"
    else:
        patch_counts = rng.poisson(expected)
        patch_counts = np.minimum(patch_counts, max_poisson_count_per_voxel).astype(int)
        expected_for_cell = expected
        sampling_replacement = True
        sampling_distribution = "poisson_intensity"

    actual_patch_count = int(patch_counts.sum())
    if actual_patch_count <= 0:
        raise RuntimeError("Step7A intensity sampling produced zero patches; lower candidate threshold or increase occurrence_rate")
    if actual_patch_count > fail_fast_patch_count_limit:
        raise RuntimeError(
            "Step7A intensity sampling exceeded fail_fast_patch_count_limit: "
            f"actual={actual_patch_count}, limit={fail_fast_patch_count_limit}. "
            "This is not truncated automatically; adjust occurrence_rate or candidate_density_quantile."
        )

    repeat_idx = np.repeat(np.arange(len(work)), patch_counts)
    selected = work.iloc[repeat_idx].drop(columns=["_IntensityBasis"], errors="ignore").reset_index(drop=True).copy()
    selected["ExpectedPatchCountForCell"] = np.repeat(expected_for_cell, patch_counts)
    selected["EffectiveCountScale"] = np.repeat(expected, patch_counts)
    selected["CountBasisEffectiveScale"] = np.repeat(expected, patch_counts)
    selected["DensityCellPatchOrdinal"] = selected.groupby(["SourceTraceIdx", "LayerGroup", "IT"]).cumcount() + 1

    return selected, {
        "sampling_mode": "poisson_intensity",
        "sampling_distribution": sampling_distribution,
        "fixed_target_count_used": False,
        "density_mass": float(np.clip(candidates["SourceDensity"].to_numpy(dtype=float), 0.0, None).sum()),
        "sampling_weight_mass": float(np.clip(candidates.get("SamplingWeight", candidates["SourceDensity"]).to_numpy(dtype=float), 0.0, None).sum()),
        "intensity_basis_column": basis_column,
        "density_power": density_power,
        "density_reference_quantile": reference_quantile,
        "max_lambda_per_voxel": max_lambda_per_voxel,
        "max_poisson_count_per_voxel": max_poisson_count_per_voxel,
        "fail_fast_patch_count_limit": fail_fast_patch_count_limit,
        "expected_patch_count": float(expected.sum()),
        "actual_patch_count": actual_patch_count,
        "positive_density_candidate_count": int(len(candidates)),
        "sampled_density_cell_count": int((patch_counts > 0).sum()),
        "sampling_replacement": sampling_replacement,
        "layer_summary": layer_summary,
    }


def sample_candidate_voxels_by_weighted_probability(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Approximate v2 weighted sampling with per-cell occurrence probability and no fixed target count."""
    if candidates.empty:
        raise RuntimeError("no candidates available for Step7A weighted-probability sampling")

    basis_column = str(config.get("probability_basis_column", "SamplingWeight"))
    if basis_column not in candidates.columns:
        basis_column = "SourceDensity"
    weights = np.clip(pd.to_numeric(candidates[basis_column], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0, None)
    if float(weights.sum()) <= 0.0:
        raise RuntimeError(f"candidate probability basis has no positive mass: {basis_column}")

    probability_scale = float(config.get("weighted_probability_scale", 0.03))
    density_power = float(config.get("weighted_probability_density_power", 1.0))
    reference_mode = str(config.get("weighted_probability_reference", "raw"))
    max_probability = float(config.get("max_cell_probability", 0.25))
    fail_fast_patch_count_limit = int(config.get("fail_fast_patch_count_limit", 250000))

    if reference_mode == "layer_quantile":
        probability = np.zeros(len(candidates), dtype=float)
        layer_summary: dict[str, Any] = {}
        reference_quantile = float(config.get("weighted_probability_reference_quantile", 0.95))
        for layer, index in candidates.groupby("LayerGroup", dropna=False).groups.items():
            idx = np.asarray(list(index), dtype=int)
            values = weights[idx]
            positive = values[values > 0.0]
            if positive.size == 0:
                continue
            reference = max(float(np.nanquantile(positive, reference_quantile)), 1.0e-9)
            layer_probability = probability_scale * np.power(values / reference, density_power)
            layer_probability = np.clip(layer_probability, 0.0, max_probability)
            probability[idx] = layer_probability
            layer_summary[str(layer)] = {
                "candidate_count": int(len(idx)),
                "basis_reference_mode": reference_mode,
                "basis_reference_quantile": reference_quantile,
                "basis_reference_value": float(reference),
                "weighted_probability_scale": probability_scale,
                "expected_patch_count": float(layer_probability.sum()),
                "probability_stats": legacy.finite_stats(layer_probability),
                "basis_stats": legacy.finite_stats(values),
            }
    else:
        probability = np.clip(probability_scale * np.power(weights, density_power), 0.0, max_probability)
        layer_summary = {}
        for layer, index in candidates.groupby("LayerGroup", dropna=False).groups.items():
            idx = np.asarray(list(index), dtype=int)
            layer_summary[str(layer)] = {
                "candidate_count": int(len(idx)),
                "basis_reference_mode": reference_mode,
                "weighted_probability_scale": probability_scale,
                "expected_patch_count": float(probability[idx].sum()),
                "probability_stats": legacy.finite_stats(probability[idx]),
                "basis_stats": legacy.finite_stats(weights[idx]),
            }

    selected_mask = rng.random(len(candidates)) < probability
    actual_patch_count = int(selected_mask.sum())
    if actual_patch_count <= 0:
        raise RuntimeError("Step7A weighted-probability sampling produced zero patches; increase weighted_probability_scale")
    if actual_patch_count > fail_fast_patch_count_limit:
        raise RuntimeError(
            "Step7A weighted-probability sampling exceeded fail_fast_patch_count_limit: "
            f"actual={actual_patch_count}, limit={fail_fast_patch_count_limit}. "
            "This is not truncated automatically; adjust weighted_probability_scale or candidate_density_quantile."
        )

    selected = candidates.loc[selected_mask].reset_index(drop=True).copy()
    selected["ExpectedPatchCountForCell"] = probability[selected_mask]
    selected["EffectiveCountScale"] = probability_scale
    selected["CountBasisEffectiveScale"] = probability_scale
    selected["DensityCellPatchOrdinal"] = selected.groupby(["SourceTraceIdx", "LayerGroup", "IT"]).cumcount() + 1
    return selected, {
        "sampling_mode": "v2_weighted_probability",
        "sampling_distribution": "bernoulli_probability_proportional_to_v2_sampling_weight",
        "fixed_target_count_used": False,
        "density_mass": float(np.clip(candidates["SourceDensity"].to_numpy(dtype=float), 0.0, None).sum()),
        "sampling_weight_mass": float(weights.sum()),
        "probability_basis_column": basis_column,
        "weighted_probability_scale": probability_scale,
        "weighted_probability_density_power": density_power,
        "weighted_probability_reference": reference_mode,
        "max_cell_probability": max_probability,
        "fail_fast_patch_count_limit": fail_fast_patch_count_limit,
        "expected_patch_count": float(probability.sum()),
        "actual_patch_count": actual_patch_count,
        "positive_density_candidate_count": int(len(candidates)),
        "sampled_density_cell_count": actual_patch_count,
        "sampling_replacement": False,
        "layer_summary": layer_summary,
    }


def sample_step7a_candidates(
    candidates: pd.DataFrame,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(config.get("domain_sampling_mode", config.get("step7a_sampling_mode", "legacy_fixed_count")))
    if mode == "poisson_intensity":
        return sample_candidate_voxels_by_intensity(candidates, config, rng)
    if mode == "v2_weighted_probability":
        return sample_candidate_voxels_by_weighted_probability(candidates, config, rng)
    return legacy.sample_candidate_voxels(candidates, config, rng)


def angle_180(value: float) -> float:
    return float(value % 180.0)


def sample_range(value: Any, rng: np.random.Generator, default: float = 0.0) -> float:
    if isinstance(value, list | tuple) and len(value) == 2:
        return float(rng.uniform(float(value[0]), float(value[1])))
    if value is None:
        return float(default)
    return float(value)


def family_layer_param(family: dict[str, Any], key: str, layer: str, default: float) -> float:
    value = family.get(key, default)
    if isinstance(value, dict):
        return float(value.get(layer, default))
    return float(value)


def choose_orientation_family(prior: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    families = list(prior.get("families", []))
    if not families:
        return {"name": "local_pca_default", "code": 1, "weight": 1.0, "base": "local_pca"}
    weights = np.asarray([max(float(item.get("weight", 0.0)), 0.0) for item in families], dtype=float)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(families), dtype=float)
    idx = int(rng.choice(np.arange(len(families)), p=weights / weights.sum()))
    return dict(families[idx])


def estimate_selected_local_orientation(
    selected: pd.DataFrame,
    density: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    density_p95 = {
        layer: max(float(group["SourceDensity"].quantile(0.95)), 1.0e-9)
        for layer, group in selected.groupby("LayerGroup", dropna=False)
    }
    orientations: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        layer = str(row["LayerGroup"])
        orientations.append(
            legacy.estimate_local_orientation(
                density=density,
                y_idx=int(row["IY"]),
                x_idx=int(row["IX"]),
                t_idx=int(row["IT"]),
                layer=layer,
                source_density=float(row["SourceDensity"]),
                density_p95=density_p95.get(layer, float(row["SourceDensity"])),
                config=config,
            )
        )
    return orientations


def apply_geologic_orientation_prior(
    selected: pd.DataFrame,
    density: np.ndarray,
    config: dict[str, Any],
    domain_name: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prior = dict(config.get("orientation_prior", {}))
    if not bool(prior.get("enabled", False)):
        return selected, {"enabled": False}

    work = selected.reset_index(drop=True).copy()
    local_orientations = estimate_selected_local_orientation(work, density=density, config=config)
    min_dip = float(prior.get("global_min_dip_deg", config.get("min_dip_deg", 35.0)))
    max_dip = float(prior.get("global_max_dip_deg", config.get("max_dip_deg", 89.0)))

    override_azimuth: list[float] = []
    override_dip: list[float] = []
    family_names: list[str] = []
    family_codes: list[int] = []
    base_azimuths: list[float] = []
    base_dips: list[float] = []
    base_sources: list[str] = []
    local_azimuths: list[float] = []
    local_dips: list[float] = []
    local_sources: list[str] = []
    azimuth_offsets: list[float] = []
    dip_offsets: list[float] = []

    for idx, row in work.iterrows():
        layer = str(row["LayerGroup"])
        local = local_orientations[idx]
        local_azimuth = float(local["azimuth"])
        local_dip = float(local["dip"])
        local_source = str(local["source"])
        template_azimuth = legacy.layer_param(config, "fallback_azimuth_deg", layer, legacy.layer_param(config, "base_azimuth_deg", layer, 60.0))
        template_dip = legacy.layer_param(config, "fallback_dip_deg", layer, legacy.layer_param(config, "base_dip_deg", layer, 70.0))

        family = choose_orientation_family(prior, rng)
        base_mode = str(family.get("base", "local_pca"))
        if base_mode == "layer_template":
            base_azimuth = float(template_azimuth)
            base_dip = float(template_dip)
            base_source = "layer_template"
        elif base_mode == "local_pca_if_planar":
            if str(local_source).startswith("local_3d_density"):
                base_azimuth = local_azimuth
                base_dip = local_dip
                base_source = local_source
            else:
                base_azimuth = float(template_azimuth)
                base_dip = float(template_dip)
                base_source = "layer_template_after_nonplanar_local_pca"
        else:
            base_azimuth = local_azimuth
            base_dip = local_dip
            base_source = local_source

        az_offset = sample_range(family.get("azimuth_offset_deg", 0.0), rng, 0.0)
        az_jitter = float(rng.normal(0.0, float(family.get("azimuth_jitter_std_deg", 0.0))))
        azimuth = angle_180(base_azimuth + az_offset + az_jitter)

        if "dip_mean_deg" in family:
            dip_mean = family_layer_param(family, "dip_mean_deg", layer, float(template_dip))
            dip_std = family_layer_param(family, "dip_std_deg", layer, 6.0)
            dip = float(rng.normal(dip_mean, dip_std))
        elif str(family.get("dip_base", "base")) == "template":
            dip = float(template_dip) + float(rng.normal(0.0, float(family.get("dip_jitter_std_deg", 6.0))))
        else:
            dip = float(base_dip) + float(rng.normal(0.0, float(family.get("dip_jitter_std_deg", 6.0))))

        family_min_dip = family_layer_param(family, "dip_min_deg", layer, min_dip)
        family_max_dip = family_layer_param(family, "dip_max_deg", layer, max_dip)
        dip = float(np.clip(dip, max(min_dip, family_min_dip), min(max_dip, family_max_dip)))

        override_azimuth.append(azimuth)
        override_dip.append(dip)
        family_names.append(str(family.get("name", "unnamed_family")))
        family_codes.append(int(family.get("code", 0)))
        base_azimuths.append(float(base_azimuth))
        base_dips.append(float(base_dip))
        base_sources.append(base_source)
        local_azimuths.append(local_azimuth)
        local_dips.append(local_dip)
        local_sources.append(local_source)
        azimuth_offsets.append(float(azimuth - base_azimuth))
        dip_offsets.append(float(dip - base_dip))

    work["OverrideAzimuthDeg"] = override_azimuth
    work["OverrideDipDeg"] = override_dip
    work["OrientationFamily"] = family_names
    work["OrientationFamilyCode"] = family_codes
    work["OrientationRule"] = str(prior.get("rule_name", f"{domain_name}_geologic_orientation_family"))
    work["OrientationBaseAzimuthDeg"] = base_azimuths
    work["OrientationBaseDipDeg"] = base_dips
    work["OrientationBaseSource"] = base_sources
    work["LocalPcaAzimuthDeg"] = local_azimuths
    work["LocalPcaDipDeg"] = local_dips
    work["LocalPcaSource"] = local_sources
    work["OrientationAzimuthOffsetDeg"] = azimuth_offsets
    work["OrientationDipOffsetDeg"] = dip_offsets
    return work, {
        "enabled": True,
        "rule_name": str(prior.get("rule_name", f"{domain_name}_geologic_orientation_family")),
        "domain": domain_name,
        "family_counts": legacy.layer_distribution(pd.Series(family_names)),
        "base_source_counts": legacy.layer_distribution(pd.Series(base_sources)),
        "local_pca_source_counts": legacy.layer_distribution(pd.Series(local_sources)),
        "azimuth_stats": legacy.finite_stats(pd.Series(override_azimuth)),
        "dip_stats": legacy.finite_stats(pd.Series(override_dip)),
    }


def attach_selected_orientation_columns(patch_df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    copy_columns = [
        "OrientationFamily",
        "OrientationFamilyCode",
        "OrientationRule",
        "OrientationBaseAzimuthDeg",
        "OrientationBaseDipDeg",
        "OrientationBaseSource",
        "LocalPcaAzimuthDeg",
        "LocalPcaDipDeg",
        "LocalPcaSource",
        "OrientationAzimuthOffsetDeg",
        "OrientationDipOffsetDeg",
    ]
    for column in copy_columns:
        if column in selected.columns and len(selected) == len(patch_df):
            patch_df[column] = selected[column].to_numpy()
    if "OrientationRule" in patch_df.columns:
        patch_df["OrientationSource"] = patch_df["OrientationRule"].astype(str)
    return patch_df


def orientation_rng_for_domain(config: dict[str, Any], domain_cfg: dict[str, Any], domain_name: str) -> np.random.Generator:
    base_seed = int(config.get("orientation_random_seed", config.get("random_seed", 20260715)))
    domain_code = int(domain_cfg.get("domain_code", 1))
    domain_offset = sum(ord(char) for char in domain_name)
    return np.random.default_rng(base_seed + domain_code * 1009 + domain_offset)


def build_domain_patches(
    domain_name: str,
    domain_cfg: dict[str, Any],
    base_config: dict[str, Any],
    trace_mapping_npz: Path,
    surfaces: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = merged_domain_config(base_config, domain_cfg)
    density_sgy_value = domain_cfg.get("density_sgy") or base_config.get("density_sgy")
    if not density_sgy_value:
        raise ValueError(f"domain {domain_name} must define density_sgy")
    density_sgy = Path(str(density_sgy_value)).resolve()
    print(f"[step7a-small] loading domain={domain_name} density={density_sgy}", flush=True)
    grid = legacy.load_density_grid(density_sgy, trace_mapping_npz)
    print(f"[step7a-small] building candidates domain={domain_name}", flush=True)
    candidates, layer_summary = build_small_candidates(grid, surfaces, config, domain_name=domain_name, domain_cfg=domain_cfg)
    print(f"[step7a-small] sampling candidates domain={domain_name}", flush=True)
    selected, sample_summary = sample_step7a_candidates(candidates, config, rng)
    selected["FractureScale"] = "small"
    selected["FractureScaleCode"] = 1
    selected["SmallDomain"] = domain_name
    selected["SmallDomainCode"] = int(domain_cfg.get("domain_code", 1))
    selected["BandContinuityMode"] = str(domain_cfg.get("band_continuity_mode", f"small_{domain_name}_density_sampling"))
    selected, orientation_summary = apply_geologic_orientation_prior(
        selected,
        grid["density"],
        config,
        domain_name,
        orientation_rng_for_domain(config, domain_cfg, domain_name),
    )
    print(f"[step7a-small] building patches domain={domain_name} count={len(selected)}", flush=True)
    patch_df, patch_summary = legacy.build_patch_table(
        selected=selected,
        density=grid["density"],
        x_values=grid["x_values"],
        y_values=grid["y_values"],
        config=config,
        rng=rng,
    )
    patch_df = attach_selected_orientation_columns(patch_df, selected)
    patch_df["SmallDomain"] = domain_name
    patch_df["SmallDomainCode"] = int(domain_cfg.get("domain_code", 1))
    patch_df["GenerationStage"] = "step7a_small_scale_domain_sampling"
    patch_df["SourceType"] = str(domain_cfg.get("source_type", f"small_{domain_name}_density"))
    patch_df["StructuralRelation"] = str(domain_cfg.get("structural_relation", f"small_{domain_name}"))
    patch_df["ConstraintLevel"] = "soft"
    patch_df["Confidence"] = normalize_patch_confidence(patch_df)
    patch_df["NeedsWellCorrection"] = 1
    audit_df = legacy.build_audit(patch_df)
    audit_df["ActionReason"] = str(domain_cfg.get("action_reason", f"small_scale_{domain_name}_from_step6d_domain_density"))
    summary = {
        "density_sgy": str(density_sgy),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "patch_count": int(len(patch_df)),
        "layer_summary": layer_summary,
        "sample_summary": sample_summary,
        "orientation_summary": orientation_summary,
        "patch_build_summary": patch_summary,
        "config": {key: value for key, value in domain_cfg.items() if key != "density_sgy"},
    }
    return candidates, selected, patch_df, {"audit": audit_df, "summary": summary}


def build_summary(
    config_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    patch_df: pd.DataFrame,
    layer_summary: dict[str, Any],
    sample_summary: dict[str, Any],
    orientation_summary: dict[str, Any],
    patch_summary: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "has_patches": len(patch_df) > 0,
        "all_small_scale": bool(patch_df["FractureScale"].astype(str).eq("small").all()),
        "csv_exists": paths["dfn_csv"].exists(),
        "raw_vtk_exists": paths["raw_vtk"].exists(),
        "audit_exists": paths["audit_csv"].exists(),
        "orientation_not_all_missing": bool(patch_df["AzimuthDeg"].notna().all() and patch_df["DipDeg"].notna().all()),
        "no_oversized_small_patches": bool(
            patch_df["LengthM"].le(float(config.get("max_small_length_check_m", 110.0))).all()
            and patch_df["HeightTimeMs"].le(float(config.get("max_small_height_check_ms", 45.0))).all()
        ),
    }
    domain_counts = legacy.layer_distribution(patch_df["SmallDomain"]) if "SmallDomain" in patch_df.columns else {}
    source_counts = legacy.layer_distribution(patch_df["SourceType"]) if "SourceType" in patch_df.columns else {}
    orientation_family_counts = legacy.layer_distribution(patch_df["OrientationFamily"]) if "OrientationFamily" in patch_df.columns else {}
    orientation_rule_counts = legacy.layer_distribution(patch_df["OrientationRule"]) if "OrientationRule" in patch_df.columns else {}
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "config_path": str(config_path),
        "generation_logic": "step7a_small_scale_from_step6d_geologic_domain_densities",
        "inputs": {
            "density_sgy": str(Path(config["density_sgy"]).resolve()) if config.get("density_sgy") else "",
            "small_domain_density_sgys": config.get("small_domain_density_sgys", {}),
            "trace_mapping_npz": str(Path(config["trace_mapping_npz"]).resolve()),
            "layer_dir": str(Path(config["layer_dir"]).resolve()),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "patch_count": int(len(patch_df)),
        "layer_summary": layer_summary,
        "sample_summary": sample_summary,
        "orientation_summary": orientation_summary,
        "patch_build_summary": patch_summary,
        "small_domain_counts": domain_counts,
        "source_type_counts": source_counts,
        "orientation_family_counts": orientation_family_counts,
        "orientation_rule_counts": orientation_rule_counts,
        "patch_stats": {
            "length_m": legacy.finite_stats(patch_df["LengthM"]),
            "height_time_ms": legacy.finite_stats(patch_df["HeightTimeMs"]),
            "area_m2": legacy.finite_stats(patch_df["PatchAreaM2"]),
            "source_density": legacy.finite_stats(patch_df["SourceDensity"]),
            "azimuth_deg": legacy.finite_stats(patch_df["AzimuthDeg"]),
            "dip_deg": legacy.finite_stats(patch_df["DipDeg"]),
        },
        "orientation_source_distribution": legacy.layer_distribution(patch_df["OrientationSource"]),
        "checks": checks,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    output_dir = Path(config["output_dir"]).resolve()
    ensure_dir(output_dir)
    paths = output_paths(output_dir)
    rng = np.random.default_rng(int(config.get("random_seed", 20260715)))

    print("[step7a-small] loading small density", flush=True)
    trace_mapping_npz = Path(config["trace_mapping_npz"]).resolve()
    print("[step7a-small] loading surfaces", flush=True)
    if config.get("density_sgy"):
        density_for_axis = Path(str(config["density_sgy"])).resolve()
    elif config.get("small_domain_density_sgys"):
        density_for_axis = Path(str(next(iter(dict(config["small_domain_density_sgys"]).values())))).resolve()
    else:
        raise ValueError("config must define density_sgy or small_domain_density_sgys")
    axis_grid = legacy.load_density_grid(density_for_axis, trace_mapping_npz)
    surfaces = legacy.attach_surface_grids(Path(config["layer_dir"]).resolve(), axis_grid["x_values"], axis_grid["y_values"])

    domain_density_sgys = dict(config.get("small_domain_density_sgys", {}))
    domain_generation = dict(config.get("small_domain_generation", {}))
    if domain_density_sgys:
        candidates_parts: list[pd.DataFrame] = []
        selected_parts: list[pd.DataFrame] = []
        patch_parts: list[pd.DataFrame] = []
        audit_parts: list[pd.DataFrame] = []
        layer_summary: dict[str, Any] = {}
        sample_summary: dict[str, Any] = {}
        patch_summary: dict[str, Any] = {}
        orientation_summary: dict[str, Any] = {}
        for domain_name, domain_path in domain_density_sgys.items():
            domain_cfg = dict(domain_generation.get(domain_name, {}))
            domain_cfg["density_sgy"] = domain_path
            candidates_i, selected_i, patch_i, extra_i = build_domain_patches(
                domain_name=domain_name,
                domain_cfg=domain_cfg,
                base_config=config,
                trace_mapping_npz=trace_mapping_npz,
                surfaces=surfaces,
                rng=rng,
            )
            candidates_parts.append(candidates_i)
            selected_parts.append(selected_i)
            patch_parts.append(patch_i)
            audit_parts.append(extra_i["audit"])
            layer_summary[domain_name] = extra_i["summary"]["layer_summary"]
            sample_summary[domain_name] = extra_i["summary"]["sample_summary"]
            orientation_summary[domain_name] = extra_i["summary"]["orientation_summary"]
            patch_summary[domain_name] = extra_i["summary"]["patch_build_summary"]
        candidates = pd.concat(candidates_parts, ignore_index=True)
        selected = pd.concat(selected_parts, ignore_index=True)
        patch_df = pd.concat(patch_parts, ignore_index=True)
        audit_df = pd.concat(audit_parts, ignore_index=True)
        patch_df["PatchID"] = [f"step7a_small_{idx + 1:06d}" for idx in range(len(patch_df))]
        audit_df["PatchID"] = patch_df["PatchID"].to_numpy()
    else:
        grid = axis_grid
        print("[step7a-small] building candidates", flush=True)
        candidates, layer_summary = build_small_candidates(grid, surfaces, config)
        print("[step7a-small] sampling candidates", flush=True)
        selected, sample_summary = sample_step7a_candidates(candidates, config, rng)
        selected["FractureScale"] = "small"
        selected["FractureScaleCode"] = 1
        selected["SmallDomain"] = "background"
        selected["SmallDomainCode"] = 1
        selected["BandContinuityMode"] = "small_background_density_sampling"
        selected, orientation_summary = apply_geologic_orientation_prior(
            selected,
            grid["density"],
            config,
            "background",
            orientation_rng_for_domain(config, {"domain_code": 1}, "background"),
        )
        print(f"[step7a-small] building patches={len(selected)}", flush=True)
        patch_df, patch_summary = legacy.build_patch_table(
            selected=selected,
            density=grid["density"],
            x_values=grid["x_values"],
            y_values=grid["y_values"],
            config=config,
            rng=rng,
        )
        patch_df = attach_selected_orientation_columns(patch_df, selected)
        patch_df["SmallDomain"] = "background"
        patch_df["SmallDomainCode"] = 1
        patch_df["GenerationStage"] = "step7a_small_scale_density_sampling"
        patch_df["SourceType"] = "small_background_density"
        patch_df["StructuralRelation"] = "small_background"
        patch_df["ConstraintLevel"] = "soft"
        patch_df["Confidence"] = normalize_patch_confidence(patch_df)
        patch_df["NeedsWellCorrection"] = 1
        audit_df = legacy.build_audit(patch_df)
        audit_df["ActionReason"] = "small_scale_background_from_step6a_small_density"
    patch_df.to_csv(paths["dfn_csv"], index=False, encoding="utf-8-sig")
    audit_df.to_csv(paths["audit_csv"], index=False, encoding="utf-8-sig")
    legacy.write_legacy_vtk(
        paths["raw_vtk"],
        patch_df,
        "step7a_small_scale_dfn_raw_time",
        display=False,
        display_z_scale=float(config.get("display_z_scale", 5.0)),
        geometry_time_scale_m_per_ms=float(config.get("geometry_time_scale_m_per_ms", config.get("orientation_time_scale_m_per_ms", 1.0))),
    )
    summary = build_summary(config_path, config, paths, candidates, selected, patch_df, layer_summary, sample_summary, orientation_summary, patch_summary)
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[step7a-small] CSV: {paths['dfn_csv']}", flush=True)
    print(f"[step7a-small] VTK: {paths['raw_vtk']}", flush=True)
    print(f"[step7a-small] status={summary['status']} patch_count={len(patch_df)}", flush=True)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

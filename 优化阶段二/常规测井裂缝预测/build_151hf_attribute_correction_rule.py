from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "near_well_attribute_samples_imaging" / "aligned_sample_csv_labeled" / "车151HF_sample.csv"
OUTPUT_JSON = ROOT / "attribute_correction_rule_151HF.json"
OUTPUT_CSV = ROOT / "attribute_correction_rule_151HF_feature_stats.csv"

INVALID_THRESHOLD = -999.0
WINDOW_IDS = [f"W{i:02d}" for i in range(63)]
ROLE_PREFIX = {
    "ant": "车西_蚂蚁体T4_T7",
    "coh": "车西_相干体T4_T7",
}


def to_num(raw: str | None) -> float:
    if raw is None or raw == "":
        return math.nan
    try:
        value = float(raw)
    except Exception:
        return math.nan
    if (not math.isfinite(value)) or value <= INVALID_THRESHOLD:
        return math.nan
    return value


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def quantile(values: list[float], q: float) -> float:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    pos = (len(finite) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return finite[lo]
    w = pos - lo
    return finite[lo] * (1.0 - w) + finite[hi] * w


def corr(xs: list[float], ys: list[float]) -> tuple[float, int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 10:
        return math.nan, len(pairs)
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    sy = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if sx == 0.0 or sy == 0.0:
        return math.nan, len(pairs)
    score = sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)
    return score, len(pairs)


def build_feature_row(row: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {
        "TIME": to_num(row.get("TIME")),
        "P10": to_num(row.get("P10")),
        "P21": to_num(row.get("P21")),
        "P33": to_num(row.get("P33")),
        "Frac_Azimuth": to_num(row.get("Frac_Azimuth")),
        "Frac_Dip": to_num(row.get("Frac_Dip")),
    }
    for role, prefix in ROLE_PREFIX.items():
        vals = [to_num(row.get(f"{prefix}_{wid}")) for wid in WINDOW_IDS]
        out[f"{role}_center"] = to_num(row.get(f"{prefix}_CENTER"))
        out[f"{role}_mean"] = mean(vals)
        out[f"{role}_min"] = min([v for v in vals if math.isfinite(v)], default=math.nan)
        out[f"{role}_max"] = max([v for v in vals if math.isfinite(v)], default=math.nan)
        out[f"{role}_p10"] = quantile(vals, 0.10)
        out[f"{role}_p90"] = quantile(vals, 0.90)
        out[f"{role}_valid_ratio"] = sum(math.isfinite(v) for v in vals) / len(vals)
    return out


def load_rows() -> list[dict[str, float]]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [build_feature_row(row) for row in reader]


def choose_rule(rows: list[dict[str, float]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    fracture_rows = [
        row
        for row in rows
        if math.isfinite(row["Frac_Azimuth"]) and math.isfinite(row["Frac_Dip"])
    ]
    candidates = [
        ("P10", "ant_mean", +1.0),
        ("P10", "ant_max", +1.0),
        ("P10", "coh_p10", -1.0),
        ("P10", "coh_mean", -1.0),
        ("P21", "ant_mean", +1.0),
        ("P21", "ant_p90", +1.0),
        ("P21", "coh_p90", -1.0),
        ("P21", "coh_max", -1.0),
    ]

    summary_rows: list[dict[str, object]] = []
    for target, feature, sign in candidates:
        score, n = corr(
            [row[target] for row in fracture_rows],
            [row[feature] for row in fracture_rows],
        )
        summary_rows.append(
            {
                "target": target,
                "feature": feature,
                "sign": sign,
                "corr_on_fracture_rows": score,
                "pair_count": n,
            }
        )

    summary_rows.sort(
        key=lambda item: abs(float(item["corr_on_fracture_rows"]))
        if math.isfinite(float(item["corr_on_fracture_rows"]))
        else -1.0,
        reverse=True,
    )

    selected_positive = [
        item for item in summary_rows
        if item["feature"].startswith("ant_")
    ][:3]
    selected_negative = [
        item for item in summary_rows
        if item["feature"].startswith("coh_")
    ][:2]
    selected = selected_positive + selected_negative
    rule = {
        "rule_name": "attribute_correction_rule_151HF_v1",
        "well_name": "车151HF",
        "purpose": "use seismic-body-derived window statistics to softly correct P10/P21-related fracture density strength",
        "selected_features": selected,
        "recommended_targets": ["P10", "P21"],
        "not_recommended_targets": ["P33", "Frac_Azimuth", "Frac_Dip"],
        "strategy": {
            "mode": "post_adjust_density_score",
            "base_score_formula": "sum(sign * standardized_feature * abs(corr))",
            "apply_scope": "only refine existing main-model density/probability output, do not replace main model",
            "priority_positive": ["ant_mean", "ant_max", "ant_p90"],
            "priority_negative": ["coh_p10", "coh_mean", "coh_p90", "coh_max"],
            "notes": [
                "ant-related features act as positive support for higher P10/P21",
                "coherence-related features act as reverse support; lower coherence implies stronger fracture likelihood",
                "curvature features are not stable enough in current aligned sample and are excluded from v1 rule",
            ],
        },
        "sample_counts": {
            "all_rows": len(rows),
            "fracture_rows": len(fracture_rows),
        },
    }
    return summary_rows, rule


def save_feature_stats(summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        return
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> int:
    rows = load_rows()
    summary_rows, rule = choose_rule(rows)
    save_feature_stats(summary_rows)
    OUTPUT_JSON.write_text(json.dumps(rule, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved rule json: {OUTPUT_JSON}")
    print(f"Saved feature stats: {OUTPUT_CSV}")
    for item in rule["selected_features"]:
        print(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

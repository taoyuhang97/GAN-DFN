from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


INVALID_THRESHOLD = -999.0
WINDOW_IDS = [f"W{i:02d}" for i in range(63)]
FEATURE_TO_PREFIX = {
    "ant_center": ("车西_蚂蚁体T4_T7", "center"),
    "ant_mean": ("车西_蚂蚁体T4_T7", "mean"),
    "ant_max": ("车西_蚂蚁体T4_T7", "max"),
    "ant_p90": ("车西_蚂蚁体T4_T7", "p90"),
    "coh_center": ("车西_相干体T4_T7", "center"),
    "coh_mean": ("车西_相干体T4_T7", "mean"),
    "coh_max": ("车西_相干体T4_T7", "max"),
    "coh_p90": ("车西_相干体T4_T7", "p90"),
    "coh_p10": ("车西_相干体T4_T7", "p10"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply 151HF-derived attribute correction rule to a predicted well CSV.")
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--attribute-csv", type=Path, required=True)
    parser.add_argument("--rule-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


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


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def extract_feature(row: dict[str, str], feature_name: str) -> float:
    if feature_name not in FEATURE_TO_PREFIX:
        return math.nan
    prefix, mode = FEATURE_TO_PREFIX[feature_name]
    if mode == "center":
        return to_num(row.get(f"{prefix}_CENTER"))
    values = [to_num(row.get(f"{prefix}_{wid}")) for wid in WINDOW_IDS]
    if mode == "mean":
        return mean(values)
    if mode == "max":
        finite = [v for v in values if math.isfinite(v)]
        return max(finite) if finite else math.nan
    if mode == "p90":
        return quantile(values, 0.90)
    if mode == "p10":
        return quantile(values, 0.10)
    return math.nan


def load_rule(rule_json: Path) -> dict:
    with rule_json.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def build_attribute_index(attribute_csv: Path) -> dict[float, dict[str, str]]:
    with attribute_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return {
            round(float(row["TIME"]), 6): row
            for row in reader
            if row.get("TIME") not in (None, "")
        }


def standardize_feature(values: list[float], value: float) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite or not math.isfinite(value):
        return math.nan
    mu = sum(finite) / len(finite)
    var = sum((v - mu) ** 2 for v in finite) / max(len(finite), 1)
    std = math.sqrt(var)
    if std < 1e-8:
        return 0.0
    return (value - mu) / std


def main() -> int:
    args = build_parser().parse_args()
    rule = load_rule(args.rule_json)
    attr_index = build_attribute_index(args.attribute_csv)

    with args.prediction_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        pred_rows = list(csv.DictReader(file_obj))
        fieldnames = list(pred_rows[0].keys()) if pred_rows else []

    feature_names = [item["feature"] for item in rule.get("selected_features", [])]
    raw_feature_values: dict[str, list[float]] = {name: [] for name in feature_names}
    row_features: list[dict[str, float]] = []

    for row in pred_rows:
        time_val = row.get("TIME")
        attr_row = None
        try:
            attr_row = attr_index.get(round(float(time_val), 6))
        except Exception:
            attr_row = None
        feat_map: dict[str, float] = {}
        for feature_name in feature_names:
            value = extract_feature(attr_row, feature_name) if attr_row is not None else math.nan
            feat_map[feature_name] = value
            raw_feature_values[feature_name].append(value)
        row_features.append(feat_map)

    extra_fields = [
        "ATTR_CORR_SCORE",
        "ATTR_CORR_ACTIVE_COUNT",
        "P10_CORR_FACTOR",
        "P21_CORR_FACTOR",
        "P10_PRED_CORR",
        "P21_PRED_CORR",
    ]
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    output_rows: list[dict[str, str]] = []
    for row, feat_map in zip(pred_rows, row_features):
        score = 0.0
        active_count = 0
        for item in rule.get("selected_features", []):
            feature_name = item["feature"]
            weight = abs(float(item["corr_on_fracture_rows"]))
            sign = float(item["sign"])
            z = standardize_feature(raw_feature_values[feature_name], feat_map[feature_name])
            if math.isfinite(z):
                score += sign * weight * z
                active_count += 1

        # Soft correction factors, clipped to avoid overcorrection.
        p10_factor = max(0.75, min(1.25, 1.0 + 0.20 * score))
        p21_factor = max(0.75, min(1.25, 1.0 + 0.15 * score))

        p10_pred = to_num(row.get("P10_PRED"))
        p21_pred = to_num(row.get("P21_PRED"))
        p10_corr = p10_pred * p10_factor if math.isfinite(p10_pred) else math.nan
        p21_corr = p21_pred * p21_factor if math.isfinite(p21_pred) else math.nan

        out = dict(row)
        out["ATTR_CORR_SCORE"] = f"{score:.6f}" if math.isfinite(score) else ""
        out["ATTR_CORR_ACTIVE_COUNT"] = str(active_count)
        out["P10_CORR_FACTOR"] = f"{p10_factor:.6f}" if math.isfinite(p10_factor) else ""
        out["P21_CORR_FACTOR"] = f"{p21_factor:.6f}" if math.isfinite(p21_factor) else ""
        out["P10_PRED_CORR"] = f"{p10_corr:.6f}" if math.isfinite(p10_corr) else ""
        out["P21_PRED_CORR"] = f"{p21_corr:.6f}" if math.isfinite(p21_corr) else ""
        output_rows.append(out)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved corrected csv: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

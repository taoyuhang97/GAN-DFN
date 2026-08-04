# T4-T7 Surface Tools

This directory contains a standalone toolset for T4/T5/T6/T7 surface-time assignment and fixed time-domain classification.

## Scope

- Read T4/T5/T6/T7 `.dat` surface files.
- Prefer `T4` old-style file by default.
- Prefer updated `T5/T6/T7` files by default.
- Assign each record the nearest surface time by Manhattan distance in `XY`.
- Classify each record with fixed rules:
  - `T4 <= TIME < T6` => `sha3_t4_t6`
  - `T6 <= TIME < T7` => `sha4_t6_t7`
  - otherwise => `out_of_target`
- Run basic validation:
  - `T4 < T5 < T6 < T7`
  - minimum adjacent thickness checks

If `T4` lacks extra metadata, treat it the same as the other surfaces: plain `X Y Z(time)` points.

## Files

- `surface_tools.py`
  - core reader, selector, assigner, classifier, validator, CLI entry
- `run_surface_classifier.py`
  - thin executable wrapper
- `demo_data/`
  - synthetic `.dat` surfaces and a small records csv for self-check

## Input format

Surface files:

- file extension: `.dat`
- one point per line
- first three whitespace-separated values are treated as `X Y Z`
- `Z` is interpreted as time

Records csv:

- required columns: `X`, `Y`, `TIME`
- other columns are preserved

## Output columns

The output csv keeps original columns and appends:

- `RecordID`
- `T4_TIME`, `T5_TIME`, `T6_TIME`, `T7_TIME`
- `T*_NEAREST_X`, `T*_NEAREST_Y`
- `T*_MANHATTAN_DISTANCE`
- `TimeDomainClass`
- `Thickness_T4_T5`, `Thickness_T5_T6`, `Thickness_T6_T7`
- `Thickness_T4_T6`, `Thickness_T6_T7_Target`
- `Check_Order_T4_T5_T6_T7`
- `Check_MinThickness_T4_T5`
- `Check_MinThickness_T5_T6`
- `Check_MinThickness_T6_T7`
- `Check_All`

## Usage

Run from repo root:

```bash
python 优化阶段二/当前任务/step0_t4_t7_surface_tools/run_surface_classifier.py \
  --records-csv 优化阶段二/当前任务/step0_t4_t7_surface_tools/demo_data/demo_records.csv \
  --layer-dir 优化阶段二/当前任务/step0_t4_t7_surface_tools/demo_data \
  --output-csv 优化阶段二/当前任务/step0_t4_t7_surface_tools/demo_data/demo_output.csv \
  --summary-json 优化阶段二/当前任务/step0_t4_t7_surface_tools/demo_data/demo_summary.json \
  --min-thickness 20
```

## Selection rule

When multiple files for the same surface code exist:

- `T4` prefers names containing `old`, `jiu`, or `legacy`
- `T5/T6/T7` prefer names containing update hints such as `20240715`, `DM_Sm`, `Ato`, `gljm`

## Demo expectations

With the bundled demo data:

- one record falls into `sha3_t4_t6`
- one record falls into `sha4_t6_t7`
- one record falls into `out_of_target`
- all order/thickness checks pass

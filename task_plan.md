# Task Plan: Step6-Step7 multiscale rebalance

Status: completed historical plan. The 2026-07-23 checkpoint is documented in `docs/正式主线项目现状与后续修改_2026-07-23.md` and `优化阶段二/正式主线/CANDIDATE_CHEYE1_VERSION_MANIFEST.json`.

## Goal

Document and then execute the next `candidate_cheye1` multiscale DFN rebalance. The new workflow must split Step6 into small, medium, large, and integration stages, then rebuild Step7A/B/C/D, Step8, and Step9 without overwriting previous results.

## Phases

- [x] Review existing Step6/Step7 multiscale documents and current implementation state.
- [x] Add a concrete execution document for the next rebalance pass.
- [x] Self-check the execution document for logical gaps and unsafe assumptions.
- [x] Implement Stage 0 input/coordinate QC.
- [x] Implement Step6A small-scale background density整理.
- [x] Implement Step6B medium corridor prior.
- [x] Implement Step6C large fault/fault-zone prior.
- [x] Implement Step6D multiscale bundle.
- [x] Rebuild Step7A small-scale DFN.
- [x] Rebuild Step7B medium-scale continuous corridor DFN.
- [x] Rebuild Step7C large fault/fault-zone DFN.
- [x] Rebuild Step7D multiscale fusion.
- [x] Run Step8 well correction.
- [x] Run Step9 final section figures and QC.
- [x] Rework Step6B medium prior to add a vertical low-coherence branch without overwriting old outputs.
- [x] Rework Step7B medium DFN to cover more valid Step6B components and reduce low-dip medium patches.
- [x] Run Step6B/Step7B preview outputs and record QC.

## Decisions

- New output suffix: `candidate_cheye1_multiscale_rebalance_v1`.
- Do not overwrite previous `candidate_cheye1_multiscale_v1`, `candidate_cheye1_multiscale_v4`, `preview_v1/v2/v3`, `detail_preserve_v4_fault_postfusion_v2`, or `fused_lowcoh_steep` outputs.
- Step6A is small/background density and starts from the current 3D density volume without retraining.
- Step6B is medium-scale corridor evidence, mainly AntTrack high values with low-coherence and curvature support.
- Step6C is large fault/fault-zone evidence, using original fault interpretation as hard constraints plus inferred steep low-coherence candidates.
- Step6D preserves scale labels and evidence sources; a flattened integrated density is only for compatibility/QC.
- Step7 must generate small, medium, and large DFN by different geometry rules before fusion.
- Step8 must not move or destroy original large fault hard constraints.
- Step9 must overlay original fault interpretation, not generated fault patches masquerading as original interpretation.

## Current Documentation

- `优化阶段二/正式主线/STEP6_MULTISCALE_REFACTOR_PLAN.md`: high-level refactor plan.
- `优化阶段二/正式主线/STEP6B_STEP7ABCD_MULTISCALE_IMPLEMENTATION.md`: current implemented multiscale v1 record.
- `优化阶段二/正式主线/STEP6_STEP7_MULTISCALE_REBALANCE_EXECUTION_PLAN.md`: next executable rebalance plan.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| planning-with-files skill path first read failed | Tried `/home/tyh/.codex/skills/.system/planning-with-files/SKILL.md` | Read correct path `/home/tyh/.codex/skills/planning-with-files/SKILL.md`. |
| Step6D failed on empty inferred large-fault CSV | Step6C produced zero inferred components, so `large_fault_component_summary.csv` had no columns | Added `read_optional_csv()` to treat empty CSV as an empty DataFrame and reran Step6D successfully. |
| Step7C lowcoh panel split failed on `np.where()` unpack | Treated `inferred_component_id` as `(y,x,time)` grid | Fixed `build_lowcoh_component_panels()` to read current Step6C `(trace,time)` component grid and map trace index to X/Y. |

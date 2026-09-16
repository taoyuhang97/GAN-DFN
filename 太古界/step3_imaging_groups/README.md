# Step 3 Imaging Supervision Samples

# Step 3 Imaging Supervision Samples

> 本目录同时保留砂砾岩正式主线的 README 与脚本。**太古界实际入口是
> `build_taigu_imaging_groups.py`**（配置 `configs/taigu_step3_imaging_groups.json`，
> 输出 `output/taigu_step3_imaging_v3`），它读取 Step2 v3 的
> `taigu_step2_segment_manifest.csv` 聚合各段并按 **TVD** 贴成像密度/点位标签；`groups/*.csv`
> 是 Step4 的唯一监督输入。构建后必须运行 `validate_taigu_rebuild_delivery.py`
> （列合同 + 逐井放行 + 点位账目 + 覆盖完整度，`delivery_acceptance_summary.json`
> 的 `status=pass` 为放行条件）。下方正文是砂砾岩 `formal_rebuild` 的参考说明，
> 不能直接用于太古界。

本次同时落地第三步（监督门控 + 产状点吸附放宽）：

- **产状点吸附放宽**：只要落在常规测井覆盖内就吸附到最近采样点，不再因"离网格超容差"丢弃；
  `point_mapping_qc.csv` 增加 `GridStepM/AttachResidualMedianM/AttachResidualP95M/AttachResidualMaxM/AttachedBeyondHalfStepRows`；
- **监督层级**：组 CSV 与清单新增 `SupervisionTier`（`strong` / `presence_only` / `audit_only`），
  由 `supervision_status` 映射、可按井覆盖（配置键 `supervision_tier`），供 Step4 做监督门控。


## 太古界 v4（2026-09-16，第一步 + 第二步修正）

新入口：`configs/taigu_step3_imaging_groups_v4.json` → 输出 `output/taigu_step3_imaging_v4`；
v3 产物保留不动，供旧链条复现。

本次修正五件事：

1. **多密度源合并口径**：改为"按源声明深度范围（`depth_min_m`/`depth_max_m`）裁剪 + 同深度取最大值"。
   旧写法 `concat → sort_values → drop_duplicates(keep="last")` 在深度区间完全重叠的两个源之间
   会随机丢值（埕北313 实测丢掉一半，且换排序方式结果在 1246 / 620 / 0 之间跳）。
2. **成像解释窗口显式登记**：每个密度源 = 一个解释窗口（`WindowID`，如 `埕北313_W01`），
   窗口的深度范围来自配置声明；组 CSV 与组清单新增 `ImagingWindowID`，分组时按窗口断开。
   登记表见 `imaging_window_registry.csv`。
3. **填充区不再当作"密度 0"**：文件为写满深度轴而填的 0 一律裁掉，下游得到的是**无资料**
   （`DensitySupportStatus=outside_imaging_window`）而不是"明确没有裂缝"的强负样本。
   各源裁掉多少填充行见 `density_source_merge_audit.csv` 的 `PaddingRowsDropped`。
4. **密度标定按窗口做**："每条缝 = 1 条"，`Density` 输出标定后的线密度（条/米），
   原值保留在 `DensityRaw`，系数在 `DensityScaleFactor`（该行所属窗口的系数）。
   按井的汇总结算保留在 `density_calibration_audit.csv` 做对照。
5. **段外成像数据不再静默丢弃**：密度段外部分记入 `imaging_out_of_log_coverage.csv`，
   未吸附的产状点明细记入 `unmapped_imaging_points.csv`（供 Step8 井控使用）；
   `point_mapping_qc.csv` 增加 `DroppedOutsideLogCoverage` / `DroppedBeyondGridTolerance` 两列。

新增审计产物：`density_source_merge_audit.csv`、`density_calibration_audit.csv`、
`imaging_window_registry.csv`、`imaging_out_of_log_coverage.csv`、`unmapped_imaging_points.csv`。

验收命令：

```bash
python3 太古界/step3_imaging_groups/build_taigu_imaging_groups.py \
  --config 太古界/step3_imaging_groups/configs/taigu_step3_imaging_groups_v4.json --replace-output
python3 太古界/step3_imaging_groups/validate_taigu_rebuild_delivery.py \
  --config 太古界/step3_imaging_groups/configs/taigu_step3_imaging_groups_v4.json
```

本步骤的正式成果是 `formal_rebuild`，用于构造成像测井强监督训练样本。

## 正式口径

- `Density` 是主监督标签。
- 外部成像井 `车660-1/车660-2/车662/车663` 的 `Density` 只取成像统计文件中的 `FVDC`。
- 矿区内成像井 `车151HF/车页1导眼` 的属性底板只取 Step2 正式样本，旧 labeled 样本只用于取 `P10` 标签来源。
- `HasFracture = Density > 0`，只是辅助存在标签。
- `GT_POINT_FLAG` 是原始裂缝点位按 `MD/DEPT` 映射到样本网格后的点位旗标，不等同于 `HasFracture`。
- 外部井补样必须与 Step2 主表和 `3x3_context` 同构；旧 `SEIS_0~SEIS_62` 只允许作为恢复 `SeisAmp` 九点上下文的中间来源，不作为正式输出字段。
- 六个成像井段都必须具有与各自 Step2/Step2-like 主表逐行同 `DEPT` 的 `*_gr_resistivity.csv`；车660两个无目标段曲线的井段允许为全空表。
- GR/电阻率值保留在 Step2 补充表中，不重复写入 `groups/*.csv`；Step3 manifest 只记录文件路径和各 group 的有效覆盖数。
- 缺失的 `Coherence/AntTrack/CurvatureMax/CurvaturePos` 及其 `3x3` 上下文置空，保留后续补体数据空间。
- 不生成冗余总训练表，训练输入只使用 `groups/*.csv`。

## 正式输入

- 矿区内底板：`step2_real_well_t4_t7_samples/output/formal_all_wells/{车151HF,车页1导眼}`
- 外部井 around/LAS/裂缝密度/裂缝点位：详见 `output/formal_rebuild/source_manifest.csv`

## 正式输出

- `output/formal_rebuild/groups/*.csv`
- `output/formal_rebuild/sample_group_manifest.csv`
- `output/formal_rebuild/source_manifest.csv`
- `output/formal_rebuild/formal_data_contract_columns.csv`
- `output/formal_rebuild/delivery_group_qc.csv`
- `output/formal_rebuild/delivery_outer_step2_contract_qc.csv`
- `output/formal_rebuild/delivery_gr_resistivity_qc.csv`
- `output/formal_rebuild/delivery_acceptance_summary.json`

## 运行

```bash
bash 优化阶段二/正式主线/step3_imaging_supervision_samples/run_formal_step3_rebuild.sh
```

构建器会先重建四个外部 Step2-like 底板，再调用 Step2 公共补充模块生成对应 GR/电阻率文件，然后重建标签 group 和 manifest。验证器要求补充表与底板 `DEPT` 完全一致，并核对 group 覆盖统计。

验收要求：`delivery_acceptance_summary.json` 中 `status=pass`。

## MD-v2 GR/电阻率分析底板

三口成像井的 MD 域测井、原始密度网格和裂缝点重建使用：

```bash
python 优化阶段二/正式主线/step3_imaging_supervision_samples/build_md_domain_analysis_samples.py
```

输出位于 `output/md_v2_analysis`。该产品明确为 MD 域，不借用其他井的时深关系，也不伪造 `TVD/X/Y/TIME`。

## 禁用成果

- `output/formal_imaging_supervision/imaging_supervision_main.csv` 不是当前正式 Step3 输入。
- `build_imaging_supervision_samples.py` 是旧总表式实现，不再作为当前正式主线入口。
- `step3_real_well_prediction` 是错误放入正式主线的调试基线，不能作为正式 Step3 或 Step4 输入。

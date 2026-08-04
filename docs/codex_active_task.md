# Codex Active Task

Status: done
Updated: 2026-07-02 03:11 CST
CurrentPhase: 全部完成

## Current Goal

完成优化阶段二正式主线，从当前仍在修改的 `Step 4 专家库与真实井预测` 开始，逐步打通：

`Step 4 专家库真实井预测 -> Step 5 单源井虚拟测井 -> Step 6 T4-T7统一样本包 -> Step 7 demo裂缝密度体 -> Step 8 初始DFN -> Step 9 井控校正DFN`

当前不能把已有 `step4_acceptance_summary.json status=pass` 视为最终完成。它只说明已有阶段性输出；用户已确认 Step 4 仍在修改，后续提醒和续跑必须从 Step 4 收尾开始。

本任务采用连续执行模式：Codex 不需要在每一步完成后等待用户确认。每个阶段完成后，Codex 必须自行运行检查、判断验收结果、更新本文档，然后继续下一阶段，直到全部项目修改完成并把 `Status` 改为 `done`。

## Working Directory

`/home/tyh/projects/petroleum/code/GAN-DFN`

## Source Of Truth

优先级从高到低：

1. 用户最新口径：Step 4 仍在修改，未最终完成；后续要求 Codex 连续执行完整流程，不在阶段之间等待人工确认。
2. 本文档当前状态账和下一步。
3. `优化阶段二/正式主线/step4_expert_real_well_prediction/README.md`
4. `优化阶段二/正式主线/step3_imaging_supervision_samples/README.md`
5. `优化阶段二/正式主线/MIGRATION_RULES.md`
6. `优化阶段二/当前任务/data_inventory/step3_5正式实现状态_2026-07-01.md`
7. `优化阶段二/当前任务/data_inventory/项目交接总表_2026-07-01.md`

若旧状态文档与 README 或用户最新口径冲突，以用户最新口径和正式主线 README 为准。

## Non-Negotiable Rules

- `Density` 是主监督；`HasFracture` 和点位只作为辅助、后处理和最终 DFN 校正信息。
- 当前目标只做矿区内 `T4-T7`，分组固定为 `T4-T6=沙三段`、`T6-T7=沙四段`。
- 成像井分层若与新层位解释冲突，以旧项目人工分层为准。
- `step3_real_well_prediction` 是调试基线，不得作为正式成果或下游输入。
- Step 3 当前正式输入是 `step3_imaging_supervision_samples/output/formal_rebuild/groups/*.csv`，不是旧的 `formal_imaging_supervision/imaging_supervision_main.csv`。
- Step 4 当前修改完成前，不推进 Step 5 全量和任何下游正式结果。
- 虚拟测井必须是单源井局部外推，不允许跨井融合。
- 虚拟点属性必须按自身 `X/Y/TIME` 从体数据重新取值，不能复制源井属性。
- 距离只控制搜索范围和置信度，不能主控 `Density`。
- 密度体必须是属性驱动预测，不使用空间平滑扩散作为正式方法。

## Continuous Execution Policy

- 不要在 Step 4、Step 5、Step 6、Step 7、Step 8、Step 9 阶段之间询问用户是否继续。
- 每个阶段结束后先执行本阶段验收检查；检查通过后立刻更新本文档，并进入下一阶段。
- 若检查失败，先定位并修复，再重新运行本阶段；不要把失败状态推进到下游。
- 只有遇到以下情况才停止并向用户说明：
  - 正式输入文件缺失，且无法从项目文档或代码中确定替代路径。
  - 同一个阶段连续修复三轮仍无法通过验收。
  - 需要执行破坏性操作，例如删除正式成果、重置 git、覆盖无法再生的数据。
  - 沙箱或系统权限阻止关键命令运行，且无法通过当前允许的方式继续。
  - 用户在对话中给出新的相互冲突要求。
- 不要把“验收通过”理解成只看脚本退出码；还必须检查输出文件、字段、统计摘要和项目硬口径。

## Current State

### Done / Stable Enough

- [x] Step 1 层位框架与样本挂接已有正式工具。
- [x] Step 2 真实井 `T4-T7` 四表样本已有正式底板：`优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_all_wells`
- [x] Step 3 当前正式 `formal_rebuild` 已生成，并且 `delivery_acceptance_summary.json` 中 `status=pass`。
- [x] Step 4 专家库与真实井连续 `Density` 预测已按最终合约重新运行并通过自检：
  - 运行命令：`python3 优化阶段二/正式主线/step4_expert_real_well_prediction/train_and_predict_expert_library.py --config 优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_expert_prediction.json`
  - 输出目录：`优化阶段二/正式主线/step4_expert_real_well_prediction/output/formal_well_expert_library`
  - `step4_acceptance_summary.json status=pass`
  - `predicted_well_count=54`，`skipped_non_ok_well_count=11`，`total_predicted_rows=376083`
  - `all_wells_t4_t7_density_prediction.csv` 已包含 `Density`、`HasFracture`、`PredDensity`、`PredFractureProb`
  - `all_wells_t4_t7_fracture_points.csv` 已包含 `Density`、`HasFracture`，有效点位行数为 `4091`
  - `formal_constraints.uses_step3_formal_groups=true`，`uses_old_step3_total_table=false`，`holdout_in_training=false`
  - `downstream_contract.checks` 全部为 `true`
- [x] Step 5 单源井虚拟测井已完成全量运行并通过自检：
  - 配置：`优化阶段二/正式主线/step5_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json`
  - 运行命令：`/home/tyh/anaconda3/envs/gan-dfn/bin/python 优化阶段二/正式主线/step5_single_source_virtual_wells/build_single_source_virtual_wells.py --config 优化阶段二/正式主线/step5_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json --progress-every 25`
  - 输出目录：`优化阶段二/正式主线/step5_single_source_virtual_wells/output/formal_single_source_virtual_wells`
  - `real_well_prediction_csv` 已指向 `step4_expert_real_well_prediction/output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`
  - `source_well_count=54`，`candidate_virtual_well_count=1350`，`training_sample_count=9402075`
  - `density_logic=single_source_attribute_continuity`，`distance_role=candidate_range_and_confidence_only`，`source_mixing=forbidden`
  - `DensitySourceLogic` 未出现 `copy_and_distance_decay`
  - `virtual_well_training_summary.csv` 中 `SingleSourceCheck` 失败数为 `0`
  - 属性非空数：`SeisAmp=9402075`，`Coherence=9393696`，`AntTrack=9400830`，`CurvatureMax=9399662`，`CurvaturePos=9399662`
  - `mean_point_confidence=0.6571260945108353`，`mean_attribute_continuity=0.8955722916651514`
  - 为避免半成品覆盖正式结果，Step 5 脚本已改为流式 `.tmp` 写出并在成功结束后替换正式输出；同时复用批量近邻查询以支撑全量运行。
- [x] Step 6 真实井 + 虚拟井 `T4-T7` 统一样本包已完成并通过自检：
  - 配置：`优化阶段二/正式主线/step5_unified_samples_t4_t7/configs/formal_unified_t4_t7_density_samples.json`
  - 运行命令：`/home/tyh/anaconda3/envs/gan-dfn/bin/python 优化阶段二/正式主线/step5_unified_samples_t4_t7/build_unified_t4_t7_density_samples.py --config 优化阶段二/正式主线/step5_unified_samples_t4_t7/configs/formal_unified_t4_t7_density_samples.json`
  - 输出目录：`优化阶段二/正式主线/step5_unified_samples_t4_t7/output`
  - `unified_t4_t7_density_samples_summary.json status=pass`
  - `total_rows=9778158`，其中 `real_well=376083`、`virtual_well=9402075`
  - 层段分布：`沙三段=8428602`，`沙四段=1349556`
  - `DensityLabel` 非空数为 `9778158`，`refined_fracture_point_count=4091`
  - 源井统计：真实井 `54`，虚拟井源井 `54`，虚拟轨迹井 `1350`
  - 字段契约、`SourceKind` 分布、层段范围、属性覆盖和虚拟井单源追溯检查均为 `true`；`virtual_summary_check.single_source_bad_count=0`
- [x] Step 7 demo 区块属性驱动裂缝密度体已完成并通过自检：
  - 配置：`优化阶段二/正式主线/step6_demo_density_volume/configs/formal_candidate_a_density_volume.json`
  - 运行命令：`/home/tyh/anaconda3/envs/gan-dfn/bin/python 优化阶段二/正式主线/step6_demo_density_volume/build_candidate_a_attribute_density_volume.py --config 优化阶段二/正式主线/step6_demo_density_volume/configs/formal_candidate_a_density_volume.json`
  - 输出目录：`优化阶段二/正式主线/step6_demo_density_volume/output`
  - 密度体：`candidate_a_t4_t7_predicted_density_volume.csv`
  - 目标属性：`candidate_a_target_trace_attributes.csv`
  - `candidate_a_density_volume_summary.json status=pass`
  - `target_trace_count=57600`，`predicted_volume_rows=109626`
  - 层段分布：`沙三段=54813`，`沙四段=54813`
  - `PredDensity` 统计：`min=0.0`，`max=5.35313772422463`，`mean=0.3324608734747592`，`median=0.2143743062178131`
  - 模型为 `attribute_driven_regression`，特征来自 Step 6 统一样本和目标 SGY 属性统计；检查确认未使用空间平滑扩散、距离衰减密度或旧候选密度体作为标签。
  - 近井对应检查已完成：`150m` 半径内 `沙三段=3089` 行、`沙四段=3155` 行。
- [x] Step 8 初始 DFN 已完成并通过自检：
  - 配置：`优化阶段二/正式主线/step7_initial_dfn/configs/formal_initial_dfn.json`
  - 运行命令：`/home/tyh/anaconda3/envs/gan-dfn/bin/python 优化阶段二/正式主线/step7_initial_dfn/build_initial_dfn_from_density_volume.py --config 优化阶段二/正式主线/step7_initial_dfn/configs/formal_initial_dfn.json`
  - 输出目录：`优化阶段二/正式主线/step7_initial_dfn/output`
  - 初始 DFN CSV：`initial_dfn_fracture_patches.csv`
  - 初始 DFN VTK：`initial_dfn_raw_time.vtk`、`initial_dfn_display.vtk`
  - 审计：`initial_dfn_generation_audit.csv`
  - `initial_dfn_summary.json status=pass`
  - 裂缝片数量：`9153`，来源密度体单元数：`8902`
  - 分层统计：`沙三段=1918`，`沙四段=7235`
  - `density_mass=36446.355715543956`，`expected_patch_count=9111.588928885989`，实际数量与密度质量控制一致。
  - 宏观分布检查通过：`沙三段` 密度质量占比 `0.207680607581477`、裂缝片占比 `0.20954878182016826`；`沙四段` 密度质量占比 `0.792319392418523`、裂缝片占比 `0.7904512181798318`。
  - 每个裂缝片均包含 `PatchID`、`DensityCellID`、`SourceTraceIdx`、`SourceDensity` 等追溯字段；Step 4 点位无方位/倾角字段，方位/倾角使用分层默认模板并已在审计中记录。
- [x] Step 9 井控校正 DFN 已完成并通过最终验收：
  - 配置：`优化阶段二/正式主线/step8_dfn_well_correction/configs/formal_well_control_correction.json`
  - 运行命令：`/home/tyh/anaconda3/envs/gan-dfn/bin/python 优化阶段二/正式主线/step8_dfn_well_correction/correct_dfn_with_well_controls.py --config 优化阶段二/正式主线/step8_dfn_well_correction/configs/formal_well_control_correction.json`
  - 输出目录：`优化阶段二/正式主线/step8_dfn_well_correction/output`
  - 最终 DFN CSV：`well_corrected_dfn_fracture_patches.csv`
  - 最终 DFN VTK：`well_corrected_dfn_raw_time.vtk`、`well_corrected_dfn_display.vtk`
  - 审计：`well_control_correction_audit.csv`
  - `well_corrected_dfn_summary.json status=pass`
  - 初始 DFN 裂缝片数 `9153`，井控校正后裂缝片数 `9644`
  - 候选 A 内井控裂缝点 `583` 个，覆盖 `9` 口井；其中 `沙三段=399`、`沙四段=184`
  - 校正动作：新增井控裂缝片 `491`，调整已有裂缝片 `92`；每个动作均写入 `well_control_correction_audit.csv`
  - 井控匹配检查：校正前匹配距离 `mean=97.19064878089083`、`median=85.49250997568153`；校正后 `mean=0.0`、`median=0.0`、`max=0.0`
  - 宏观分布保持检查通过：`沙三段` 密度质量占比 `0.207680607581477`、最终裂缝片占比 `0.23807548734964745`；`沙四段` 密度质量占比 `0.792319392418523`、最终裂缝片占比 `0.7619245126503525`；最大差异 `0.030394879768170457` 小于 `0.05`
  - 远井区域保持由 Step 7 密度体控制：井控动作占初始 DFN 的 `0.06369496339997816`，未进行井点过度外推。

### In Progress

- [x] 全部阶段完成。

## Next Step

全部完成；提醒循环可按 `Status: done` 和 `--stop-when-done` 停止。

## Detailed Implementation Plan

### Phase A. Step 4 修改收尾与最终验收

目标：把 Step 4 从“已有阶段性输出”推进到“自检验收通过的正式完成”。

入口文件：

- `优化阶段二/正式主线/step4_expert_real_well_prediction/train_and_predict_expert_library.py`
- `优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_expert_prediction.json`
- `优化阶段二/正式主线/step4_expert_real_well_prediction/README.md`

正式输入：

- Step 3 groups：`优化阶段二/正式主线/step3_imaging_supervision_samples/output/formal_rebuild/groups`
- Step 2 全井底板：`优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_all_wells`

应做动作：

1. 先读当前用户最新要求和本文件，明确 Step 4 仍需收尾，不把旧 `status=pass` 误判为最终完成。
2. 检查 Step 4 代码是否仍满足：
   - 使用 `formal_rebuild/groups/*.csv`
   - 不使用旧 `formal_imaging_supervision/imaging_supervision_main.csv`
   - 不使用 `step3_real_well_prediction`
   - `车页1导眼` 作为 holdout 不进入训练专家
   - 按井留一或 holdout 验证没有点级随机泄漏
3. 根据当前未完成项修改 Step 4 训练、专家选择、阈值、点位细化或输出字段。
4. 运行：

```bash
python 优化阶段二/正式主线/step4_expert_real_well_prediction/train_and_predict_expert_library.py \
  --config 优化阶段二/正式主线/step4_expert_real_well_prediction/configs/formal_expert_prediction.json
```

5. 验收输出至少包括：
   - `output/formal_well_expert_library/step4_acceptance_summary.json`
   - `output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`
   - `output/formal_well_expert_library/all_wells_t4_t7_fracture_points.csv`
   - `output/formal_well_expert_library/all_wells_t4_t7_prediction_summary.csv`
   - `output/formal_well_expert_library/feature_contract.csv`
   - `output/formal_well_expert_library/train_holdout_split_manifest.csv`
6. 验收通过条件：
   - summary `status=pass`
   - `formal_constraints.uses_step3_formal_groups=true`
   - `formal_constraints.uses_old_step3_total_table=false`
   - `holdout_in_training=false`
   - 输出井数、预测点数、跳过井清单合理
   - `Density` 非空率、范围、按井统计不出现明显异常
   - Step 4 README、配置、输出摘要与当前正式口径一致
   - 自检通过后可直接进入 Step 5，不需要用户再次确认

完成后更新本文档：

- 把 `CurrentPhase` 改为 `Step 5 全量运行`
- 在 `Done` 中写明 Step 4 最终运行命令、输出目录、关键验收数值
- 在 `Next Step` 中写 Step 5 配置切换与冒烟

### Phase B. Step 5 单源井虚拟测井

前置条件：Step 4 自检验收通过。

入口文件：

- `优化阶段二/正式主线/step5_single_source_virtual_wells/build_single_source_virtual_wells.py`
- `优化阶段二/正式主线/step5_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json`

必须先修正：

- 当前 Step 5 配置仍可能指向旧输出：
  `step4_expert_real_well_prediction/output/formal_expert_prediction/all_wells_t4_t7_density_prediction.csv`
- 正式应优先指向：
  `step4_expert_real_well_prediction/output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`

执行顺序：

1. 修改 Step 5 配置中的 `real_well_prediction_csv`。
2. 小样本冒烟：

```bash
python 优化阶段二/正式主线/step5_single_source_virtual_wells/build_single_source_virtual_wells.py \
  --config 优化阶段二/正式主线/step5_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json \
  --max-source-wells 1 --max-points-per-well 20
```

3. 检查冒烟结果：
   - `virtual_well_build_audit.json` 中 `real_well_prediction_csv` 指向 `formal_well_expert_library`
   - `density_logic=single_source_attribute_continuity`
   - `distance_role=candidate_range_and_confidence_only`
   - `source_mixing=forbidden`
   - `virtual_well_training_summary.csv` 中 `SingleSourceCheck` 全部为 `pass`
4. 全量运行：

```bash
python 优化阶段二/正式主线/step5_single_source_virtual_wells/build_single_source_virtual_wells.py \
  --config 优化阶段二/正式主线/step5_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json
```

5. 全量验收：
   - 候选虚拟井数量与源井数、`5x5` 网格逻辑一致
   - 虚拟点属性非空率合理，不能重现早期 demo 体数据全空问题
   - `DensitySourceLogic` 不出现 `copy_and_distance_decay`
   - 距离字段只影响 `PointConfidence`
   - 抽查 3-5 口井，确认属性连续性和密度空间变化合理

完成后更新本文档：

- 把 `CurrentPhase` 改为 `Step 6 统一样本包`
- 在 `Done` 中写明 Step 5 全量运行命令、输出目录、候选虚拟井数量、样本数量、单源性检查结果
- 继续执行 Phase C，不等待用户确认

### Phase C. Step 6 T4-T7 统一样本包

目标：把 Step 4 真实井预测结果和 Step 5 虚拟井弱监督样本统一成后续密度体训练入口。

建议正式目录：

- `优化阶段二/正式主线/step5_unified_samples_t4_t7`

可参考旧入口，但不要直接照搬旧路径：

- `优化阶段二/当前任务/step2_target_block_and_density_samples/build_density_training_samples.py`
- `优化阶段二/当前任务/step2_t4_t7_density_package/build_t4_t7_density_sample_package.py`

实现要点：

1. 新增正式脚本，例如：
   `build_unified_t4_t7_density_samples.py`
2. 输入：
   - 真实井：`step4_expert_real_well_prediction/output/formal_well_expert_library/all_wells_t4_t7_density_prediction.csv`
   - 真实井点位：`all_wells_t4_t7_fracture_points.csv`
   - 虚拟井：`step5_single_source_virtual_wells/output/formal_single_source_virtual_wells/virtual_well_training_samples.csv`
3. 统一字段：
   - `SampleID`
   - `SourceKind`：`real_well` / `virtual_well`
   - `SourceWellName`
   - `TrackWellName`
   - `WellName`
   - `X/Y/TIME/TVD/DEPT`
   - `StrataName` 或 `LayerGroup`
   - `Density`
   - `DensityLabel`
   - `HasFracture`
   - `PointConfidence`
   - `SeisAmp/Coherence/AntTrack/CurvatureMax/CurvaturePos`
   - 对应 `Mean/Std/Min/Max/ValidCount`
4. 真实井样本 `PointConfidence=1.0`，虚拟井使用 Step 5 输出。
5. 输出：
   - `unified_t4_t7_density_samples.csv`
   - `unified_t4_t7_density_samples_summary.json`
   - `unified_t4_t7_field_contract.csv`
   - `unified_t4_t7_qc.csv`
6. 验收：
   - `DensityLabel` 非空
   - `SourceKind` 分布合理
   - `LayerGroup/StrataName` 只包含沙三段、沙四段
   - 虚拟井单源性仍可追溯
   - 真实井和虚拟井字段同构

完成后更新本文档：

- 把 `CurrentPhase` 改为 `Step 7 demo裂缝密度体`
- 在 `Done` 中写明统一样本数量、真实井/虚拟井分布、字段契约检查结果
- 继续执行 Phase D，不等待用户确认

### Phase D. Step 7 demo 区块属性驱动裂缝密度体

目标：基于统一样本包，在 `3000m x 3000m` demo 区块、`T4-T7` 空间内预测连续裂缝密度体。

建议正式目录：

- `优化阶段二/正式主线/step6_demo_density_volume`

可参考：

- `优化阶段二/当前任务/step3_candidate_a_density_volume/build_candidate_a_predicted_density_volume.py`

实现要点：

1. 不使用 `build_candidate_a_density_volume.py` 的平滑/扩散式密度体作为正式逻辑。
2. 以统一样本包训练属性驱动模型。
3. 目标区块默认候选 A：
   - `X=[568567.32, 571567.32]`
   - `Y=[4199778.50, 4202778.50]`
4. 每个目标道点按层段提取 SGY 属性统计：
   - `SeisAmp`
   - `Coherence`
   - `AntTrack`
   - `CurvatureMax`
   - `CurvaturePos`
   - 如可用，保留裂缝反演属性
5. 输出：
   - `candidate_a_t4_t7_predicted_density_volume.csv`
   - `candidate_a_target_trace_attributes.csv`
   - `candidate_a_density_volume_summary.json`
6. 验收：
   - 输出只包含沙三段、沙四段
   - 密度范围非负且统计合理
   - 不是全零、全常数或仅由距离平滑形成
   - 真实井附近与真实井预测结果有可解释对应关系

完成后更新本文档：

- 把 `CurrentPhase` 改为 `Step 8 初始DFN`
- 在 `Done` 中写明密度体输出文件、目标道点数、层段分布、密度统计
- 继续执行 Phase E，不等待用户确认

### Phase E. Step 8 初始 DFN

目标：把 demo 裂缝密度体转成初始 DFN 对象。

建议正式目录：

- `优化阶段二/正式主线/step7_initial_dfn`

实现要点：

1. 新增脚本，例如：
   `build_initial_dfn_from_density_volume.py`
2. 输入：
   - Step 7 密度体 CSV
   - 真实井裂缝点位：Step 4 `all_wells_t4_t7_fracture_points.csv`
   - 层位/层段边界
3. 初始规则：
   - `Density` 控制裂缝片出现强度或采样概率
   - 沙三段、沙四段分层处理
   - 方位/倾角优先使用 Step 4 点位细化输出中可用字段；缺失时使用局部统计或默认参数并记录
   - 输出必须可追溯到密度体单元
4. 输出：
   - 初始 DFN CSV/VTK
   - `initial_dfn_summary.json`
   - `initial_dfn_generation_audit.csv`
5. 验收：
   - 裂缝片数量与密度体统计一致
   - 坐标范围在 demo 区块内
   - 分层归属合理
   - 可视化或截面抽查不出现明显离群

完成后更新本文档：

- 把 `CurrentPhase` 改为 `Step 9 井控校正DFN`
- 在 `Done` 中写明初始 DFN 输出文件、裂缝片数量、分层统计、审计结果
- 继续执行 Phase F，不等待用户确认

### Phase F. Step 9 井控校正 DFN

目标：用真实井点位和井轨迹约束校正初始 DFN。

建议正式目录：

- `优化阶段二/正式主线/step8_dfn_well_correction`

实现要点：

1. 新增脚本，例如：
   `correct_dfn_with_well_controls.py`
2. 输入：
   - 初始 DFN
   - Step 4 `all_wells_t4_t7_fracture_points.csv`
   - Step 2 真实井轨迹/样本底板
3. 校正逻辑：
   - 井附近真实裂缝点位作为硬约束
   - 调整或补充穿井裂缝片
   - 保持远井区域由密度体控制，不用井点过度外推
4. 输出：
   - 井控校正 DFN CSV/VTK
   - `well_corrected_dfn_summary.json`
   - `well_control_correction_audit.csv`
5. 最终验收：
   - 真实井穿越处裂缝点位匹配改善
   - 不破坏 Step 7 密度体宏观分布
   - 每个校正动作可审计

完成后更新本文档：

- 把 `CurrentPhase` 改为 `全部完成`
- 把 `Status` 改为 `done`
- 在 `Done` 中写明最终 DFN 输出文件、井控匹配检查、最终验收摘要
- 停止提醒循环

## Verification

每次提醒恢复后先运行或检查：

```bash
git status --short
python3 -m py_compile scripts/codex_task_reminder.py
```

Step 4 修改完成后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step4_expert_real_well_prediction/output/formal_well_expert_library/step4_acceptance_summary.json
```

Step 5 之后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step5_single_source_virtual_wells/output/formal_single_source_virtual_wells/virtual_well_build_audit.json
```

Step 6 之后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step5_unified_samples_t4_t7/output/unified_t4_t7_density_samples_summary.json
```

Step 7 之后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step6_demo_density_volume/output/candidate_a_density_volume_summary.json
```

Step 8 之后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step7_initial_dfn/output/initial_dfn_summary.json
```

Step 9 之后检查：

```bash
python3 -m json.tool 优化阶段二/正式主线/step8_dfn_well_correction/output/well_corrected_dfn_summary.json
```

## Reminder Command

推荐用定时提醒继续当前最近的 Codex 会话：

```bash
python3 scripts/codex_task_reminder.py watch \
  --interval 10m \
  --mode codex-exec-resume \
  --include-task-text \
  --stop-when-done
```

如果当前 Codex 对话运行在 tmux 会话 `demo` 的 pane 中，先用下面命令找到 pane id：

```bash
tmux list-panes -t demo -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_current_command} #{pane_current_path}'
```

然后在另一个 tmux window/pane 中运行提醒脚本，例如目标 pane 是 `demo:0.0`：

```bash
cd /home/tyh/projects/petroleum/code/GAN-DFN

python3 scripts/codex_task_reminder.py watch \
  --interval 10m \
  --mode tmux \
  --tmux-target demo:0.0 \
  --include-task-text \
  --stop-when-done
```

先 dry-run 查看提示内容：

```bash
python3 scripts/codex_task_reminder.py once --mode print --include-task-text
```

## Resume Prompt

请继续执行本文档中的项目任务。先读取 `docs/codex_active_task.md`，确认 `Status`、`CurrentPhase`、`Current State`、`Next Step`、`Continuous Execution Policy` 和 `Detailed Implementation Plan`。当前 Step 4 仍在修改，不能把已有 `status=pass` 输出当作最终完成。请从 Step 4 收尾开始，完成一个阶段就自行运行验收检查；验收通过后更新本文档并直接进入下一阶段，不要等待用户中间确认。只有遇到缺失输入、连续失败三轮、破坏性操作、权限阻断或用户新冲突要求时才停止。全部阶段完成并验收后，把 `Status` 改为 `done`。

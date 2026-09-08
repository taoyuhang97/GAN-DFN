# Step 5A 普通曲率主导的单源虚拟井

本步骤从当前 Step4 v3 的最终合并曲线出发，在真实井周围构造单源虚拟轨迹，并生成供 Step5B 使用的弱监督样本。正式版本为 `formal_curvature_led_v2`，旧 `formal_single_source_virtual_wells` 仅是历史结果，不能继续作为下游输入。

## 为什么需要重构

旧实现存在以下问题：

- 配置仍读取已经失效的 54 井旧 Step4 结果，而当前正式 Step4 v3 是 38 井、按 `WellName + DEPT` 唯一的合并结果。
- 每口源井生成 25 条虚拟轨迹，旧正式结果达到约 940 万行和 27 GB；真正的问题不是25条轨迹本身，而是虚拟样本权重过大以及同一批宽字段被重复写入多个文件。
- 旧相似度门槛几乎不淘汰样点，并把不可信转移直接写成 `Density=0`，制造了大量假阴性。
- 旧代码由 `Density > 0` 反推 `HasFracture`。当前 Step4 的连续期望密度几乎处处为正，这会错误地把几乎全部虚拟样点标为裂缝。
- 虚拟点移动了 `X/Y`，但沿用源井 `TIME`；当地层有倾斜时，虚拟点会落到错误层内位置。
- 最大正曲率参与旧正式合同，但当前任务要求普通对齐曲率体作为小尺度裂缝的主要地震依据。

## 当前逻辑

1. 只通过 Step4 v3 manifest 读取 `all_wells_t4_t7_merged_density_prediction.csv`，要求 `WellName + DEPT` 唯一且 `PredictionValid=1`。
2. 每口井建立完整的 `5 x 5` 地震道邻域，共25条虚拟轨迹。中心格点是离测井位置最近的地震道；由于测井轨迹通常不与实际地震道重合，它仍然属于虚拟测井，不能删除。
3. 真实井 `X/Y/TIME` 完全不改。虚拟轨迹移动 `X/Y` 后，先计算源点在沙三 `T4-T6` 或沙四 `T6-T7` 内的相对位置，再用虚拟位置的局部层面重算 `VirtualTIME`。
4. 内部计算 `SourceTIME`、`VirtualTIME`、`TimeShiftMs` 和层内相对位置用于门控和审计；这些中间量不再逐行写入正式训练表。层面无效、相对位置越界或时间移动超过配置上限的样点不生成标签。
5. 在虚拟 `X/Y/VirtualTIME` 上重新采样地震振幅、相干、蚂蚁体和普通 `CurvatureMax`。普通曲率必须有效，并以 0.65 权重作为主要相似性证据；不存在统一的原始曲率高值阈值。
6. `CurvaturePos` 不进入本步骤正式输入、输出或相似性公式。

## 双标签口径

- `PresenceLabel`：继承 Step4 `HasFracture`，表示是否存在裂缝。
- `DensityLabel`：只在 `PresenceLabel=1` 时保存条件裂缝密度，值为源井密度乘以转移连续性。
- `PresenceLabel=0` 时 `DensityLabel` 保持空值，不把“无裂缝”伪装成一个连续密度观测。
- 相似性、曲率、层面或时间移动不合格的样点标记为 `unlabelled_low_confidence`，只进入可选审计中间表，不进入正式训练表。
- 正式训练表本身只包含通过门控的样点，因此不再重复保存逐行 `LabelStatus`。

虚拟样本权重为 `PointConfidence / 25`。因此同一个真实样点的全部25个虚拟子样点总权重最多等于一个真实样点，避免虚拟扩增主导后续训练。

正式 `virtual_well_training_samples.csv` 只保留15列：源样点、源井、虚拟轨迹、`X/Y/TIME`、地层、双标签、置信度、样本权重和4种地震属性。DEPT/TVD、时间映射明细、专家/曲线对来源、连续性中间量和3×3统计量只用于内部计算或JSON审计，不再复制到数百万行训练表。

## 输出

默认只发布必要文件：

- `virtual_well_index.csv`
- `virtual_well_training_samples.csv`
- `virtual_well_training_summary.csv`
- `virtual_well_build_audit.json`

只有显式设置 `write_intermediate_tables=true` 时，才额外输出属性、3x3 上下文、弱标签和置信度明细表。

## 车页1导眼虚拟测井说明图

`build_cheye1_virtual_well_presentation.py` 只用于说明 Step5A 的虚拟测井构造，复用井周 `200 m` 的 Step9 剖面几何和 T4-T7 层位框架，**不读取或叠加 DFN、断层或 Step8 成果**。

图件写入正式 Step5A 输出目录下的 `virtual_well_presentation/`：

当前配置生成的 PPT 版位于其下的 `ppt_200m/`，严格使用车页1导眼轨迹外扩 ±200m 窗口；XZ/YZ 横向道数上限为 82/98 道，但实际使用窗口内存在的有效地震道（当前为 XZ=34、YZ=50 道），不对数据进行插值扩道。PNG 为 300 dpi、15×10 cm（4500×3000 像素）。

- `01_plan_virtual_well_grid.png`：真实车页1导眼与其 `5 x 5`、共25口虚拟测井地震道位置；橙色菱形为中心虚拟井。
- `02-03`：XZ/YZ 地震振幅背景上的真实井证据。黄色为真实井轨迹；青色短线是具有倾向倾角的成像测井真实裂缝投影；洋红圆点是仅有位置的常规测井预测裂缝，且自动排除成像测井覆盖段。
- `04-05`：XZ/YZ 曲率体背景上的 Step5 虚拟测井构造。紫色菱形为虚拟测井裂缝事件中心，仅表示位置，不代表倾向倾角；中心虚拟井使用橙色强调。
- `06-07`：三类证据的综合核对图。

Step4/Step5 的连续正样本会按时间连续性先合并为裂缝事件，每个事件只绘制一个中心点，避免把高采样率标签误画成具有姿态的裂缝线。成像裂缝才保留由 `Frac_Azimuth/Frac_Dip` 计算的方向投影。

```bash
python 优化阶段二/正式主线/step5a_single_source_virtual_wells/build_cheye1_virtual_well_presentation.py \
  --config 优化阶段二/正式主线/step5a_single_source_virtual_wells/configs/formal_cheye1_virtual_well_presentation.json
```

## 运行

```bash
python 优化阶段二/正式主线/step5a_single_source_virtual_wells/build_single_source_virtual_wells.py \
  --config 优化阶段二/正式主线/step5a_single_source_virtual_wells/configs/formal_single_source_virtual_wells.json
```

隔离冒烟测试可同时使用 `--output-dir`、`--max-source-wells` 和 `--max-points-per-well`，不会改动正式或历史结果。

## 后续 Step6 口径

Step5仍提供连续密度监督，最终目标仍是连续三维裂缝密度体，不是只输出 0/1 离散体。后续 Step6 应采用两阶段模型：先学习 `PresenceLabel`，再只在正样本上学习 `DensityLabel`，普通曲率作为必需的主要地震特征；验证必须按源井分组，真实井及其所有虚拟子井进入同一折。

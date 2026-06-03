# 本项目 DFN 效果评估实验清单（共识指标版，2026-05-11）

## 1. 简化原则

本版只保留石油地质、地球物理和 DFN 建模文献中最常见、解释成本最低、共识度最高的指标。目标不是把所有可能的指标都算一遍，而是形成一条甲方容易接受的证据链：

`井上能对上 -> 裂缝发育强弱合理 -> 裂缝面积强度合理 -> 方向和地质控制关系合理`

第一轮只保留 4 类核心指标：

| 核心指标 | 汇报含义 | 为什么保留 |
|---|---|---|
| 井上符合性 | 预测裂缝在井上能不能和成像测井/标定裂缝对上 | 井资料是最硬的验证资料，文献中最常见 |
| 裂缝数量与密度 | 哪些层、哪些单元裂缝更发育 | 条数和密度是地质人员最容易理解的裂缝强弱指标 |
| P32 / 面积强度 | 单位体积里有多少裂缝面积 | DFN 领域高频指标，比单纯条数更能代表裂缝规模 |
| 产状与地质控制一致性 | 倾向、倾角、层位差异、断层附近增强是否符合认识 | 玫瑰图、倾角分布、层位/断层控制是地球物理解释常用证据 |

第一轮不作为主指标的内容：

| 暂不作为主指标 | 原因 |
|---|---|
| KL/JS/Wasserstein 等复杂分布距离 | 解释成本高，汇报时不如均值、中位数、玫瑰图和柱状图直观 |
| 全区精确拓扑连通性 | 计算量大，且需要生产动态或流动模拟结果支撑解释 |
| 等效孔隙度/等效渗透率 | 需要裂缝开度、渗透率或生产资料标定，否则假设过多 |
| AUC/F1 等机器学习指标 | 可用于测井预测模型说明，但不能直接证明 DFN 地质合理 |
| DFN/体素互转保真 | 适合技术验证，不适合作为第一轮甲方效果评估主线 |
| 大/中/小尺度比例 | 可作为展示辅助，不再作为第一轮主评价指标 |

## 2. 基准路径

共享原始数据只读根目录：

`SHARED_ROOT=/data/shared/project-oil/wx数据/砂砾岩`

当前第三部分主运行目录：

`RUN_ROOT=/home/tyh/data/project-oil/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/后台执行/full113_conf_up_traininfer_20260506_142956`

当前训练/验证/测试划分目录：

`SPLIT_ROOT=/home/tyh/data/project-oil/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/验证_新划分/full113_conf_up_fixed_split_79_17_17_20260506`

第二部分单元 DFN 样本目录：

`SOURCE_UNIT_DFN_ROOT=/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分`

建议评估输出目录：

`EVAL_ROOT=/home/tyh/data/project-oil/砂砾岩/优化阶段一/研究内容三/DFN效果评估/full113_conf_up_20260511`

## 3. 第一轮只做 3 个验证实验

`E00` 只做结果清点，不作为效果指标。正式效果验证只做 `E01-E03` 三个实验。

### E00 结果版本与数量流转清点

目的：

先确认当前训练、推理、合并、后处理和断层融合结果来自同一轮流程，避免后续把不同版本结果混在一起。

读取数据：

| 类型 | 路径 |
|---|---|
| 训练汇总 | `RUN_ROOT/train_layerwise/layerwise_baseline_layer_training_summary.json` |
| 测试评估汇总 | `RUN_ROOT/eval/eval_test/aggregated/evaluation_summary.json` |
| 全区推理汇总 | `RUN_ROOT/production/full_infer/aggregated/production_inference_summary_shard_*.json` |
| 单元合并汇总 | `RUN_ROOT/merge/merge_full/merge_summary.json` |
| 区域后处理汇总 | `RUN_ROOT/postprocess/postprocess_full/regional_postprocess_summary.json` |
| 断层融合汇总 | `RUN_ROOT/fault_postfusion/fault_postfusion_full/regional_fault_postfusion_pipeline_summary.json` |

输出：

| 文件 | 内容 |
|---|---|
| `00_inventory/pipeline_counts.csv` | 各阶段单元数、裂缝片数、断层面片数 |
| `00_inventory/artifact_paths.csv` | 本轮评估实际读取的结果文件 |
| `00_inventory/pipeline_count_flow.png` | 数量流转图 |

### E01 井上符合性验证

目的：

验证测井裂缝预测结果是否能和成像测井/真实标定裂缝对上。这个实验回答“井上是否可信”。

读取数据：

| 类型 | 路径 |
|---|---|
| 方案对比总表 | `SHARED_ROOT/优化阶段一/方案对比/研究内容一/fracture_compare_summary.csv` |
| 车页1导眼真实裂缝 | `SHARED_ROOT/优化阶段一/方案对比/研究内容一/details/raw_vs_actual_cheye1_actual_standardized.csv` |
| 车页1导眼预测裂缝 | `SHARED_ROOT/优化阶段一/方案对比/研究内容一/details/raw_vs_actual_cheye1_predicted_standardized.csv` |
| 车151HF真实裂缝 | `SHARED_ROOT/优化阶段一/方案对比/研究内容一/details/optimized_vs_actual_che151hf_actual_standardized.csv` |
| 车151HF预测裂缝 | `SHARED_ROOT/优化阶段一/方案对比/研究内容一/details/optimized_vs_actual_che151hf_predicted_standardized.csv` |
| 成像测井裂缝 | `SHARED_ROOT/优化阶段一/研究内容一/成像测井/裂缝提取/*.csv` |

只保留指标：

| 指标 | 说明 |
|---|---|
| 裂缝数量误差 | 预测条数与真实条数差多少 |
| 深度命中率 | 真实裂缝附近是否有预测裂缝 |
| 漏判率 | 真实裂缝中没有预测到的比例 |
| 倾向/倾角差异 | 倾向按 0-360 度周期角度计算，倾角直接比较 |

输出：

| 文件 | 内容 |
|---|---|
| `01_well_validation/well_count_depth_metrics.csv` | 各井数量误差、深度命中率、漏判率 |
| `01_well_validation/well_orientation_metrics.csv` | 倾向、倾角差异 |
| `01_well_validation/depth_track_true_vs_pred.png` | 沿井深真实/预测裂缝对比 |
| `01_well_validation/orientation_rose_true_vs_pred.png` | 真实/预测倾向玫瑰图 |

汇报说法：

“先看井上能不能对上。真实裂缝在哪些深度出现，预测结果是否也在附近出现；真实和预测的裂缝方向是否大体一致。”

### E02 单元 DFN 与样本 DFN 对比

目的：

在测试单元上验证预测 DFN 是否接近已有样本 DFN。这个实验回答“模型生成的单元 DFN 是否像训练体系里的真实样本”。

读取数据：

| 类型 | 路径 |
|---|---|
| 单元评估总表 | `RUN_ROOT/eval/eval_test/aggregated/unit_evaluation.csv` |
| 层位评估总表 | `RUN_ROOT/eval/eval_test/aggregated/unit_layer_evaluation.csv` |
| 层位密度评估 | `RUN_ROOT/eval/eval_test/aggregated/layer_density_evaluation.csv` |
| 测试单元真实 DFN | `RUN_ROOT/eval/eval_test/units/<UnitID>/ground_truth_unit_patches.csv` |
| 测试单元预测 DFN | `RUN_ROOT/eval/eval_test/units/<UnitID>/predicted_unit_patches.csv` |

只保留指标：

| 指标 | 说明 |
|---|---|
| 裂缝数量误差 | 真实单元和预测单元裂缝条数差异 |
| 分层裂缝密度误差 | 按层位统计 `条/100ms`，比单纯条数更公平 |
| P32 误差 | 裂缝面积总和 / 单元或层段体积 |
| 产状一致性 | 倾向玫瑰图、倾角直方图是否接近 |

输出：

| 文件 | 内容 |
|---|---|
| `02_unit_validation/unit_count_density_metrics.csv` | 测试单元数量和密度误差 |
| `02_unit_validation/unit_layer_density_metrics.csv` | 测试单元分层密度误差 |
| `02_unit_validation/unit_p32_metrics.csv` | 单元和层位 P32 对比 |
| `02_unit_validation/unit_orientation_metrics.csv` | 倾向、倾角统计 |
| `02_unit_validation/count_density_true_vs_pred.png` | 数量/密度真实预测对比图 |
| `02_unit_validation/p32_true_vs_pred.png` | P32 真实预测对比图 |
| `02_unit_validation/orientation_rose_true_vs_pred.png` | 真实/预测方向玫瑰图 |

汇报说法：

“在没有参与训练的测试单元上，重点看裂缝数量、分层密度、P32 和方向是否接近样本 DFN。”

### E03 全矿区地质一致性验证

目的：

验证最终矿区 DFN 是否符合基本地质认识：不同层位裂缝发育强度不同，断层附近裂缝更发育并向外过渡，整体方向组系有规律。这个实验回答“全区结果是否像一个受层位和断层控制的地质体，而不是随机裂缝云”。

读取数据：

| 类型 | 路径 |
|---|---|
| 无断层区域 DFN | `RUN_ROOT/postprocess/postprocess_full/regional_dfn_postprocessed.vtk` |
| 后处理裂缝表/汇总 | `RUN_ROOT/postprocess/postprocess_full/*summary*.json` |
| 全区层位密度明细 | `RUN_ROOT/analysis/full_infer_layer_density/full_infer_layer_density_control_raw.csv` |
| 全区层位密度汇总 | `RUN_ROOT/analysis/full_infer_layer_density/full_infer_layer_density_summary.csv` |
| 含断层最终 DFN | `RUN_ROOT/fault_postfusion/fault_postfusion_full/03_fault_fused/fault_postfusion/regional_dfn_fault_embedded_raw.vtk` |
| 断层融合裂缝表 | `RUN_ROOT/fault_postfusion/fault_postfusion_full/03_fault_fused/fault_postfusion/regional_dfn_fault_embedded_fractures.csv` |
| 断层融合汇总 | `RUN_ROOT/fault_postfusion/fault_postfusion_full/regional_fault_postfusion_pipeline_summary.json` |

只保留指标：

| 指标 | 说明 |
|---|---|
| 分层裂缝密度 | 各层位 `条/100ms`，体现层位差异 |
| 分层 P32 | 各层位裂缝面积强度，体现裂缝规模差异 |
| 分层产状 | 各层位倾向玫瑰图、倾角分布 |
| 断层距离带密度 | 0-50ms、50-100ms、100-200ms、>200ms 的裂缝密度变化 |

输出：

| 文件 | 内容 |
|---|---|
| `03_regional_validation/layer_density_p32_summary.csv` | 全区分层密度和 P32 |
| `03_regional_validation/orientation_by_layer.csv` | 分层倾向、倾角统计 |
| `03_regional_validation/fault_distance_band_density.csv` | 距断层不同距离带的裂缝密度 |
| `03_regional_validation/layer_density_p32_bar.png` | 分层密度/P32 柱状图 |
| `03_regional_validation/orientation_rose_by_layer.png` | 分层方向玫瑰图 |
| `03_regional_validation/fault_distance_density_curve.png` | 距断层距离与裂缝密度关系图 |
| `03_regional_validation/vtk/regional_no_fault_for_display.vtk` | 无断层区域 DFN 展示版 |
| `03_regional_validation/vtk/regional_fault_embedded_for_display.vtk` | 含断层最终 DFN 展示版 |
| `03_regional_validation/vtk/fault_influence_only.vtk` | 只包含断层面和断层诱导裂缝 |

汇报说法：

“全矿区只重点讲三件事：层位之间裂缝发育强弱不同，断层附近裂缝更密集并向外过渡，裂缝方向有稳定的优势组系。”

## 4. 汇报交付物

第一轮评估只交付以下内容：

| 交付物 | 说明 |
|---|---|
| `DFN_effect_evaluation_report.md` | 中文评估小结 |
| `DFN_effect_evaluation_core_tables.xlsx` | 核心表格合集 |
| `all_core_metrics_summary.json` | 机器可读核心指标汇总 |
| `figures_for_presentation/` | 汇报图 |
| `vtk_for_demonstration/` | 展示 VTK |

建议 PPT 只放 4 张图：

1. 井上真实裂缝与预测裂缝沿井深对比图。
2. 测试单元真实/预测裂缝密度和 P32 对比图。
3. 测试单元或全区分层倾向玫瑰图。
4. 全矿区分层密度与断层距离密度曲线图。

## 5. 后续可选增强

只有在第一轮 4 类指标能够支撑当前汇报后，再考虑补充以下内容：

| 可选增强 | 适用条件 |
|---|---|
| 大/中/小尺度比例 | 需要专门解释多尺度裂缝展示效果时再用 |
| 地震属性一致性 | 确认存在相干、方差、曲率、蚂蚁体等独立属性体后再做 |
| 拓扑连通性 | 需要讨论裂缝网络连通性或渗流通道时再做 |
| 等效渗透率/等效孔隙度 | 有开度、渗透率、试井或生产资料标定后再做 |
| DFN/体素互转保真 | 技术答辩或模型表达方式说明时再做 |

## 6. 执行顺序

建议按以下顺序执行：

1. `E00`：清点本轮结果版本和数量流转。
2. `E01`：做井上符合性验证。
3. `E02`：做测试单元 DFN 与样本 DFN 对比。
4. `E03`：做全矿区层位、P32、产状和断层控制一致性验证。

如果这 4 步结果能讲通，就先形成汇报版本，不再追加复杂指标。

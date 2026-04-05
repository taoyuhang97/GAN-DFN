# 优化阶段一代码整理（2026-04-02）

本文档基于当前仓库代码现状整理“优化阶段一”相关内容，重点回答四件事：

1. 当前阶段一相关代码分布在哪里
2. 主流程已经收敛成什么样
3. 各阶段主要输入输出文件是什么
4. 还存在哪些未完全收口的地方

本文档偏向“当前代码导航”和“接手顺序”，不替代已有实验结果文档。

## 1. 结论先行

当前“优化阶段一”已经不再只是单一的裂缝存在性实验，而是沿着下面这条链路逐步收口：

`数据准备 -> 分地层两阶段裂缝预测 -> 专家库验证/部署 -> 虚拟测井与单元 DFN -> DFN 标准化表达与监督基线`

从代码结构看，相关内容已经分成 5 个层次：

- `数据精简/优化阶段一`
  - 负责样本重建、裂缝标签补充、成像井裂缝信息提取
- `模型训练/优化阶段一`
  - 负责第一部分主流程，已经是当前最稳定的阶段一入口
- `小范围DFN生成/优化阶段一`
  - 负责把井尺度结果转到单元尺度，形成虚拟测井、单元归属、单元 DFN 预览
- `研究内容三/优化阶段一`
  - 负责 DFN 表达互转、训练样本打包、监督基线
- `docs/`
  - 已有阶段进展、交接和第二部分设计说明

如果现在要继续推进阶段一，建议优先从 `模型训练/优化阶段一/基于地层约束的测井裂缝预测` 读起，再读 `小范围DFN生成/优化阶段一`，最后读 `研究内容三/优化阶段一`。

## 2. 目录分层

### 2.1 数据准备层

目录：

- `数据精简/优化阶段一`

主要脚本：

- `new_data_around_well.py`
  - 井震对齐、插值构样、常规测井/成像测井样本构建
- `new_data_around_well_other.py`
  - 其他成像井样本构建
- `new_fracture_exist_sample.py`
  - 将裂缝方位和密度字段并入样本
- `new_fracture_exist_sample_other.py`
  - 其他井的裂缝标签构建
- `车151HF裂缝信息提取.py`
- `车页1导眼裂缝信息提取.py`
- `che660-1_GR_add.py`

这一层仍然保留较强的“原始实验脚本”特征，主要价值是给第一部分提供样本和标签来源。

### 2.2 第一部分主流程层

目录：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测`

这是当前阶段一最核心的目录，已经形成“分地层 + 两阶段 + 专家模型库 + 部署/批量汇总”的统一实现。

按职责可以分成 4 组。

第一组，训练与验证包装脚本：

- `run_manual_strata_lstm_experiment.py`
  - 手工分层后的第一阶段训练/验证包装
- `run_manual_strata_raw_point_refine.py`
  - 手工分层后的第二阶段训练/验证包装
- `run_outer_holdout_expert_validation.py`
  - 外层留一，先建专家库，再做整井串联验证

第二组，部署与批处理入口：

- `run_conventional_log_strata_validation.py`
  - 单井完整部署验证入口
- `run_conventional_log_strata_segmentation_only.py`
  - 只做层位归属/切分验证
- `run_batch_existing_conventional_log_prediction.py`
  - 对现有常规测井批量部署
- `run_aggregate_existing_conventional_log_prediction.py`
  - 对批量结果做总表汇总
- `run_deploy_strata_expert_prediction.py`
  - 旧版单脚本式部署入口，功能已被模块化版本吸收

第三组，底层实验入口：

- `run_lstm_experiment.py`
- `run_point_refine_experiment.py`
- `run_density_experiment.py`
- `run_strata_lstm_experiment.py`

第四组，共享模块：

- `workflow_paths.py`
- `manual_strata_workflow/common.py`
- `manual_strata_workflow/dataset.py`
- `strata_expert_deploy/runtime.py`
- `strata_expert_deploy/strata_resolution.py`
- `strata_expert_deploy/stage_prediction.py`
- `strata_expert_deploy/pipeline.py`

### 2.3 第二部分衔接层

目录：

- `小范围DFN生成/优化阶段一`

这部分负责把第一部分的井尺度结果转成单元尺度约束和单元 DFN。

主要脚本：

- `stage2_virtual_well_dfn.py`
  - 统一单元网格、生成单元中心虚拟测井、直接输出参数化裂缝片
- `虚拟测井/generate_unit_virtual_well.py`
  - 面向单元的虚拟测井生成
- `虚拟测井/build_wells.py`
  - 更早期的虚拟井实验脚本
- `虚拟裂缝预测/run_batch_virtual_well_fracture_prediction.py`
  - 对虚拟测井批量跑第一部分流程
- `单元裂缝归属/build_unit_fracture_assignment.py`
  - 把裂缝点/段/层位拆分到单元包
- `单元裂缝归属/merge_virtual_fractures_into_unit_packages.py`
  - 合并真实井和虚拟井的单元裂缝结果
- `单元DFN构建/build_unit_dfn_preview.py`
  - 从单元裂缝种子构建可视化预览和面片
- `单元DFN构建/run_batch_unit_dfn_preview.py`
  - 批量跑单元 DFN 预览

### 2.4 第三部分准备层

目录：

- `研究内容三/优化阶段一`

这部分已经不是第一部分预测本身，而是阶段一成果向标准化表达、训练样本和基线模型的延伸。

主要子目录：

- `DFN体素互转实验`
  - `dfn_patches_to_voxel.py`
  - `voxel_to_dfn_patches.py`
  - `run_dfn_voxel_roundtrip_experiment.py`
- `DFN实例表达互转实验`
  - `dfn_patches_to_instance_label.py`
  - `instance_label_to_dfn_patches.py`
  - `run_dfn_instance_roundtrip_experiment.py`
- `GAN训练准备/训练样本打包`
  - `build_sparse_instance_gan_dataset.py`
  - `build_instance_gan_dataset.py`
- `GAN训练准备/窗口级实例统计`
  - `analyze_instance_window_distribution.py`
- `G_DFN监督基线`
  - `build_unit_level_split.py`
  - `train_supervised_baseline.py`
  - `infer_and_evaluate_baseline.py`
  - `infer_production_units_baseline.py`

## 3. 当前主流程

### 3.1 样本与手工分层

当前第一部分已经把 6 口关键井的手工分层写死在共享模块里：

- `车151HF`
- `车660-1`
- `车660-2`
- `车662`
- `车663`
- `车页1导眼`

对应代码位置：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/common.py`

分层与数据导出由下面两个共享函数负责：

- `build_labeled_dataset()`
- `export_filtered_strata_dataset()`

对应代码位置：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/dataset.py`

这意味着当前第一部分的训练、验证、外层留一和部署，全都建立在统一的手工分层口径上。

### 3.2 第一阶段：裂缝发育段/存在性预测

第一阶段底层模型脚本仍然是：

- `模型训练/优化阶段一/裂缝存在性分析/LSTM/imaging_well_to_fracture_cnn_lstm_test_1.py`

但当前稳定入口已经上收为：

- `run_manual_strata_lstm_experiment.py`

当前包装逻辑是：

1. 读取样本
2. 按手工层位切分
3. 统计每口井在每个地层中的有效序列数
4. 按地层筛选有效井
5. 支持全其他井训练或按距离选训练井
6. 调用 `run_lstm_experiment.py` 训练/验证
7. 输出 `manual_strata_lstm_summary.csv` 和 `summary.json`

### 3.3 第二阶段：裂缝点位细化

第二阶段底层主脚本是：

- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/raw_point_guided_segment_refine.py`

当前稳定入口是：

- `run_manual_strata_raw_point_refine.py`

当前包装逻辑是：

1. 读取和第一阶段一致的手工分层样本
2. 从第一阶段目录读取 `verify_*` 结果
3. 按地层建立第二阶段输入数据
4. 按 `config_profile` 或 `config_profile_map_json` 选择不同地层的 refine profile
5. 生成 `manual_strata_refine_summary.csv` 和 `summary.json`

第二阶段已经不只是输出点位，还开始输出密度强度、尺度和方向相关字段，作为后续 DFN 的中间约束。

### 3.4 专家模型库与外层留一

当前阶段一真正的流程验证入口是：

- `run_outer_holdout_expert_validation.py`

这份脚本做了 4 件关键事情：

1. 用训练井重新训练第一阶段库和第二阶段库
2. 建立跨地层的专家注册表
3. 根据目标井与专家井的地震/测井相似度选模
4. 对留出的整井做“第一阶段 + 第二阶段”整链验证

从代码结构上看，这份脚本已经承担了“阶段一系统集成验证”的角色。

### 3.5 单井部署、批量部署和汇总

当前阶段一对外更实用的入口是：

- `run_conventional_log_strata_validation.py`
- `run_batch_existing_conventional_log_prediction.py`
- `run_aggregate_existing_conventional_log_prediction.py`

其中：

- 单井脚本负责给单口常规测井跑完整流程
- 批量脚本负责扫描目录并批量预测
- 汇总脚本负责把所有井的最终结果并成总表

### 3.6 第二部分衔接

当前第二部分最清晰的一条主干是：

1. 消费第一部分部署目录
2. 读取 `whole_well_pred_fracture_points.csv`、`whole_well_pred_segment_summary.csv` 和层位范围
3. 把真实井点位映射到单元
4. 在相邻单元插值虚拟测井样本
5. 把真实点和虚拟点统一为 `dfn_seed_points.csv`
6. 用方向和密度字段生成参数化裂缝片

对应入口脚本：

- `小范围DFN生成/优化阶段一/stage2_virtual_well_dfn.py`

### 3.7 第三部分准备

当前第三部分已经不是空设计，而是有明确代码落地：

1. 先把 `unit_dfn_patches.csv` 转成体素或实例标签
2. 再把这些标准化表达转回裂缝片，做 round-trip 检查
3. 然后将窗口级数据打包成 `.npz` 训练样本
4. 最后训练一个监督 baseline 验证“地震窗口 -> 裂缝实例/裂缝片”是否可学

## 4. 当前关键输入输出契约

### 4.1 第一部分单井最终输出

`run_conventional_log_strata_validation.py` 已固定输出 5 个最终文件：

- `final_log_with_fractures.csv`
- `final_fracture_points.csv`
- `final_fracture_segments.csv`
- `final_strata_segmentation.csv`
- `prediction_meta.json`

这 5 个文件已经是当前第一部分最稳定的单井结果口径。

### 4.2 第一部分批量汇总输出

`run_aggregate_existing_conventional_log_prediction.py` 已固定汇总输出：

- `all_final_fracture_points.csv`
- `all_final_fracture_segments.csv`
- `all_final_strata_segmentation.csv`
- `all_predicted_well_summary.csv`
- `aggregation_meta.json`

如果第二部分要消费第一部分结果，优先用这组总表。

### 4.3 第二部分主干输出

`stage2_virtual_well_dfn.py` 当前固定输出：

- `unit_index.csv`
- `mapped_control_points.csv`
- `virtual_well_samples.csv`
- `dfn_seed_points.csv`
- `unit_constraint_summary.csv`
- `unit_dfn_patches.csv`
- `unit_dfn_summary.csv`
- `run_summary.json`

这组文件已经构成“井尺度结果 -> 单元裂缝种子 -> 参数化裂缝片”的最小闭环。

### 4.4 第三部分训练样本与实验输出

当前研究内容三的关键契约有三类。

第一类，DFN 标准化表达实验：

- `orig_voxel_volume.npz`
- `roundtrip_summary.json`
- `instance_label_volume.npz`

第二类，GAN/监督训练样本：

- `aggregated/sample_manifest.csv`
- `aggregated/unit_manifest.csv`
- `aggregated/dataset_summary.json`

第三类，监督基线训练结果：

- `checkpoints/best_model.pt`

## 5. 当前已完成的改进点

结合代码和已有文档，当前阶段一已经完成的改进主要有：

- 第一部分已经从分散脚本收口到统一目录 `模型训练/优化阶段一/基于地层约束的测井裂缝预测`
- 手工分层、井名归一化、数据导出已沉到共享模块，而不是散在各脚本里重复实现
- 第一阶段和第二阶段都已经有统一包装入口，而不只是底层实验脚本
- 外层留一验证已经升级为“训练专家库 + 相似度选模 + 两阶段串联”的整体流程
- 部署逻辑已经从单一长脚本拆为 `runtime / strata_resolution / stage_prediction / pipeline`
- 常规测井单井预测、批量预测、汇总三类入口已经分开
- 第二部分已经有统一脚本把第一部分输出转成虚拟测井和单元 DFN 裂缝片
- 第三部分已经补上了体素互转、实例互转、稀疏训练样本打包和监督 baseline

## 6. 当前还没完全收口的地方

从当前代码看，仍然存在 6 个明显的未收口点。

### 6.1 默认路径仍然大量硬编码

虽然 `workflow_paths.py` 已经统一了项目根目录和底层脚本路径，但很多入口脚本默认数据目录、结果目录、实验记录文档仍直接写死在 `E:` 或 `D:` 盘。

这意味着：

- 仓库内部相对路径已经在收口
- 但数据和结果路径仍然强依赖本机环境

### 6.2 数据准备层还没并到统一主流程

`数据精简/优化阶段一` 仍然是若干专用脚本集合，尚未像第一部分那样形成统一入口和统一输出契约。

### 6.3 第一部分存在“新模块”和“旧入口”并存

当前已经有：

- `strata_expert_deploy/pipeline.py`

同时还保留：

- `run_deploy_strata_expert_prediction.py`
- `strata_expert_deploy/_legacy_run_deploy_strata_expert_prediction.py`

说明部署逻辑已经模块化，但旧入口还没有完全退场。

### 6.4 第一部分验证/部署脚本仍偏大

尤其是：

- `run_conventional_log_strata_validation.py`
- `run_outer_holdout_expert_validation.py`

这两个文件已经承担了较多职责，后续如果继续扩展，最好进一步拆成“数据准备、选模、评价、结果落盘”几个更稳定的模块。

### 6.5 第二部分存在新旧两套路子

当前第二部分同时存在：

- `stage2_virtual_well_dfn.py` 这条新主干
- `虚拟测井/build_wells.py` 这类更早的实验脚本

后续需要明确哪些保留为历史实验，哪些继续作为主线演进。

### 6.6 第三部分已落地，但还没有统一总入口

研究内容三已经具备多个子实验的 README 和脚本，但还没有一份总的“阶段一从单元 DFN 到基线训练”的导航文档。

## 7. 建议阅读顺序

如果后续要继续接手阶段一，建议按这个顺序阅读和定位：

1. `docs/session_handoff_2026-03-26.md`
2. `docs/stage_progress_2026-03-25.md`
3. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/common.py`
4. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/dataset.py`
5. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_manual_strata_lstm_experiment.py`
6. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_manual_strata_raw_point_refine.py`
7. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_outer_holdout_expert_validation.py`
8. `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/pipeline.py`
9. `小范围DFN生成/优化阶段一/stage2_virtual_well_dfn.py`
10. `研究内容三/优化阶段一` 下各 README

## 8. 当前整理建议

如果继续做“阶段一代码整理”，优先级建议如下：

1. 先补一份阶段一总索引，把 `数据准备 -> 第一部分 -> 第二部分 -> 研究内容三` 连起来
2. 再统一第一部分新旧入口，只保留一个主部署入口
3. 再把默认路径逐步收进配置文件或环境变量
4. 最后才考虑继续搬目录或重命名历史脚本

## 9. 对应参考文档

当前与本整理互补的文档有：

- `docs/stage_progress_2026-03-22.md`
- `docs/stage_progress_2026-03-25.md`
- `docs/session_handoff_2026-03-26.md`
- `docs/stage2_virtual_well_dfn_design_2026-03-26.md`

其中：

- `stage_progress_2026-03-25.md` 更偏阶段结论
- `session_handoff_2026-03-26.md` 更偏当前第一部分交接
- `stage2_virtual_well_dfn_design_2026-03-26.md` 更偏第二部分设计
- 本文档更偏“当前代码地图”

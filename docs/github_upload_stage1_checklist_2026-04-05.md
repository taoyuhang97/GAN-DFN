# GitHub上传清单（第一轮优化后，2026-04-05）

## 1. 文档目的

本文档用于在上传 GitHub 前，对当前仓库中的文件做三类整理：

- 建议上传：当前第一轮优化后主链路确实在使用的代码和文档
- 建议保留但标记历史：仍有参考价值，但已不是当前主入口
- 建议不上传：本地临时结果、缓存、中间产物、与主链路无关的杂项

这份清单面向“先把第一轮优化后主链路整理清楚再上传”的目标。

## 2. 当前建议上传的核心目录

建议优先上传以下目录：

- `docs/`
- `数据精简/优化阶段一/`
- `模型训练/优化阶段一/`
- `小范围DFN生成/优化阶段一/`
- `研究内容三/优化阶段一/`

建议保留仓库根目录中的以下基础文件：

- `README.md`
- `requirements.txt`
- `.gitignore`

## 3. 建议上传：第一部分

### 3.1 第一部分主入口

建议上传：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_manual_strata_lstm_experiment.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_manual_strata_raw_point_refine.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_outer_holdout_expert_validation.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_conventional_log_strata_validation.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_batch_existing_conventional_log_prediction.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_aggregate_existing_conventional_log_prediction.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_conventional_log_strata_segmentation_only.py`

### 3.2 第一部分共享模块

建议上传：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/workflow_paths.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/common.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/manual_strata_workflow/dataset.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/runtime.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/strata_resolution.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/stage_prediction.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/pipeline.py`

### 3.3 第一部分底层核心脚本

建议上传：

- `模型训练/优化阶段一/裂缝存在性分析/LSTM/imaging_well_to_fracture_cnn_lstm_test_1.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/raw_point_guided_segment_refine.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/nearest_imaging_well_predict_then_refine.py`

### 3.4 第一部分前置数据准备

建议上传：

- `数据精简/优化阶段一/new_data_around_well.py`
- `数据精简/优化阶段一/new_data_around_well_other.py`
- `数据精简/优化阶段一/new_fracture_exist_sample.py`
- `数据精简/优化阶段一/new_fracture_exist_sample_other.py`
- `数据精简/优化阶段一/车151HF裂缝信息提取.py`
- `数据精简/优化阶段一/车页1导眼裂缝信息提取.py`
- `数据精简/优化阶段一/che660-1_GR_add.py`

## 4. 建议上传：第二部分

### 4.1 当前主干脚本

建议上传：

- `小范围DFN生成/优化阶段一/stage2_virtual_well_dfn.py`

### 4.2 虚拟测井与虚拟裂缝

建议上传：

- `小范围DFN生成/优化阶段一/虚拟测井/generate_unit_virtual_well.py`
- `小范围DFN生成/优化阶段一/虚拟裂缝预测/run_batch_virtual_well_fracture_prediction.py`

### 4.3 单元裂缝归属

建议上传：

- `小范围DFN生成/优化阶段一/单元裂缝归属/build_unit_fracture_assignment.py`
- `小范围DFN生成/优化阶段一/单元裂缝归属/merge_virtual_fractures_into_unit_packages.py`

### 4.4 单元DFN构建

建议上传：

- `小范围DFN生成/优化阶段一/单元DFN构建/build_unit_dfn_preview.py`
- `小范围DFN生成/优化阶段一/单元DFN构建/run_batch_unit_dfn_preview.py`

## 5. 建议上传：第三部分

### 5.1 DFN表达验证

建议上传：

- `研究内容三/优化阶段一/DFN体素互转实验/roundtrip_common.py`
- `研究内容三/优化阶段一/DFN体素互转实验/dfn_patches_to_voxel.py`
- `研究内容三/优化阶段一/DFN体素互转实验/voxel_to_dfn_patches.py`
- `研究内容三/优化阶段一/DFN体素互转实验/run_dfn_voxel_roundtrip_experiment.py`
- `研究内容三/优化阶段一/DFN实例表达互转实验/instance_roundtrip_common.py`
- `研究内容三/优化阶段一/DFN实例表达互转实验/dfn_patches_to_instance_label.py`
- `研究内容三/优化阶段一/DFN实例表达互转实验/instance_label_to_dfn_patches.py`
- `研究内容三/优化阶段一/DFN实例表达互转实验/run_dfn_instance_roundtrip_experiment.py`

### 5.2 训练样本组织

建议上传：

- `研究内容三/优化阶段一/GAN训练准备/窗口级实例统计/analyze_instance_window_distribution.py`
- `研究内容三/优化阶段一/GAN训练准备/训练样本打包/build_sparse_instance_gan_dataset.py`
- `研究内容三/优化阶段一/GAN训练准备/训练样本打包/build_instance_gan_dataset.py`

### 5.3 监督基线与生产推理

建议上传：

- `研究内容三/优化阶段一/G_DFN监督基线/baseline_common.py`
- `研究内容三/优化阶段一/G_DFN监督基线/baseline_model.py`
- `研究内容三/优化阶段一/G_DFN监督基线/build_unit_level_split.py`
- `研究内容三/优化阶段一/G_DFN监督基线/train_supervised_baseline.py`
- `研究内容三/优化阶段一/G_DFN监督基线/infer_and_evaluate_baseline.py`
- `研究内容三/优化阶段一/G_DFN监督基线/infer_production_units_baseline.py`

### 5.4 第三部分辅助脚本

建议上传：

- `研究内容三/优化阶段一/G_DFN监督基线/attach_fault_surfaces_to_units.py`
- `研究内容三/优化阶段一/G_DFN监督基线/merge_unit_dfn_vtks.py`
- `研究内容三/优化阶段一/G_DFN监督基线/scale_merged_dfn_vtk_uniform.py`
- `研究内容三/优化阶段一/G_DFN监督基线/prepare_small_validation_unit_list.py`
- `研究内容三/优化阶段一/G_DFN监督基线/build_quick_cpu_subset_split.py`
- `研究内容三/优化阶段一/G_DFN监督基线/train_quick_cpu_baseline.py`

## 6. 建议保留但标记历史

这些文件建议不要直接删除，可以保留在仓库中，但最好在后续 README 或目录说明里标明“历史实验/旧入口”：

### 6.1 第一部分历史/过渡文件

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_deploy_strata_expert_prediction.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_lstm_experiment.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_strata_lstm_experiment.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_point_refine_experiment.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/run_density_experiment.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/_append_density_result_docx.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/测井倾向倾角可视化.py`
- `模型训练/优化阶段一/基于地层约束的测井裂缝预测/strata_expert_deploy/_legacy_run_deploy_strata_expert_prediction.py`

### 6.2 第二部分历史文件

- `小范围DFN生成/优化阶段一/虚拟测井/build_wells.py`

### 6.3 第三部分保留对照文件

- `研究内容三/优化阶段一/GAN训练准备/训练样本打包/build_instance_gan_dataset.py`
  - 可保留为旧版稠密标签对照方案

## 7. 建议不上传

### 7.1 临时运行结果目录

建议不上传：

- `_tmp/`
- `_tmp_ascii_point_refine/`
- `_tmp_autostrata_check/`
- `_tmp_deploy_check/`
- `_tmp_inspect_raw_labels/`
- `_tmp_inspect_raw_labels_ascii/`
- `_tmp_shared_lstm_dryrun_after_move/`
- `_tmp_shared_lstm_dryrun_v1/`
- `_tmp_shared_refine_dryrun_after_move/`
- `_tmp_shared_refine_dryrun_v1/`
- `_tmp_stage2_s4_profile_compare_v5/`
- `_tmp_stage2_virtual_well_dfn_demo/`

### 7.2 Python缓存与临时语法检查文件

建议不上传：

- `__pycache__/`
- 所有子目录中的 `__pycache__/`
- `tmp_syntax_check.py`
- `_tmp_syntax_check.py`
- `_tmp_syntax_check_no_bom.py`

### 7.3 本地产生的临时记录文件

建议不上传：

- `_tmp_experiment_record_20260328.docx`
- `_tmp_stage2_s4_profile_compare_v5.docx`
- `_outer_holdout_w660_2_v3_exist_map.json`
- `_tmp_exist_exp_map_w6602_v5.json`

### 7.4 IDE与本地环境文件

建议不上传：

- `.idea/`

## 8. 文档建议

建议与主代码一起上传以下文档：

- `docs/stage1_file_inventory_2026-04-05.md`
- `docs/stage1_code_map_2026-04-02.md`
- `docs/stage_progress_2026-03-25.md`
- `docs/session_handoff_2026-03-26.md`
- `docs/stage2_virtual_well_dfn_design_2026-03-26.md`

如果你希望 GitHub 首页更容易理解，建议后续在 `README.md` 中补一段：

- 当前主链路说明
- 第一、二、三部分对应目录
- 这份上传清单文档的链接

## 9. 上传前最小检查表

上传前建议逐项确认：

- 第一部分主入口文件是否齐全
- 第二部分主干脚本是否齐全
- 第三部分表达验证、样本打包、基线脚本是否齐全
- 历史/过渡文件是否决定保留
- `_tmp*`、`__pycache__`、`.idea` 是否未纳入上传范围
- `docs/` 中是否附上代码结构说明文档

## 10. 当前建议

如果你的目标是“先把第一轮优化后当前真正使用的主链路传上去”，最稳妥的做法是：

1. 先上传第一、二、三部分主链路代码
2. 再上传说明文档
3. 历史/过渡文件保留，但在后续 README 中标注
4. 所有 `_tmp*`、缓存、IDE 文件不上传

这样最容易让 GitHub 上的代码结构看起来清楚，也最方便后续继续整理。

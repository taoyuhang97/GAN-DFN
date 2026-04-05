# 第一轮优化后代码文件清单（2026-04-05）

## 1. 文档目的

本文档用于在上传 GitHub 前，快速核对“第一轮优化后当前主链路实际使用的代码文件”。  
整理原则如下：

- 只列当前主链路中涉及的代码文件
- 按第一部分、第二部分、第三部分拆开说明
- 每个文件只说明当前主要职责
- 单独标出“辅助/过渡文件”，避免与主用文件混淆

本文档不包含以下内容：

- `__pycache__`
- `_tmp`、`_tmp_*`
- 图片、表格、`docx`、`pptx`
- 纯说明性 `README` 文件

## 2. 当前主链路总览

当前第一轮优化后的整体代码链路可概括为：

`数据准备 -> 第一部分测井裂缝预测 -> 第二部分单元DFN构建 -> 第三部分DFN标准化表达与基线验证`

三部分之间的关系是：

- 第一部分负责把成像测井裂缝认识迁移到常规测井，输出井尺度裂缝结果
- 第二部分负责把井尺度裂缝结果转成单元尺度裂缝种子和单元 DFN
- 第三部分负责验证 DFN 标签表达方式、组织训练样本，并建立监督基线与生产推理原型

## 3. 第一部分

第一部分对应目录：

- `模型训练/优化阶段一/基于地层约束的测井裂缝预测`

第一部分前置数据准备目录：

- `数据精简/优化阶段一`

### 3.1 当前主用入口文件

- `run_manual_strata_lstm_experiment.py`
  - 第一阶段主入口
  - 按手工分层组织样本
  - 训练/验证裂缝发育段或裂缝存在性模型

- `run_manual_strata_raw_point_refine.py`
  - 第二阶段主入口
  - 在第一阶段预测出的发育段内部细化裂缝点位

- `run_outer_holdout_expert_validation.py`
  - 外层留一验证入口
  - 重新训练专家库，并对整井执行“第一阶段 + 第二阶段”串联验证

- `run_conventional_log_strata_validation.py`
  - 单井完整部署入口
  - 输出最终单井结果：
    `final_log_with_fractures.csv`
    `final_fracture_points.csv`
    `final_fracture_segments.csv`
    `final_strata_segmentation.csv`
    `prediction_meta.json`

- `run_batch_existing_conventional_log_prediction.py`
  - 批量部署入口
  - 对现有常规测井目录逐井执行完整预测

- `run_aggregate_existing_conventional_log_prediction.py`
  - 批量结果汇总入口
  - 汇总输出总表：
    `all_final_fracture_points.csv`
    `all_final_fracture_segments.csv`
    `all_final_strata_segmentation.csv`
    `all_predicted_well_summary.csv`
    `aggregation_meta.json`

- `run_conventional_log_strata_segmentation_only.py`
  - 只做层位切分和归层验证
  - 不执行完整裂缝预测

### 3.2 第一部分底层核心脚本

- `模型训练/优化阶段一/裂缝存在性分析/LSTM/imaging_well_to_fracture_cnn_lstm_test_1.py`
  - 第一阶段底层模型脚本
  - 负责裂缝发育段/存在性预测

- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/raw_point_guided_segment_refine.py`
  - 第二阶段底层主脚本
  - 负责段内裂缝点位细化

- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/nearest_imaging_well_predict_then_refine.py`
  - 专家井选择与两阶段串联预测底层脚本

### 3.3 第一部分共享模块

- `workflow_paths.py`
  - 统一项目根目录与底层脚本路径解析

- `manual_strata_workflow/common.py`
  - 井名归一化
  - 手工分层配置
  - JSON/参数解析辅助

- `manual_strata_workflow/dataset.py`
  - 构建手工分层样本
  - 导出按地层切分后的数据集

- `strata_expert_deploy/runtime.py`
  - 部署期通用工具函数

- `strata_expert_deploy/strata_resolution.py`
  - 目标井地层归属与区间切分

- `strata_expert_deploy/stage_prediction.py`
  - 部署期第二阶段预测封装

- `strata_expert_deploy/pipeline.py`
  - 当前模块化部署入口
  - 将分层、选模、预测串起来

### 3.4 第一部分前置数据准备文件

- `数据精简/优化阶段一/new_data_around_well.py`
  - 构建井震对齐样本
  - 生成常规测井/成像测井联合输入数据

- `数据精简/优化阶段一/new_data_around_well_other.py`
  - 其他成像井样本构建版本

- `数据精简/优化阶段一/new_fracture_exist_sample.py`
  - 将裂缝方位、密度等标签并入样本

- `数据精简/优化阶段一/new_fracture_exist_sample_other.py`
  - 其他井裂缝标签构建版本

- `数据精简/优化阶段一/车151HF裂缝信息提取.py`
  - 提取车151HF原始裂缝解释信息

- `数据精简/优化阶段一/车页1导眼裂缝信息提取.py`
  - 提取车页1导眼原始裂缝解释信息

- `数据精简/优化阶段一/che660-1_GR_add.py`
  - 为特定样本补充 `GR` 字段

### 3.5 第一部分辅助/过渡文件

这些文件仍有参考价值，但不建议与当前主入口混为一类：

- `run_deploy_strata_expert_prediction.py`
  - 较早的单脚本部署入口
  - 当前功能已被 `strata_expert_deploy/pipeline.py` 吸收

- `run_lstm_experiment.py`
  - 第一阶段底层实验包装脚本

- `run_strata_lstm_experiment.py`
  - 更早的分层实验入口

- `run_point_refine_experiment.py`
  - 第二阶段实验包装脚本

- `run_density_experiment.py`
  - 密度分支实验脚本

- `_append_density_result_docx.py`
  - 密度实验结果文档追加脚本

- `测井倾向倾角可视化.py`
  - 倾向倾角结果可视化脚本

## 4. 第二部分

第二部分对应目录：

- `小范围DFN生成/优化阶段一`

### 4.1 当前主用入口文件

- `stage2_virtual_well_dfn.py`
  - 当前第二部分主干脚本
  - 统一单元网格编号
  - 读取第一部分部署结果
  - 生成单元中心虚拟测井
  - 汇总真实/虚拟裂缝种子
  - 输出参数化裂缝片

### 4.2 虚拟测井与虚拟裂缝文件

- `虚拟测井/generate_unit_virtual_well.py`
  - 面向单元批量生成虚拟测井
  - 输出可兼容第一部分流程的虚拟井样本

- `虚拟裂缝预测/run_batch_virtual_well_fracture_prediction.py`
  - 对虚拟测井批量执行第一部分预测流程
  - 生成虚拟裂缝结果

### 4.3 单元裂缝归属与合并文件

- `单元裂缝归属/build_unit_fracture_assignment.py`
  - 将真实井裂缝点、裂缝段、分层井段拆分归属到单元
  - 生成单元级裂缝数据包

- `单元裂缝归属/merge_virtual_fractures_into_unit_packages.py`
  - 合并真实井单元裂缝和虚拟井裂缝结果

### 4.4 单元DFN构建文件

- `单元DFN构建/build_unit_dfn_preview.py`
  - 根据裂缝种子、层位和地震梯度生成单元 DFN 面片
  - 支持预览图和 VTK 导出

- `单元DFN构建/run_batch_unit_dfn_preview.py`
  - 批量执行单元 DFN 预览构建

### 4.5 第二部分辅助/历史文件

- `虚拟测井/build_wells.py`
  - 更早期的虚拟井实验脚本
  - 当前建议作为历史参考，不作为主入口

## 5. 第三部分

第三部分对应目录：

- `研究内容三/优化阶段一`

### 5.1 DFN表达验证文件

#### DFN体素互转实验

- `DFN体素互转实验/roundtrip_common.py`
  - 体素互转共用函数

- `DFN体素互转实验/dfn_patches_to_voxel.py`
  - 裂缝片参数表转体素

- `DFN体素互转实验/voxel_to_dfn_patches.py`
  - 体素结果回转裂缝片参数表

- `DFN体素互转实验/run_dfn_voxel_roundtrip_experiment.py`
  - 一键执行 `DFN -> 体素 -> DFN` round-trip

#### DFN实例表达互转实验

- `DFN实例表达互转实验/instance_roundtrip_common.py`
  - 实例标签互转共用函数

- `DFN实例表达互转实验/dfn_patches_to_instance_label.py`
  - 裂缝片参数表转实例标签

- `DFN实例表达互转实验/instance_label_to_dfn_patches.py`
  - 实例标签回转裂缝片

- `DFN实例表达互转实验/run_dfn_instance_roundtrip_experiment.py`
  - 一键执行 `DFN -> 实例标签 -> DFN` round-trip

### 5.2 训练样本准备文件

- `GAN训练准备/窗口级实例统计/analyze_instance_window_distribution.py`
  - 统计窗口级裂缝实例分布
  - 为切窗和参数选择提供依据

- `GAN训练准备/训练样本打包/build_sparse_instance_gan_dataset.py`
  - 当前主用样本打包脚本
  - 输出轻量稀疏实例 `.npz` 样本

- `GAN训练准备/训练样本打包/build_instance_gan_dataset.py`
  - 旧版稠密标签打包脚本
  - 当前更多用于历史对照

### 5.3 监督基线主用文件

- `G_DFN监督基线/baseline_common.py`
  - 训练与推理共用函数

- `G_DFN监督基线/baseline_model.py`
  - 3D U-Net 监督基线模型定义

- `G_DFN监督基线/build_unit_level_split.py`
  - 按单元划分 train/val/test

- `G_DFN监督基线/train_supervised_baseline.py`
  - 监督基线训练主脚本

- `G_DFN监督基线/infer_and_evaluate_baseline.py`
  - 窗口预测
  - 单元回拼
  - 定量评估

- `G_DFN监督基线/infer_production_units_baseline.py`
  - 面向新单元的生产推理脚本

### 5.4 第三部分辅助文件

- `G_DFN监督基线/attach_fault_surfaces_to_units.py`
  - 将断层面挂接到单元结果

- `G_DFN监督基线/merge_unit_dfn_vtks.py`
  - 合并多个单元的 VTK 面片结果

- `G_DFN监督基线/scale_merged_dfn_vtk_uniform.py`
  - 统一缩放合并后的 VTK

- `G_DFN监督基线/prepare_small_validation_unit_list.py`
  - 生成小规模验证单元清单

- `G_DFN监督基线/build_quick_cpu_subset_split.py`
  - 生成 CPU 快速验证子集

- `G_DFN监督基线/train_quick_cpu_baseline.py`
  - 快速 CPU 基线训练脚本

## 6. 当前建议的检查顺序

如果上传前需要先做代码核对，建议按下面顺序检查：

1. 第一部分主入口和共享模块
2. 第一部分前置数据准备文件
3. 第二部分主干：
   `stage2_virtual_well_dfn.py`
   `build_unit_fracture_assignment.py`
   `build_unit_dfn_preview.py`
4. 第三部分三组核心文件：
   DFN表达互转
   样本打包
   监督基线

## 7. 上传前建议

如果本次上传的目标是先把“第一轮优化后主链路代码”整理清楚，建议你优先确认以下几点：

- 第一部分主入口是否只保留当前实际使用版本
- 第二部分中 `build_wells.py` 这类历史实验脚本是否需要保留
- 第三部分中旧版稠密标签脚本是否作为历史方案说明保留
- `docs/` 中是否需要同步保留本说明文档，方便 GitHub 阅读者理解目录角色

如果后续还要继续做仓库整理，建议在此基础上再补一份：

- “主链路文件”
- “历史实验文件”
- “临时结果文件不上传”

三类边界更清晰的 GitHub 上传说明。

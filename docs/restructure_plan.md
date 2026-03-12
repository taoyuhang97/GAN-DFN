# Restructure Plan

这个文件给出“当前脚本放在哪里更合适”的迁移建议，目标是逐步工程化，而不是一次性重构。

## 原则

1. 先保留现有脚本可运行
2. 先抽公共函数，再迁移入口脚本
3. 路径配置优先，目录重命名滞后
4. 模型产物、图件、日志统一进 `outputs/`

## 当前目录到目标目录的映射

| 当前目录/脚本 | 当前职责 | 目标位置 |
|---|---|---|
| `file_check.py` | 原始文件存在性检查 | `scripts/audit/check_raw_files.py` |
| `curve_check.py` | LAS 曲线项统计 | `scripts/audit/check_las_curves.py` |
| `file_read_data.py` | DLIS/FMI 提取 | `scripts/dataset/export_fmi_from_dlis.py` |
| `seismic_extract.py` | 井位关联最近地震道并导出 tracecube | `scripts/dataset/export_well_tracecube.py` |
| `数据精简/read_trace_header_xy.py` | trace header 坐标提取 | `scripts/dataset/export_trace_header_xy.py` |
| `数据精简/data_around_well.py` | 井周联合样本构建 | `scripts/dataset/build_well_samples.py` |
| `数据精简/fracture_exist_sample.py` | 裂缝标签生成 | `scripts/dataset/build_fracture_labels.py` |
| `模型训练/裂缝存在性分析/` | 裂缝存在性建模 | `scripts/train/presence/` |
| `模型训练/裂缝位置分析/` | 裂缝位置建模 | `scripts/train/position/` |
| `模型训练/裂缝角度预测/` | 裂缝角度建模 | `scripts/train/orientation/` |
| `小范围DFN生成/单元地震信息抽取/` | 单元地震块提取 | `scripts/dfn/extract_unit_seismic/` |
| `小范围DFN生成/测井所在单元DFN构建/` | 井所在单元 DFN 构建 | `scripts/dfn/build_unit_dfn/` |
| `小范围DFN生成/断层裂缝片生成/` | 断层网格切割与裂缝片处理 | `scripts/dfn/fault_patch/` |
| `小范围DFN生成/fracture_fault_fusion.py` | 断层与裂缝融合 | `scripts/dfn/fuse_fault_and_fracture.py` |
| `研究内容三/mesh_to_volume.py` | DFN 转体素 | `scripts/voxel/mesh_to_volume.py` |
| `研究内容三/volume_to_mesh.py` | 体素转网格 | `scripts/voxel/volume_to_mesh.py` |
| `研究内容三/3D-GAN_input_data_generate.py` | GAN 输入样本生成 | `scripts/gan/prepare_input.py` |
| `研究内容三/train_and_generate.py` | 3D-GAN 训练与推理 | `scripts/gan/train_and_infer.py` |
| `研究内容三/fusion_unit_DFNs.py` | 多单元 DFN 合并 | `scripts/dfn/merge_unit_dfns.py` |

## 建议抽出来的公共模块

### `src/fracture_sample_gen/io/`

- `las.py`
  - 读取 LAS 曲线
  - 提取井口坐标
- `well_track.py`
  - 读取井斜轨迹
- `timedepth.py`
  - 读取时深关系
- `segy.py`
  - 读取 trace header
  - 按 `TraceIdx` 导出地震道
- `dlis.py`
  - 读取 DLIS/FMI

### `src/fracture_sample_gen/preprocessing/`

- `trace_match.py`
  - KDTree 最近道搜索
  - 地震道排序
- `sample_builder.py`
  - 井周窗口样本构造
  - 地震振幅插值

### `src/fracture_sample_gen/labeling/`

- `fmi_label.py`
  - 从 FMI 结果生成深度段
  - 生成 `FRACTURE_FLAG`

### `src/fracture_sample_gen/models/`

- `xgboost_presence.py`
- `xgboost_orientation.py`
- `lstm_presence.py`

### `src/fracture_sample_gen/dfn/`

- `fracture_generation.py`
- `fault_patch.py`
- `dfn_merge.py`
- `visualization.py`

### `src/fracture_sample_gen/voxel/`

- `mesh_to_volume.py`
- `volume_to_mesh.py`
- `voxel_utils.py`

## 第一阶段落地建议

第一阶段只做这 4 件事：

1. 新增 `configs/paths.example.yaml`
2. 把所有硬编码路径改成从配置读取
3. 把重复的读取函数从脚本中抽到 `src/`
4. 保留旧目录不动，只新增 `scripts/` 入口

这样风险最低，也最容易逐步验证。

## 第二阶段落地建议

第二阶段再做：

1. 把稳定脚本迁移到 `scripts/`
2. 把统计图、模型文件、日志统一到 `outputs/`
3. 给每条研究链路补最小 README

## 第三阶段落地建议

第三阶段再考虑：

1. 精简重复目录中的“优化阶段一”副本
2. 统一中英文命名风格
3. 拆分训练与推理脚本
4. 补基础命令行参数

## 建议保留为 legacy 的部分

下面这类文件建议暂时不要急着重构：

- 一次性分析脚本
- 可视化验证脚本
- 已经产出论文图件的脚本
- 强依赖历史路径的数据修复脚本

可以统一放到未来的 `legacy/` 目录，避免打断已有研究流程。

## 最小可执行工程化目标

如果只做最小整理，建议先达到这个状态：

```text
scripts/dataset/build_well_samples.py
scripts/dataset/build_fracture_labels.py
scripts/train/presence/train_xgboost.py
scripts/dfn/build_unit_dfn.py
scripts/voxel/mesh_to_volume.py
scripts/gan/train_and_infer.py
src/fracture_sample_gen/...
configs/paths.example.yaml
outputs/
```

达到这一步，这个仓库就会从“研究脚本堆”提升成“可复现的研究工程”。

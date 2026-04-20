# DFN实例表达互转实验

本目录用于验证“中心点 + 参数场”实例表达方案，不修改原有 `DFN体素互转实验` 的 occupancy 连通体流程。

## 目标

- 用“中心槽位 + 几何参数”表达单元内裂缝片
- 不再依赖 occupancy 连通域回转实例
- 继续输出原始裂缝片、回转裂缝片和对比 VTK 面片

## 脚本

- `dfn_patches_to_instance_label.py`
  - 将 `unit_dfn_patches.csv` 编码为 `instance_label_volume.npz`
- `instance_label_to_dfn_patches.py`
  - 将实例标签解码为 `roundtrip_patches.csv`
- `run_dfn_instance_roundtrip_experiment.py`
  - 端到端完成编码、解码、对比输出和实验记录写入

## 标签

当前标签通道：

- `center_heatmap`
- `center_count`
- `offsets`
- `normals`
- `u_dirs`
- `lengths`
- `heights`
- `confidence`
- `source_patch_index`

默认 `slots_per_voxel = 4`，允许同一体素存放多个裂缝中心实例。

## 输出

默认输出目录：

- `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\DFN实例表达互转实验`

实验记录：

- `D:\项目\石油开采\断缝储实验\实验记录20260328.docx`

## 示例

```powershell
py -3.12 .\研究内容三\优化阶段一\DFN实例表达互转实验\run_dfn_instance_roundtrip_experiment.py --unit-id BX69_BY28 --run-name smoke_20260330_slot4_xy24_z02 --xy-resolution 24 --z-step-ms 0.2 --slots-per-voxel 4 --center-threshold 0.5 --thickness-vox-for-compare 1.0
```

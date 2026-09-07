# Step 7B 3D Initial DFN

本步骤是 `step7_initial_dfn` 的并行增强版本，专门消费 Step6B 输出的三维裂缝密度 SGY，先面向 `candidate_cheye1` 构造真正受 `X-Y-Time` 密度体控制的初始 DFN。

旧 Step7 保留不动，用于复现二维层段密度表路径；Step7B 输出字段保持与 Step8 井控校正兼容。

## 输入

- `candidate_cheye1_3d_predicted_density.sgy`：Step6B 输出的三维裂缝密度体。
- `candidate_cheye1_3d_trace_mapping.npz`：SGY trace 到 `TraceIdx/X/Y/IX/IY` 的映射。
- T4-T7 层位目录：用于把三维密度体限制在沙三段、沙四段。

## 运行

原始三维密度版本：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7b_initial_dfn_3d/build_initial_dfn_from_3d_density_sgy.py \
  --config 优化阶段二/正式主线/step7b_initial_dfn_3d/configs/formal_initial_dfn_3d_candidate_cheye1.json
```

低相干引导版本，输出到独立目录，不覆盖原始结果：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7b_initial_dfn_3d/build_initial_dfn_from_3d_density_sgy.py \
  --config 优化阶段二/正式主线/step7b_initial_dfn_3d/configs/formal_initial_dfn_3d_candidate_cheye1_lowcoh.json
```

## 核心逻辑

- 读取 SGY 并重排成 `Y × X × Time` 三维网格。
- 用 T4-T6、T6-T7 分别约束沙三段、沙四段有效体素。
- 按层段密度分位数提取高密度候选体素，并做三维连通域过滤，减少孤立噪声。
- 按体素密度加权抽样裂缝片中心，中心时间直接来自三维密度网格，不再在层段窗口内随机生成。
- 可选低相干引导：读取相干体 SGY，将低相干黑色异常转换为权重，影响候选体素评分和抽样概率；裂缝片尺寸仍由原三维裂缝密度控制。
- 在裂缝中心邻域内用三维 PCA 估计局部高密度带的空间方向，并转换为 `AzimuthDeg/DipDeg`。
- 裂缝片尺寸随密度增大而增大：`LengthM` 和 `HeightTimeMs` 均使用层段基础尺寸乘以密度因子。
- VTK 几何真正使用 `DipDeg` 生成倾斜矩形面；旧 Step7 只是把倾角写成属性。

## 输出

默认原始三维密度输出目录：`output/candidate_cheye1`。

低相干引导输出目录：`output/candidate_cheye1_lowcoh`。

- `initial_dfn_fracture_patches.csv`：与 Step8 兼容的初始 DFN 裂缝片表。
- `initial_dfn_raw_time.vtk`：时间域原始 VTK。
- `initial_dfn_generation_audit.csv`：逐裂缝片审计表。
- `initial_dfn_summary.json`：输入网格、候选体素、采样和质量检查摘要。

## 关键配置

- `candidate_density_quantile`：按层段选择高密度候选体素的分位数阈值。
- `min_component_voxels`：三维连通体最小体素数。
- `count_scale/min_patch_count/max_patch_count`：控制最终裂缝片数量。
- `base_length_m/base_height_time_ms`：层段基础裂缝片尺寸。
- `length_density_gain/height_density_gain`：密度越高，裂缝片越大的放大系数。
- `orientation_window_xy_cells/orientation_window_time_samples`：局部 PCA 方向估计窗口。
- `coherence_guidance`：低相干引导开关和权重参数；当前正式 lowcoh 配置使用低相干权重最大 3 倍，并保持总裂缝片数量仍由原三维密度质量控制。

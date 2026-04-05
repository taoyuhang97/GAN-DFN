# 第二部分设计说明（2026-03-26）

## 1. 目标

本轮新增脚本：

- `小范围DFN生成/优化阶段一/stage2_virtual_well_dfn.py`

它的职责不是替代旧的实验脚本，而是先把第二部分最核心的三件事收口到一条可运行链路里：

1. 统一单元网格编号
2. 基于第一部分输出构建单元中心虚拟测井样本
3. 将点位/密度/方向约束转成单元级 DFN 裂缝片

## 2. 当前统一口径

### 2.1 单元划分与编号

- 直接复用 `trace_header_xy.csv`
- 单元大小沿用历史脚本的 `25 x 25` 道
- 单元步长沿用历史脚本的 `24` 道
- 单元编号格式统一为 `BX{block_x}_BY{block_y}`

这样新脚本与 `well_fracture_split.py`、`extract_block.py` 的区块口径保持一致。

### 2.2 虚拟测井的表示方式

当前采用“单元中心垂向样本井”口径，而不是井轨迹样式：

- `X/Y` 取单元中心
- 每个地层单独插值 `DepthMin/DepthMax`
- 在层内按固定步长采样，形成虚拟井样本序列

当前每个虚拟井样本可直接携带：

- `VirtualAC`
- `VirtualGR`
- `PredDensityStrength`
- `PredDensitySizeScale`
- `PredAzimuth`
- `PredDip`
- `PredOrientationConfidence`

因此它既可以被视为“虚拟测井样本”，也可以直接作为后续单元 DFN 的先验约束轨迹。

### 2.3 井深域到单元空间的映射

当前分成两类：

- 真实点位：第一部分输出中的 `X/Y/TIME` 直接映射到单元
- 虚拟样本：由单元中心 + 插值后的层位范围生成

真实点位和虚拟样本最终都统一为 `dfn_seed_points.csv`，作为 DFN 裂缝片生成的共同输入。

## 3. DFN 几何参数定义

### 3.1 方向参数

- 裂缝片方向直接由 `PredAzimuth + PredDip` 控制
- 当需要插值时，先转为法向量，再做加权平均，再还原回方位角/倾角

### 3.2 密度与尺度参数

当前支持三种模式：

- `size`
  - 密度只控制裂缝片尺寸
- `count`
  - 密度只控制裂缝片数量
- `hybrid`
  - 密度同时控制尺寸和数量

默认使用 `hybrid`，因为这更接近交接文档中“密度既可能影响裂缝数量，也可能影响裂缝片尺度”的过渡口径。

### 3.3 裂缝片表达

当前输出为参数化裂缝片，而不是体素：

- 裂缝中心 `CenterX/CenterY/CenterTIME`
- 方位 `Azimuth`
- 倾角 `Dip`
- 长度 `PatchLength`
- 高度 `PatchHeight`
- 四个顶点 `V1..V4`

对应文件：

- `unit_dfn_patches.csv`
- `unit_dfn_summary.csv`

## 4. 当前输出文件

脚本当前会输出：

- `unit_index.csv`
- `mapped_control_points.csv`
- `virtual_well_samples.csv`
- `dfn_seed_points.csv`
- `unit_constraint_summary.csv`
- `unit_dfn_patches.csv`
- `unit_dfn_summary.csv`
- `run_summary.json`

## 5. 当前边界

本轮实现优先解决的是“第二部分主干打通”，因此有意保留了几个后续扩展点：

- 当前虚拟井仍是单元中心垂向样本井，还不是实际井轨迹样式
- 当前层位范围插值依赖已有井的层位/点位约束，还没有接入断层曲面
- 当前 DFN 输出是裂缝片参数表达，尚未同步输出体素表达
- 当前裂缝片未做跨层裁切与断层后处理

## 6. 建议的下一步

建议后续继续沿这条线补强，而不是回到旧的单脚本实验态：

1. 接入多井部署结果，提升虚拟井的空间插值稳定性
2. 将层位范围插值替换为断层/层位曲面控制
3. 增加体素化输出，与 `研究内容三` 的体素/网格流程对接
4. 在生成裂缝片后加入跨层裁切和断层融合

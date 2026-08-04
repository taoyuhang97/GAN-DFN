# Sample Horizon Attach

这个目录提供一个独立的“样本层位挂接/验证入口”，用于在不改动现有 `step1_unified_samples` builder 的前提下，给已有点级样本 CSV 增加 `T4-T7` 分层结果与基础合法性摘要。

它是后续多井样本构建前的前置步骤：

- 先把单井点级样本按现有 `step1` 结果挂上层位与目标层段标记
- 再进入多井拼接、筛样和训练流程

## 文件

- `attach_sample_horizons.py`
  - 输入已有点级 CSV，至少要求 `X/Y/TIME`
  - 直接调用 `step0_t4_t7_surface_tools` 的真实层位面挂接逻辑
  - 输出带 `T4Time/T5Time/T6Time/T7Time/LayerGroup/InTargetT4T7/SurfaceOrderValid/InputRowValid` 的样本表
- `configs/cheye1_from_step1.json`
  - 基于 `车页1导眼_unified_point_table.csv` 和真实 `/层位` 目录的最小跑通配置
- `output/`
  - 示例输出目录

## 当前挂接口径

- 不修改旧的 `step1` builder
  - 但层位挂接不再反推 `step1` 结果，而是直接读取真实 `T4/T5/T6/T7` 层位面
- `T4-T7` 界面时间恢复方式
  - 对每个输入点，按 `XY` 最近曼哈顿距离从真实层位面读取 `T4/T5/T6/T7` 时间
- 默认层位选择规则
  - `T4` 优先旧版
  - `T5/T6/T7` 优先 2024 更新版
- 样本行层位挂接方式
  - `T4 <= TIME < T6` 记为 `沙三段`
  - `T6 <= TIME < T7` 记为 `沙四段`
  - 其余记为 `OUT_OF_TARGET`
- `InTargetT4T7`
  - 当前定义为样本 `TIME` 落在 `T4-T7` 目标层段内，且当前行层位顺序/厚度检查通过
- `SurfaceOrderValid`
  - 检查当前行 `T4 < T5 < T6 < T7`

## 用法

在仓库根目录执行：

```bash
python 优化阶段二/当前任务/sample_horizon_attach/attach_sample_horizons.py \
  --config 优化阶段二/当前任务/sample_horizon_attach/configs/cheye1_from_step1.json
```

示例输出：

- `优化阶段二/当前任务/sample_horizon_attach/output/车页1导眼_horizon_attached.csv`
- `优化阶段二/当前任务/sample_horizon_attach/output/车页1导眼_horizon_attached_summary.json`

## 输入要求

- 输入 CSV 至少包含：
  - `X`
  - `Y`
  - `TIME`

## 输出核心字段

- `T4Time`
- `T5Time`
- `T6Time`
- `T7Time`
- `LayerGroup`
- `InTargetT4T7`
- `SurfaceOrderValid`
- `InputRowValid`
- `ReferenceMatchDistance`

## 已知限制

- 当前脚本按真实层位面做点级最近曼哈顿距离挂接，不做区域面插值。
- `T4` 当前依赖未平滑版本；平滑版补充后应优先替换。
- 层位文件本身只解决垂向分层，不解决目标区域平面范围选择。

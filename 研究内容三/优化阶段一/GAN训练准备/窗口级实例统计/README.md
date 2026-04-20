# 窗口级实例统计

本实验用于在 GAN 训练前统计单元 DFN 实例表达在窗口级的分布特征。

## 目标

- 统计每个窗口内的裂缝实例数量和单体素冲突强度
- 为 `slots_per_voxel = K` 提供定量依据
- 比较不同 `Z` 窗口长度对冲突分布的影响
- 给出第一批训练单元建议

## 脚本

- `analyze_instance_window_distribution.py`

## 默认输入

- 单元 DFN 根目录：
  - `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容二\单元DFN构建\批量生成`
- 地震道头：
  - `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\研究内容一\trace_header_xy.csv`

## 默认输出

- `E:\项目\石油项目\断缝储\原始数据\wx数据\砂砾岩\优化阶段一\研究内容三\GAN训练准备\窗口级实例统计`

## 示例

```powershell
py -3.12 .\研究内容三\优化阶段一\GAN训练准备\窗口级实例统计\analyze_instance_window_distribution.py --window-sizes 128 160 --overlap-ratio 0.5
```

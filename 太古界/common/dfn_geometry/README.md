# 太古界 DFN 公共几何模块

本目录保存Step7系列共同使用的底层能力，不是一个可独立执行的流程步骤。

当前主要能力包括：

- 读取裂缝密度体并重建属性网格；
- 根据候选中心生成有倾向、倾角和尺寸的裂缝片；
- 输出时间域VTK；
- 生成裂缝片统计与审计信息。

正式中尺度DFN入口位于`step7b_multiscale_initial_dfn`；大尺度DFN入口位于
`step7c_large_fault_dfn`。公共代码放在这里是为了避免再出现两个Step7B目录。


---

## 口径变更（2026-09-22，azi1）

产状统一为 **真倾向方位 0–360（自北顺时针）+ 倾角**，唯一实现见
`太古界/common/orientation_frame/convention.py`（换算规则与验证见该目录 README）。

- `DipAzimuthDeg` = 唯一真值字段；`AzimuthDeg` 降级为派生走向 `(DipAzimuthDeg-90)%180`。
- 第三维为 `TIME`（向下为正）的帧一律走 `*_depth` 变体换算。
- 旧口径开关（`family_azimuth_semantics` / `ridge_azimuth_semantics` /
  `vertex_convention` / `dfn_azimuth_semantics`）已删除，不再保留双口径分支。
- 背景：`太古界/太古界流程梳理与问题记录_20260915.md` §0.37；
  施工方案：`太古界/太古界产状口径统一施工方案_20260922.md`。

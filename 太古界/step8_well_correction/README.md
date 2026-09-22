# 太古界 Step8：井控校正与井轨迹展示件

## md1（2026-09-22）：验收中的"成像区域校正"新增"不适用"分支

背景：Step1/2/3 改成 MD 口径后，埕北古斜405 的成像解释段（MD 3676.6–3998.0，TIME 2746.8–2866.2，
借用 CBGX22 时深）整体落在"上部复合层"窗口之上（层顶 TIME 2950.7 / MD 4179.7），
312 个成像控制点中 301 个被判为层外，本区块因此**没有可用的成像井控**，
`imaging_well_region_correction` 无处可施。

处理（需求方 2026-09-22 决策：按当前 MD 位置推进，汇报材料中注明 405 不在目标深度范围）：

- 新增内部账目 `config["_imaging_control_layer_qc"]`：
  `imaging_control_count_input` / `imaging_control_inside_layer_count` / `imaging_control_outside_layer_count`；
- `summary["correction_logic"]["imaging_well_region_correction"]` 增加
  `status`（`applied` / `not_applicable_no_candidates_within_radius` /
  `not_applicable_no_imaging_controls` / `not_applicable_imaging_controls_outside_target_layer` /
  `disabled`）与 `skip_reason`；
- 验收 check `imaging_well_region_correction_applied` 在**"不适用且 skip_reason 非空"**时判通过，
  其余情况仍判失败——即不允许"无账目地静默通过"。

md1 实测：`status=pass`，`control_point_count=11`（全部 `step4_predicted_event`），
成像区域校正 `status=not_applicable_imaging_controls_outside_target_layer`，
`skip_reason=all_imaging_controls_fall_outside_the_assigned_target_layer_window`，
`imaging_control_count_input=301 / inside=0 / outside=301`。

> 该分支只是把"本区块没有可用成像井控"这件事**显式记账**；405 的井震标定/时深（甲方标注
> "时深=无、井斜=无"）仍是下一轮第一优先，见问题记录 §0.33。

## 正式井控入口

- `correct_taigu_multiscale_dfn_with_well_controls.py`
- `configs/taigu_step8_attribute_multiscale_v1.json`

旧版 `correct_taigu_dfn_with_well_controls.py` 及其配置 `taigu_step8_v1.json`、`smoke_v2.json`
已于 2026-09-15 移入 `太古界/_archive_20260915/C_legacy_v1_chain/`，只作历史对照。

## 井轨迹辅助模块

`export_taigu_well_trajectories_vtk.py`读取太古界 Step2 每井目录下的多个分段 CSV，按井合并、去重并限制到 Step8 配置的目标区。坐标合同为 `X/Y` 米、`Z=TIME(ms)`，不取绝对值、不做显示缩放。

```bash
python3 太古界/step8_well_correction/export_taigu_well_trajectories_vtk.py \
  --config 太古界/step8_well_correction/configs/taigu_step8_attribute_multiscale_v1.json
```

输出到对应 Step8 结果目录：

- `well_trajectories_raw_time.vtk`：每口井一条折线，含 `WellID/WellName/WellNameASCII`。
- `well_heads_raw_time.vtk`：每口井一个井顶点，含 `WellID/WellName/WellNameASCII/TemporaryTimeDepth` 点属性，用于 ParaView 批量显示井点和井名。
- `well_trajectories_annotations.csv`：井名、井顶标签坐标、深度范围、点数、时深来源和临时时深标记。
- `well_trajectories_export.json`：范围、井数、井名、坐标口径和排除井说明。

`label_taigu_well_trajectories_paraview.py`是当前正式的 ParaView 展示脚本，由用户已在砂砾岩矿区实际使用的合并几何版适配而来：全部轨迹、全部井顶球和全部 ASCII 井名分别合并为1个对象，总计3个对象，不随井数增长。脚本会显式更新管线、渲染并记录三类几何的 bounds 和 Visibility。`plot_taigu_well_name_map.py`用于生成带中文井名的平面图。

Windows ParaView 运行时，将以下文件放到同一目录，脚本会优先自动读取同目录数据：

- `label_taigu_well_trajectories_paraview.py`
- `well_trajectories_raw_time.vtk`
- `well_trajectories_annotations.csv`

推荐脚本不依赖 `well_heads_raw_time.vtk`；该文件作为独立井顶点数据成果继续保留。

当前 5 km demo 内输出埕北古406和埕北古斜405两口井；405保留 `temporary_neighbor_time_depth=1`。


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

# 太古界 Step8：井控校正与井轨迹展示件

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

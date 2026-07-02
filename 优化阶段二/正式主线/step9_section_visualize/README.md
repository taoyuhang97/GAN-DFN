# Step 9 Section Visualize

本目录保存优化阶段二正式主线最终成果的过井 DFN 剖面可视化过程。

## 入口

过井邻域剖面：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_demo_well_section_visualization.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_demo_well_section_visualization.json
```

全区尺度投影剖面：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_all_area_section_visualization.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_all_area_section_visualization.json
```

过井属性体剖面：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_well_attribute_section_visualization.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_well_attribute_section_visualization.json
```

全矿区尺度过井属性体剖面：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_mine_attribute_section_visualization.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_attribute_section_visualization.json
```

## 输入

- 最终井控校正 DFN：`优化阶段二/正式主线/step8_dfn_well_correction/output/well_corrected_dfn_raw_time.vtk`
- 正式真实井 `T4-T7` 样本：`优化阶段二/正式主线/step2_real_well_t4_t7_samples/output/formal_all_wells`
- 层位界面：`/data/shared/project-oil/wx数据/砂砾岩/层位/T4/T5/T6/T7*.dat`

## 选井逻辑

配置默认不指定井名，脚本会在候选 A demo 区块内自动选择轨迹点覆盖最多的真实井，并排除 `车页1导眼`。

当前正式运行选中：`车58`，区块内轨迹点 `12071/12071`，覆盖比例 `1.0`。

## 分层与界面

- 绘制界面：`T4`、`T5`、`T6`、`T7`
- 正式层段：`T4->T6` 为沙三段，`T6->T7` 为沙四段
- `T5` 只作为沙三段内部界面线显示，不单独形成正式层段

## 输出

过井邻域剖面输出在 `output/around_well`。该版本按参考井轨迹曲线和 `half_width=50m` 筛选 DFN 片段，用于查看井周边剖面：

- `output/around_well/demo_well_t4_t7_projection_summary.json`
- `output/around_well/demo_well_t4_t7_projection_segments.csv`
- `output/around_well/demo_well_t4_t7_surface_section_curves.csv`
- `output/around_well/well_trajectory_projected.csv`
- `output/around_well/section_xz_t4_t7.png` / `.svg`
- `output/around_well/section_yz_t4_t7.png` / `.svg`
- `output/around_well/section_xz_yz_t4_t7_combined.png` / `.svg`

全区尺度投影剖面输出在 `output/all`。该版本保留 `车58` 井轨迹作为参考线，但不按井距或 `half_width` 筛选 DFN 片段，纳入全部落在正式 `T4-T6/T6-T7` 层段内的最终 DFN 片段：

- `output/all/all_area_t4_t7_projection_summary.json`
- `output/all/all_area_t4_t7_projection_segments.csv`
- `output/all/all_area_t4_t7_surface_section_curves.csv`
- `output/all/well_trajectory_projected.csv`
- `output/all/section_xz_t4_t7.png` / `.svg`
- `output/all/section_yz_t4_t7.png` / `.svg`
- `output/all/section_xz_yz_t4_t7_combined.png` / `.svg`

属性体剖面输出在 `output/attribute_sections/<井名>/<section_tag>`，与 DFN 的 `around_well`、`all` 目录隔离，避免后续新增剖面类型时混淆。当前正式配置固定井 `车58`，`section_tag=anttrack_coherence_t4_t7`，输出包括：

- `well_trajectory_projected.csv`
- `attribute_section_surface_curves.csv`
- `coherence_section_xz_t4_t7_samples.csv`
- `coherence_section_yz_t4_t7_samples.csv`
- `coherence_section_xz_t4_t7.png` / `.svg`
- `coherence_section_yz_t4_t7.png` / `.svg`
- `coherence_section_xz_yz_t4_t7_combined.png` / `.svg`
- `anttrack_section_xz_t4_t7_samples.csv`
- `anttrack_section_yz_t4_t7_samples.csv`
- `anttrack_section_xz_t4_t7.png` / `.svg`
- `anttrack_section_yz_t4_t7.png` / `.svg`
- `anttrack_section_xz_yz_t4_t7_combined.png` / `.svg`
- `attribute_section_summary.json`

全矿区尺度属性体剖面输出在 `output/mine_attribute_sections/<井名>/<section_tag>`，当前配置 `section_tag=anttrack_coherence_t4_t7_mine_extent`，横向范围不再按 candidate A 裁剪，而是使用全矿区 `trace_header_xy.csv` 范围并按 `axis_sample_count=720` 重采样。

属性体配色口径：

- `AntTrack` 使用黑白色谱，当前数据中裂缝发育段 `AntTrack` 均值更低、更负，因此以黑色表示更高裂缝发育响应，白色表示弱发育或不发育响应。
- `Coherence` 使用地震记录常用红白蓝 `seismic` 色谱。

## 当前验收

最近一次过井邻域剖面运行结果：`status=pass`。

- 选中井：`车58`
- 筛选 DFN 片段：`608`
- XZ 投影片段：`238`
- YZ 投影片段：`370`
- 层段统计：`T4->T6=290`，`T6->T7=318`
- 绘制界面：`T4`、`T5`、`T6`、`T7`
- 检查项全部为 `true`：选井未使用排除井、选井位于 demo 区块、存在剖面片段、界面齐全、层段只保留正式分组

最近一次全区尺度投影剖面运行结果：`status=pass`。

- 参考井：`车58`
- 入选 DFN 片：`9609`
- 投影片段：`19218`，其中 XZ `9609`、YZ `9609`
- 层段统计：`T4->T6=6564`，`T6->T7=12654`
- 绘制界面：`T4`、`T5`、`T6`、`T7`
- 检查项全部为 `true`：存在剖面片段、界面齐全、层段只保留正式分组、未使用井距筛选、每个入选 DFN 片均生成 XZ/YZ 投影

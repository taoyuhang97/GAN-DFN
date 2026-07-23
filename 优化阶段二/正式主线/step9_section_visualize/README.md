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

车页1导眼 `candidate_cheye1` DFN-相干体剖面最终8图：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_cheye1_dfn_coherence_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_candidate_cheye1_dfn_coherence_sections.json
```

车页1导眼 `candidate_cheye1_3d` 三维密度 DFN-相干体剖面最终8图：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_cheye1_dfn_coherence_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_candidate_cheye1_3d_dfn_coherence_sections.json
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

车页1导眼地质背景扩展剖面使用配置：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/qc_geological_background_volumes.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_candidate_cheye1_geological_background_v1.json

/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_well_geological_attribute_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_candidate_cheye1_geological_background_v1.json

/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_well_seismic_amplitude_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_candidate_cheye1_geological_background_v1.json
```

该版本只使用 `AntTrack`、`CurvatureMax` 和 `SeisAmp`，不读取 `CurvaturePos`，也不生成综合曲率。输出位于 `output/candidate_cheye1_geological_background_v1/`，包含蚂蚁体、最大曲率体、振幅变密度、振幅波形+变面积的 XZ/YZ 共8张 PNG。剖面数值保存为压缩 NPZ，不展开为大体积 CSV。

全矿区尺度过车页1导眼地震振幅剖面使用配置：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step9_section_visualize/build_well_seismic_amplitude_sections.py \
  --config 优化阶段二/正式主线/step9_section_visualize/configs/formal_mine_cheye1_seismic_amplitude_v1.json
```

该配置不设置 demo `target_block`，横向使用全矿区地震道范围并重采样为720点；波形+变面积图最多显示240道。结果单独写入 `output/mine_cheye1_seismic_amplitude_v1/`。

原始道数保留版本使用 `formal_candidate_cheye1_geological_background_v2_full_traces.json`、`formal_mine_cheye1_geological_attributes_v2_full_traces.json` 和 `formal_mine_cheye1_seismic_amplitude_v2_full_traces.json`。这些配置将 `axis_sample_count=0`，并将 `wiggle_max_trace_count=0` 设置为不限制道数：蚂蚁体、相干体、最大曲率体、变密度和波形图都使用剖面横向全部原始坐标，不进行720点重采样或180/240道抽样。结果分别写入 `output/candidate_cheye1_geological_background_v2_full_traces/`、`output/mine_cheye1_geological_attributes_v2_full_traces/` 和 `output/mine_cheye1_seismic_amplitude_v2_full_traces/`。

矿区地震振幅全道分幅版本使用 `formal_mine_cheye1_seismic_amplitude_v3_full_traces_panels.json`。XZ总图宽12000像素并连续分为4幅，YZ总图宽9000像素并连续分为3幅，每幅宽3000像素；波形摆幅系数为0.6，总图和分幅图同时输出PNG与SVG。曲面几何仍为随井轨迹变化的 `XZ=(x,Ywell(t),t)`、`YZ=(Xwell(t),y,t)`。

当前层位显示统一读取 `common/horizon_trace_table/output/formal_horizon_trace_table_v1/horizon_trace_table.npy`。XZ对每个X道坐标求解 `T=Horizon[TraceIdx(x,Ywell(T))]`，YZ对每个Y道坐标求解 `T=Horizon[TraceIdx(Xwell(T),y)]`；不再使用500/900个层位采样点，也不再从四个层位文件临时建立最近邻曲线。源表层序反转不会被强制排序，相关数量写入各输出summary。

车页1导眼 `candidate_cheye1` 最终8图输出在 `output/candidate_cheye1/cheye1_dfn_coherence_sections`，只保留 PNG，不生成 SVG 或 combined 图：

车页1导眼 `candidate_cheye1_3d` 最终8图输出在 `output/candidate_cheye1_3d/cheye1_dfn_coherence_sections`，命名和图片内容与 `candidate_cheye1` 版本一致，但 DFN 输入来自 Step7B/Step8 的三维密度路径。

- `01_dfn_section_xz_t4_t7.png`
- `02_dfn_section_yz_t4_t7.png`
- `03_coherence_section_xz_t4_t7.png`
- `04_coherence_section_yz_t4_t7.png`
- `05_coherence_dfn_overlay_xz_t4_t7.png`
- `06_coherence_dfn_overlay_yz_t4_t7.png`
- `07_coherence_dfn_overlay_200m_xz_t4_t7.png`
- `08_coherence_dfn_overlay_200m_yz_t4_t7.png`
- `section_summary.json` 为验收摘要，不属于最终图片。

上述8图中的青色粗线为 Step3 `formal_rebuild/groups/车页1导眼_*.csv` 对应成像测井井段轨迹；洋红色圆点为 `GT_POINT_FLAG=1` 的原始成像测井裂缝点位，即现实井上解释裂缝映射到样本网格后的标签；洋红色短线为 `Frac_Azimuth/Frac_Dip` 投影到当前 XZ/YZ 剖面后的视倾角符号。该标签不是 Step4 专家预测后的井控裂缝点。

属性体配色口径：

- `AntTrack` 使用黑白色谱，当前统一解释口径为 `AntTrack` 高值表示裂缝发育响应更强、低值表示弱发育或不发育；显示时应使用高值黑色、低值白色，保证黑色对应更强裂缝发育响应。
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

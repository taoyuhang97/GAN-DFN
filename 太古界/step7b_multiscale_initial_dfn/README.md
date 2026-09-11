# Step 7B Multiscale Initial DFN

本目录是太古界唯一的正式 Step7B 实现目录。裂缝片几何、VTK导出和统计等公共能力已迁移到
`common/dfn_geometry`，不再保留第二个Step7B目录。

## 定位

新版 Step7B 的任务不再是只从一个三维裂缝密度体中抽样裂缝片，而是接收多尺度证据，生成带有尺度标签的初始 DFN：

- 小尺度：背景裂缝，主要来自三维裂缝密度体。
- 中尺度：裂缝带/小断层组合，主要来自蚂蚁体高值，并由低相干、曲率辅助约束。
- 大尺度：断层/断裂带，主要来自低相干连续带；后续正式版还会接入原始断层解释硬约束。

当前中尺度正式流程由 `build_medium_scale_dfn_v4.py` 实现，当前待验收配置为
`configs/taigu_step7b_medium_v5_scale_separation.json`。它只消费Step6B v5的中尺度连通体，
与小尺度Step7A、未完成的大尺度和最终融合模块分离。

Step6B v5保存的属性道号、10 ms时间轴和有效Top/Middle/Base层位窗口是Step7B的唯一上游合同。
Step7B不再重新读取OBN道号或重新执行层位填补。VTK的Z坐标直接使用TWT `TIME(ms)`。

v5将中尺度片长目标中位数设为80至100 m、上限145 m，脊线片间距设为50 m，
全局最小中心间距设为45 m。片高仍由局部裂缝带时间厚度控制，不随片长同比压缩。

## 运行

预览版 v3：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7b_multiscale_initial_dfn/build_multiscale_initial_dfn_preview.py \
  --config 优化阶段二/正式主线/step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_multiscale_preview_v3.json
```

预览版 v2：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7b_multiscale_initial_dfn/build_multiscale_initial_dfn_preview.py \
  --config 优化阶段二/正式主线/step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_multiscale_preview_v2.json
```

早期预览版 v1：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7b_multiscale_initial_dfn/build_multiscale_initial_dfn_preview.py \
  --config 优化阶段二/正式主线/step7b_multiscale_initial_dfn/configs/formal_candidate_cheye1_multiscale_preview_v1.json
```

## 当前预览版输入

- 当前 `lowcoh_steep` 三维密度体。
- 相干体。
- 蚂蚁体。
- 最大曲率体。
- 最大正曲率体。
- T4-T7 层位。

## 输出

默认输出目录：

`output/candidate_cheye1_multiscale_preview_v3/`

主要文件：

- `initial_dfn_fracture_patches.csv`
- `initial_dfn_raw_time.vtk`
- `initial_dfn_generation_audit.csv`
- `fracture_band_centerlines_raw_time.vtk`
- `fracture_band_summary.csv`
- `initial_dfn_summary.json`

## 与早期 Step7B 原型的差异

- 旧 Step7B：从一个三维密度体中抽样，尺度标签主要由连通体形态后判别。
- 新 Step7B：先按小/中/大尺度证据分组，再分别生成裂缝片。
- 早期原型不再作为独立步骤保留，可复用能力已归入公共几何模块。

## v2 修正点

- 大中尺度裂缝片倾角不再固定为单一值，而是根据局部带状结构和属性强度动态计算。
- `fracture_band_centerlines_raw_time.vtk` 改为连接最终裂缝片中心点，和实际 DFN 片位置保持一致。
- 大中尺度裂缝片中心距按片长控制，减少同一 band 内裂缝片相互脱节的问题。

## v3 修正点

- 大中尺度裂缝片的倾向/倾角改为由局部候选裂缝带点云 PCA 计算，不再预设固定倾角。
- 裂缝片大小由局部发育带宽度、局部时间厚度和局部密度共同控制。
- 同时输出两类中心线：`fracture_band_centerlines_raw_time.vtk` 表示候选裂缝带骨架，`fracture_patch_centerlines_raw_time.vtk` 表示最终裂缝片中心线。
- 当前 v3 暴露出一个上游问题：大尺度低相干候选中存在近水平层状结构，导致局部 PCA 计算出低倾角裂缝片。这不应在 Step7B 中强行改成陡倾角，后续应在 Step6C 大尺度先验中先过滤层界面。

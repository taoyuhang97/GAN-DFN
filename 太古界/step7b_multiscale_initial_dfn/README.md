# Step 7B Multiscale Initial DFN

本目录是新版 Step7B 的独立实现目录，旧目录 `step7b_initial_dfn_3d` 保留不动。

## 定位

新版 Step7B 的任务不再是只从一个三维裂缝密度体中抽样裂缝片，而是接收多尺度证据，生成带有尺度标签的初始 DFN：

- 小尺度：背景裂缝，主要来自三维裂缝密度体。
- 中尺度：裂缝带/小断层组合，主要来自蚂蚁体高值，并由低相干、曲率辅助约束。
- 大尺度：断层/断裂带，主要来自低相干连续带；后续正式版还会接入原始断层解释硬约束。

当前脚本是预览版，目的是先看新版 Step7B 的 DFN 视觉效果。正式 Step6A/Step6B/Step6C/Step6D 完成后，再把输入切换为 Step6D 输出的多尺度密度包。

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

## 与旧 Step7B 的差异

- 旧 Step7B：从一个三维密度体中抽样，尺度标签主要由连通体形态后判别。
- 新 Step7B：先按小/中/大尺度证据分组，再分别生成裂缝片。
- 旧 Step7B 目录和输出不被覆盖。

## v2 修正点

- 大中尺度裂缝片倾角不再固定为单一值，而是根据局部带状结构和属性强度动态计算。
- `fracture_band_centerlines_raw_time.vtk` 改为连接最终裂缝片中心点，和实际 DFN 片位置保持一致。
- 大中尺度裂缝片中心距按片长控制，减少同一 band 内裂缝片相互脱节的问题。

## v3 修正点

- 大中尺度裂缝片的倾向/倾角改为由局部候选裂缝带点云 PCA 计算，不再预设固定倾角。
- 裂缝片大小由局部发育带宽度、局部时间厚度和局部密度共同控制。
- 同时输出两类中心线：`fracture_band_centerlines_raw_time.vtk` 表示候选裂缝带骨架，`fracture_patch_centerlines_raw_time.vtk` 表示最终裂缝片中心线。
- 当前 v3 暴露出一个上游问题：大尺度低相干候选中存在近水平层状结构，导致局部 PCA 计算出低倾角裂缝片。这不应在 Step7B 中强行改成陡倾角，后续应在 Step6C 大尺度先验中先过滤层界面。

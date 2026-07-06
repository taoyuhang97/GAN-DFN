# Step 7C Fused DFN

本步骤融合二维层段密度网格 DFN 与三维密度体 DFN，输出兼容 Step8 的初始 DFN。

## 输入

- 二维密度网格 DFN：`step7_initial_dfn/output/candidate_cheye1/initial_dfn_fracture_patches.csv`
- 三维密度体 DFN：`step7b_initial_dfn_3d/output/candidate_cheye1/initial_dfn_fracture_patches.csv`

## 逻辑

- 三维 DFN 全部保留，标记为 `FusionSource=3d_primary`。
- 二维 DFN 按同层段三维邻域裂缝片进行姿态纠正，方位角使用 0-180 轴向双角加权平均，倾角使用加权平均。
- 二维片如果缺少足够三维邻居，则仅保留高密度分位数以上的补充片。
- 二维补充片与三维主片过近时删除，避免重复计数。
- 只输出 `raw_time` VTK，不输出 display VTK。

## 运行

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step7c_fused_dfn/build_fused_dfn_from_2d_3d.py \
  --config 优化阶段二/正式主线/step7c_fused_dfn/configs/formal_fused_candidate_cheye1.json
```

## 输出

默认输出目录：`output/candidate_cheye1`。

- `fused_initial_dfn_fracture_patches.csv`
- `fused_initial_dfn_raw_time.vtk`
- `fused_initial_dfn_generation_audit.csv`
- `fused_initial_dfn_summary.json`

后续 Step8 可将 `initial_dfn_csv` 指向 `fused_initial_dfn_fracture_patches.csv`。

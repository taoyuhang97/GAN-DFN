# Step 6B 3D Density SGY

本步骤是 Step6 的并行增强版本，先面向 `candidate_cheye1` 构造真正的 `X-Y-T` 三维裂缝密度体。

## 运行

小规模 smoke test：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step6b_demo_density_volume_3d/build_candidate_3d_density_sgy.py \
  --config 优化阶段二/正式主线/step6b_demo_density_volume_3d/configs/formal_candidate_cheye1_3d_density_sgy.json \
  --max-train-chunks 1 \
  --max-target-traces 5 \
  --output-suffix _smoke
```

正式后台运行建议用 tmux：

```bash
tmux new -s step6b_cheye1_3d
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step6b_demo_density_volume_3d/build_candidate_3d_density_sgy.py \
  --config 优化阶段二/正式主线/step6b_demo_density_volume_3d/configs/formal_candidate_cheye1_3d_density_sgy.json \
  2>&1 | tee 优化阶段二/正式主线/step6b_demo_density_volume_3d/output/candidate_cheye1/run.log
```

## 模型逻辑

- 训练输入：`step5b_unified_samples_t4_t7/output/unified_t4_t7_density_samples.csv`。
- 正式运行默认遍历全部真实测井和虚拟测井样本，不使用旧 Step6 的层段模型。
- 按 `沙三段`、`沙四段` 分别训练逐时间样点 `HistGradientBoostingRegressor`。
- 特征包括当前时间样点的地震属性、层段编码、层内相对时间、距层顶/层底时间和层厚。
- 样本权重：真实井样本权重更高，虚拟井样本降权，避免虚拟样本数量主导训练。

## 输出

正式输出目录：`output/candidate_cheye1`。

- `candidate_cheye1_3d_predicted_density.sgy`：三维裂缝密度 SGY，只包含 demo 区内 traces。
- `candidate_cheye1_3d_trace_mapping.npz`：输出 trace 序号到原始 `TraceIdx/X/Y/IX/IY` 的轻量映射。
- `candidate_cheye1_3d_density_models.joblib`：分层模型和特征中位数。
- `candidate_cheye1_3d_density_sgy_summary.json`：训练、采样轴、SGY 和 QC 摘要。

不输出逐点 CSV。SGY 继承原始地震体 trace header，只写入 `candidate_cheye1` demo 区内 trace；demo 区外 trace 不写入输出文件。

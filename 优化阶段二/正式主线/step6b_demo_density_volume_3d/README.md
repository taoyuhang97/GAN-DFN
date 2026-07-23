# Step 6 三维密度与多尺度先验

本目录先面向 `candidate_cheye1` 构造 `X-Y-T` 基础三维裂缝密度体，再拆分为 Step6A 小尺度背景、Step6B 中尺度裂缝带、Step6C 大尺度断层/断裂带和 Step6D 多尺度整合。

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

## 当前多尺度主线

- Step6A：`build_step6a_small_background.py`
- Step6B：`build_step6b_medium_corridor_prior.py`
- Step6C：`build_step6c_large_fault_prior.py`
- Step6D：`build_step6d_multiscale_bundle.py`

当前检查点的 Step6B 使用蚂蚁体主分支和独立陡倾低相干分支。低相干候选仍需通过垂向延伸、倾角、线性和层状异常过滤，不能把所有低相干区域直接转换为裂缝带。

当前统一 Step6D 输出：

`output/candidate_cheye1_final_lowcoh_vertical_v1/step6d_bundle/`

现有输入对齐和定位 QC 已通过。当前基础密度可以作为小尺度背景使用，但真正的留一井重新训练验证尚未执行，因此不能据此宣称模型已经具备充分的跨井泛化能力。

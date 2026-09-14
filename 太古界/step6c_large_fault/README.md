# Step6C 太古界大尺度断层先验

本步骤生成大尺度断裂先验，不直接生成最终 DFN 面片。输入网格统一使用
`太古界/common/trace_contract` 的曲率体属性网格，时间窗口统一为三层位合同：
上部复合层顶 `T-a-1` -> 太古界顶 `Art_1` -> 风化壳底 `Art_d1-1`。

## 两类来源

- 人工解释断层：读取 `common/fault_preprocessing_300m/patches`，按 `FaultName`
  选择目标区域内的 300 m 片，保留原始三角面，并栅格化为硬约束。
- 属性推断断裂：使用相干体、蚂蚁体和曲率体。相干体低值为主证据，蚂蚁体和曲率体为辅助证据。
  AntTrack 的 `-1` 是有效弱响应，不作为缺失值删除。

人工断层与属性推断断裂分别输出，靠近且方向一致的属性组件只作为人工断层的支持，避免重复计数。
本步骤不使用地震振幅，也不把 Step6A 密度值作为大尺度断裂评分依据。

## 运行

```bash
python code/build_step6c_large_fault_prior.py \
  --config configs/formal_candidate_cheye1_multiscale_density_v1.json
```

配置中的 `large_evidence` 明确使用绝对 `1800-4800 ms`、2 ms 采样轴，不能改回砂砾岩的 T4-T7 轴。

主要结果包括 `original_fault_*`、`inferred_fault_*` 和合并后的 `large_fault_*` SGY/NPZ/VTK/CSV，
以及 `large_fault_qc.json`。

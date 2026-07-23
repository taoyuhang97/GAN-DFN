# 逐地震道层位表

该目录构造可被后续剖面任务复用的逐道T4-T7层位时间表。

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/common/horizon_trace_table/build_horizon_trace_table.py \
  --config 优化阶段二/正式主线/common/horizon_trace_table/configs/formal_horizon_trace_table_v1.json
```

正式输出为结构化NumPy文件 `horizon_trace_table.npy`，字段为 `TraceIdx/T4/T5/T6/T7`，时间单位为TWT ms。表按 `TraceIdx` 排序，行号等于 `TraceIdx`，可使用 `np.load(path, mmap_mode="r")` 按道查询。

T4由短名解释点进行8邻点IDW插值；T5-T7使用长名平滑层位文件，先匹配到地震网格，少量缺失格点再取最近有效层位格点。该版本不保存层位来源距离、有效掩膜或层序质量数组，也不强制修改层位顺序。

正式配置默认启用 `tqdm`。T4插值和T5-T7长文件映射均显示完成比例、处理速度和预计剩余时间；未安装 `tqdm` 时自动退回逐块文本进度。

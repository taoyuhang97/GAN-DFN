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

## 10 km v2层位合同

`build_horizon_contract_v2.py`以v1逐道表为不可变输入，生成`formal_horizon_trace_table_v2`。它只在10 km目标区处理已确认的`T7<=T6`异常：保留T6、令T7=T6并显式设置`ShasiPresent=0`，不会交换层位或人为增加沙四厚度。目标区外的层位值保持不变。

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/common/horizon_trace_table/build_horizon_contract_v2.py \
  --config 优化阶段二/正式主线/common/horizon_trace_table/configs/formal_horizon_trace_table_v2.json
```

v2表新增`SurfaceOrderValid/ShasanPresent/ShasiPresent/HorizonCorrectionCode`字段，并生成修正审计、倒置连通组件、车571/车572邻域QC、厚度图，以及2 ms和10 ms的紧凑层窗索引NPZ。紧凑索引记录每条道各地层的起止样点，不展开为占用数百MB的三维布尔数组。

`horizon_contract.py`是Step6-Step9唯一的程序化层位入口。它负责按`TraceIdx`精确对齐、校验2/10 ms样点轴、分块应用T4-T7掩膜、生成Step7网格层位以及提供Step8/Step9的X/Y查询。统一接入可运行：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/common/horizon_trace_table/qc_wp1_horizon_integration.py \
  --master-config 优化阶段二/正式主线/step6b_demo_density_volume_3d/configs/formal_demo_10km_multiscale_flow_v2.json \
  --output 优化阶段二/正式主线/common/horizon_trace_table/output/formal_horizon_trace_table_v2/wp1_horizon_integration_qc.json
```

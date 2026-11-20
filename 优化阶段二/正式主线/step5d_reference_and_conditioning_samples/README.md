# 高保真裂缝参考模型与直接条件化样本

本步骤从砂砾岩、太古界 Step8 最终 DFN 中构造两套互不混用的数据产品。

## 数据产品

### 高保真裂缝参考模型

- 砂砾岩：200 组，`优化阶段二/正式主线/output/high_fidelity_reference_models_sand_v1/`。
- 太古界：40 组，`太古界/output/high_fidelity_reference_models_taigu_v1/`。
- 一组对应一条真实井或虚拟井轨迹，以及与该轨迹真实相交的全部 Step8 面片。
- 严格排除此前 416/208 地震网格样本使用的 16 口砂砾岩源井和 8 口太古界源井及其虚拟井。
- 交切采用井轨迹相邻线段与面片两个三角形的三维精确求交，不使用中心距离阈值代替。

### 直接条件化样本

- 砂砾岩：80 组，`优化阶段二/正式主线/output/direct_conditioning_samples_sand_v1/`。
- 太古界：40 组，`太古界/output/direct_conditioning_samples_taigu_v1/`。
- 每组对应一个 300 m x 300 m XY 单元。面片按中心坐标唯一归入单元，因此不同样本之间不重复面片。
- 单元及其外扩 100 m 安全缓冲区内不得出现 Step8 正式真实井轨迹点。
- 在满足面片数和尺度数门槛的单元中，采用确定性空间最远点策略选样，使样本覆盖整个矿区，而不是集中到少数高密度位置。

## 输出合同

高保真根目录：

```text
candidate_index.csv       所有剩余井轨迹的实际交切检查和是否入选
model_index.csv           200/40 个正式模型及尺度统计
models/<GroupID>/
  patches.csv             正式数据记录，保留 Step8 原始全部列
  trajectory.csv          该组井轨迹 X/Y/TIME
  fractures.vtk           裂缝片可视化文件
  trajectory.vtk          井轨迹可视化文件
  metadata.json           来源、PatchID、尺度、时间范围等审计信息
summary.json
config_resolved.json
COMPLETE
```

直接条件化根目录：

```text
candidate_cells.csv       所有占用单元、井距、资格和入选状态
sample_index.csv          80/40 个正式样本及尺度统计
samples/<SampleID>/
  patches.csv             正式数据记录，保留 Step8 原始全部列
  fractures.vtk           裂缝片可视化文件
  metadata.json           单元边界、井距、PatchID、时间范围等审计信息
summary.json
config_resolved.json
COMPLETE
```

`patches.csv` 是机器训练和后续处理使用的正式记录；VTK 只用于检查模型形态，不进入样本计数。

## 运行

完整流程：

```bash
bash 优化阶段二/正式主线/step5d_reference_and_conditioning_samples/run_formal_reference_and_conditioning_v1.sh
```

完成后运行验收器：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step5d_reference_and_conditioning_samples/validate_fracture_reference_samples.py \
  --report 优化阶段二/正式主线/step5d_reference_and_conditioning_samples/formal_validation_report.json
```

单组小规模验证：

```bash
/home/tyh/anaconda3/envs/gan-dfn/bin/python \
  优化阶段二/正式主线/step5d_reference_and_conditioning_samples/build_fracture_reference_samples.py \
  --config 优化阶段二/正式主线/step5d_reference_and_conditioning_samples/configs/formal_sand_direct_conditioning_v1.json \
  --target-count 3 \
  --output-dir /tmp/direct_sand_demo \
  --overwrite
```

正式构建默认拒绝覆盖已有目录；仅在所有输入和输出范围已确认时使用 `--overwrite`。运行脚本会使用该选项重新构建四个正式目录。

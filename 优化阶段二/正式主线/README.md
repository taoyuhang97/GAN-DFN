# 优化阶段二 正式主线

本目录只存放当前项目最终确认保留的正式代码。

这里不保存：

- 中间调试脚本
- 过程性验证脚本
- 已确认废弃的方法原型
- 仅用于思考和对比的临时产物

正式主线的目标是：

- 从原始数据整理开始
- 到真实井裂缝预测
- 到 Step5A 正式单源虚拟测井构造
- 到 Step5B 真实井与虚拟井 `T4-T7` 统一样本
- 到 demo 裂缝密度体
- 到初始 DFN
- 到井控校正后的 DFN
- 到最终剖面可视化

当前正式执行顺序：

```text
step1_surface_framework
  -> step2_real_well_t4_t7_samples
  -> step3_imaging_supervision_samples
  -> step4_expert_real_well_prediction
  -> step5a_single_source_virtual_wells
  -> step5b_unified_samples_t4_t7
  -> step6_demo_density_volume
  -> step7_initial_dfn
  -> step8_dfn_well_correction
  -> step9_section_visualize
```

迁移原则：

- 每一批代码迁入前，必须先通过“有效性 + 必要性 + 口径一致性”检查
- 未通过检查的代码，不进入本目录

当前本目录仍在建设中。

## 当前保留范围

本目录当前保留的内容只包括：

- 正式脚本
- 正式配置
- 步骤说明文档
- 必要的目录占位文件

以下内容不再保留在正式主线代码区：

- 已确认错误放入正式主线的调试链路
- 已废弃的旧入口脚本
- 与当前正式输入契约不一致的旧输出目录
- `__pycache__` 等解释器缓存

## GitHub 上传口径

正式主线下各步骤 `output/` 属于生成产物，不作为 GitHub 上传内容。原因：

- 体积大
- 可由正式脚本重新生成
- 容易混入历史结果，干扰正式代码目录

上传 GitHub 时，默认只上传：

- `README.md`
- `MIGRATION_RULES.md`
- 各 step 的 `*.py`
- 各 step 的 `configs/*.json`
- 各 step 的步骤说明文档

不上传：

- 各 step `output/`
- 本地缓存
- 历史调试结果

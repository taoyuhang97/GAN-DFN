# 第一轮优化阶段代码补充说明（2026-05-04）

## 1. 文档目的

本文档用于说明本次上传到 `upload/first-round-improvement-20260420` 分支的代码范围、核心改动点以及明确排除项，避免把本地运行残留、会话记录或临时评估结果误传到正式仓库。

本次说明只对应“第一轮修改”主线，不回顾最初可行性实验方案的旧实现。

## 2. 本次上传范围

本次上传主要覆盖以下三类正式代码：

### 2.1 分层 GAN / 监督基线主链补充

位于：

- `研究内容三/优化阶段一/G_DFN监督基线/`

本次补充和修改的重点包括：

- 增加按岩层组织模型、训练和推理的主流程脚本。
- 增加分层模型注册表，用于集中管理不同岩层的模型路径、阈值和覆盖关系。
- 增加固定单元划分配置，避免运行期间样本划分漂移。
- 调整训练、推理、评估和生产批量生成逻辑，使其能够适配“先分层、后层内生成、再单元拼接”的流程。
- 补充后台批量运行脚本，统一当前第一轮优化阶段的执行入口。

对应文件包含：

- `baseline_common.py`
- `build_unit_level_split.py`
- `infer_and_evaluate_baseline.py`
- `infer_production_units_baseline.py`
- `train_supervised_baseline.py`
- `layer_model_registry.py`
- `train_layerwise_models.py`
- `run_layerwise_full_pipeline.py`
- `start_layerwise_full_pipeline_nohup.sh`
- `start_demo_bx6_13_by28_35_nohup.sh`
- `fixed_unit_splits/demo_bx6_13_by28_35_fixed_split.csv`

### 2.2 数据打包与实例表达兼容补充

位于：

- `研究内容三/优化阶段一/GAN训练准备/训练样本打包/`
- `研究内容三/优化阶段一/DFN体素互转实验/`
- `研究内容三/优化阶段一/DFN实例表达互转实验/`

本次改动主要用于配合新的分层样本组织与实例表达流程，保证训练样本、推理结果以及 VTK/实例表达之间的字段兼容。

对应文件包含：

- `build_sparse_instance_gan_dataset.py`
- `roundtrip_common.py`
- `instance_roundtrip_common.py`

### 2.3 单元融合、区域后处理与断层后融合改造

位于：

- `研究内容三/优化阶段一/单元DFN融合/`
- `研究内容三/优化阶段一/单元DFN融合/区域断层后融合/`

本次改动重点包括：

- 调整单元后处理与多尺度后处理逻辑。
- 将“零裂缝单元”视为合法结果，避免流程在空结果处异常终止。
- 收紧部分层段的生成与解码约束，尤其是底部层段的单独阈值控制。
- 调整区域断层后融合逻辑，扩展断层影响边界与断层控制区域。
- 增加断层影响展示导出脚本，便于单独查看“断层及其影响裂缝片”。
- 调整最终导出逻辑，使 VTK 中用于着色的 `patchArea` 更贴近最终保留裂缝片本身，而不是混入前处理阶段的历史面积信息。

对应文件包含：

- `geophysical_postprocess.py`
- `geophysical_postprocess_multiscale.py`
- `run_regional_postprocess_multiscale.py`
- `build_fault_surface_fragments_from_raw_patches.py`
- `fault_postfusion_common.py`
- `fuse_faults_into_regional_dfn_v2.py`
- `run_regional_fault_postfusion_pipeline_v2.py`
- `export_fault_influence_display_vtk.py`

## 3. 当前改动逻辑摘要

从当前第一轮修改主线看，代码逻辑已经从“统一单模型直接生成单元 DFN”转向“以地层/岩层约束为核心的分层生成与后融合”：

1. 先按新的层位文件完成单元与样本的层段拆分。
2. 再按岩层差异组织训练样本，并训练多个层内模型。
3. 推理时先对目标单元做相同的层段拆分，再调用对应层内模型生成该层的裂缝结果。
4. 将层内结果回拼成完整单元 DFN。
5. 对批量单元结果执行区域拼接、多尺度后处理和断层后融合。
6. 最终输出适合后续展示和再处理的区域 DFN 结果。

这条链路的目标不是单纯追求模型分数，而是让最终区域 DFN 的层内裂缝发育强度、空间分布和断层影响关系更接近地质解释需求。

## 4. 本次明确不上传的内容

以下内容属于本地记录、临时评估或运行产物，本次明确不纳入正式提交：

- `record/`
- `tmp_eval_smoke/`
- `/data/shared/project-oil/...` 下的输入数据、运行结果和预览产物
- `__pycache__/`
- `_tmp*.py`
- `tmp_syntax_check.py`
- `error_log.txt`

其中：

- `record/` 用于保留协作过程记录，不属于正式流程代码。
- `tmp_eval_smoke/` 属于本地临时评估目录，不应作为正式结果上传。
- `/data/shared/project-oil/...` 为服务器数据与运行输出目录，本就不应进入 Git 仓库。

## 5. 提交建议

本次建议只提交正式代码、正式脚本和正式说明文档，不提交本地运行残留。  
若后续继续推进区域级 DFN 结果优化，应继续沿用这一边界，避免仓库被中间结果污染。

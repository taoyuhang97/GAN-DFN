# GitHub上传记录 2026-03-22

## 背景

- 仓库路径：`D:\Code\Python\石油项目\断缝储\断缝储_裂缝样本生成`
- 当前分支：`imaging_well_to_fracture`
- 远程仓库：`origin = https://github.com/taoyuhang97/GAN-DFN.git`
- 记录时间：2026-03-22

## 本次上传范围

### 第一阶段：裂缝发育段预测

- `模型训练/优化阶段一/裂缝存在性分析/LSTM/imaging_well_to_fracture_cnn_lstm_test_1.py`
  - 支持按环境变量切换实验参数
  - 支持基于井距离选择训练井
  - 增加缺失测井值清洗
  - 增加基于 `accuracy / F1 / IoU` 的阈值搜索与模型选择
  - 支持关闭密度回归、关闭域对抗
  - 输出训练井、验证井与配置到实验目录
- `模型训练/优化阶段一/裂缝存在性分析/data_distribution_analysis.py`
  - 支持按环境变量筛选分析步骤
  - 增加热力图绘制兼容逻辑
  - 支持按环境变量覆盖测井特征列表

### 第二阶段：裂缝点位细化

- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/build_segment_count_dataset.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/fracture_density_predict.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/fracture_point_refine_by_density.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/segment_count_predict_then_refine.py`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/raw_point_guided_segment_refine.py`
  - 已实现段级总量标定
  - 已实现边界补偿 `prob_shoulder`
  - 已实现选择性后处理下调
  - 已增加 `count_error_pct / overall_count_error_pct`
  - 已加入“基于井距离的训练加权”开关：
    - `--train-distance-weight-mode`
    - `--train-distance-csv`
    - `--train-distance-weight-power`
    - `--train-distance-weight-min`
    - `--train-distance-weight-max`
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/nearest_imaging_well_predict_then_refine.py`
  - 支持先选最近成像井模型，再串联第一阶段与第二阶段预测
- `模型训练/优化阶段一/裂缝位置分析/基于密度的裂缝点位分析/裂缝点位精细化目标与评价标准.txt`

### 实验辅助脚本

- `run_lstm_experiment.py`
- `run_density_experiment.py`
- `run_point_refine_experiment.py`
- `rebuild_experiment_docx_20260319.py`
- `_append_density_result_docx.py`

### 文档

- `docs/session_handoff_2026-03-19.md`
- `docs/github_upload_2026-03-22.md`

## 当前阶段性结论

- 第一阶段仍采用“按井留一验证 + 基于距离筛训练井”的逻辑。
- 第二阶段当前主方案不是逐点密度回归，而是：
  1. 用第一阶段预测裂缝发育段；
  2. 对预测段做段级点数标定；
  3. 在段内按概率形状分配点位。
- 第二阶段在加入边界补偿和选择性后处理后，代表实验为：
  - `raw_point_tweedie_bexpand0403_sel6602_dual_pmpl_probmean_q70_predcap060_v1`
- 在此基础上加入“训练井距离加权”后，代表实验为：
  - `raw_point_tweedie_bexpand0403_sel6602_dual_pmpl_probmean_q70_predcap060_distw_inv070130_v1`
- 目前距离加权带来小幅正向收益：
  - `overall_count_error_pct: 22.77% -> 21.78%`

## 未纳入 Git 的内容

- 本地临时运行结果目录：`_local_experiment_runs/`
- Python 缓存：`__pycache__/`、`*.pyc`
- 临时语法检查脚本：`_tmp*.py`、`tmp_syntax_check.py`

## 提交与推送

- 提交哈希：待提交
- 提交说明：待提交
- 推送结果：待推送

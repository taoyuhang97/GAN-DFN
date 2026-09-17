# Step6D 太古界多尺度密度融合

本目录只负责融合 Step6A、Step6B、Step6C 的独立输出，不负责重新识别裂缝或断层。
融合权重和去重规则待各尺度验证完成后单独确认。

## v4 正式口径（2026-09-17）

配置：`configs/taigu_step6d_multiscale_v4.json`（`compact_multiscale_flow=false`，完整模式）

| 项 | 内容 |
|---|---|
| 小尺度输入 | `taigu_step6a_small_background_v4`（0.4 模型密度体 + 0.6 曲率体的融合分数，见问题记录 0.24） |
| 中尺度输入 | `taigu_medium_v7_anttrack_led_v4`（Step6B 蚂蚁体主导的裂缝带） |
| 大尺度输入 | `taigu_step6c_large_v3_attribute_v3_v4`（Step6C 断层先验；时间窗已修正为 2624–4048 ms） |
| 尺度权重 | small 0.35 / medium 0.55 / large 1.0 |
| 时间轴 | 统一重采样到 Step6A 的 2 ms 轴（6B/6C 原为 10 ms） |
| 主产物 | `final_small_density.sgy`（供 Step7A）、`small_background_score.sgy`、`scale_label.sgy`、`multiscale_bundle_summary.json` |
| 契约字段 | summary 中 `model_contract_version=taigu_step6a_attribute_v3_v4`、`output_paths.density_sgy` |

Step7A 的小尺度密度体输入已由 Step6A 模型密度体切换为 **本目录的 `final_small_density.sgy`**。

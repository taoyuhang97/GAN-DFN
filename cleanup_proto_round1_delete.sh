#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--execute" ]]; then
  echo "Dry run only. Use --execute to delete listed paths."
  DRY_RUN=1
else
  DRY_RUN=0
fi

DELETE_PATHS=(
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容一/XGboost/裂缝存在预测"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容一/测井"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容一/井斜"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容三/3D_GAN输入"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容三/3D_GAN输出/体素文件"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容三/训练用数据"
  "/data/shared/project-oil/wx数据/砂砾岩/研究内容二/单元实验"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝位置预测"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/裂缝角度预测"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/井斜"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/测井"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/outer_holdout"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/outer_holdout_uniform_0p2ms"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/stage1_lstm"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/stage1_lstm_uniform_0p2ms"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/stage2_refine"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/stage2_refine_uniform_0p2ms"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容一/两阶段裂缝预测流程结果/常规测井验证"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/虚拟测井批量生成"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/虚拟裂缝预测"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/虚拟裂缝预测_补充空白单元_20260606"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/processed_0.2s_all_wells_collums"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/虚拟测井构建/相对深度重采样数据"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容二/单元DFN构建/批量生成_新层位重拆分"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/GAN训练准备"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/后台全流程"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/临时对比"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/快速子集"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/调参验证"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/G_DFN监督基线/推理评估"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/DFN体素互转实验"
  "/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/DFN实例表达互转实验"
  "/data/shared/project-oil/GAN-step3-result-20260511/visualization"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/eval"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/logs"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/merge"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/postprocess"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4_apply_check_BX38_BY8_BX39_BY8"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4_apply_check_BX38_BY8_BX39_BY8_sparsecap"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4_loop_full"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4_loop_full_iter04_05"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_rebalance_v4_plan_check_sparsecap"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_transition_smoothed"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_transition_smoothed_strong"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/boundary_transition_smoothed_v2"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/cross_boundary_smoothed"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/distribution_resampled"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/final_dfn_boundary_rebalance_v4_full"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/final_scale_split_dfn"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/final_scale_split_dfn_v2"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/final_scale_split_dfn_v3"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/full_infer_layer_density"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/orientation_dehomogenized_spatial"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/orientation_dehomogenized_v2"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/p32_calibration_conservative"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/p32_calibration_spatial_r2"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/postprocess_progress_smoke"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/postprocess_quality"
  "/data/shared/project-oil/GAN-step3-result-20260511/full113_conf_up_traininfer_20260506_142956/analysis/zero_layer_filled_v2"
)

allowed_path() {
  local path="$1"
  [[ "$path" == /data/shared/project-oil/wx数据/砂砾岩/研究内容一/* ]] && return 0
  [[ "$path" == /data/shared/project-oil/wx数据/砂砾岩/研究内容二/* ]] && return 0
  [[ "$path" == /data/shared/project-oil/wx数据/砂砾岩/研究内容三/* ]] && return 0
  [[ "$path" == /data/shared/project-oil/wx数据/砂砾岩/优化阶段一/* ]] && return 0
  [[ "$path" == /data/shared/project-oil/GAN-step3-result-20260511/* ]] && return 0
  return 1
}

for path in "${DELETE_PATHS[@]}"; do
  if ! allowed_path "$path"; then
    echo "REFUSE unsafe path: $path" >&2
    exit 2
  fi
done

echo "Deletion candidates: ${#DELETE_PATHS[@]}"
for path in "${DELETE_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    du -sh "$path" 2>/dev/null || true
  else
    echo "missing $path"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

echo "Deleting listed paths..."
for path in "${DELETE_PATHS[@]}"; do
  if [[ -e "$path" ]]; then
    echo "rm -rf $path"
    rm -rf -- "$path"
  fi
done
echo "Deletion complete."

# DFN 体素互转实验

默认代码目录：

- `研究内容三/优化阶段一/DFN体素互转实验`

默认结果目录：

- `/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/研究内容三/DFN体素互转实验`

默认实验记录文档：

- `/data/shared/project-oil/wx数据/砂砾岩/优化阶段一/实验记录/实验记录20260328.docx`

脚本：

- `dfn_patches_to_voxel.py`
  - 单元裂缝片参数表转体素。
- `voxel_to_dfn_patches.py`
  - 体素回拟合裂缝片参数表。
- `run_dfn_voxel_roundtrip_experiment.py`
  - 一键执行 `DFN -> 体素 -> DFN` round-trip，并把总结追加到实验记录文档。

推荐先用一键脚本：

```powershell
py -3.12 .\研究内容三\优化阶段一\DFN体素互转实验\run_dfn_voxel_roundtrip_experiment.py --unit-id BX69_BY28 --xy-resolution 24 --z-step-ms 0.2 --thickness-vox 1.0 --channels occupancy
```

输出目录结构：

- `<output_root>\<run_name>\<UnitID>\input_patches.csv`
- `<output_root>\<run_name>\<UnitID>\input_patches_raw_time.vtk`
- `<output_root>\<run_name>\<UnitID>\input_patches_display.vtk`
- `<output_root>\<run_name>\<UnitID>\orig_voxel_volume.npz`
- `<output_root>\<run_name>\<UnitID>\orig_voxel_preview.vtk`
- `<output_root>\<run_name>\<UnitID>\roundtrip_patches.csv`
- `<output_root>\<run_name>\<UnitID>\roundtrip_patches_raw_time.vtk`
- `<output_root>\<run_name>\<UnitID>\roundtrip_patches_display.vtk`
- `<output_root>\<run_name>\<UnitID>\patch_compare_raw_time.vtk`
- `<output_root>\<run_name>\<UnitID>\patch_compare_display.vtk`
- `<output_root>\<run_name>\<UnitID>\patch_vtk_mappings.json`
- `<output_root>\<run_name>\<UnitID>\roundtrip_voxel_volume.npz`
- `<output_root>\<run_name>\<UnitID>\roundtrip_voxel_preview.vtk`
- `<output_root>\<run_name>\<UnitID>\roundtrip_summary.json`
- `<output_root>\<run_name>\<UnitID>\roundtrip_config.json`

说明：

- `z-step-ms` 默认按当前 DFN 原始时间采样 `0.2ms` 设置。
- 这里的 `z-step-ms` 是 DFN 标签体素化分辨率，不必和后续地震输入的 `1ms` 采样间隔一致。

# 10 km多尺度DFN流程节点记录（2026-07-28）

## 节点范围

- 分支：`second-round-optimization`
- 正式版本：`formal_demo_10km_multiscale_flow_v1`
- Demo范围：X `571250-581250`，Y `4196500-4206500`
- Step6A保持2 ms；Step6B、Step6C及多尺度上下文使用10 ms。
- Step6A证据权重由配置控制：连续裂缝密度50%，普通曲率体50%。

## 当前流程状态

- Step6A、Step6B、Step6C、Step6D均已通过正式验收。
- Step7A小尺度DFN：97,396片。
- Step7B中尺度DFN：47片，覆盖20/20个候选组件。
- Step7C大尺度DFN：12,763片，其中断层及影响带12,724片、低相干补充39片。
- Step7D融合结果：110,206片。
- Step8井控校正结果：110,394片，263个井控片，全部验收项通过。
- Step9车页1正式20张矿区/井周剖面已生成并通过验收。

## 本节点关键修改

1. 将10 km流程的Step6B/C改为10 ms紧凑组件处理，避免5.25亿体素上的重复全体扫描。
2. Step6D与Step7A-D改用紧凑中间合同，减少内存、磁盘和重复扫描。
3. Step7B改为每个候选组件至少生成一个裂缝片，并增加空间最远点抽样、局部PCA和整体PCA回退。
4. Step7C使用组件分组和面板化生成断层面、损伤带及低相干补充片。
5. Step8对有限但倒置的局部层位窗进行排序修复，再纳入硬井控中心；本次修复2片。
6. Step7A、Step7B、Step7C正式配置开启三维VTK输出，并提供从现有CSV直接导出的脚本。

## 三维内部检查产品

- Step7A：`step7a_small_scale_dfn/output/formal_demo_10km_multiscale_flow_v1/small_dfn_raw_time.vtk`
- Step7B：`step7b_multiscale_initial_dfn/output/formal_demo_10km_multiscale_flow_v1/medium_dfn_raw_time.vtk`
- Step7C完整结果：`step7c_large_fault_dfn/output/formal_demo_10km_multiscale_flow_v1/large_fault_and_damage_raw_time.vtk`
- Step7C断层及影响带：`large_fault_only_and_influence_raw_time.vtk`
- Step7C低相干补充：`large_lowcoh_component_panels_raw_time.vtk`

以上运行结果位于Git忽略的`output/`目录，不上传GitHub；GitHub保存生成逻辑、配置和节点记录。

## 下一轮重点问题

1. 小尺度数量达到97,396片，在矿区尺度和三维视图中可能过密，需要检查空间聚集、方向分布和尺寸是否过度离散。
2. 中尺度只有47片，虽然覆盖20个候选组件，但数量、空间覆盖和与蚂蚁体/低相干走廊的连续性可能不足。
3. 大尺度12,763片中绝大多数来自原始断层面碎片，需要判断这是合理的曲面离散，还是拆分过细导致数量和显示权重失衡。
4. Step6连续密度与曲率、成像测井标签的关联仍偏弱，当前流程主要用于先跑通并检查多尺度几何效果。
5. Step8在车571、车572位置发现T6/T7局部倒置；当前修复保证几何合同有效，但原始层位/时深资料仍需单独复核。

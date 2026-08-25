# -*- coding: utf-8 -*-
"""把几何变换（缩放/平移）直接烘焙进 legacy ASCII VTK 文件。

用途：当 ParaView 的 Transform 过滤器不可用时，在服务器上先把变换写进文件，
新生成的 VTK 加载后即为变换后的结果，无需再在 ParaView 里操作。

用法：
  python3 apply_vtk_transform.py --input 输入.vtk --output 输出.vtk \
      --scale 1 1 -1 --translate 0 0 0

参数说明：
  --scale      三个分量依次为 X/Y/Z 缩放，如 Z 翻转用 "1 1 -1"
  --translate  X/Y/Z 平移（米；Z 为 TWT 毫秒）

适用于 well_trajectories_raw_time.vtk、大中小尺度 DFN、断层面、
融合 DFN 等所有 legacy ASCII POLYDATA 文件。
"""

import argparse
from pathlib import Path

import vtk


def transform_vtk(input_path: str, output_path: str,
                  scale=(1.0, 1.0, 1.0), translate=(0.0, 0.0, 0.0)):
    """读取 legacy VTK，应用缩放/平移后写出新文件。返回 (点数, 单元数, 变换前Z, 变换后Z)。"""
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(input_path)
    # 默认只读活动标量，必须全部打开，否则面积/倾向倾角/地层等属性会在入口丢失
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllNormalsOn()
    reader.ReadAllTCoordsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    src = reader.GetOutput()
    if src.GetNumberOfPoints() == 0:
        raise RuntimeError("输入文件读取为空：%s" % input_path)

    tr = vtk.vtkTransform()
    tr.Translate(translate)
    tr.Scale(scale)

    if isinstance(src, vtk.vtkPolyData):
        transform_filter = vtk.vtkTransformPolyDataFilter()
    else:
        transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetInputData(src)
    transform_filter.SetTransform(tr)
    transform_filter.Update()
    out = transform_filter.GetOutput()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(out)
    writer.SetFileTypeToASCII()
    writer.Write()

    b0 = src.GetBounds()
    b1 = out.GetBounds()
    return {
        "points": int(src.GetNumberOfPoints()),
        "cells": int(src.GetNumberOfCells()),
        "z_before": (float(b0[4]), float(b0[5])),
        "z_after": (float(b1[4]), float(b1[5])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="烘焙几何变换到 legacy VTK 文件")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--translate", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()

    info = transform_vtk(args.input, args.output, args.scale, args.translate)
    print("输入: %s" % args.input)
    print("  变换: scale=%s translate=%s" % (args.scale, args.translate))
    print("  变换前 Z: %.1f ~ %.1f  变换后 Z: %.1f ~ %.1f"
          % (info["z_before"][0], info["z_before"][1],
             info["z_after"][0], info["z_after"][1]))
    print("  点数/单元数: %d / %d" % (info["points"], info["cells"]))
    print("已写出: %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

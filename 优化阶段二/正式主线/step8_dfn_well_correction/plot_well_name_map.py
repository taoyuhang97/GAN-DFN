# -*- coding: utf-8 -*-
"""生成井位 + 井轨迹 + 中文井名的平面标注图（ParaView 渲染不可用时的备选）。

井名字号默认 34pt（原 8.5pt 的 4 倍），并自动避让：每个标签在井点周围
尝试 8 个方向 + 多档距离，选择不与已放置标签重叠的位置；密集区用引导线
把标签连回井点，保证标签之间不互相覆盖。

用法（在服务器上）：
  python3 plot_well_name_map.py --csv <annotations.csv> --vtk <trajectories.vtk> \
      --out <output.png> --title "标题" [--transparent] [--bare]

坐标约定与 DFN 一致：X/Y 为米（平面俯视，北在上），Z 为 TWT 毫秒。
--bare 时画布固定为区块范围（无坐标轴/标题/图例），可直接叠加到 DFN 图上。
"""

import argparse
import csv
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache-tyh")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patheffects import withStroke
import numpy as np


# 井名字号：原 8.5pt 的 4 倍
WELL_LABEL_FONT_SIZE = 34.0
# 标签与井点之间的最小间隙（显示像素）
LABEL_GAP_PX = 10.0
# 引导线阈值：井点与标签边缘距离超过该值才画引导线（像素）
LEADER_MIN_PX = 20.0


def setup_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [family]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return family
    return None


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_vtk_polylines(path):
    """读取 legacy ASCII POLYDATA：返回每条折线的 (N,3) 点数组列表（按单元顺序）。"""
    with open(path, "r", encoding="utf-8") as handle:
        lines = [ln.rstrip("\n") for ln in handle]
    points = []
    polylines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith("POINTS"):
            count = int(line.split()[1])
            pts = np.array(
                [[float(v) for v in lines[i + j].split()[:3]] for j in range(count)],
                dtype=np.float64,
            )
            points = pts
            i += count
        elif line.startswith("LINES"):
            _, m, _ = line.split()
            for _ in range(int(m)):
                idx = [int(v) for v in lines[i].split()[1:]]
                polylines.append(points[np.asarray(idx, dtype=int)])
                i += 1
        elif line.startswith("CELL_DATA"):
            break
    return polylines


def rects_overlap(a, b, margin=0.0):
    return not (
        a[2] + margin <= b[0] or b[2] + margin <= a[0]
        or a[3] + margin <= b[1] or b[3] + margin <= a[1]
    )


def rect_area(a):
    return max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)


def rect_intersect_area(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return max(ix, 0.0) * max(iy, 0.0)


def inside_axes(ax, box, renderer, margin=4.0):
    ext = ax.get_window_extent(renderer)
    return (
        box[0] >= ext.x0 + margin and box[2] <= ext.x1 - margin
        and box[1] >= ext.y0 + margin and box[3] <= ext.y1 - margin
    )


def refine_placements(items, anchors, renderer, ax, max_iters=200):
    """松弛式标签微调：每次迭代先汇总所有重叠产生的斥力，再统一移动。

    items: [[name, x, y, cx, cy, box], ...]，box 为可变的 [x0,y0,x1,y1]。
    标签之间、标签与井点之间都有斥力；阻尼系数随时间衰减，逐步收敛。
    标签始终被约束在画布内。
    """
    ext = ax.get_window_extent(renderer)
    margin = 4.0
    n = len(items)
    anchor_boxes = [(p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6) for p in anchors]

    def clamp(box):
        if box[0] < ext.x0 + margin:
            s = ext.x0 + margin - box[0]
            box[0] += s
            box[2] += s
        if box[2] > ext.x1 - margin:
            s = ext.x1 - margin - box[2]
            box[0] += s
            box[2] += s
        if box[1] < ext.y0 + margin:
            s = ext.y0 + margin - box[1]
            box[1] += s
            box[3] += s
        if box[3] > ext.y1 - margin:
            s = ext.y1 - margin - box[3]
            box[1] += s
            box[3] += s

    def push_vec(a, b, sep=3.0):
        """返回把 b 推离 a 的位移向量（沿最小重叠轴）。"""
        ox = min(a[2], b[2]) - max(a[0], b[0])
        oy = min(a[3], b[3]) - max(a[1], b[1])
        if ox <= oy:
            d = 1.0 if (b[0] + b[2]) / 2.0 >= (a[0] + a[2]) / 2.0 else -1.0
            return [(ox + sep) * d, 0.0]
        d = 1.0 if (b[1] + b[3]) / 2.0 >= (a[1] + a[3]) / 2.0 else -1.0
        return [0.0, (oy + sep) * d]

    for it in range(max_iters):
        forces = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i][5], items[j][5]
                if not rects_overlap(a, b, margin=2.0):
                    continue
                v = push_vec(a, b)
                forces[j][0] += v[0]
                forces[j][1] += v[1]
                forces[i][0] -= v[0] * 0.5
                forces[i][1] -= v[1] * 0.5
        for i in range(n):
            box_i = items[i][5]
            for k in range(n):
                if k == i:
                    continue
                if rects_overlap(box_i, anchor_boxes[k]):
                    v = push_vec(anchor_boxes[k], box_i)
                    forces[i][0] += v[0]
                    forces[i][1] += v[1]

        total = sum(abs(f[0]) + abs(f[1]) for f in forces)
        if total < 1e-6:
            break
        damping = max(0.05, 1.0 - it / float(max_iters))
        for i in range(n):
            dx = forces[i][0] * damping
            dy = forces[i][1] * damping
            if abs(dx) < 0.05 and abs(dy) < 0.05:
                continue
            box = items[i][5]
            box[0] += dx
            box[2] += dx
            box[1] += dy
            box[3] += dy
            clamp(box)

    # 定向清理：对仍重叠的标签对做全额推力（带微小抖动跳出局部极小）
    for it in range(80):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i][5], items[j][5]
                if not rects_overlap(a, b, margin=2.0):
                    continue
                v = push_vec(a, b, sep=4.0)
                jitter = ((it * 7 + i * 13 + j * 29) % 7) - 3
                dx = v[0] + (jitter if v[0] != 0.0 else 0.0)
                dy = v[1] + (jitter if v[1] != 0.0 else 0.0)
                b[0] += dx
                b[2] += dx
                b[1] += dy
                b[3] += dy
                clamp(b)
                a[0] -= v[0] * 0.5
                a[2] -= v[0] * 0.5
                a[1] -= v[1] * 0.5
                a[3] -= v[1] * 0.5
                clamp(a)
                moved = True
        if not moved:
            break

    for item in items:
        b = item[5]
        item[3] = (b[0] + b[2]) / 2.0
        item[4] = (b[1] + b[3]) / 2.0


def place_well_labels(ax, renderer, wells, fontsize):
    """贪心避让布局：返回 [(name, x_data, y_data, cx_disp, cy_disp, box_disp), ...]。

    wells: [(name, x, y), ...]，坐标为数据坐标。
    标签以 (cx, cy) 为中心绘制（ha/va 均为 center）。
    候选位置：先试井点周围 8 个相邻位置，再按小角度步长做螺旋搜索；
    找不到完全无重叠的位置时，选"重叠面积最小且距井点最近"的候选兜底。
    """
    anchors = [ax.transData.transform((x, y)) for (_, x, y) in wells]
    sizes = []
    for name, _, _ in wells:
        dummy = ax.text(0.0, 0.0, name, fontsize=fontsize, ha="center", va="center")
        bb = dummy.get_window_extent(renderer)
        sizes.append((bb.width, bb.height))
        dummy.remove()

    ext = ax.get_window_extent(renderer)
    # 搜索半径取画布对角线的 1.5 倍，保证最角落的井也能把标签放到远处空区域
    max_radius = math.hypot(ext.width, ext.height) * 1.5
    gap = LABEL_GAP_PX

    def make_candidates(px, py, w, h):
        half_w, half_h = w / 2.0, h / 2.0
        cands = []
        # 相邻 8 个位置（右/左/上/下/四个角）
        for ox, oy in (
            (half_w + gap, 0.0),
            (-(half_w + gap), 0.0),
            (0.0, half_h + gap),
            (0.0, -(half_h + gap)),
            (half_w + gap, half_h + gap),
            (-(half_w + gap), half_h + gap),
            (half_w + gap, -(half_h + gap)),
            (-(half_w + gap), -(half_h + gap)),
        ):
            cands.append((px + ox, py + oy))
        # 螺旋候选：半径逐步增大，每圈 24 个方向
        r = gap
        angle = 0.0
        step_angle = math.radians(15.0)
        step_radius = 10.0
        while r <= max_radius:
            for _ in range(24):
                cx = px + math.cos(angle) * r
                cy = py + math.sin(angle) * r
                # 跳过会把井点盖在标签框内部的候选
                if abs(cx - px) <= half_w and abs(cy - py) <= half_h:
                    angle += step_angle
                    continue
                cands.append((cx, cy))
                angle += step_angle
            r += step_radius
        return cands

    occupied = []
    results = []
    for i, ((name, x, y), (w, h)) in enumerate(zip(wells, sizes)):
        px, py = anchors[i]
        half_w, half_h = w / 2.0, h / 2.0
        candidates = make_candidates(px, py, w, h)

        def box_of(cx, cy):
            return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

        chosen = None
        chosen_box = None
        for cx, cy in candidates:
            box = box_of(cx, cy)
            if not inside_axes(ax, box, renderer):
                continue
            if any(rects_overlap(box, ob, margin=2.0) for ob in occupied):
                continue
            if any(
                rects_overlap(box, (p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6))
                for j, p in enumerate(anchors) if j != i
            ):
                continue
            chosen = (cx, cy)
            chosen_box = box
            break

        if chosen is None:
            # 兜底：画布内选择"总重叠面积最小 + 距离最近"的候选
            best = None
            best_score = None
            for cx, cy in candidates:
                box = box_of(cx, cy)
                if not inside_axes(ax, box, renderer):
                    continue
                overlap_area = sum(rect_intersect_area(box, ob) for ob in occupied)
                for j, p in enumerate(anchors):
                    if j == i:
                        continue
                    overlap_area += rect_intersect_area(
                        box, (p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6)
                    )
                dist = math.hypot(cx - px, cy - py)
                score = (overlap_area, dist)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (cx, cy, box)
            if best is not None:
                chosen, chosen_box = (best[0], best[1]), best[2]
            else:
                chosen, chosen_box = (px, py), box_of(px, py)

        occupied.append(chosen_box)
        results.append([name, x, y, chosen[0], chosen[1], list(chosen_box)])

    refine_placements(results, anchors, renderer, ax)
    return results


def plot_well_map(rows, polylines, out_path, title, transparent=False, bare=False):
    family = setup_cjk_font()
    wells = []
    for row in rows:
        try:
            wells.append((row["WellName"], float(row["LabelX"]), float(row["LabelY"])))
        except Exception:
            continue

    xs = [w[1] for w in wells]
    ys = [w[2] for w in wells]
    dx = (max(xs) - min(xs)) * 0.02 or 1000.0
    dy = (max(ys) - min(ys)) * 0.02 or 1000.0
    xlim = (min(xs) - dx, max(xs) + dx)
    ylim = (min(ys) - dy, max(ys) + dy)

    if bare:
        # 画布比例 = 数据范围比例，零边距，保证叠加对齐
        data_w = xlim[1] - xlim[0]
        data_h = ylim[1] - ylim[0]
        fig_w = 12.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * data_h / data_w), dpi=150)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax.set_axis_off()
    else:
        fig, ax = plt.subplots(figsize=(12, 9), dpi=150)
        ax.set_xlabel("X (m)  → 向东增大")
        ax.set_ylabel("Y (m)  → 向北增大（图上自上而下减小）")
        ax.set_title(title, fontsize=14, pad=12)
        ax.grid(True, linestyle="--", alpha=0.35)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # 井轨迹平面投影（细线）
    for pl in polylines:
        ax.plot(pl[:, 0], pl[:, 1], color="#4a7ebb", lw=0.8, alpha=0.55, zorder=2)

    # 标签避让布局
    placements = place_well_labels(ax, renderer, wells, WELL_LABEL_FONT_SIZE)

    # 井点
    ax.scatter(xs, ys, s=26, marker="o", facecolor="crimson",
               edgecolor="white", linewidths=1.0, zorder=5, label="井位（井顶）")
    if not bare:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # 引导线 + 井名
    for name, x, y, cx, cy, box in placements:
        anchor_px, anchor_py = ax.transData.transform((x, y))
        # 井点(显示坐标)到标签框最近点的距离，超过阈值才画引导线
        nearest_x = min(max(anchor_px, box[0]), box[2])
        nearest_y = min(max(anchor_py, box[1]), box[3])
        dist = math.hypot(nearest_x - anchor_px, nearest_y - anchor_py)
        if dist > LEADER_MIN_PX:
            line_x = ax.transData.inverted().transform((nearest_x, nearest_y))
            ax.plot([x, line_x[0]], [y, line_x[1]], color="#555555",
                    lw=1.2, alpha=0.85, zorder=4)
        tx, ty = ax.transData.inverted().transform((cx, cy))
        ax.text(tx, ty, name, fontsize=WELL_LABEL_FONT_SIZE, ha="center", va="center",
                color="#111111", zorder=6,
                path_effects=[withStroke(linewidth=3.5, foreground="white")])

    out_path = str(out_path)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    if bare:
        fig.savefig(out_path, transparent=transparent)
    else:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, transparent=transparent)
    plt.close(fig)
    print("已生成：%s（%d 口井，字体 %s，字号 %.0fpt，透明=%s，纯净叠加=%s）"
          % (out_path, len(wells), family, WELL_LABEL_FONT_SIZE, transparent, bare))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--vtk", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="井位与井轨迹平面图")
    parser.add_argument("--transparent", action="store_true",
                        help="输出透明背景 PNG")
    parser.add_argument("--bare", action="store_true",
                        help="去掉坐标轴/标题/图例，只保留井轨迹、井点和井名")
    args = parser.parse_args()

    rows = read_csv(args.csv)
    polylines = read_vtk_polylines(args.vtk)
    print("CSV 井数：%d，VTK 折线数：%d" % (len(rows), len(polylines)))
    if len(rows) != len(polylines):
        print("警告：井数与折线数不一致，请检查 VTK/CSV 是否成对")
    plot_well_map(rows, polylines, args.out, args.title,
                  transparent=args.transparent, bare=args.bare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

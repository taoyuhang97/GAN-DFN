import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator, interp1d
import warnings

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
# 原有DFN结果目录（不包含两口井的区域）
PREDICTED_DIR_OLD = r"D:\康宁畅\中国石油\三阶段解释验证\第三阶段\结果"

# 新的DFN结果目录（包含车页1导眼井的区域）
PREDICTED_DIR_NEW = r"D:\康宁畅\中国石油\三阶段解释验证\第三阶段\结果\bx75_by21_t1_t7_20260409"

WELL_DATA_DIR = r"D:\康宁畅\中国石油\三阶段解释验证\第三阶段\两口测井"
FAULT_DIR = r"D:\康宁畅\中国石油\三阶段解释验证\第三阶段\层位"
OUTPUT_DIR = r"D:\康宁畅\中国石油\三阶段解释验证\第三阶段\验证结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 读取断层数据（仅T开头）====================

def read_dat_surface(filepath):
    """读取.dat断层文件"""
    data_lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    data_lines.append([x, y, z])
                except:
                    continue

    if not data_lines:
        return None
    return pd.DataFrame(data_lines, columns=['X', 'Y', 'Z'])


def get_fault_surfaces():
    """获取所有T开头的断层曲面，并预创建插值器"""
    fault_surfaces = {}
    for file in os.listdir(FAULT_DIR):
        if not file.startswith('T') or not file.endswith('.dat'):
            continue
        name = file.replace('.dat', '')
        df = read_dat_surface(os.path.join(FAULT_DIR, file))
        if df is not None and len(df) > 0:
            interp = NearestNDInterpolator(df[['X', 'Y']].values, df['Z'].values)
            fault_surfaces[name] = {
                'df': df,
                'interp': interp,
                'x_min': df['X'].min(),
                'x_max': df['X'].max(),
                'y_min': df['Y'].min(),
                'y_max': df['Y'].max()
            }
    print(f"找到 {len(fault_surfaces)} 个T开头断层")
    return fault_surfaces


def get_surface_depth(x, y, fault_surface):
    """使用预创建的插值器获取曲面深度"""
    try:
        if (x < fault_surface['x_min'] or x > fault_surface['x_max'] or
                y < fault_surface['y_min'] or y > fault_surface['y_max']):
            return None
        z = fault_surface['interp'](x, y)
        return float(z) if not np.isnan(z) else None
    except:
        return None


def assign_fault_zone(x, y, time, fault_surfaces):
    """根据TIME判断属于哪个断层区间"""
    fault_depths = []
    for name, surface in fault_surfaces.items():
        depth = get_surface_depth(x, y, surface)
        if depth is not None:
            fault_depths.append((depth, name))

    if not fault_depths:
        return "Unknown"

    fault_depths.sort(key=lambda x: x[0])

    if time < fault_depths[0][0]:
        return f"Above_{fault_depths[0][1]}"
    elif time > fault_depths[-1][0]:
        return f"Below_{fault_depths[-1][1]}"
    else:
        for i in range(len(fault_depths) - 1):
            if fault_depths[i][0] <= time <= fault_depths[i + 1][0]:
                return f"Between_{fault_depths[i][1]}_{fault_depths[i + 1][1]}"

    return "Unknown"


def classify_points(df, fault_surfaces, desc=""):
    """为数据点添加地层分类"""
    if df is None or len(df) == 0:
        return None

    df = df.copy()
    zones = []
    total = len(df)

    print(f"  正在为{desc}添加地层分类 ({total} 个点)...")

    for idx, row in df.iterrows():
        x = row['X']
        y = row['Y']
        time = row['TIME']
        zone = assign_fault_zone(x, y, time, fault_surfaces)
        zones.append(zone)

        if (idx + 1) % 500 == 0:
            print(f"    进度: {idx + 1}/{total}")

    df['Fault_Zone'] = zones
    print(f"    完成!")
    return df


# ==================== 2. 读取实际井数据 ====================

def load_well_sample_data(well_name):
    """加载井的_sample.csv数据"""
    file_path = os.path.join(WELL_DATA_DIR, f"{well_name}_sample.csv")
    if not os.path.exists(file_path):
        print(f"  警告: 找不到 {file_path}")
        return None

    df = pd.read_csv(file_path)
    print(f"  加载 {well_name}_sample.csv: {len(df)} 行")

    if 'X' not in df.columns or 'Y' not in df.columns or 'TIME' not in df.columns:
        print(f"  错误: 缺少必要的坐标列 (X, Y, TIME)")
        return None

    return df


# ==================== 3. 读取预测结果 ====================

def load_predicted_patches(pred_dir):
    """加载预测的裂缝单元数据"""
    if not os.path.exists(pred_dir):
        print(f"  警告: 目录不存在 {pred_dir}")
        return None

    all_patches = []
    for root, dirs, files in os.walk(pred_dir):
        for file in files:
            if file == 'predicted_unit_patches.csv':
                filepath = os.path.join(root, file)
                df = pd.read_csv(filepath)
                df['Source'] = os.path.basename(root)
                all_patches.append(df)
                print(f"  加载: {os.path.basename(root)} - {len(df)} 个裂缝")

    if not all_patches:
        return None

    combined = pd.concat(all_patches, ignore_index=True)
    print(f"  总计: {len(combined)} 个裂缝单元")

    combined = combined.rename(columns={
        'CenterX': 'X',
        'CenterY': 'Y',
        'CenterTIME': 'TIME'
    })

    return combined


# ==================== 4. 沿井轨迹计算预测裂缝密度 ====================

def calculate_predicted_density_along_well(well_df, patches_df, search_radius=100, step_m=10):
    """沿井轨迹计算预测裂缝密度"""
    if len(well_df) < 2 or len(patches_df) == 0:
        return None, None

    patch_centers = patches_df[['X', 'Y']].values
    tree = cKDTree(patch_centers)

    well_points = well_df[['X', 'Y', 'TIME']].dropna().values
    well_points = well_points[well_points[:, 2].argsort()]

    sampled_points = []
    for i in range(len(well_points) - 1):
        p1 = well_points[i]
        p2 = well_points[i + 1]
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        n_steps = max(1, int(dist / step_m))

        for j in range(n_steps + 1):
            t = j / n_steps if n_steps > 0 else 0
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            time = p1[2] + t * (p2[2] - p1[2])
            sampled_points.append([x, y, time])

    sampled_df = pd.DataFrame(sampled_points, columns=['X', 'Y', 'TIME'])

    densities = []
    for _, point in sampled_df.iterrows():
        distances, indices = tree.query([point['X'], point['Y']], k=min(50, len(patches_df)))
        if np.isscalar(indices):
            indices = [indices]
        nearby = sum(1 for d in distances if d <= search_radius)
        densities.append(nearby)

    sampled_df['Predicted_Density'] = densities

    interp_density = interp1d(sampled_df['TIME'], sampled_df['Predicted_Density'],
                              kind='linear', fill_value='extrapolate', bounds_error=False)
    well_densities = interp_density(well_df['TIME'].values)

    return well_densities, sampled_df


# ==================== 5. 统计分布对比（按井） ====================

def plot_azimuth_rose(actual_azimuths, pred_azimuths, well_name, zone_name, output_dir):
    """方位角玫瑰图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})

    if len(actual_azimuths) > 0:
        az_rad = np.radians(actual_azimuths)
        n, bins = np.histogram(az_rad, bins=36, range=(0, 2 * np.pi))
        theta = np.linspace(0, 2 * np.pi, 36)
        axes[0].bar(theta, n, width=2 * np.pi / 36, alpha=0.7, color='coral')
        axes[0].set_title(f'{well_name} - 实际裂缝')

    if len(pred_azimuths) > 0:
        az_rad = np.radians(pred_azimuths)
        n, bins = np.histogram(az_rad, bins=36, range=(0, 2 * np.pi))
        theta = np.linspace(0, 2 * np.pi, 36)
        axes[1].bar(theta, n, width=2 * np.pi / 36, alpha=0.7, color='steelblue')
        axes[1].set_title(f'{well_name} - 预测裂缝')

    plt.suptitle(f'方位角分布 - {zone_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{well_name}_{zone_name}_azimuth.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_dip_histogram(actual_dips, pred_dips, well_name, zone_name, output_dir):
    """倾角分布直方图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(actual_dips) > 0:
        axes[0].hist(actual_dips, bins=20, alpha=0.7, color='coral', edgecolor='black')
        axes[0].set_xlabel('倾角 (度)')
        axes[0].set_ylabel('频数')
        axes[0].set_title(f'{well_name} - 实际裂缝')
        axes[0].grid(True, alpha=0.3)

    if len(pred_dips) > 0:
        axes[1].hist(pred_dips, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].set_xlabel('倾角 (度)')
        axes[1].set_ylabel('频数')
        axes[1].set_title(f'{well_name} - 预测裂缝')
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'倾角分布 - {zone_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{well_name}_{zone_name}_dip.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_density_curve(well_df, pred_densities, well_name, zone_name, output_dir):
    """P10与预测裂缝密度曲线对比"""
    if well_df is None or pred_densities is None:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    if 'P10' in well_df.columns:
        ax.plot(well_df['TIME'], well_df['P10'], 'o-', color='coral',
                markersize=2, linewidth=1, label='实际 P10')

    ax.plot(well_df['TIME'], pred_densities, 's-', color='steelblue',
            markersize=2, linewidth=1, label='预测裂缝密度')

    ax.set_xlabel('TIME (ms)')
    ax.set_ylabel('裂缝密度 / 数量')
    ax.set_title(f'{well_name} - 裂缝密度对比 ({zone_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if 'P10' in well_df.columns:
        valid_mask = (well_df['P10'] > 0) & (pred_densities > 0)
        if valid_mask.sum() > 5:
            corr = np.corrcoef(well_df['P10'][valid_mask], pred_densities[valid_mask])[0, 1]
            ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{well_name}_{zone_name}_density.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_comparison(well_df, pred_densities, well_name, zone_name, output_dir):
    """P10 vs 预测密度 散点图"""
    if 'P10' not in well_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(well_df['P10'], pred_densities, alpha=0.5, color='steelblue', s=10)
    ax.set_xlabel('实际 P10')
    ax.set_ylabel('预测裂缝密度')
    ax.set_title(f'{well_name} - P10 vs 预测密度 ({zone_name})')
    ax.grid(True, alpha=0.3)

    valid_mask = (well_df['P10'] > 0) & (pred_densities > 0)
    if valid_mask.sum() > 5:
        z = np.polyfit(well_df['P10'][valid_mask], pred_densities[valid_mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(well_df['P10'][valid_mask].min(), well_df['P10'][valid_mask].max(), 50)
        ax.plot(x_line, p(x_line), 'r--', label=f'趋势线 (斜率={z[0]:.3f})')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{well_name}_{zone_name}_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ==================== 6. 分析单口井 ====================

def analyze_well(well_name, patches_df, fault_surfaces, output_dir):
    """分析单口井的验证结果"""
    print(f"\n{'=' * 60}")
    print(f"处理井: {well_name}")
    print(f"{'=' * 60}")

    # 加载井数据
    well_df = load_well_sample_data(well_name)
    if well_df is None:
        print(f"  井数据为空，跳过")
        return None

    # 为井数据添加地层分类
    well_df = classify_points(well_df, fault_surfaces, f"{well_name}_sample")

    # 为预测结果添加地层分类
    patches_df = classify_points(patches_df, fault_surfaces, "预测裂缝")

    # 创建输出目录
    well_output_dir = os.path.join(output_dir, well_name)
    os.makedirs(well_output_dir, exist_ok=True)
    os.makedirs(os.path.join(well_output_dir, "statistics"), exist_ok=True)
    os.makedirs(os.path.join(well_output_dir, "density"), exist_ok=True)

    # 获取所有地层
    all_zones = patches_df['Fault_Zone'].unique()
    all_zones = [z for z in all_zones if z != "Unknown"]

    well_summary = []

    for zone in all_zones:
        print(f"\n  处理地层: {zone}")

        # 预测数据
        pred_zone = patches_df[patches_df['Fault_Zone'] == zone]
        pred_azimuths = pred_zone['Azimuth'].values
        pred_dips = pred_zone['Dip'].values

        # 实际井数据
        well_zone = well_df[well_df['Fault_Zone'] == zone]
        if len(well_zone) == 0:
            print(f"    该井在此地层无数据，跳过")
            continue

        # 提取裂缝属性（P10 > 0 的位置）
        fracture_points = well_zone[well_zone['P10'] > 0]
        actual_azimuths = fracture_points['Frac_Azimuth'].dropna().values
        actual_dips = fracture_points['Frac_Dip'].dropna().values

        print(f"    预测裂缝: {len(pred_zone)} 条")
        print(f"    实际裂缝点: {len(actual_azimuths)} 个")

        # 计算沿井轨迹的预测裂缝密度
        try:
            pred_densities, _ = calculate_predicted_density_along_well(well_zone, pred_zone, search_radius=100)

            if pred_densities is not None:
                plot_density_curve(well_zone, pred_densities, well_name, zone,
                                   os.path.join(well_output_dir, "density"))
                plot_scatter_comparison(well_zone, pred_densities, well_name, zone,
                                        os.path.join(well_output_dir, "density"))
        except Exception as e:
            print(f"    密度计算失败: {e}")

        # 方位角玫瑰图
        if len(actual_azimuths) > 0 or len(pred_azimuths) > 0:
            plot_azimuth_rose(actual_azimuths, pred_azimuths, well_name, zone,
                              os.path.join(well_output_dir, "statistics"))

        # 倾角直方图
        if len(actual_dips) > 0 or len(pred_dips) > 0:
            plot_dip_histogram(actual_dips, pred_dips, well_name, zone,
                               os.path.join(well_output_dir, "statistics"))

        # 汇总统计
        well_summary.append({
            '地层': zone,
            '预测裂缝数': len(pred_zone),
            '预测方位角均值': np.mean(pred_azimuths) if len(pred_azimuths) > 0 else np.nan,
            '预测方位角标准差': np.std(pred_azimuths) if len(pred_azimuths) > 0 else np.nan,
            '预测倾角均值': np.mean(pred_dips) if len(pred_dips) > 0 else np.nan,
            '预测倾角标准差': np.std(pred_dips) if len(pred_dips) > 0 else np.nan,
            '实际裂缝点': len(actual_azimuths),
            '实际方位角均值': np.mean(actual_azimuths) if len(actual_azimuths) > 0 else np.nan,
            '实际倾角均值': np.mean(actual_dips) if len(actual_dips) > 0 else np.nan
        })

    # 保存该井的汇总报告
    if well_summary:
        well_summary_df = pd.DataFrame(well_summary)
        well_summary_df.to_csv(os.path.join(well_output_dir, "summary.csv"), index=False)
        print(f"\n  汇总报告已保存: {well_output_dir}/summary.csv")

    return well_summary


# ==================== 7. 主函数 ====================

def main():
    print("=" * 80)
    print("裂缝预测结果验证 - 按井分开")
    print("=" * 80)

    # 1. 加载断层数据
    print("\n1. 加载断层数据...")
    fault_surfaces = get_fault_surfaces()

    # 2. 选择要分析的井
    print("\n2. 选择要分析的井:")
    print("   1. 车页1导眼 (使用新的DFN结果)")
    print("   2. 车151HF (使用原有DFN结果)")
    print("   3. 两口井都分析")

    choice = input("请输入选择 (1/2/3): ").strip()

    # 3. 加载预测结果
    print("\n3. 加载预测结果...")

    all_results = []

    if choice in ['1', '3']:
        print("\n加载车页1导眼区域的DFN结果...")
        patches_df_new = load_predicted_patches(PREDICTED_DIR_NEW)
        if patches_df_new is not None:
            print(f"  预测裂缝: {len(patches_df_new)} 个单元")
            result = analyze_well("车页1导眼", patches_df_new, fault_surfaces, OUTPUT_DIR)
            if result:
                all_results.extend(result)
        else:
            print("  错误: 无法加载车页1导眼区域的DFN结果")

    if choice in ['2', '3']:
        print("\n加载车151HF区域的DFN结果...")
        patches_df_old = load_predicted_patches(PREDICTED_DIR_OLD)
        if patches_df_old is not None:
            print(f"  预测裂缝: {len(patches_df_old)} 个单元")
            result = analyze_well("车151HF", patches_df_old, fault_surfaces, OUTPUT_DIR)
            if result:
                all_results.extend(result)
        else:
            print("  错误: 无法加载车151HF区域的DFN结果")

    # 4. 保存总体汇总报告
    if all_results:
        all_summary_df = pd.DataFrame(all_results)
        all_summary_df.to_csv(os.path.join(OUTPUT_DIR, "all_wells_summary.csv"), index=False)
        print("\n总体汇总报告已保存: all_wells_summary.csv")

    print("\n" + "=" * 80)
    print("验证完成！")
    print(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

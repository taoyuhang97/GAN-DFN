import os
import pandas as pd
from pathlib import Path

# 定义目录列表
directories = [
    r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测/峰值检测方法/测井",
    r"/data/shared/project-oil/wx数据/砂砾岩/研究内容一/裂缝位置预测/峰值检测方法/井斜"
]

# 存储结果
results = []

for directory in directories:
    if os.path.exists(directory):
        for ext in ['*.csv']:  # 查找CSV文件
            for file_path in Path(directory).rglob(ext):
                try:
                    # 使用pd.read_csv读取CSV文件
                    df = pd.read_csv(file_path)
                    # 直接从文件名提取井名（去掉"_"后面的部分）
                    well_name = file_path.name.split('_')[0]

                    results.append({
                        '文件路径': str(file_path),
                        '文件名': file_path.name,  # 保留完整文件名
                        '井名': well_name,  # 添加井名列
                        '记录数': len(df)  # 不需要减1，因为pd.read_csv已经排除了标题行
                    })
                    print(f"{file_path.name}: {len(df)} 条记录")
                except Exception as e:
                    print(f"读取失败: {file_path.name} - {e}")

# 保存结果
if results:
    result_df = pd.DataFrame(results)
    result_df.to_excel("CSV文件记录统计.xlsx", index=False)
    print(f"\n统计完成! 共 {len(results)} 个文件")

    # 按井名分组统计
    well_summary = result_df.groupby('井名').agg({
        '文件数': ('文件名', 'count'),
        '总记录数': ('记录数', 'sum')
    }).reset_index()

    well_summary.to_excel("按井名统计.xlsx", index=False)
    print("已生成按井名统计的汇总表")
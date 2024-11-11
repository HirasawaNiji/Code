import pandas as pd

# 读取故障记录文件，替换文件路径为本地路径
file_path = 'M102故障记录.csv'
df = pd.read_csv(file_path, encoding='gbk')

# 将 '月份' 和 '日期' 列合并为完整的 '日期' 列，以便计算发生频率
df['完整日期'] = pd.to_datetime(df['月份'].astype(str) + '-' + df['日期'].astype(str), format='%m-%d')

# 计算数据集中总天数，用于计算发生频率
total_days = df['完整日期'].nunique()

print(total_days)

# 分组计算每种故障类别的统计信息
fault_stats = df.groupby('故障类别').agg(
    occurrence_count=('故障类别', 'size'),
    average_duration=('持续时长（秒）', 'mean')
).reset_index()

# 计算每种故障类别的发生频率（次/天）
fault_stats['occurrence_frequency'] = fault_stats['occurrence_count'] / total_days

# 计算所有故障的总体统计
overall_stats = {
    '故障类别': '总体',
    'occurrence_count': df.shape[0],
    'average_duration': df['持续时长（秒）'].mean(),
    'occurrence_frequency': df.shape[0] / total_days
}

# 将总体统计添加到 DataFrame
fault_stats = pd.concat([fault_stats, pd.DataFrame([overall_stats])], ignore_index=True)

# 输出统计结果
print(fault_stats)

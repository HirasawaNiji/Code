import pandas as pd
import numpy as np

file_path = 'M102故障记录.csv' 
data = pd.read_csv(file_path, encoding='GBK')

# 转换月份和日期列为字符串格式，便于统计
data['月份'] = data['月份'].astype(str)
data['日期'] = data['日期'].astype(str)

fault_types = ['A1', 'A2', 'A3', 'A4']

# 统计总次数和平均持续时长
summary = data.groupby(['月份', '日期', '故障类别']).agg(
    总次数=('故障类别', 'count'), 平均持续时长=('持续时长（秒）', 'mean')
).reset_index()

# 平均持续时长保留两位小数
summary['平均持续时长'] = summary['平均持续时长'].round(2)

# 确保所有的故障类别都有记录
# 将原始数据中的月份和日期与故障类别进行笛卡尔积来补全类别，但不补全日期
unique_month_date = summary[['月份', '日期']].drop_duplicates()
all_combinations = unique_month_date.assign(key=1).merge(
    pd.DataFrame({'故障类别': fault_types, 'key': 1}), on='key'
).drop('key', axis=1)

# 将补全类别后的数据与统计数据合并
summary_complete = all_combinations.merge(
    summary, on=['月份', '日期', '故障类别'], how='left'
)

# 填充缺失值
summary_complete['总次数'] = summary_complete['总次数'].fillna(0).astype(int)
summary_complete['平均持续时长'] = summary_complete['平均持续时长'].where(summary_complete['总次数'] != 0, np.nan)

# 将月份和日期重新转换为整数类型并按其排序
summary_complete['月份'] = summary_complete['月份'].astype(int)
summary_complete['日期'] = summary_complete['日期'].astype(int)
summary_complete = summary_complete.sort_values(by=['月份', '日期']).reset_index(drop=True)

# 导出统计结果
summary_complete.to_csv('M102故障统计.csv', index=False, encoding='GBK')

print("故障统计已导出为 'M101故障统计.csv'")



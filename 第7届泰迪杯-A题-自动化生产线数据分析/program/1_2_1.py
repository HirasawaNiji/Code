import pandas as pd

file_path = 'M102故障记录.csv'
data = pd.read_csv(file_path, encoding='GBK')

fault_types = data['故障类别'].unique()

# 筛选每种故障类别第25次出现的行
fault_25th_occurrences = []
for fault_type in fault_types:
    fault_rows = data[data['故障类别'] == fault_type].reset_index(drop=True)
    if len(fault_rows) >= 25:  # 检查是否至少有25次出现
        fault_25th_occurrences.append(fault_rows.iloc[24])  # 第25次是索引24

fault_25th_df = pd.DataFrame(fault_25th_occurrences)
print("各故障第25次出现的数据：")
print(fault_25th_df)

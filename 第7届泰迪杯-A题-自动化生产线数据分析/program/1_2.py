import pandas as pd

file_path = 'M101.csv' 
data = pd.read_csv(file_path, encoding='GBK')

# 去除列名中的空格
data.columns = data.columns.str.strip()

data = data[['月份', '日期', '时间', '故障类别']]

# 只保留故障类别不为空的行
fault_data = data[data['故障类别'].notna()]

# 用于存储每次故障的开始时间和持续时长
fault_records = []
current_fault_start = None
current_fault_type = None
fault_start_month = None
fault_start_date = None

# 设置时间间隔阈值
time_threshold = 60

for i in range(len(fault_data)):
    row = fault_data.iloc[i]
    
    current_time = int(row['时间'])
    
    if current_fault_start is None:
        # 新的故障开始
        current_fault_start = current_time
        current_fault_type = row['故障类别']
        fault_start_month = row['月份']
        fault_start_date = row['日期']
    
    # 检查下一行是否同一故障类别的连续记录
    if i < len(fault_data) - 1:
        next_row = fault_data.iloc[i + 1]
        next_time = int(next_row['时间'])
        
        # 如果下一行类别不同或时间间隔超过阈值，记录当前故障的开始时间和持续时长
        if (next_row['故障类别'] != current_fault_type) or (abs(next_time - current_time) > time_threshold):
            duration = current_time - current_fault_start+1  # 计算故障持续时长
            fault_records.append([
                fault_start_month, fault_start_date, 
                current_fault_start, 
                current_fault_type,
                duration
            ])
            # 重置以获取下一个故障
            current_fault_start = None
    else:
        # 最后一行的故障记录
        duration = current_time - current_fault_start +1
        fault_records.append([
            fault_start_month, fault_start_date, 
            current_fault_start, 
            current_fault_type,
            duration
        ])

output_df = pd.DataFrame(fault_records, columns=['月份', '日期', '开始时间', '故障类别', '持续时长（秒）'])
output_df.to_csv('M101故障记录.csv', index=False, encoding='GBK')

print("故障记录已导出为 'M101故障记录.csv'")


import pandas as pd

data_M101 = pd.read_csv('E:/code/taidi/taidi2024/附件1/M101.csv', encoding = 'gbk')
data_M102 = pd.read_csv('E:/code/taidi/taidi2024/附件1/M102.csv', encoding = 'gbk')

# 定义一个函数计算每日有效工作时长
def calculate_effective_hours(data):
    # 假设每小时一条记录，"故障类别"列中空值表示正常工作，否则表示故障
    data['有效工作时长'] = data.apply(lambda row: 1 if pd.isna(row['故障类别']) else 0, axis=1)
    # 按月份和日期分组，每天的有效工作时长（小时）
    daily_hours = data.groupby(['月份', '日期'])['有效工作时长'].sum().reset_index()
    return daily_hours/3600

# 分别计算M101和M102的有效工作时长
M101_daily_hours = calculate_effective_hours(data_M101)
M102_daily_hours = calculate_effective_hours(data_M102)

# 将结果合并为表7的格式
table7 = M101_daily_hours.merge(M102_daily_hours, on=['月份', '日期'], suffixes=('_M101', '_M102'))

# 计算日平均有效工作时长（不按月份区分）
average_M101 = M101_daily_hours['有效工作时长'].mean()
average_M102 = M102_daily_hours['有效工作时长'].mean()

# 创建表8的DataFrame
table8 = pd.DataFrame({
    '生产线': ['M101', 'M102'],
    '日平均有效工作时长（小时/天）': [average_M101, average_M102]
})

# 将结果保存到Excel文件中
output_path = 'E:/code/taidi/taidi2024/附件2/result1_4.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    table7.to_excel(writer, index=False, sheet_name='表7-每日有效工作时长')
    table8.to_excel(writer, index=False, sheet_name='表8-日平均有效工作时长')

print(f"结果已保存到 {output_path}")

import pandas as pd

file_path = 'M101.csv'  
data = pd.read_csv(file_path, encoding='GBK')

# 按月份和日期分组，并获取每组中最后一个时间的合格和不合格产品累计数
latest_data = data.sort_values(by=['日期', '时间']).groupby(['月份', '日期']).last().reset_index()
output_data = latest_data[['月份', '日期', '合格产品累计数', '不合格产品累计数']]
output_data.to_csv('M101_selected.csv', index=False,encoding='GBK')

print("导出完成，文件保存为 M101_selected.csv")


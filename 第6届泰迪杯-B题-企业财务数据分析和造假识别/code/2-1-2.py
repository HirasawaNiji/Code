import pandas as pd

# 读取 Excel 文件
file_path = 'LR_new.csv'
df = pd.read_csv(file_path,encoding='utf-8')
df['Accper'] = pd.to_datetime(df['Accper'])

# 提取出 2019 年 9 月的数据
def Average_caculating(month):
    month_data = df[(df['Accper'].dt.year == 2018)
                        & (df['Accper'].dt.month == month)]

    # 计算B001000000列的均值
    result = month_data.groupby('Indnme')['利润率'].mean()
    pd.set_option('display.float_format', lambda x: '%.5f' % x)

    return result

for i in range(1,13):
    print(f'第{i}月的利润率平均值为{Average_caculating(i)}')



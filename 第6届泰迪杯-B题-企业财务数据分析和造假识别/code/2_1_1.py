import pandas as pd
df = pd.read_csv('LR_new.csv', encoding='utf-8')
df['Accper'] = pd.to_datetime(df['Accper'])

# 选择2019年9月的数据
september_data = df[(df['Accper'].dt.year == 2019)
                    & (df['Accper'].dt.month == 9)]

# 计算输出B001000000列的均值
result = september_data.groupby('Indnme')['B001000000'].mean()
pd.set_option('display.float_format', lambda x: '%.5f' % x)
print(result)


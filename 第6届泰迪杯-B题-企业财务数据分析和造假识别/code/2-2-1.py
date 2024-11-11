import pandas as pd
df = pd.read_csv('LR_new.csv', encoding='uft-8')
df['Accper'] = pd.to_datetime(df['Accper'])

# 选择2019年9月的数据
september_data = df[(df['Accper'].dt.year == 2019)
                    & (df['Accper'].dt.month == 9)]
september_data = september_data[september_data['Indnme'] == '金融']

# 计算利润率列的均值
result = september_data.groupby('Nindnme')['利润率'].mean()
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# 对结果进行降序排序并输出
result = result.sort_values(ascending=False)
print(result)


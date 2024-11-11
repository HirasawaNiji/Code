import pandas as pd

df = pd.read_csv('LR_new.csv', encoding='utf-8')

df['Accper'] = pd.to_datetime(df['Accper'])

# 选择2019年9月的数据
september_data = df[(df['Accper'].dt.year == 2019)
                    & (df['Accper'].dt.month == 9)]
september_data = september_data[september_data['Nindnme'] == '证券、期货业']

result = september_data['利润率'].sort_values(ascending=False)

# 输出结果
print(result)


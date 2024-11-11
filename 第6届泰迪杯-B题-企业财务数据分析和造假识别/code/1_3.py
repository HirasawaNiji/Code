import pandas as pd

lr_2 = pd.read_csv("LR_2.csv")
missing_percentage = lr_2.isnull().mean()
selected_columns = missing_percentage[missing_percentage < 0.7].index

lr_3 = lr_2[selected_columns]
lr_3.to_csv("LR_3.csv", encoding='GBK', index=False)

print("处理后数据的列数：", lr_3.shape[1])

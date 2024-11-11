import pandas as pd

lr_3 = pd.read_csv("LR_3.csv")
lr_4 = lr_3.dropna()
lr_4.to_csv("LR_4.csv", encoding='utf-8', index=False)

print("处理后数据的行数：", lr_4.shape[0])


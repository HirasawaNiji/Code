import pandas as pd

lr_4 = pd.read_csv("LR_4_new.csv")
lr_4['Accper'] = pd.to_datetime(
    lr_4['Accper'], format='%Y/%m/%d').dt.strftime('%Y-%m-%d')

lr_4.to_csv("LR_5.csv", encoding='utf-8', index=False)


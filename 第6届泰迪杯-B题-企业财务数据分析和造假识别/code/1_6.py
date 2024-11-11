import pandas as pd

lr_5 = pd.read_csv("LR_5.csv", encoding='gbk')
lr_5['利润率'] = (lr_5['B001000000'] / lr_5['B001100000'])
lr_5['资产负债率'] = (lr_5['A002000000'] / lr_5['A001000000'])

lr_new = lr_5[
    (lr_5['利润率'].between(-3, 3)) &
    (lr_5['资产负债率'].between(-3, 3))
]

lr_new.to_csv("LR_new.csv", encoding='gbk', index=False)

print("处理后的数据行数和列数：", lr_new.shape)
print("前5个企业的利润率和资产负债率：")
print(lr_new[['利润率', '资产负债率']].head())


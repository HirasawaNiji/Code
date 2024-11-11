import pandas as pd

lr_1 = pd.read_csv("LR_1.csv")

zcfz = pd.read_csv("ZCFZ.csv")
zcfz_subset = zcfz.loc[zcfz['Typrep'].isin(['A002000000', 'A001000000']), ['Stkcd', 'Accper', 'Typrep', 'A002000000', 'A001000000']]

lr_1 = pd.merge(lr_1, zcfz_subset, how='left', on=['Stkcd', 'Accper', 'Typrep'])

stk_ind = pd.read_csv("Stk_ind.csv")
stk_ind_subset = stk_ind[['Stkcd', 'Indnme', 'Nindnme']]

lr_1 = pd.merge(lr_1, stk_ind_subset, how='left', on='Stkcd')
lr_1.to_csv("LR_2.csv", encoding='utf-8', index=False)

print("合并后的数据行数和列数：", lr_1.shape)
import pandas as pd

lr = pd.read_csv("LR.csv")
selected_columns = ['Stkcd',	'Accper',	'Typrep',
                    'B001000000',	'B001100000',	'B001101000',	'B001200000',	
                    'B001201000',	'B001207000',	'B001209000',	'B001210000',	
                    'B001211000',	'B001212000',	'B001303000',	'B002300000',]

lr_subset = lr[selected_columns]
lr_filtered = lr_subset[lr_subset['Typrep'] == 'A']
lr_filtered.to_csv("LR_1.csv", encoding='utf-8', index=False)
print("筛选后的数据行数和列数：", lr_filtered.shape)


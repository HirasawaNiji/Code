import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


data_M101 = pd.read_csv('E:/code/taidi/taidi2024/附件1/M101.csv', encoding = 'gbk')
#data_M102 = pd.read_csv('E:/code/taidi/taidi2024/附件1/M102.csv', encoding = 'gbk')


# 数据预处理：计算每天的推出数量、抓取数量和抓取故障次数
daily_data = data_M101.groupby(['月份', '日期']).agg(
    推出累计数=('推出累计数', 'max'),
    抓取累计数=('抓取累计数', 'max'),
    抓取故障次数=('抓取状态', lambda x: (x == -1).sum())
).reset_index()
# 计算相关性矩阵
correlation_matrix = daily_data[['推出累计数', '抓取累计数', '抓取故障次数']].corr()
print("相关性矩阵：\n", correlation_matrix)

# 绘制热力图来展示相关性矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#plt.title('生产线 M101 的变量相关性热力图')
plt.show()

sns.set_theme(style="whitegrid")

# 绘制散点图矩阵，使用cividis配色
pairplot = sns.pairplot(
    daily_data[['推出累计数', '抓取累计数', '抓取故障次数']],
    palette="cividis",
    diag_kind="kde",   # 对角线使用核密度估计
    plot_kws={'alpha': 0.7, 's': 60}  # 设置点的透明度和大小
)

# 设置图表标题和调整布局
#pairplot.fig.suptitle("推出累计数、抓取累计数与抓取故障次数的负相关关系", y=1.02)
plt.show()

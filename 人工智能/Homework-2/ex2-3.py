import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 生成训练集
x_train = np.arange(-1, 1.05, 0.05)
y_train = 1.2 * np.sin(np.pi * x_train) - np.cos(2.4 * np.pi * x_train)

# 生成测试集
x_test = np.arange(-1, 1.01, 0.01)
y_test = 1.2 * np.sin(np.pi * x_test) - np.cos(2.4 * np.pi * x_test)


# 定义隐藏神经元数量
hidden_neurons = [1, 2, 5, 7, 10, 50]

# 创建一个大图，2行3列
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 将axes展平，方便按顺序访问
axes = axes.flatten()

for i, n in enumerate(hidden_neurons):
    # 创建MLP模型
    mlp = MLPRegressor(hidden_layer_sizes=(n,), activation='tanh',
                        solver='adam', max_iter=5000,
                        random_state=123)
    
    # 训练模型
    mlp.fit(x_train.reshape(-1, 1), y_train)
    
    # 预测测试集
    y_pred = mlp.predict(x_test.reshape(-1, 1))
    
    # 在对应的子图里画图
    axes[i].plot(x_test, y_test, label='True Function', color='blue')
    axes[i].plot(x_test, y_pred, label='MLP Prediction', color='red')
    axes[i].set_title(f'MLP with {n} hidden neurons')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].legend()
    
# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
# 定义范围外的输入
x_extrap = np.array([-1.5, 1.5]).reshape(-1, 1)

# 预测
y_extrap = mlp.predict(x_extrap)

print(f'x = -1.5, y = {y_extrap[0]}')
print(f'x = 1.5, y = {y_extrap[1]}')

# 创建一个大图，2行3列
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 将axes展平，方便按顺序访问
axes = axes.flatten()

for i, n in enumerate(hidden_neurons):
    # 创建带有正则化的MLP模型
    mlp = MLPRegressor(hidden_layer_sizes=(n,), activation='tanh',
                        solver='lbfgs', max_iter=5000, alpha=0.001,
                        random_state=123)
    
    # 训练模型
    mlp.fit(x_train.reshape(-1, 1), y_train)
    
    # 预测测试集
    y_pred = mlp.predict(x_test.reshape(-1, 1))
    
    # 在对应的子图里画图
    axes[i].plot(x_test, y_test, label='True Function', color='blue')
    axes[i].plot(x_test, y_pred, label='MLP Prediction', color='red')
    axes[i].set_title(f'MLP with {n} hidden neurons and alpha=0.001')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
    axes[i].legend()
    
# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

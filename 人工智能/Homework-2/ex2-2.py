import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 定义函数
def func(x):
    return 1.2 * np.sin(np.pi * x) - np.cos(2.4 * np.pi * x)

# 生成训练和测试数据
x_train = np.arange(-1, 1, 0.05)
x_test = np.arange(-1, 1, 0.01)

y_train = func(x_train)
y_test = func(x_test)

# 尝试不同的隐藏层神经元数量
neurons = [1, 2, 5, 7, 10, 50]
for n in neurons:
    model = Sequential()
    model.add(Dense(n, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # 训练模型
    model.fit(x_train[:, np.newaxis], y_train, epochs=5000, verbose=0)
    
    # 评估模型
    y_pred_train = model.predict(x_train[:, np.newaxis])
    y_pred_test = model.predict(x_test[:, np.newaxis])
    
    # 绘制训练和测试结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_train, y_train, label='True Function (Train)')
    plt.plot(x_train, y_pred_train, label=f'MLP Prediction (n={n})')
    plt.legend()
    plt.title(f'Training Data with {n} Neurons')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_test, y_test, label='True Function (Test)')
    plt.plot(x_test, y_pred_test, label=f'MLP Prediction (n={n})')
    plt.legend()
    plt.title(f'Test Data with {n} Neurons')
    
    plt.show()

    # 计算 x=-1.5 和 x=1.5 时的输出
    x_out = np.array([-1.5, 1.5]).reshape(-1, 1)
    y_out = model.predict(x_out)
    print(f"Predictions outside the training range with {n} neurons: {y_out}")
import numpy as np
import matplotlib.pyplot as plt

# 激活函数：阶跃函数
def step_function(v):
    return 1 if v > 0 else 0

# 感知机训练算法
def perceptron_train(inputs, targets, eta, epochs):
    weights = np.random.rand(inputs.shape[1])  # 初始化随机权重
    bias = np.random.rand()  # 初始化随机偏置
    weight_history = []

    for epoch in range(epochs):
        for i, x in enumerate(inputs):
            # 计算感知机输出
            v = np.dot(weights, x) + bias
            y = step_function(v)

            # 更新规则
            error = targets[i] - y
            weights += eta * error * x
            bias += eta * error

            # 保存权重和偏置的变化轨迹
            weight_history.append((weights.copy(), bias))

    return weights, bias, weight_history

# 绘制数据分布和决策边界
def plot_xor(inputs, targets, weights, bias, title):
    plt.figure(figsize=(6, 6))
    # 绘制数据点
    for i, (x1, x2) in enumerate(inputs):
        if targets[i] == 0:
            plt.scatter(x1, x2, color="red", label="Class 0" if i == 0 else "")
        else:
            plt.scatter(x1, x2, color="blue", label="Class 1" if i == 0 else "")
    # 绘制决策边界
    x = np.linspace(-0.1, 1.1, 100)
    if len(weights) == 2:
        y = -(weights[0] * x + bias) / weights[1]
        plt.plot(x, y, color="green", label="Decision Boundary")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()

# XOR 数据集
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 输入
targets = np.array([0, 1, 1, 0])  # 目标输出

# 学习率和训练轮数
eta = 1.0
epochs = 10

# 使用感知机算法训练
weights, bias, weight_history = perceptron_train(inputs, targets, eta, epochs)

# 绘制结果
print(f"Final Weights: {weights}, Final Bias: {bias}")
plot_xor(inputs, targets, weights, bias, "XOR with Single-Layer Perceptron")

# 权重轨迹图
w1_history = [w[0][0] for w in weight_history]
w2_history = [w[0][1] for w in weight_history]
bias_history = [w[1] for w in weight_history]

plt.figure()
plt.plot(w1_history, label="w1")
plt.plot(w2_history, label="w2")
plt.plot(bias_history, label="bias")
plt.title("Weight Trajectories for XOR")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.show()

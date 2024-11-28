import numpy as np
import matplotlib.pyplot as plt

# 激活函数：阶跃函数
def step_function(v):
    return 1 if v > 0 else 0

# 感知机训练算法
def perceptron_train(inputs, targets, eta, epochs):
    # 初始化权重和偏置
    weights = np.random.rand(inputs.shape[1])
    bias = np.random.rand()
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

            # 保存每一步的权重和偏置
            weight_history.append((weights.copy(), bias))

    return weights, bias, weight_history

# 绘制权重更新轨迹
def plot_weights(weight_history, title, input_dim):
    w1_history = [w[0][0] for w in weight_history]
    if input_dim > 1:
        w2_history = [w[0][1] for w in weight_history]
    else:
        w2_history = None
    bias_history = [w[1] for w in weight_history]

    plt.figure()
    plt.plot(w1_history, label="w1")
    if w2_history is not None:
        plt.plot(w2_history, label="w2")
    plt.plot(bias_history, label="bias")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# 数据定义
logic_functions = {
    "AND": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "targets": np.array([0, 0, 0, 1]),
    },
    "OR": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "targets": np.array([0, 1, 1, 1]),
    },
    "COMPLEMENT": {
        "inputs": np.array([[0], [1]]),
        "targets": np.array([1, 0]),
    },
}

# 学习率和迭代次数
eta = 1.0
epochs = 20

# 对每个逻辑函数进行训练
for logic, data in logic_functions.items():
    inputs, targets = data["inputs"], data["targets"]
    weights, bias, weight_history = perceptron_train(inputs, targets, eta, epochs)
    print(f"Logic Function: {logic}")
    print(f"Final Weights: {weights}, Final Bias: {bias}")
    plot_weights(weight_history, f"Weight Trajectory for {logic}", inputs.shape[1])

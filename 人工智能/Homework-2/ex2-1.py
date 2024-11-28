import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def target_function(x):
    return 1.2 * torch.sin(np.pi * x) - torch.cos(2.4 * np.pi * x)

# 创建训练集和测试集
x_train = torch.arange(-1, 1.05, 0.05).unsqueeze(1)  # 训练集，转为列向量
y_train = target_function(x_train)

x_test = torch.arange(-1, 1.01, 0.01).unsqueeze(1)   # 测试集
y_test = target_function(x_test)

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, n_hidden):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(1, n_hidden)
        self.output = nn.Linear(n_hidden, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# 定义带正则化的 MLP 模型
class RegularizedMLP(MLP):
    def __init__(self, n_hidden, regularization):
        super(RegularizedMLP, self).__init__(n_hidden)
        self.regularization = regularization

    def get_regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(param**2)
        return self.regularization * reg_loss

# 训练和测试模型的函数
def train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test, epochs=500):
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        if isinstance(model, RegularizedMLP):
            loss += model.get_regularization_loss()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_pred = model(x_test).squeeze()
    return y_pred

# 绘制结果
def plot_results(x_test, y_test, y_pred, title):
    plt.figure()
    plt.plot(x_test.numpy(), y_test.numpy(), label="True Function")
    plt.plot(x_test.numpy(), y_pred.numpy(), label="MLP Output")
    plt.title(title)
    plt.legend()
    plt.show()

# 实验配置
hidden_neurons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]
results = {}

# (a) 顺序模式训练（BP 算法）
for n in hidden_neurons:
    model = MLP(n)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    results[f"BP_n_{n}"] = y_pred
    plot_results(x_test, y_test, y_pred, title=f"MLP with {n} Hidden Neurons (BP)")

# (b) 批量模式训练（带正则化）
for n in hidden_neurons:
    model = RegularizedMLP(n, regularization=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    results[f"Batch_Reg_n_{n}"] = y_pred
    plot_results(x_test, y_test, y_pred, title=f"MLP with {n} Hidden Neurons (Regularized Batch)")

# 将所有 BP 算法结果绘制在一张图上
plt.figure(figsize=(12, 6))
for n in hidden_neurons:
    model = MLP(n)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    plt.plot(x_test.numpy(), y_pred.numpy(), label=f"n={n}")

plt.plot(x_test.numpy(), y_test.numpy(), 'k--', label="True Function")  # 真实函数
plt.title("MLP Outputs with BP Algorithm (Different Hidden Neurons)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# 将所有带正则化结果绘制在一张图上
plt.figure(figsize=(12, 6))
for n in hidden_neurons:
    model = RegularizedMLP(n, regularization=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    plt.plot(x_test.numpy(), y_pred.numpy(), label=f"n={n}")

plt.plot(x_test.numpy(), y_test.numpy(), 'k--', label="True Function")  # 真实函数
plt.title("MLP Outputs with Regularization (Different Hidden Neurons)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

bp_neurons = [1, 2, 3, 4, 5, 7, 9, 10, 50]
plt.figure(figsize=(15, 15))
for i, n in enumerate(bp_neurons):
    plt.subplot(3, 3, i + 1)
    model = MLP(n)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    plt.plot(x_test.numpy(), y_test.numpy(), 'k--', label="True Function")  # 真实函数
    plt.plot(x_test.numpy(), y_pred.numpy(), label=f"n={n}")  # 模型输出
    plt.title(f"BP Algorithm (n={n})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()

# 带正则化的 3x3 组合图
plt.figure(figsize=(15, 15))
for i, n in enumerate(bp_neurons):
    plt.subplot(3, 3, i + 1)
    model = RegularizedMLP(n, regularization=0.01)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    y_pred = train_and_evaluate(model, optimizer, criterion, x_train, y_train, x_test, y_test)
    plt.plot(x_test.numpy(), y_test.numpy(), 'k--', label="True Function")  # 真实函数
    plt.plot(x_test.numpy(), y_pred.numpy(), label=f"n={n}")  # 模型输出
    plt.title(f"Regularized (n={n})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()

# 范围外数据预测
x_out_of_range = torch.tensor([[-1.5], [1.5]])
with torch.no_grad():
    for n in hidden_neurons:
        model = MLP(n)
        y_out_of_range = model(x_out_of_range)
        print(f"Predictions for x = -1.5 and x = 1.5 (n={n}): {y_out_of_range.flatten().numpy()}")

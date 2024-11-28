import numpy as np
import matplotlib.pyplot as plt

# 数据集
data = np.array([[0.8, -1], [1.6, -4], [3.0, -5], [4.0, -6], [5.0, -9]])
X = data[:, 0]
d = data[:, 1]

# LLS 方法
X_design = np.vstack([X, np.ones_like(X)]).T  # 构造设计矩阵
w, b = np.linalg.lstsq(X_design, d, rcond=None)[0]  # 求解 w 和 b

# 画出拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(X, d, color="blue", label="Data Points")
plt.plot(X, w * X + b, color="red", label=f"LLS Fit: y = {w:.2f}x + {b:.2f}")
plt.title("Linear Fit using Least Squares Method")
plt.xlabel("x")
plt.ylabel("d")
plt.legend()
plt.grid()
plt.show()

# LMS 方法
np.random.seed(0)  # 固定随机种子
w_lms, b_lms = np.random.randn(), np.random.randn()  # 随机初始化权重和偏置
learning_rate = 0.02
epochs = 200

# 存储权重和偏置的变化轨迹
w_trajectory, b_trajectory = [w_lms], [b_lms]

# LMS 训练
for epoch in range(epochs):
    for i in range(len(X)):
        y_pred = w_lms * X[i] + b_lms
        error = d[i] - y_pred
        w_lms += learning_rate * error * X[i]
        b_lms += learning_rate * error
        w_trajectory.append(w_lms)
        b_trajectory.append(b_lms)

# 画出拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(X, d, color="blue", label="Data Points")
plt.plot(X, w_lms * X + b_lms, color="green", label=f"LMS Fit: y = {w_lms:.2f}x + {b_lms:.2f}")
plt.title("Linear Fit using LMS Algorithm")
plt.xlabel("x")
plt.ylabel("d")
plt.legend()
plt.grid()
plt.show()

# 画出权重和偏置的变化轨迹
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(w_trajectory, color="purple")
plt.title("Weight (w) Trajectory")
plt.xlabel("Iterations")
plt.ylabel("Weight (w)")

plt.subplot(1, 2, 2)
plt.plot(b_trajectory, color="orange")
plt.title("Bias (b) Trajectory")
plt.xlabel("Iterations")
plt.ylabel("Bias (b)")

plt.tight_layout()
plt.show()


# 打印权重和偏置
print(f"LLS Method: w = {w:.4f}, b = {b:.4f}")
print(f"LMS Method: w = {w_lms:.4f}, b = {b_lms:.4f}")

# 画出两种方法的拟合结果
plt.figure(figsize=(8, 6))
plt.scatter(X, d, color="blue", label="Data Points")
plt.plot(X, w * X + b, color="red", label="LLS Fit")
plt.plot(X, w_lms * X + b_lms, color="green", linestyle="--", label="LMS Fit")
plt.title("Comparison of LLS and LMS Methods")
plt.xlabel("x")
plt.ylabel("d")
plt.legend()
plt.grid()
plt.show()




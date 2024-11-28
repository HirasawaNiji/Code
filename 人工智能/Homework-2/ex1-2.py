import numpy as np
import matplotlib.pyplot as plt

# 定义 Rosenbrock 函数及其梯度和 Hessian 矩阵
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

def rosenbrock_hessian(x, y):
    hxx = 2 - 400 * (y - x**2) + 1200 * x**2
    hxy = -400 * x
    hyy = 200
    return np.array([[hxx, hxy], [hxy, hyy]])

# 梯度下降法实现
def gradient_descent(start, learning_rate, tol=1e-6, max_iter=10000):
    path = [start]
    x, y = start
    for _ in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        x, y = np.array([x, y]) - learning_rate * grad
        path.append([x, y])
        if rosenbrock(x, y) < tol:
            break
    return np.array(path), len(path) - 1

# 牛顿法实现
def newton_method(start, tol=1e-6, max_iter=10000):
    path = [start]
    x, y = start
    for _ in range(max_iter):
        grad = rosenbrock_gradient(x, y)
        hess = rosenbrock_hessian(x, y)
        step = np.linalg.solve(hess, grad)
        x, y = np.array([x, y]) - step
        path.append([x, y])
        if rosenbrock(x, y) < tol:
            break
    return np.array(path), len(path) - 1

# 设置初始条件
np.random.seed(42)
start_point = np.random.rand(2)  # 随机初始化点 (x, y) ∈ (0, 1)

# 梯度下降法
path_gd, iter_gd = gradient_descent(start=start_point, learning_rate=0.001)

# 牛顿法
path_newton, iter_newton = newton_method(start=start_point)

# 绘图
plt.figure(figsize=(12, 6))

# 梯度下降法路径图
plt.subplot(1, 2, 1)
x_vals, y_vals = path_gd[:, 0], path_gd[:, 1]
plt.plot(x_vals, y_vals, marker="o", label="Gradient Descent Path")
plt.scatter(1, 1, color="red", label="Global Minimum (1,1)")
plt.title("Gradient Descent Path")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# 牛顿法路径图
plt.subplot(1, 2, 2)
x_vals, y_vals = path_newton[:, 0], path_newton[:, 1]
plt.plot(x_vals, y_vals, marker="o", label="Newton's Method Path")
plt.scatter(1, 1, color="red", label="Global Minimum (1,1)")
plt.title("Newton's Method Path")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 输出迭代次数
print(iter_gd, iter_newton)

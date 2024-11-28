import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def gradient(x, y):
    df_dx = -2 * (1 - x) - 400 * x * (y - x**2)
    df_dy = 200 * (y - x**2)
    return np.array([df_dx, df_dy])

def gradient_descent(eta, tol=1e-6, max_iter=10000):
    np.random.seed(42)
    x, y = np.random.rand(2)  # 随机初始化
    trajectory = [(x, y)]
    for _ in range(max_iter):
        grad = gradient(x, y)
        x, y = x - eta * grad[0], y - eta * grad[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory)

# 参数
eta_small = 0.001
eta_large = 0.1

# 运行梯度下降
trajectory_small = gradient_descent(eta_small)
trajectory_large = gradient_descent(eta_large)

# 绘制轨迹
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(trajectory_small[:, 0], trajectory_small[:, 1], marker='o', markersize=3, label='Small Eta')
plt.title("Small Learning Rate (η=0.001)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(trajectory_large[:, 0], trajectory_large[:, 1], marker='o', markersize=3, label='Large Eta')
plt.title("Large Learning Rate (η=0.1)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()

def hessian(x, y):
    d2f_dx2 = 2 - 400 * (y - 3 * x**2)
    d2f_dy2 = 200
    d2f_dxdy = -400 * x
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])

def newton_method(tol=1e-6, max_iter=10000):
    np.random.seed(42)
    x, y = np.random.rand(2)
    trajectory = [(x, y)]
    for _ in range(max_iter):
        grad = gradient(x, y)
        hess = hessian(x, y)
        delta = np.linalg.solve(hess, -grad)
        x, y = x + delta[0], y + delta[1]
        trajectory.append((x, y))
        if np.linalg.norm(grad) < tol:
            break
    return np.array(trajectory)

trajectory_newton = newton_method()

# 绘制牛顿法轨迹
plt.figure()
plt.plot(trajectory_newton[:, 0], trajectory_newton[:, 1], marker='o', markersize=3, label='Newton Method')
plt.title("Newton's Method Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

library(VGAM)

# 设置参数
a <- 2
b <- 2
n <- 1000  # 样本量

# 逆变换法生成Pareto分布随机样本
u <- runif(n)  # 生成[0,1]均匀分布的随机数
x <- b / (u^(1/a))  # 逆变换公式

# 绘制密度直方图和使用VGAM包中的dpowerpareto函数绘制Pareto分布密度曲线
hist(x, probability = TRUE, main = "Pareto分布样本的密度直方图", col = "lightblue")
curve(dpareto(x, shape = a, scale = b), col = "red", add = TRUE)


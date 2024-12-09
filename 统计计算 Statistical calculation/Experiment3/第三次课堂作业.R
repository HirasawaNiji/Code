# 定义找零函数
coinChange <- function(coins, amount) {
  dp <- rep(amount + 1, amount + 1)  # 初始化 dp 数组，填充大于任何可能答案的值
  dp[1] <- 0
  
  for (a in 1:amount) {
    for (coin in coins) {
      if (a >= coin) {
        dp[a + 1] <- min(dp[a + 1], dp[a - coin + 1] + 1)
      }
    }
  }
  
  if (dp[amount + 1] > amount) {
    return(-1)
  } else {
    return(dp[amount + 1])
  }
}
  
  # 测试输入和输出
  test_cases <- list(
    list(coins = c(1, 2, 5), amount = 14),  # 测试1
    list(coins = c(3, 7, 8), amount = 25),  # 测试2
    list(coins = c(2, 5, 7), amount = 30)   # 测试3
  )
  
  # 遍历测试用例并输出结果
  for (i in seq_along(test_cases)) {
    coins <- test_cases[[i]]$coins
    amount <- test_cases[[i]]$amount
    result <- coinChange(coins, amount)
    cat("输入: coins =", coins, ", amount =", amount, "\n")
    cat("输出:", result, "\n\n")
  }
  
  # 加载随机数种子（可选，不同运行结果会不同）
  set.seed(123)
  
  # 找到最后一个山峰的函数
  findLastPeak <- function(heights) {
    n <- length(heights)
    
    # 从倒数第二个元素开始检查，最后一个元素不算
    for (i in seq(n - 1, 2, by = -1)) {
      if (heights[i] > heights[i - 1] && heights[i] > heights[i + 1]) {
        return(c("山峰高度" = heights[i], "位置" = i))
      }
    }
    
    return(NULL)
  }
  
  # 随机生成5个数组并找到每个数组的最后一个山峰
  for (j in 1:5) {
    random_heights <- sample(0:20, 10, replace = TRUE)
    cat("第", j, "个随机数组:", random_heights, "\n")
    
    result <- findLastPeak(random_heights)
    cat("最后一个山峰的高度:", result["山峰高度"], "位置:", result["位置"], "\n\n")
  }
  
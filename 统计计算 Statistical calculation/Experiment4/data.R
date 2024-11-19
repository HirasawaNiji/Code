# 1. 读取本地数据 data.csv
data <- read.csv("D:/统计计算实验/第四次上机/data.csv", header = TRUE, stringsAsFactors = FALSE)

# 2. 查看数据前 10 行
print("数据前 10 行：")
print(head(data, 10))

# 3. 读取 salary 列
if ("salary" %in% names(data)) {
  library(stringr)  # 加载 stringr 包用于字符串处理
  
  # 提取 salary 列的最小值和最大值
  salary_split <- str_extract_all(data$salary, "\\d+")  # 提取数字部分
  min_salary <- sapply(salary_split, function(x) ifelse(length(x) > 0, as.numeric(x[1]), NA))  # 最小值
  max_salary <- sapply(salary_split, function(x) ifelse(length(x) > 1, as.numeric(x[2]), NA))  # 最大值
  
  # 将提取的最小值和最大值作为新列
  data$min_salary <- min_salary
  data$max_salary <- max_salary
  
  # 计算平均值并替换 salary 列
  avg_salary <- rowMeans(cbind(min_salary, max_salary), na.rm = TRUE)
  data$salary <- avg_salary  # 替换 salary 列为平均值
  
  # 查看数据后 10 行
  print("前后 10 行（salary 列已处理为平均值）：")
  print(head(data, 10))
  print(tail(data, 10))
}

# 4. 根据学历分组，计算平均工资并打印出来
if ("education" %in% names(data)) {
  library(dplyr)
  
  avg_salary_by_education <- data %>%
    group_by(education) %>%
    summarise(avg_salary = mean(salary, na.rm = TRUE))  
  
  print("按学历分组的平均工资：")
  print(avg_salary_by_education)
} 
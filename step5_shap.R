# 加载必要的包
library(fastshap)
library(shapviz)
library(dplyr)
library(ggplot2)

# 读取清洗后的数据
heart_data <- read.csv("data/heart_clean.csv")

# 将分类变量转为因子（与步骤4的模型一致）
heart_data$sex <- as.factor(heart_data$sex)
heart_data$cp <- as.factor(heart_data$cp)
heart_data$fbs <- as.factor(heart_data$fbs)
heart_data$restecg <- as.factor(heart_data$restecg)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope <- as.factor(heart_data$slope)
heart_data$thal <- as.factor(heart_data$thal)
heart_data$target <- as.factor(heart_data$target)

# 拟合逻辑回归模型（与步骤4相同）
model <- glm(target ~ ., data = heart_data, family = binomial())

# 定义预测包装器（输出概率）
pred_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "response")
}

# 选择用于 SHAP 计算的数据（全部数据，若慢可采样前200行）
X_explain <- heart_data %>% dplyr::select(-target)

set.seed(123)
shap_values <- fastshap::explain(
  model,
  X = X_explain,
  pred_wrapper = pred_wrapper,
  nsim = 50,      # 蒙特卡洛模拟次数，可适当增减
  adjust = TRUE
)

# 转换为 shapviz 对象
shp <- shapviz(shap_values, X = X_explain)

# 保存 SHAP summary plot (beeswarm)
png("output/shap_summary.png", width = 8, height = 6, units = "in", res = 150)
sv_importance(shp, kind = "beeswarm") +
  labs(title = "SHAP Summary Plot - MACE Prediction")
dev.off()

# 保存 SHAP bar plot (特征重要性)
png("output/shap_bar.png", width = 8, height = 5, units = "in", res = 150)
sv_importance(shp, kind = "bar") +
  labs(title = "SHAP Feature Importance (Bar Plot)")
dev.off()

# 打印完成信息
print("✅ 步骤5完成：SHAP 图已保存至 output/shap_summary.png 和 output/shap_bar.png")
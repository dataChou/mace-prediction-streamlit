# 1. 加载包
library(fastshap)
library(shapviz)
library(dplyr)
library(ggplot2)

# 2. 读取数据
heart_data <- read.csv("data/heart_clean.csv")

# 3. 将分类变量转为因子（与 Python 独热编码对应）
# 注意：这些变量在原始数据中虽然是数字编码，但本质是分类
heart_data$sex <- as.factor(heart_data$sex)
heart_data$cp <- as.factor(heart_data$cp)
heart_data$fbs <- as.factor(heart_data$fbs)
heart_data$restecg <- as.factor(heart_data$restecg)
heart_data$exang <- as.factor(heart_data$exang)
heart_data$slope <- as.factor(heart_data$slope)
heart_data$thal <- as.factor(heart_data$thal)

# 目标变量转为因子（二分类）
heart_data$target <- as.factor(heart_data$target)

# 4. 拟合逻辑回归（glm 会自动创建虚拟变量，参考水平为第一个因子水平）
model <- glm(target ~ ., data = heart_data, family = binomial())

# 查看模型摘要（可选）
summary(model)

# 5. 定义预测包装器（输出概率）
pred_wrapper <- function(object, newdata) {
  predict(object, newdata, type = "response")
}

# 6. 准备解释数据（去除 target 列）
X_explain <- heart_data %>% dplyr::select(-target)

# 7. 计算 SHAP 值（使用所有数据，若较慢可采样前200行）
set.seed(123)
shap_values <- fastshap::explain(
  model,
  X = X_explain,
  pred_wrapper = pred_wrapper,
  nsim = 50,
  adjust = TRUE
)

# 8. 转换为 shapviz 对象
shp <- shapviz(shap_values, X = X_explain)

# 9. 保存 SHAP summary plot (beeswarm)
png("output/shap_summary.png", width = 8, height = 6, units = "in", res = 150)
sv_importance(shp, kind = "beeswarm") +
  labs(title = "SHAP Summary Plot - MACE Prediction (Corrected)")
dev.off()

# 10. 保存 SHAP bar plot (特征重要性)
png("output/shap_bar.png", width = 8, height = 5, units = "in", res = 150)
sv_importance(shp, kind = "bar") +
  labs(title = "SHAP Feature Importance (Bar Plot)")
dev.off()

# 11. 打印完成信息
print("修正后的 SHAP 图已保存至 output/shap_summary.png 和 output/shap_bar.png")